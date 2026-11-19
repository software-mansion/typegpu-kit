import {
  cameras,
  history,
  Particle,
  particles,
  passes,
  pipelines,
  shading,
  targets,
  type CameraBase,
  type Depth,
  type Frame,
  type Lighting,
  type SharedBindings,
  type Targets,
} from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { perlin3d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

type Vec3 = readonly [number, number, number];

/**
 * The air the beams light, as a grid of cells along the view: columns across the screen, and
 * slices spaced by distance, finer near the eye, from `NEAREST` to `FARTHEST` metres. Each frame
 * samples the sun's shadow at `SAMPLES` points in each cell and keeps `BLEND` of them over what
 * the cell held before. The blinds stripe the light finer than a cell, so a cell needs many
 * samples, spread over the frames by `SPREAD`, to settle.
 */
const GRID = [160, 90, 64] as const;
const NEAREST = 0.1;
const FARTHEST = 14;
const BLEND = 0.05;
const SAMPLES = 4;
const WORKGROUP = 8;
/** Roberts' R3 steps, which move a cell's samples on each time so they cover it evenly. */
const SPREAD = d.vec3f(0.8191725, 0.6710436, 0.5497005);
/** How strongly dust scatters light forward, from 0, the same every way, towards 1. */
const FORWARD = 0.4;
/** The share of the sunlight crossing a metre of air that the dust scatters. */
const DENSITY = 0.35;
/** How many motes drift in the room, and how long each lives, in seconds. */
const MOTES = 4000;
const MOTE_LIFE = 40;
/** A mote's width in metres and how much brighter it shines than the air around it. */
const MOTE_SIZE = 0.004;
const MOTE_GLOW = 6;
/** How fast the air turns the motes, in metres per second squared, and over what span, in metres. */
const SWIRL = 0.02;
const EDDY = 1.5;
/** The room's air, inside its walls, floor and ceiling, as its lowest and highest corners. */
const LOW: Vec3 = [-2.55, 0.02, -3.45];
const HIGH: Vec3 = [3.7, 2.95, 4.55];

const Air = d.struct({
  /** The room's air as a box, its lowest and highest corners. */
  low: d.vec3f,
  frame: d.f32,
  high: d.vec3f,
  /** The share of this frame's light in each cell, 1 when the grid holds none from before. */
  blend: d.f32,
});
const air = tgpu.accessor(Air);
const ScatterLayout = tgpu.bindGroupLayout({
  before: { texture: d.texture3d(d.f32) },
  into: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
  smooth: { sampler: 'filtering' },
});
const SumLayout = tgpu.bindGroupLayout({
  scattered: { texture: d.texture3d(d.f32) },
  summed: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
});
const BeamsLayout = tgpu.bindGroupLayout({
  summed: { texture: d.texture3d(d.f32) },
  depth: { texture: d.textureDepth2d() },
  smooth: { sampler: 'filtering' },
});
const Seen = d.struct({ color: d.vec4f });

/** Henyey and Greenstein's phase function: the share of light scattered at an angle's cosine. */
const phase = (cosine: number) => {
  'use gpu';
  const g2 = FORWARD * FORWARD;
  return (1 - g2) / (4 * Math.PI * std.pow(1 + g2 - 2 * FORWARD * cosine, 1.5));
};

/** The sunlight dust scatters towards an eye looking along `direction`, per metre of air lit. */
const scatteredAlong = (direction: d.v3f) => {
  'use gpu';
  const env = shading.settings();
  return (
    env.sunColor * (env.sunIntensity * DENSITY * phase(std.dot(shading.towardsSun(), direction)))
  );
};

const sunlitAt = (position: d.v3f) => {
  'use gpu';
  const at = shading.Point();
  at.position = d.vec3f(position);
  return shading.sunShare(at);
};

const clipBefore = history.previously(cameras.toClip);

/** The distance from the eye along the view of a slice's place in the grid, from 0 to 1. */
const distanceAt = (slice: number) => {
  'use gpu';
  return NEAREST * std.pow(FARTHEST / NEAREST, slice);
};
const sliceAt = (distance: number) => {
  'use gpu';
  return std.log(std.max(distance, NEAREST) / NEAREST) / std.log(FARTHEST / NEAREST);
};

const rayAt = (uv: d.v2f) => {
  'use gpu';
  return cameras.rayOf(cameras.current.$, d.vec2f(uv.x * 2 - 1, 1 - uv.y * 2));
};

/**
 * Lights each cell of the grid by the sunlight the dust scatters towards the eye, sampled at a
 * few points in the cell, over what the cell held last frame where it was then.
 */
const scatter = tgpu.computeFn({
  workgroupSize: [WORKGROUP, WORKGROUP, 1],
  in: { id: d.builtin.globalInvocationId },
})(({ id }) => {
  'use gpu';
  if (id.x >= GRID[0] || id.y >= GRID[1] || id.z >= GRID[2]) return;
  const cells = d.vec3f(GRID[0], GRID[1], GRID[2]);
  randf.seed3(d.vec3f(id));
  const start = d.vec3f(randf.sample(), randf.sample(), randf.sample());
  let light = d.vec3f();
  for (let i = d.i32(0); i < SAMPLES; i++) {
    const at =
      (d.vec3f(id) + std.fract(start + (air.$.frame * SAMPLES + d.f32(i)) * SPREAD)) / cells;
    const ray = rayAt(at.xy);
    const ahead = std.dot(ray.direction, cameras.forwardOf(cameras.current.$));
    const point = ray.origin + ray.direction * (distanceAt(at.z) / ahead);
    const inside = std.all(std.ge(point, air.$.low)) && std.all(std.le(point, air.$.high));
    if (inside) light += scatteredAlong(ray.direction) * (sunlitAt(point) / SAMPLES);
  }

  // Find where the cell's centre was in last frame's grid
  const centre = (d.vec3f(id) + 0.5) / cells;
  const middle = rayAt(centre.xy);
  const along =
    distanceAt(centre.z) / std.dot(middle.direction, cameras.forwardOf(cameras.current.$));
  const clip = clipBefore(middle.origin + middle.direction * along);
  const before = d.vec3f((clip.xy / clip.w) * d.vec2f(0.5, -0.5) + 0.5, sliceAt(clip.w));
  const kept = std.all(std.ge(before, d.vec3f())) && std.all(std.le(before, d.vec3f(1)));
  if (kept) {
    const was = std.textureSampleLevel(ScatterLayout.$.before, ScatterLayout.$.smooth, before, 0);
    light = std.mix(was.xyz, light, air.$.blend);
  }
  std.textureStore(ScatterLayout.$.into, id, d.vec4f(light, 1));
});

/** Sums each column's cells from the eye outwards, so a cell holds the light in front of it. */
const sum = tgpu.computeFn({
  workgroupSize: [WORKGROUP, WORKGROUP, 1],
  in: { id: d.builtin.globalInvocationId },
})(({ id }) => {
  'use gpu';
  if (id.x >= GRID[0] || id.y >= GRID[1]) return;
  const ray = rayAt((d.vec2f(id.xy) + 0.5) / d.vec2f(GRID[0], GRID[1]));
  const ahead = std.dot(ray.direction, cameras.forwardOf(cameras.current.$));
  let total = d.vec3f();
  let near = d.f32(0);
  for (let k = d.u32(0); k < GRID[2]; k++) {
    const far = distanceAt(d.f32(k + 1) / GRID[2]);
    const cell = std.textureLoad(SumLayout.$.scattered, d.vec3u(id.xy, k), 0).xyz;
    total += cell * ((far - near) / ahead);
    std.textureStore(SumLayout.$.summed, d.vec3u(id.xy, k), d.vec4f(total, 1));
    near = far;
  }
});

/** Adds the light the air scatters in front of each pixel's surface onto the scene. */
const over: passes.Drawing<typeof passes.ScreenVertex, d.Infer<typeof Seen>> = {
  ...passes.fullScreen,
  label: 'sunbeams',
  pixelOf: (corner) => {
    'use gpu';
    const size = std.textureDimensions(BeamsLayout.$.depth);
    const texel = std.min(d.vec2u(corner.uv * d.vec2f(size)), size - 1);
    const depth = std.textureLoad(BeamsLayout.$.depth, d.vec2i(texel), 0);
    // A cell's centre holds the light up to its far side
    const slice = sliceAt(cameras.distanceOf(cameras.current.$, depth)) - 0.5 / GRID[2];
    const at = d.vec3f(corner.uv, std.max(slice, 0.5 / GRID[2]));
    const light = std.textureSampleLevel(BeamsLayout.$.summed, BeamsLayout.$.smooth, at, 0).xyz;
    return Seen({ color: d.vec4f(light, 0) });
  },
  color: passes.shaded,
  blend: pipelines.blend.add,
  draw(pass) {
    pass.draw(3);
  },
};

/** A mote turns slowly on the air, a different eddy for each. */
const drift = (mote: Particle, dt: number) => {
  'use gpu';
  const at = mote.position / EDDY + d.vec3f(mote.seed * 7, mote.age * 0.05, 0);
  const push = d.vec3f(
    perlin3d.sample(at),
    perlin3d.sample(at + d.vec3f(31, 0, 0)),
    perlin3d.sample(at + d.vec3f(0, 0, 57)),
  );
  const next = Particle(mote);
  next.velocity = mote.velocity + push * (SWIRL * dt);
  const inside =
    std.all(std.ge(mote.position, air.$.low)) && std.all(std.le(mote.position, air.$.high));
  if (!inside) next.life = 0;
  return next;
};

/** A mote glints where the sun reaches it, most when the eye looks towards the sun. */
const glint = (mote: Particle, local: d.v2f) => {
  'use gpu';
  const edge = std.saturate(1 - std.length(local));
  const t = mote.age / mote.life;
  const fade = std.saturate(t * 10) * std.saturate((1 - t) * 10);
  const direction = std.normalize(mote.position - cameras.current.$.position);
  const light = scatteredAlong(direction) * ((sunlitAt(mote.position) * MOTE_GLOW) / DENSITY);
  return d.vec4f(light * (edge * edge * fade * mote.color.w), 0);
};

export interface AirOptions {
  readonly camera: CameraBase;
  readonly light: Lighting;
  /** The targets the scene drew, whose `hdr` takes the beams. */
  readonly screen: Targets<{ hdr: targets.Spec }>;
  /** The scene's depth, where each pixel's view of the air ends. */
  readonly depth: () => Depth;
  /** The bindings the scene shares, which hold the sun's shadow. */
  readonly shared: SharedBindings;
  /** The length of a simulation step in seconds. */
  readonly step: number;
}

/**
 * The dust in the classroom's air: beams where the sun's shadow lets its light through the
 * windows, gathered in a grid along the view, and motes drifting through them.
 */
export function createAir(root: TgpuRoot, options: AirOptions) {
  const { camera: view, light, screen, depth, shared, step } = options;
  const settings = root.createUniform(Air, { low: LOW, frame: 0, high: HIGH, blend: 1 });

  const volume = () =>
    root
      .createTexture({ size: [...GRID], format: 'rgba16float', dimension: '3d' })
      .$usage('storage', 'sampled');
  const scattered = [volume(), volume()] as const;
  const summed = volume();
  const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const sampled = (texture: ReturnType<typeof volume>) => texture.createView(d.texture3d(d.f32));
  const writes = (texture: ReturnType<typeof volume>) =>
    texture.createView(d.textureStorage3d('rgba16float', 'write-only'));
  // Each frame scatters into one grid over the other, then sums the one it wrote
  const scatterings = [0, 1].map((i) =>
    root.createBindGroup(ScatterLayout, {
      before: sampled(scattered[1 - i]),
      into: writes(scattered[i]),
      smooth,
    }),
  );
  const sums = [0, 1].map((i) =>
    root.createBindGroup(SumLayout, { scattered: sampled(scattered[i]), summed: writes(summed) }),
  );
  const lit = root.pipe((cfg) => light.inject()(view.inject()(cfg)).with(air, settings));
  const scattering = lit.createComputePipeline({ compute: scatter });
  const summing = lit.createComputePipeline({ compute: sum });

  const adding = passes
    .create(root, screen, { hdr: passes.color }, { label: 'sunbeams' })
    .pipe(view.inject());
  const summedView = sampled(summed);
  const beamsOf = memo((deep: Depth) =>
    root.createBindGroup(BeamsLayout, {
      summed: summedView,
      depth: deep.createView(d.textureDepth2d()),
      smooth,
    }),
  );
  const added = adding.add({
    ...over,
    draw(pass) {
      pass.setBindGroup(BeamsLayout, beamsOf(depth()));
      pass.draw(3);
    },
  });

  const motes = particles.create(root, view, {
    label: 'motes',
    capacity: MOTES,
    step,
    blend: 'add',
    settings: { gravity: [0, -0.003, 0], drag: 0.5 },
    update: tgpu.fn(drift).pipe((cfg) => cfg.with(air, settings)),
    shade: tgpu
      .fn(glint)
      .pipe(light.inject())
      .pipe((cfg) => cfg.with(air, settings)),
  });
  const center: Vec3 = [(LOW[0] + HIGH[0]) / 2, (LOW[1] + HIGH[1]) / 2, (LOW[2] + HIGH[2]) / 2];
  const area: Vec3 = [(HIGH[0] - LOW[0]) / 2, (HIGH[1] - LOW[1]) / 2, (HIGH[2] - LOW[2]) / 2];
  const cloud = { position: center, area, size: MOTE_SIZE, life: MOTE_LIFE, variance: 0.5 };
  motes.burst({ ...cloud, count: MOTES, aged: true });
  motes.stream(MOTES / MOTE_LIFE, cloud);

  const groups = [Math.ceil(GRID[0] / WORKGROUP), Math.ceil(GRID[1] / WORKGROUP), GRID[2]] as const;
  let last: number | undefined;

  return {
    /** The motes, for the scene's pass. */
    drawing: motes.drawing,
    /** Spawns, moves and lists the motes, `steps` fixed steps, then shows them `ahead` seconds on. */
    simulate(pass: TgpuComputePass, steps: number, ahead: number) {
      motes.spawn(pass);
      for (let i = 0; i < steps; i++) motes.step(pass);
      motes.compact(pass, ahead);
    },
    /** Lights the air and adds it onto the scene the frame drew. Record it after the scene's pass. */
    record(frame: Frame) {
      // The grid before is missing when the last frame lit no air
      const kept = last !== undefined && frame.index === last + 1;
      settings.patch({ frame: frame.index % 1024, blend: kept ? BLEND : 1 });
      const into = frame.index % 2;
      const pass = frame.encoder.beginComputePass();
      pass.setBindGroup(shared.layout, shared.group());
      pass.setPipeline(scattering);
      pass.setBindGroup(ScatterLayout, scatterings[into]);
      pass.dispatchWorkgroups(...groups);
      pass.setPipeline(summing);
      pass.setBindGroup(SumLayout, sums[into]);
      pass.dispatchWorkgroups(groups[0], groups[1]);
      pass.end();
      adding.record(frame, added, { clear: false });
      last = frame.index;
    },
    initAsync: () => pipelines.compile([scattering, summing, adding, motes]),
    destroy() {
      for (const texture of [...scattered, summed]) texture.destroy();
      motes.destroy();
    },
  };
}
