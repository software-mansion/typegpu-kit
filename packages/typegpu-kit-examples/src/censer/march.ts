import {
  cameras,
  pipelines,
  shading,
  targets,
  type CameraBase,
  type Frame,
  type Lighting,
  type SizeSource,
  type post,
} from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { randf } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import type { Censer } from './censer.ts';
import { dispatch, groups, kernel, volume } from './cells.ts';
import type { Smoke } from './smoke.ts';

/** Cells along each edge of a brick, the unit the march skips empty air in. */
const BRICK = 8;
/**
 * Smoke stops light more than in proportion to how much there is, so a dense core shades into
 * wisps that let the dark through: `EXTINCTION` per metre at `REFERENCE` smoke, rising as its
 * `CURVE` power, and easing off toward `DENSEST` per metre so even the core shows the shadows
 * inside it. Smoke under `EMPTY` stops nothing, and bricks holding no more are skipped.
 */
const EXTINCTION = 2;
const REFERENCE = 0.05;
const CURVE = 1.5;
const DENSEST = 6;
const EMPTY = 0.002;
/** Toward the key, smoke shades as if this many times denser, so its folds show light and shadow. */
const SHADOW = 3;
/** How much light reaches a point through smoke: a share that scatters on through thick smoke. */
const DEEP = 0.2;
const DEEP_SHARE = 0.2;
/** The censer's body stops light this much per metre, its edge softened over `SOFT` metres. */
const BODY = 60;
const SOFT = 0.06;
/** A light cell looks toward the key in steps of this many cells, at most `LIGHT_STEPS` of them. */
const LIGHT_STRIDE = 2;
const LIGHT_STEPS = 128;
/**
 * The march's steps in metres: short where the density changes fast, so edges stay sharp, and up
 * to `FAR` through even smoke, each changing it by about `EDGE` per metre. Never more than
 * `MOST_STEPS` a pixel.
 */
const NEAR = 0.01;
const FAR = 0.04;
const EDGE = 0.4;
const MOST_STEPS = 400;
/** Smoke fades out this far inside the box's open faces, so it leaves without an edge. */
const FADE = 0.24;
/** A broad lobe and a forward one, with its share, so smoke before the light glows at its rim. */
const BROAD = 0.2;
const FORWARD = 0.75;
const FORWARD_SHARE = 0.15;
/** The faint cool light smoke takes in its shade, as a share of the key's. */
const SHADE = d.vec3f(0.035, 0.04, 0.05);
/** The embers' light on the smoke near the vents: colour, strength and reach in metres. */
const EMBER = d.vec3f(1, 0.36, 0.09);
const EMBER_POWER = 0.5;
const EMBER_REACH = 0.2;

/**
 * The march runs at this fraction of the output's size, each of its pixels marching up to the
 * nearest surface among the output pixels it covers, and an output pixel takes the marched ones
 * around it whose depths match its own within `MATCH` of its distance.
 */
const REDUCE = 2;
const MATCH = 0.03;

const marchLayout = tgpu.bindGroupLayout({
  depth: { texture: d.textureDepth2d() },
});
const compositeLayout = tgpu.bindGroupLayout({
  picture: { texture: d.texture2d(d.f32) },
  depth: { texture: d.textureDepth2d() },
  marched: { texture: d.texture2d(d.f32) },
  nearest: { texture: d.texture2d(d.f32), sampleType: 'unfilterable-float' },
});

/** Henyey and Greenstein's phase function, times 4π so light scattered evenly is 1. */
const lobe = (cosine: number, g: number) => {
  'use gpu';
  return (1 - g * g) / std.pow(1 + g * g - 2 * g * cosine, 1.5);
};

/** The density that stops light at a point, per metre, from the smoke it holds. */
const densityOf = (smoke: number) => {
  'use gpu';
  const thick = std.max(smoke - EMPTY, 0) / REFERENCE;
  const linear = std.pow(thick, CURVE) * EXTINCTION;
  return DENSEST * (1 - std.exp(-linear / DENSEST));
};

/**
 * The smoke lit by one key light from the scene's sun and by the embers in the censer, marched at
 * half the output's size through each pixel's view up to the scene's depth and laid over the
 * picture by depth.
 */
export function createMarch(
  root: TgpuRoot,
  smoke: Smoke,
  censer: Censer,
  options: {
    view: CameraBase;
    light: Lighting;
    size: SizeSource;
    picture: () => post.FloatTexture;
    depth: () => targets.Depth;
  },
) {
  const { view, light, size, picture, depth } = options;
  const [nx, ny, nz] = smoke.cells;
  const bricks = [nx / BRICK, ny / BRICK, nz / BRICK] as const;
  const output = targets.create(root, size, { hdr: { format: 'rgba16float', sampled: true } });
  const reduced = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true, scale: 1 / REDUCE },
    nearest: { format: 'r32float', sampled: true, scale: 1 / REDUCE },
  });
  const smooth = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  // Per cell of the air's grid the key light let through to it and the embers' light; per brick
  // the most smoke near it
  const lit = volume(root, smoke.cells, 'rgba16float', 'march light');
  const occupied = volume(root, bricks, 'rgba16float', 'march bricks');

  const low = d.vec3f(...smoke.low);
  const high = d.vec3f(...smoke.high);
  const { cell } = smoke;
  const extent = d.vec3f(nx * cell, ny * cell, nz * cell);
  const brickSize = cell * BRICK;
  const brickCount = d.vec3i(...bricks);

  /** How much of the smoke at a point shows, fading out toward the box's open faces. */
  const fadeAt = (position: d.v3f) => {
    'use gpu';
    const inside = std.min(position - low, high - position);
    const edge = std.min(std.min(inside.x, inside.y), inside.z);
    return std.smoothstep(0, FADE, edge);
  };
  /** The smoke at a point, faded out toward the box's open faces. */
  const smokeAt = (position: d.v3f) => {
    'use gpu';
    return smoke.smokeAt(position) * fadeAt(position);
  };

  /**
   * The smoke averaged over the air's cells at a point, sampled between them, so light and bricks
   * need not read every finer cell.
   */
  const coarseAt = (position: d.v3f) => {
    'use gpu';
    const uvw = (position - low) / extent;
    return std.textureSampleLevel(smoke.averaged.$, smooth.$, uvw, 0).x * fadeAt(position);
  };

  /**
   * For each cell, the key light that reaches it through the smoke and the censer, and the embers'.
   * Only cells in bricks holding smoke look toward the key: the march reads no others.
   */
  const lightCells = kernel(({ id }) => {
    'use gpu';
    if (id.x >= nx || id.y >= ny || id.z >= nz) return;
    const at = low + (d.vec3f(id) + 0.5) * cell;
    const ember = censer.middle() + censer.turn(d.vec3f(0, 0.18, 0));
    const near = std.length(at - ember) / EMBER_REACH;
    const glow = EMBER_POWER / (1 + near * near);
    if (std.textureLoad(occupied.read.$, d.vec3i(id) / BRICK, 0).x < EMPTY) {
      std.textureStore(lit.write.$, id, d.vec4f(1, glow, 0, 0));
      return;
    }
    const toward = shading.towardsSun();
    let depthOf = d.f32(0);
    for (let i = d.i32(0); i < LIGHT_STEPS; i++) {
      const q = at + toward * ((d.f32(i) + 0.5) * cell * LIGHT_STRIDE);
      if (std.any(std.gt(q, high)) || std.any(std.lt(q, low))) break;
      // The censer's body casts a shadow, softened over a few centimetres at its edge
      const body = std.saturate(0.5 - censer.distance(q) / SOFT);
      depthOf += (densityOf(coarseAt(q)) * SHADOW + body * BODY) * cell * LIGHT_STRIDE;
    }
    const through = std.mix(std.exp(-depthOf), std.exp(-depthOf * DEEP), DEEP_SHARE);
    std.textureStore(lit.write.$, id, d.vec4f(through, glow, 0, 0));
  });

  /** For each brick, the most smoke in it and a cell around it, where the march must look. */
  const occupancy = kernel(({ id }) => {
    'use gpu';
    if (id.x >= bricks[0] || id.y >= bricks[1] || id.z >= bricks[2]) return;
    const first = d.vec3i(id) * BRICK - 1;
    const last = d.vec3i(nx - 1, ny - 1, nz - 1);
    let most = d.f32(0);
    for (let z = d.i32(0); z < BRICK + 2; z++) {
      for (let y = d.i32(0); y < BRICK + 2; y++) {
        for (let x = d.i32(0); x < BRICK + 2; x++) {
          const at = std.clamp(first + d.vec3i(x, y, z), d.vec3i(), last);
          const shown = fadeAt(low + (d.vec3f(at) + 0.5) * cell);
          most = std.max(most, std.textureLoad(smoke.averaged.$, at, 0).x * shown);
        }
      }
    }
    std.textureStore(occupied.write.$, id, d.vec4f(most, 0, 0, 0));
  });

  /** The view distance of the nearest surface among the output pixels a reduced one covers. */
  const nearestOf = (reducedPixel: d.v2i) => {
    'use gpu';
    const last = d.vec2i(std.textureDimensions(marchLayout.$.depth)) - 1;
    let nearest = d.f32(3.4e38);
    for (const y of tgpu.unroll([0, 1])) {
      for (const x of tgpu.unroll([0, 1])) {
        const pixel = std.min(reducedPixel * REDUCE + d.vec2i(x, y), last);
        nearest = std.min(
          nearest,
          cameras.distanceOf(cameras.current.$, std.textureLoad(marchLayout.$.depth, pixel, 0)),
        );
      }
    }
    return nearest;
  };

  const march = tgpu.fragmentFn({
    in: { uv: d.vec2f, position: d.builtin.position },
    // The smoke a reduced pixel marched, and the view distance it marched up to in `x`
    out: { smoke: d.vec4f, nearest: d.vec4f },
  })(({ uv, position }) => {
    'use gpu';
    const camera = cameras.current.$;
    const ray = cameras.rayOf(camera, uv * d.vec2f(2, -2) + d.vec2f(-1, 1));
    const nearest = nearestOf(d.vec2i(position.xy));
    const end = nearest / std.max(0.1, std.dot(ray.direction, cameras.forwardOf(camera)));

    // Where the view crosses the smoke's box
    const inverse = d.vec3f(1) / ray.direction;
    const near = (low - ray.origin) * inverse;
    const far = (high - ray.origin) * inverse;
    const enter = std.max(
      std.max(std.min(near.x, far.x), std.min(near.y, far.y)),
      std.max(std.min(near.z, far.z), 0),
    );
    const exit = std.min(
      std.min(std.max(near.x, far.x), std.max(near.y, far.y)),
      std.min(std.max(near.z, far.z), end),
    );
    if (exit <= enter) return { smoke: d.vec4f(0, 0, 0, 1), nearest: d.vec4f(nearest, 0, 0, 0) };

    randf.seed2(std.floor(position.xy));
    const offset = randf.sample();
    const env = shading.settings();
    const cosine = std.dot(ray.direction, shading.towardsSun());
    const phase = std.mix(lobe(cosine, BROAD), lobe(cosine, FORWARD), FORWARD_SHARE);
    const key = env.sunColor * (env.sunIntensity * phase);
    const shade = SHADE * env.sunIntensity;

    let gathered = d.vec3f();
    let through = d.f32(1);
    let t = enter + offset * NEAR;
    let step = d.f32(NEAR);
    let before = d.f32(0);
    for (let i = d.i32(0); i < MOST_STEPS; i++) {
      if (t >= exit || through < 0.004) break;
      const world = ray.origin + ray.direction * t;
      const brick = std.clamp(
        d.vec3i(std.floor((world - low) / brickSize)),
        d.vec3i(),
        brickCount - 1,
      );
      if (std.textureLoad(occupied.read.$, brick, 0).x < EMPTY) {
        // Skip to where the view leaves the brick
        const corner = low + d.vec3f(brick) * brickSize;
        const a = (corner - ray.origin) * inverse;
        const b = (corner + brickSize - ray.origin) * inverse;
        const out = std.min(std.min(std.max(a.x, b.x), std.max(a.y, b.y)), std.max(a.z, b.z));
        t = std.max(out, t) + offset * NEAR + 1e-4;
        step = NEAR;
        before = 0;
        continue;
      }
      const density = densityOf(smokeAt(world).x);
      step = std.clamp((EDGE * step) / (std.abs(density - before) + 1e-4), NEAR, FAR);
      before = density;
      if (density > 1e-3) {
        // The last step is cut where the view meets the censer or leaves the box
        const length = std.min(step, exit - t);
        const here = std.textureSampleLevel(lit.read.$, smooth.$, (world - low) / extent, 0);
        const fade = std.exp(-density * length);
        const scattered = key * here.x + shade + EMBER * here.y;
        gathered += scattered * (through * (1 - fade));
        through *= fade;
      }
      t += step;
    }
    return { smoke: d.vec4f(gathered, through), nearest: d.vec4f(nearest, 0, 0, 0) };
  });

  /** The smoke marched at the reduced size over the picture, at the output's. */
  const composite = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })(({ position }) => {
    'use gpu';
    const pixel = d.vec2i(position.xy);
    const seen = std.textureLoad(compositeLayout.$.picture, pixel, 0).xyz;
    const own = cameras.distanceOf(
      cameras.current.$,
      std.textureLoad(compositeLayout.$.depth, pixel, 0),
    );
    const last = d.vec2i(std.textureDimensions(compositeLayout.$.marched)) - 1;
    const at = position.xy / REDUCE - 0.5;
    const base = d.vec2i(std.floor(at));
    const f = at - std.floor(at);
    let sum = d.vec4f();
    let weight = d.f32(0);
    let best = d.vec4f(0, 0, 0, 1);
    let closest = d.f32(3.4e38);
    for (const y of tgpu.unroll([0, 1])) {
      for (const x of tgpu.unroll([0, 1])) {
        const marched = std.clamp(base + d.vec2i(x, y), d.vec2i(), last);
        const gap = std.abs(std.textureLoad(compositeLayout.$.nearest, marched, 0).x - own);
        const value = std.textureLoad(compositeLayout.$.marched, marched, 0);
        const near = std.mix(1 - f, f, d.vec2f(x, y));
        const w = near.x * near.y * std.exp(-gap / (MATCH * own));
        sum += value * w;
        weight += w;
        if (gap < closest) {
          closest = gap;
          best = d.vec4f(value);
        }
      }
    }
    const laid = std.select(best, sum / weight, weight > 1e-4);
    return d.vec4f(seen * laid.w + laid.xyz, 1);
  });

  const lit3d = root.pipe(view.inject()).pipe(light.inject());
  const kernels = {
    light: lit3d.createComputePipeline({ compute: lightCells }),
    occupancy: root.createComputePipeline({ compute: occupancy }),
    march: lit3d.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: march,
      targets: { smoke: { format: 'rgba16float' }, nearest: { format: 'r32float' } },
    }),
    composite: root.pipe(view.inject()).createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: composite,
      targets: { format: 'rgba16float' },
    }),
  };
  const marching = memo((deep: targets.Depth, matter: Smoke['matter']) =>
    kernels.march.with(root.createBindGroup(marchLayout, { depth: deep })).with(matter),
  );
  const compositing = memo(
    (color: post.FloatTexture, deep: targets.Depth, marched: typeof reduced.textures) =>
      kernels.composite.with(
        root.createBindGroup(compositeLayout, {
          picture: color,
          depth: deep,
          marched: marched.hdr,
          nearest: marched.nearest,
        }),
      ),
  );
  const attachments = { colorAttachments: { view: output.views.hdr } };
  const reducedAttachments = {
    colorAttachments: [{ view: reduced.views.hdr }, { view: reduced.views.nearest }],
  };
  const cellGroups = groups(smoke.cells);
  const brickGroups = groups(bricks);

  return {
    get output() {
      return output.textures.hdr;
    },
    initAsync: () => pipelines.compile(Object.values(kernels)),
    /** Records finding the bricks the smoke fills and lighting it, after the smoke's `advect`. */
    light(pass: TgpuComputePass) {
      pass.setPipeline(kernels.occupancy);
      dispatch(pass, brickGroups);
      pass.setPipeline(kernels.light);
      dispatch(pass, cellGroups);
    },
    /** Marches the smoke at the reduced size, then lays it over the picture into `output`. */
    record(frame: Frame) {
      reduced.fit();
      output.fit();
      reducedAttachments.colorAttachments[0].view = reduced.views.hdr;
      reducedAttachments.colorAttachments[1].view = reduced.views.nearest;
      const marchPass = frame.encoder.beginRenderPass(reducedAttachments);
      marchPass.setPipeline(marching(depth(), smoke.matter));
      marchPass.draw(3);
      marchPass.end();
      attachments.colorAttachments.view = output.views.hdr;
      const pass = frame.encoder.beginRenderPass(attachments);
      pass.setPipeline(compositing(picture(), depth(), reduced.textures));
      pass.draw(3);
      pass.end();
    },
    destroy() {
      output.destroy();
      reduced.destroy();
      lit.texture.destroy();
      occupied.texture.destroy();
    },
  };
}
