import { pipelines, type CameraBase, type Feed, Particle, particles, tone } from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

/** Feed texels per side of a cell that can shed an ember. */
const CELL = 4;
const WORKGROUP = 64;
const CAPACITY = 16_384;

/** Upward pull in feed texels per second squared: embers rise like sparks over a fire. */
const LIFT = 70;
const DRAG = 1.1;
const SPEED = { least: 18, most: 60, across: 22 };
const LIFE_SECONDS = { least: 0.9, most: 2.6 };
const SIZE_TEXELS = 2.6;
/** Sideways push of the air the embers drift through, in texels per second squared. */
const SWIRL = 160;
const SWIRL_SCALE = 0.012;

const HOT = d.vec3f(6, 2.6, 0.6);
const COOL = d.vec3f(2.4, 0.4, 0.04);

export const Shedding = d.struct({
  /** Change in brightness, from 0 to 1, that counts as motion. */
  sensitivity: d.f32,
  /** Chance a moving edge cell sheds an ember per camera frame. */
  rate: d.f32,
  /** Mixes each ember's colour from fire, 0, to the colour of what shed it, 1. */
  tint: d.f32,
  /** Seconds of simulation, which move the air. */
  seconds: d.f32,
  /** 0 on the first frame after a reset, which only records brightness. */
  primed: d.u32,
});

export const SHEDDING = { sensitivity: 0.08, rate: 0.35, tint: 0.12, seconds: 0, primed: 0 };

/**
 * Sheds embers from the feed's moving edges. Brightness is kept per cell between camera frames,
 * and a cell sheds when its brightness changes where the picture has an edge.
 */
export function createEmbers(
  root: TgpuRoot,
  options: {
    readonly camera: CameraBase;
    readonly step: number;
    readonly feed: Feed;
    readonly size: readonly [number, number];
  },
) {
  const { camera, step, feed, size } = options;
  const grid = d.vec2u(size[0] / CELL, size[1] / CELL);
  const cells = grid.x * grid.y;
  const height = size[1];

  const reading = feed.texture.createView(d.texture2d(d.f32));
  const brightness = root.createMutable(d.arrayOf(d.f32, cells));
  const settings = root.createUniform(Shedding, SHEDDING);

  const drift = (ember: Particle, dt: number) => {
    'use gpu';
    const next = Particle(ember);
    const air = d.vec2f(next.position.x, next.position.y) * SWIRL_SCALE;
    const push = perlin2d.sample(air + d.vec2f(0, settings.$.seconds * 0.4));
    next.velocity.x += push * SWIRL * dt;
    return next;
  };

  const embers = particles.create(root, camera, {
    capacity: CAPACITY,
    step,
    dimensions: 2,
    update: drift,
    shade: particles.glowingDisc,
    settings: { gravity: [0, LIFT, 0], drag: DRAG },
  });

  const lumaAt = (texel: d.v2i) => {
    'use gpu';
    const at = std.clamp(texel, d.vec2i(), d.vec2i(size[0] - 1, size[1] - 1));
    return tone.luminance(std.textureLoad(reading.$, at, 0).rgb);
  };

  const shed = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP],
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= cells) return;
    const cell = d.vec2i(d.i32(gid.x % grid.x), d.i32(gid.x / grid.x));
    const centre = cell * CELL + d.i32(CELL / 2);

    const here = (lumaAt(centre) + lumaAt(centre - 1)) * 0.5;
    const across = lumaAt(centre + d.vec2i(CELL, 0)) - lumaAt(centre - d.vec2i(CELL, 0));
    const down = lumaAt(centre + d.vec2i(0, CELL)) - lumaAt(centre - d.vec2i(0, CELL));
    const edge = std.length(d.vec2f(across, down));

    const before = brightness.$[gid.x];
    brightness.$[gid.x] = here;
    if (settings.$.primed === 0) return;

    const sensitivity = settings.$.sensitivity;
    const moved = std.smoothstep(sensitivity * 0.5, sensitivity * 2, std.abs(here - before));
    const outlined = std.smoothstep(0.04, 0.25, edge);
    randf.seed3(d.vec3f(d.vec2f(cell), settings.$.seconds));
    if (randf.sample() >= moved * outlined * settings.$.rate) return;

    const texel = d.vec2f(centre) + (randf.inUnitCircle() * CELL) / 2;
    const seen = srgbToLinear(std.textureLoad(reading.$, d.vec2i(texel), 0).rgb);
    const heat = randf.sample();
    const fire = std.mix(COOL, HOT, heat);
    const shade = seen * (tone.luminance(fire) / std.max(tone.luminance(seen), 0.05));
    const speed = SPEED.least + (SPEED.most - SPEED.least) * randf.sample();
    embers.emit(
      Particle({
        position: d.vec3f(texel.x, height - texel.y, 0),
        age: 0,
        velocity: d.vec3f((randf.sample() * 2 - 1) * SPEED.across, speed, 0),
        life: LIFE_SECONDS.least + (LIFE_SECONDS.most - LIFE_SECONDS.least) * randf.sample(),
        color: d.vec4f(std.mix(fire, shade, settings.$.tint), 1),
        size: SIZE_TEXELS * (0.6 + heat * 0.8),
        seed: randf.sample(),
      }),
    );
  });

  const shedder = root.createComputePipeline({ compute: shed });
  const groups = Math.ceil(cells / WORKGROUP);
  let primed = false;

  return {
    embers,
    settings,
    initAsync: () => pipelines.compile([embers, shedder]),

    /** Compares a new picture with the last and sheds embers where it moved. */
    shed(pass: TgpuComputePass, seconds: number) {
      settings.patch({ seconds, primed: primed ? 1 : 0 });
      primed = true;
      pass.setPipeline(shedder);
      pass.dispatchWorkgroups(groups);
    },

    /** Forgets the last picture, so the next only records brightness. */
    reset() {
      primed = false;
    },

    destroy() {
      embers.destroy();
      brightness.buffer.destroy();
    },
  };
}
