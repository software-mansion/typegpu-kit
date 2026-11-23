import { cameras, Particle, particles, pipelines, shading, type Lighting } from '@typegpu/kit';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

type Vec3 = readonly [number, number, number];

/** Flames, embers and smoke puffs a second from each brazier. */
const RATE = { flames: 60, embers: 5, smoke: 5 };
/** Where the flames are hottest, white-yellow, then orange, then a dull red as they fade. */
const CORE = d.vec3f(7, 4.4, 1.6);
const BODY = d.vec3f(3.2, 0.9, 0.15);
const DYING = d.vec3f(0.5, 0.06, 0.01);
const EMBER = [7, 2.6, 0.6, 1] as const;
/** Metres across a flame and an ember. Anything smaller than `EMBER_BELOW` is an ember. */
const FLAME_SIZE = 0.3;
const EMBER_SIZE = 0.03;
const EMBER_BELOW = 0.06;
/** A flame's height over its width. */
const TALL = 2;
/** The share of the light reaching it that a puff of smoke scatters, and its opacity. */
const SOOT = 0.0015;
const PUFF = 0.22;
/** Metres a second a puff widens by, and the breeze that leans the smoke along x. */
const SPREAD = 0.3;
const BREEZE = 0.12;

/** Flames draw in towards the middle as they climb; embers flutter. */
const rise = (flame: Particle) => {
  'use gpu';
  let next = Particle(flame);
  const flutter = flame.age * 9 + flame.seed * 40;
  if (flame.size < EMBER_BELOW) {
    next.velocity += d.vec3f(std.sin(flutter) * 0.06, 0.02, std.cos(flutter) * 0.06);
  } else {
    next.velocity = d.vec3f(next.velocity.x * 0.9, next.velocity.y + 0.035, next.velocity.z * 0.9);
  }
  return next;
};

const burn = (flame: Particle, local: d.v2f) => {
  'use gpu';
  if (flame.size < EMBER_BELOW) return particles.glowingDisc(flame, local);
  const age = flame.age / flame.life;
  // A tongue: round at its foot, narrowing to a tip
  const up = local.y * 0.5 + 0.5;
  const across = local.x / std.max(1 - up * up * 0.85, 0.05);
  const edge = std.saturate(1 - std.length(d.vec2f(across, local.y)));
  const heat = std.mix(CORE, BODY, std.smoothstep(0, 0.45, age));
  const color = std.mix(heat, DYING, std.smoothstep(0.45, 1, age));
  const fade = edge * edge * std.smoothstep(0, 0.1, age) * (1 - age);
  return d.vec4f(color * (fade * 0.5), 0);
};

/** Flames stand upright, twice as tall as wide, and shrink as they rise; embers stay round. */
const shrink = (flame: Particle, corner: d.v2f) => {
  'use gpu';
  const tall = std.select(d.f32(1), TALL, flame.size >= EMBER_BELOW);
  const across = cameras.current.$.right * corner.x + d.vec3f(0, corner.y * tall, 0);
  const size = flame.size * (1 - (flame.age / flame.life) * 0.6);
  return flame.position + across * (size * 0.5);
};

/** The moon lights the smoke as little as it lights the dark under the braziers. */
const sunless: shading.SunShadowHook = () => {
  'use gpu';
  return d.f32(0);
};

/**
 * Flames and embers rising from each brazier rim in `rims`, and smoke above them that their light
 * lights.
 */
export function createFire(
  root: TgpuRoot,
  view: cameras.CameraBase,
  light: Lighting,
  options: { rims: readonly Vec3[]; step: number },
) {
  const { rims, step } = options;
  const fire = particles.create(root, view, {
    capacity: 768,
    step,
    update: rise,
    place: shrink,
    shade: burn,
    label: 'fire',
    settings: { gravity: [0, 0, 0], drag: 0.6 },
  });
  const smoke = particles.create(root, view, {
    capacity: 384,
    step,
    blend: 'over',
    label: 'smoke',
    update: tgpu
      .fn((puff: Particle, dt: number) => {
        'use gpu';
        let next = Particle(puff);
        next.size = puff.size + SPREAD * dt;
        next.velocity = d.vec3f(BREEZE, puff.velocity.y, puff.velocity.z);
        next.color = d.vec4f(shading.incoming(puff.position) * SOOT, puff.color.w);
        return next;
      })
      .pipe(light.inject())
      .with(shading.sunShadow, sunless),
    settings: { gravity: [0, 0, 0], drag: 0.15 },
  });

  for (const [x, y, z] of rims) {
    const flames = {
      position: [x, y + 0.02, z],
      area: [0.25, 0.04, 0.25],
      velocity: [0, 0.8, 0],
      scatter: 0.14,
      size: FLAME_SIZE,
      life: 0.65,
      variance: 0.4,
    } as const;
    fire.stream(RATE.flames, flames);
    fire.stream(RATE.embers, {
      position: [x, y + 0.2, z],
      area: [0.2, 0.05, 0.2],
      velocity: [0, 1.7, 0],
      scatter: 0.5,
      color: EMBER,
      size: EMBER_SIZE,
      life: 3,
      variance: 0.5,
    });
    const puff = {
      position: [x, y + 1.5, z],
      area: [0.15, 0.1, 0.15],
      velocity: [0, 0.6, 0],
      scatter: 0.08,
      color: [0, 0, 0, PUFF],
      size: 0.5,
      life: 6,
      variance: 0.3,
    } as const;
    // The first frame finds the braziers alight, a puff of smoke over each
    fire.burst({ ...flames, count: RATE.flames * flames.life, aged: true });
    smoke.burst({ ...puff, count: 1 });
    smoke.stream(RATE.smoke, puff);
  }

  return {
    initAsync: () => pipelines.compile([fire, smoke]),
    /** Smoke first, so the flames add their light over it. */
    drawings: [smoke.drawing, fire.drawing],

    /** Spawns what the streams owe, takes `steps` steps and lists the particles to draw. */
    record(pass: TgpuComputePass, steps: number, ahead: number) {
      fire.spawn(pass);
      smoke.spawn(pass);
      for (let s = 0; s < steps; s++) {
        fire.step(pass);
        smoke.step(pass);
      }
      fire.compact(pass, ahead);
      smoke.compact(pass, ahead);
    },

    destroy() {
      fire.destroy();
      smoke.destroy();
    },
  };
}
