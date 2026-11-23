import { cameras, clock, environment, particles, type CameraBase } from '@typegpu/kit';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';

// Colours are in linear light
const ABYSS = d.vec3f(0, 0.002, 0.007);
const DEEP = d.vec3f(0.003, 0.025, 0.045);
const SURFACE = d.vec3f(0.3, 0.62, 0.66);
const SNOW = d.vec3f(0.55, 0.7, 0.75);
const CYAN = d.vec3f(0.1, 1, 0.8);
const BLUE = d.vec3f(0.1, 0.35, 1);

/** Direction toward the light, above and right of a camera resting on +z. */
const SUN = std.normalize(d.vec3f(0.75, 1, -0.45));
const SIDE = std.normalize(std.cross(SUN, d.vec3f(0, 0, 1)));
const FRONT = std.cross(SIDE, SUN);

/** Half-extents of the snow volume. Large enough to contain the camera. */
const BOX = d.vec3f(14, 8, 14);
/** Height of the water's surface, which glints far above. */
const SURFACE_Y = 9;
/** The share of flecks that are bioluminescent motes. */
const MOTES = 0.006;
/** Minimum and maximum snow-fleck width in focus, and mote width, in world units. */
const FLECK_SIZE = [0.018, 0.058] as const;
const MOTE_SIZE = 0.22;
/** Minimum and maximum snow sinking speed in world units per second. */
const SINK = [0.08, 0.2] as const;
/** World units a second the pointer drags a fleck at most. */
const DRAG = 1.5;
/** Drag decay rate per second. After one second, `exp(-SETTLE)` of the pointer drag remains. */
const SETTLE = 1.3;

/** Maximum pointer speed in view widths per second. Limits impulses when the pointer enters. */
const FLING = 3;
/** Pointer-velocity smoothing rate per second. */
const EASING = 10;
/** Pointer speed threshold in view widths per second. Slower motion counts as still. */
const STILL = 1e-3;

export const Water = d.struct({
  /** Brightness of the light shafts. */
  shafts: d.f32,
  /** Brightness of the light glinting through the waves on the surface far above. */
  glints: d.f32,
  /** Brightness of the marine snow. */
  snow: d.f32,
  /** Brightness of the bioluminescent motes. */
  motes: d.f32,
  /** Water extinction per world unit. Transmitted light is `exp(-fog × distance)`. */
  fog: d.f32,
  /** Focus distance in world units. Nearer flecks blur. */
  focus: d.f32,
  /** Blur growth in world units per unit closer than the focus distance. */
  blur: d.f32,
  /** How strongly the pointer drags the flecks it passes. */
  stir: d.f32,
});

const Pointer = d.struct({
  /** Pointer coordinates from 0 to 1, measured from the view top left. */
  at: d.vec2f,
  /** Pointer velocity in view units per second. */
  motion: d.vec2f,
});

/**
 * Light-shaft brightness along a direction. Shafts run parallel to the light and fan outward in
 * perspective. Surface motion makes them sway.
 */
const shafts = (ray: d.v3f, t: number) => {
  'use gpu';
  const toward = std.saturate(std.dot(ray, SUN));
  const across = ray - SUN * std.dot(ray, SUN);
  const around = std.normalize(d.vec2f(std.dot(across, SIDE), std.dot(across, FRONT)) + 0.0001);
  const broad = perlin2d.sample(around * 2.5 + d.vec2f(t * 0.021, t * -0.013));
  const fine = perlin2d.sample(around * 13 + d.vec2f(t * -0.04, t * 0.05) + 11);
  const beams = std.smoothstep(-0.15, 0.6, broad) * std.saturate(0.35 + fine * 1.2);

  // Blur shafts near the light instead of letting them pinch to a point
  const blurred = std.mix(0.35, beams, std.smoothstep(0.03, 0.35, std.length(across)));

  return blurred * toward * toward * toward * std.smoothstep(-0.2, 0.6, ray.y);
};

/** Snow-fleck colour and blur flattening from 0 to 1. */
const Fleck = particles.schema('Fleck', { flat: d.f32 });
type Fleck = d.InferGPU<typeof Fleck>;

const isMote = (fleck: Fleck) => {
  'use gpu';
  return fleck.seed > 1 - MOTES;
};

/** Draws a fleck as a soft dot flattening into a blurred disc, or a glowing mote core and halo. */
const speck = (fleck: Fleck, local: d.v2f) => {
  'use gpu';
  const r2 = std.dot(local, local);
  const dome = std.saturate(1 - r2);
  const snow = std.mix(dome * dome, std.smoothstep(1, 0.55, r2) * 0.55, fleck.flat);
  const mote = std.exp(-r2 * 9) + dome * dome * 0.08;
  const shape = std.select(snow, mote, isMote(fleck));

  return d.vec4f(fleck.color.xyz * shape, 0);
};

export const WATER = {
  shafts: 0.55,
  glints: 3,
  snow: 0.35,
  motes: 1,
  fog: 0.13,
  focus: 5,
  blur: 0.035,
  stir: 0.45,
};

/**
 * Draws deep water with light shafts, surface glints and `count` flecks of marine snow and
 * glowing motes, stepped every `step` seconds. The pointer stirs the particles.
 */
export function createAbyss(
  root: TgpuRoot,
  options: { camera: CameraBase; time: TgpuUniform<d.F32>; step: number; count: number },
) {
  const { camera, time, step, count } = options;
  const settings = root.createUniform(Water, WATER);
  const pointer = root.createUniform(Pointer);

  // Cached Perlin gradients avoid per-sample hashing
  const noise = perlin2d.staticCache({ root, size: d.vec2u(64, 64) });

  /** The water's light along `ray`: deep blue below, shafts and the glinting surface above. */
  const backdrop = (ray: d.v3f) => {
    'use gpu';
    const toward = std.saturate(std.dot(ray, SUN));
    const t = time.$;

    let color = std.mix(ABYSS, DEEP, std.smoothstep(-0.5, 0.8, ray.y));
    color += SURFACE * (std.pow(toward, 6) * 0.08 + std.pow(toward, 24) * 0.25);
    color += SURFACE * (shafts(ray, t) * settings.$.shafts);

    if (ray.y > 0.3) {
      const distance = (SURFACE_Y - cameras.current.$.position.y) / ray.y;
      const at = cameras.current.$.position.xz + ray.xz * distance;
      const swell = perlin2d.sampleWithGradient(at * 0.8 + d.vec2f(t * 0.11, t * 0.06)).yz;
      const ripple = perlin2d.sampleWithGradient(at * 5 + d.vec2f(t * -0.21, t * 0.15) + 5).yz;
      const slope = swell * 0.2 + ripple * 0.09;
      const bent = std.normalize(ray + d.vec3f(slope.x, 0, slope.y));
      const glint = std.pow(std.saturate(std.dot(bent, SUN)), 50);
      const fade = std.exp(-distance * settings.$.fog) * std.smoothstep(0.3, 0.8, ray.y);
      color += SURFACE * (glint * fade * settings.$.glints);
    }

    return color;
  };

  /**
   * Wraps sinking snow and hovering motes around the volume. Pointer drag decays toward their
   * normal drift. Distance from the eye controls size, colour and blur from birth.
   */
  const drift = (fleck: Fleck, dt: number) => {
    'use gpu';
    randf.seed(fleck.seed);
    const grain = randf.sample();
    const phase = randf.sample() * 40;
    const sink = randf.sample();
    const hue = randf.sample();
    const rate = randf.sample();
    const mote = isMote(fleck);
    const age = fleck.age;

    const sway = d.vec3f(std.sin(age * 0.5 + phase) * 0.07, 0, std.cos(age * 0.4 + phase) * 0.05);
    const hover = std.sin(age * 0.23 + phase) * 0.03;
    const calm = sway + d.vec3f(0, std.select(-std.mix(SINK[0], SINK[1], sink), hover, mote), 0);

    // Drag water near the pointer ray only while the pointer moves
    const view = cameras.current.$;
    const offset = fleck.position - view.position;
    const ray = cameras.rayOf(view, d.vec2f(pointer.$.at.x, 1 - pointer.$.at.y) * 2 - 1).direction;
    const along = std.dot(offset, ray);
    const aside = offset - ray * along;
    const near = std.exp(-std.dot(aside, aside) / (0.006 * along * along + 0.001));
    const depth = std.smoothstep(0.5, 2, along) * std.smoothstep(14, 5, along);
    const stirring = std.min(1, std.length(pointer.$.motion) * 30);
    const motion = (pointer.$.motion * d.vec2f(2, -2)) / cameras.lensOf(view);
    const pushed = (view.right * motion.x + view.up * motion.y) * (along * settings.$.stir);
    const drag = pushed * std.min(1, DRAG / (std.length(pushed) + 0.0001));
    const pull = std.saturate(near * depth * stirring * 7 * dt);
    const velocity = calm + (fleck.velocity - calm) * std.exp(-SETTLE * dt);

    const position = std.fract((fleck.position + BOX) / (BOX * 2)) * (BOX * 2) - BOX;
    const inside = BOX - std.abs(position);
    const distance = std.length(offset);
    // Keep motes away from the eye to prevent large foreground halos
    const fade =
      std.exp(-distance * settings.$.fog) *
      std.smoothstep(0.4, std.select(d.f32(1.4), 3.5, mote), distance) *
      std.smoothstep(0, 1.5, std.min(inside.x, std.min(inside.y, inside.z)));

    // Conserve light as blur spreads a fleck over a larger disc. Keep distant flecks at least a
    // pixel or two wide to prevent flicker
    const sharp = std.select(std.mix(FLECK_SIZE[0], FLECK_SIZE[1], grain * grain), MOTE_SIZE, mote);
    const blurred = sharp + settings.$.blur * std.max(settings.$.focus - distance, 0);
    const size = std.max(blurred, distance * 0.0025);
    const spread = (sharp * sharp) / (size * size);

    const direction = offset / distance;
    const backlit = std.pow(std.saturate(std.dot(direction, SUN)), 6);
    const light = 0.5 + 3 * shafts(direction, time.$) + backlit * 1.5;
    const height = 0.7 + 0.5 * std.smoothstep(-BOX.y, BOX.y, position.y);
    const glow = 0.5 + 0.5 * std.sin(age * (0.4 + rate * 0.6) + phase);
    const pulse = 0.25 + 2 * glow * glow * glow * glow;

    const snow = SNOW * (settings.$.snow * light * height);
    const lit = std.mix(CYAN, BLUE, hue) * (settings.$.motes * pulse);
    const shade = std.select(snow, lit, mote) * (fade * spread);
    const flat = std.saturate((size - sharp) / (size * 0.6));

    return Fleck({
      position,
      age,
      velocity: velocity + (calm + drag - velocity) * pull,
      life: fleck.life,
      color: d.vec4f(shade, 1),
      size,
      seed: fleck.seed,
      flat,
    });
  };

  const snow = particles.create(root, camera, {
    capacity: count,
    particle: Fleck,
    step,
    blend: 'add',
    update: drift,
    born: (fleck: Fleck) => {
      'use gpu';
      return drift(fleck, d.f32(0));
    },
    shade: speck,
    // `drift` controls particle movement, so disable gravity and drag
    settings: { gravity: [0, 0, 0], drag: 0 },
  });

  // Write pointer position and smoothed velocity only when they change
  const pointed = { at: d.vec2f(), motion: d.vec2f() };
  const aim = { x: 0.5, y: 0.5, seen: false };
  const last = { x: 0.5, y: 0.5 };
  const eased = { x: 0, y: 0 };
  let moving = false;

  return {
    /** The water settings. */
    settings,
    /** The snow and motes. Add their drawing after `water`. */
    snow,

    /** Pointer coordinates from 0 to 1, measured from the view top left. */
    point(x: number, y: number) {
      if (!aim.seen) {
        last.x = x;
        last.y = y;
      }
      aim.x = x;
      aim.y = y;
      aim.seen = true;
    },

    /** Smooths pointer velocity and writes changes. Call once per frame. */
    update(dt: number) {
      if (!(dt > 0) || !aim.seen) return;

      const dx = aim.x - last.x;
      const dy = aim.y - last.y;
      const limit = FLING / Math.max(FLING, Math.hypot(dx, dy) / dt);
      const ease = clock.ease(dt, EASING);
      eased.x += ((dx / dt) * limit - eased.x) * ease;
      eased.y += ((dy / dt) * limit - eased.y) * ease;
      last.x = aim.x;
      last.y = aim.y;

      const still = Math.hypot(eased.x, eased.y) < STILL;
      if (still) eased.x = eased.y = 0;
      if (still && !moving && dx === 0 && dy === 0) return;

      pointed.at.x = aim.x;
      pointed.at.y = aim.y;
      pointed.motion.x = eased.x;
      pointed.motion.y = eased.y;
      pointer.write(pointed);
      moving = !still;
    },

    /** Queues initial snow for the next `spawn`. */
    fill() {
      snow.burst({ count, position: [0, 0, 0], area: [BOX.x, BOX.y, BOX.z], life: 1e9 });
    },

    /** The water behind everything, for a description piped with the camera. */
    water: environment.background(tgpu.fn(backdrop).pipe(noise.inject())),

    destroy() {
      snow.destroy();
      noise.destroy();
    },
  };
}
