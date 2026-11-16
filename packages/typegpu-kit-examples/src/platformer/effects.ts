import { images, Particle, particles, pipelines, type CameraBase } from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import { d, std, type TgpuRoot, type TgpuComputePass } from 'typegpu';
import type { Picture } from '../contract.ts';
import { pixelatedMote, TEXELS } from './sprites.ts';

/** Dust colour and opacity. Opaque dust lies on the front card, beside the sprites. */
export const DUST = [0.82, 0.76, 0.66, 1] as const;
const GOLD = [1, 0.85, 0.35, 1] as const;

/**
 * Dust a slide sprays ahead of it: motes a second, tiles from the hero's centre to its leading
 * foot, and tiles a second each flies faster than the hero and up.
 */
const SLIDING = { rate: 50, foot: 0.4, ahead: 2, up: 4 };

/** Texels along each side of a bubble's frame, and its width in tiles. */
const BUBBLE = 8;
const BUBBLE_SIZE = BUBBLE / TEXELS;
/** Seconds a bubble rises whole: at least `least`, and up to `more` seconds longer at random. */
const FLOAT = { least: 1, more: 0.5 };
/** A bubble's popping frames, each shown `hold` seconds. */
const POP = { frames: 2, hold: 0.08 };
const POPPING = POP.frames * POP.hold;
/** Tiles a second a bubble rises, at least, before a random half again. */
const RISE = 1.2;
/** A bubble's sway: radians a second and half its width in tiles. */
const SWAY = { rate: 4, reach: 0.15 };

/**
 * A bubble blown at `at`, rising and swaying until it pops. It draws on `randf`, which the
 * caller seeds.
 */
export const blown = (at: d.v2f) => {
  'use gpu';
  const seed = randf.sample();
  return Particle({
    position: d.vec3f(at, 0),
    age: 0,
    velocity: d.vec3f(0, RISE * (1 + seed * 0.5), 0),
    life: FLOAT.least + randf.sample() * FLOAT.more + POPPING,
    color: d.vec4f(1),
    size: BUBBLE_SIZE,
    seed,
  });
};

/** Pepper's two colours, as the grinder's sheet draws them. */
const PEPPER = [
  d.vec4f(0x22 / 255, 0x20 / 255, 0x34 / 255, 1),
  d.vec4f(0x4b / 255, 0x42 / 255, 0x41 / 255, 1),
];
/** Tiles a second pepper falls. */
export const FALL = 5;
/** A grain of pepper's width in tiles, two texels. */
const GRAIN = 2 / TEXELS;

/**
 * A grain of pepper dropped at `at`, falling `depth` tiles before it goes. It draws on `randf`,
 * which the caller seeds.
 */
export const grain = (at: d.v2f, depth: number) => {
  'use gpu';
  const seed = randf.sample();
  return Particle({
    position: d.vec3f(at, 0),
    age: 0,
    velocity: d.vec3f(0, -FALL, 0),
    life: depth / FALL,
    color: std.select(PEPPER[0], PEPPER[1], seed < 0.5),
    size: GRAIN,
    seed,
  });
};

/** Rises at the speed it was blown with, swaying side to side, and holds still as it pops. */
const drift = (bubble: Particle) => {
  'use gpu';
  const next = Particle(bubble);
  const sway = std.cos(bubble.age * SWAY.rate + bubble.seed * 6.2832) * SWAY.reach * SWAY.rate;
  const rise = RISE * (1 + bubble.seed * 0.5);
  next.velocity = std.select(d.vec3f(sway, rise, 0), d.vec3f(), bubble.life - bubble.age < POPPING);
  return next;
};

/** Hard-edged shrinking dot in texels of its own. */
const dot = (mote: Particle, local: d.v2f) => {
  'use gpu';
  const left = 1 - mote.age / mote.life;
  if (std.length(pixelatedMote(local, mote.size)) > left) return d.vec4f();
  return d.vec4f(mote.color.xyz * mote.color.w, mote.color.w);
};

/** A square of its colour, a grain of pepper. */
const fleck = (mote: Particle) => {
  'use gpu';
  return d.vec4f(mote.color.xyz, 1);
};

/** Additive four-pointed sparkle in texels of its own. */
const twinkle = (spark: Particle, local: d.v2f) => {
  'use gpu';
  const across = std.abs(pixelatedMote(local, spark.size));
  const star = std.max(1 - across.x * 4 - across.y, 1 - across.y * 4 - across.x);
  const light = std.saturate(star * 2) * (1 - spark.age / spark.life);
  return d.vec4f(spark.color.xyz * light, 0);
};

export interface EffectsOptions {
  /** Seconds per `step`. */
  readonly step: number;
  /** The bubble's sheet, frames of `BUBBLE` texels square in a row: whole, then popping. */
  readonly bubble: Picture;
}

/**
 * Dust kicked up by landings, skids, slides and stomps, sparkles from taken coins, bubbles the
 * sponges blow and pepper the grinders grind.
 */
export function createEffects(root: TgpuRoot, camera: CameraBase, options: EffectsOptions) {
  const { step } = options;
  const shared = { step, dimensions: 2 } as const;
  // The sheet holds display colours, as the print reads them, so it loads unconverted
  const texture = images.create(root, options.bubble, { color: 'linear', mips: false });
  const sheet = texture.createView(d.texture2d(d.f32));

  /** A bubble's texel at `local`, whole until its last `POPPING` seconds, then popping. */
  const film = (bubble: Particle, local: d.v2f) => {
    'use gpu';
    const t = d.vec2i(std.floor((local + 1) * (BUBBLE / 2)));
    if (t.x < 0 || t.x >= BUBBLE || t.y < 0 || t.y >= BUBBLE) return d.vec4f();
    const popped = POPPING - (bubble.life - bubble.age);
    const frame = std.select(
      d.i32(0),
      1 + std.min(d.i32(popped / POP.hold), POP.frames - 1),
      popped > 0,
    );
    const texel = std.textureLoad(sheet.$, d.vec2i(frame * BUBBLE + t.x, BUBBLE - 1 - t.y), 0);
    return d.vec4f(texel.xyz * texel.w, texel.w);
  };

  const dust = particles.create(root, camera, {
    ...shared,
    capacity: 512,
    blend: 'over',
    shade: dot,
    settings: { gravity: [0, 2, 0], drag: 6 },
  });
  // The game moves the slide's dust to the hero's feet each step, and the stream reads it
  const kicked = {
    position: [0, 0, 0] as [number, number, number],
    velocity: [0, 0, 0] as [number, number, number],
    area: [0.1, 0, 0] as const,
    scatter: 1.5,
    size: 0.22,
    life: 0.35,
    color: DUST,
  };
  const trail = dust.stream(0, kicked);
  const sparks = particles.create(root, camera, {
    ...shared,
    capacity: 256,
    blend: 'add',
    shade: twinkle,
    settings: { gravity: [0, -18, 0], drag: 1.5 },
  });
  const bubbles = particles.create(root, camera, {
    ...shared,
    capacity: 256,
    label: 'bubbles',
    blend: 'over',
    update: drift,
    shade: film,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  const pepper = particles.create(root, camera, {
    ...shared,
    capacity: 256,
    label: 'pepper',
    blend: 'over',
    shade: fleck,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });

  return {
    /** Kernels can `dust.emit` motes of their own. */
    dust,
    /** Add its drawing after the dust's. */
    sparks,
    /** Kernels can `bubbles.emit` a bubble `blown` where they like. */
    bubbles,
    /** Kernels can `pepper.emit` a `grain` where they like. */
    pepper,

    spawn(pass: TgpuComputePass) {
      dust.spawn(pass);
      sparks.spawn(pass);
      bubbles.spawn(pass);
      pepper.spawn(pass);
    },
    compact(pass: TgpuComputePass, ahead: number) {
      dust.compact(pass, ahead);
      sparks.compact(pass, ahead);
      bubbles.compact(pass, ahead);
      pepper.compact(pass, ahead);
    },
    initAsync: () => pipelines.compile([dust, sparks, bubbles, pepper]),

    /** Emits dust at the hero's feet, kicked `push` tiles a second along x. */
    puff(x: number, y: number, push = 0) {
      dust.burst({
        count: 6,
        position: [x, y, 0],
        velocity: [push, 0, 0],
        area: [0.25, 0, 0],
        scatter: 2.5,
        size: 0.22,
        life: 0.35,
        color: DUST,
      });
    },

    /**
     * Sprays dust diagonally up ahead of a hero sliding `speed` tiles a second along x, from the
     * leading foot of its feet at `x, y`, or stops when `speed` is 0.
     */
    slide(x: number, y: number, speed: number) {
      const way = Math.sign(speed);
      trail.rate = way === 0 ? 0 : SLIDING.rate;
      kicked.position[0] = x + way * SLIDING.foot;
      kicked.position[1] = y;
      kicked.velocity[0] = speed + way * SLIDING.ahead;
      kicked.velocity[1] = SLIDING.up;
    },

    sparkle(x: number, y: number) {
      sparks.burst({
        count: 10,
        position: [x, y, 0],
        velocity: [0, 4, 0],
        scatter: 7,
        size: 0.5,
        life: 0.5,
        color: GOLD,
      });
    },

    destroy() {
      texture.destroy();
      dust.destroy();
      sparks.destroy();
      bubbles.destroy();
      pepper.destroy();
    },
  };
}

export type Effects = ReturnType<typeof createEffects>;
