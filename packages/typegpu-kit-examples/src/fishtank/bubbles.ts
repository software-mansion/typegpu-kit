import { clock, Particle, particles, type CameraBase } from '@typegpu/kit';
import { random } from 'math/random';
import { d, std, type TgpuRoot } from 'typegpu';
import { TANK } from './water.ts';

/** Vents in the sand that bubbles puff from, in world units. */
const VENTS: [number, number, number][] = [
  [-5.2, 0.1, -2.5],
  [-4.6, 0.1, -2.1],
  [2.8, 0.1, 1.6],
  [6.1, 0.1, -3.2],
];
/** Minimum and maximum bubbles per vent burst. */
const PUFF = [1, 3] as const;
/** Each burst's spread round its vent, its bubbles' size, lifetime and colour. */
const BURST = {
  area: [0.08, 0, 0.08] as const,
  scatter: 0.2,
  size: 0.1,
  life: 20,
  color: [0.55, 0.85, 1, 0.9] as const,
};
/** Wobble frequency in radians per second and acceleration in world units per second squared. */
const WOBBLE = { rate: 7, push: 2 };
/** The share of its size a bubble grows by each second. */
const GROWTH = 0.05;
/** Brightness of a bubble's rim. */
const RIM = 0.5;
/** Highlight centre and radius relative to a unit-radius bubble, and highlight brightness. */
const HIGHLIGHT = { at: d.vec2f(-0.35, 0.4), radius: 0.25, brightness: 2 };

/** Additive shading for a thin rim and a highlight facing the light. */
const rim = (bubble: Particle, local: d.v2f) => {
  'use gpu';
  const r = std.length(local);
  const ring = std.smoothstep(0.6, 0.95, r) * (1 - std.smoothstep(0.95, 1, r));
  const highlight = std.saturate(1 - std.length(local - HIGHLIGHT.at) / HIGHLIGHT.radius);

  return d.vec4f(
    bubble.color.xyz * (ring * RIM + highlight * HIGHLIGHT.brightness) * bubble.color.w,
    0,
  );
};

/** Wobbles a bubble sideways and grows it as it rises, and bursts it at the surface. */
const rise = (bubble: Particle, dt: number) => {
  'use gpu';
  let next = Particle(bubble);
  const wobble = std.sin(bubble.age * WOBBLE.rate + bubble.seed * 40) * WOBBLE.push;
  next.velocity.x = bubble.velocity.x + wobble * dt;
  next.size = bubble.size * (1 + GROWTH * dt);
  if (bubble.position.y > TANK.y - bubble.size) next.life = 0;

  return next;
};

/**
 * Bubbles that puff from vents in the sand, wobble and grow as they rise, and burst at the top.
 * `step` is the length of a step in seconds; see `particles.Options.step`.
 */
export function createBubbles(root: TgpuRoot, camera: CameraBase, options: { step: number }) {
  const { step } = options;
  const bubbles = particles.create(root, camera, {
    step,
    capacity: 1024,
    blend: 'add',
    update: rise,
    shade: rim,
    // Buoyancy and drag give a terminal speed of 0.8 world units per second
    settings: { gravity: [0, 1.2, 0], drag: 1.5 },
  });

  const puff = clock.cadence({ hz: 5 });

  return {
    bubbles,

    /** Schedules occasional vent bursts. Call once per frame before `spawn`. */
    vent(dt: number) {
      if (!puff.due(dt)) return;

      bubbles.burst({
        ...BURST,
        count: random.int(Math.random, ...PUFF),
        position: random.choice(Math.random, VENTS),
      });
    },
  };
}
