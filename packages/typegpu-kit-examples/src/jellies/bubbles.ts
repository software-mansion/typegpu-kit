import { clock, Particle, particles, type CameraBase } from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { mulberry32 } from 'math/random';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { stirAt, SURFACE } from './water.ts';

/** Where streams of bubbles rise from, on the tank's floor. */
const VENTS: [number, number, number][] = [
  [-1.9, -2.4, -1.4],
  [1.4, -2.4, -0.4],
];
/** Wobble rate in radians a second, and its push. */
const WOBBLE = { rate: 9, push: 1.4 };

/** A bubble's thin rim, brightest on top where the light from the surface catches it. */
const film = (bubble: Particle, local: d.v2f) => {
  'use gpu';
  const r = std.length(local);
  const rim = std.smoothstep(0.55, 0.9, r) * (1 - std.smoothstep(0.9, 1, r));
  const top = std.saturate(local.y * 0.7 + 0.45);
  const spot = std.saturate(1 - std.length(local - d.vec2f(-0.2, 0.5)) / 0.28);
  const alpha = (rim * (0.12 + top * 0.5) + spot * 0.5) * bubble.color.w;
  return d.vec4f(bubble.color.xyz * (alpha * top + spot * 1.2), alpha);
};

/** Wobbles a bubble and lets the current carry it. It bursts at the surface. */
const rise = (bubble: Particle, dt: number) => {
  'use gpu';
  const next = Particle(bubble);
  const wobble = std.sin(bubble.age * WOBBLE.rate + bubble.seed * 40) * WOBBLE.push;
  const push = d.vec3f(wobble, 0, wobble * 0.6) + stirAt(bubble.position);
  next.velocity = bubble.velocity + push * dt;
  if (bubble.position.y > SURFACE) next.life = 0;
  return next;
};

/** Bubbles rising in thin streams from the floor, wobbling and stirred by the current. */
export function createBubbles(
  root: TgpuRoot,
  camera: CameraBase,
  options: { step: number; water: Inject },
) {
  const { step, water } = options;
  const bubbles = particles.create(root, camera, {
    step,
    capacity: 128,
    label: 'bubbles',
    update: tgpu.fn(rise).pipe(water),
    shade: film,
    blend: 'translucent',
    settings: { gravity: [0, 1.6, 0], drag: 2.2 },
  });

  const rng = mulberry32.create(3);
  const puff = clock.cadence({ hz: 2.5 });
  let vent = 0;

  return {
    bubbles,

    /** Releases a bubble from the next vent now and then. Call once a frame before `spawn`. */
    vent(dt: number) {
      if (!puff.due(dt)) return;
      vent = (vent + 1) % VENTS.length;
      bubbles.burst({
        count: 1,
        position: VENTS[vent],
        area: [0.03, 0, 0.03],
        scatter: 0.05,
        size: 0.012 + mulberry32.sample(rng) * 0.013,
        life: 14,
        color: [0.75, 0.95, 1, 1],
      });
    },
  };
}
