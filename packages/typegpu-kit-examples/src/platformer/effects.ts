import { particles, pipelines, type CameraBase, type Particle } from '@typegpu/kit';
import { d, std, type TgpuRoot, type TgpuComputePass } from 'typegpu';
import { pixelatedMote } from './sprites.ts';

/** Dust colour and opacity. Opaque dust lies on the front card, beside the sprites. */
export const DUST = [0.82, 0.76, 0.66, 1] as const;
const GOLD = [1, 0.85, 0.35, 1] as const;

/** Hard-edged shrinking dot in texels of its own. */
const dot = (mote: Particle, local: d.v2f) => {
  'use gpu';
  const left = 1 - mote.age / mote.life;
  if (std.length(pixelatedMote(local, mote.size)) > left) return d.vec4f();
  return d.vec4f(mote.color.xyz * mote.color.w, mote.color.w);
};

/** Additive four-pointed sparkle in texels of its own. */
const twinkle = (spark: Particle, local: d.v2f) => {
  'use gpu';
  const across = std.abs(pixelatedMote(local, spark.size));
  const star = std.max(1 - across.x * 4 - across.y, 1 - across.y * 4 - across.x);
  const light = std.saturate(star * 2) * (1 - spark.age / spark.life);
  return d.vec4f(spark.color.xyz * light, 0);
};

/** Dust kicked up by landings and stomps, and sparkles from taken coins. */
export function createEffects(root: TgpuRoot, camera: CameraBase, options: { step: number }) {
  const { step } = options;
  const shared = { step, dimensions: 2 } as const;
  const dust = particles.create(root, camera, {
    ...shared,
    capacity: 512,
    blend: 'over',
    shade: dot,
    settings: { gravity: [0, 2, 0], drag: 6 },
  });
  const sparks = particles.create(root, camera, {
    ...shared,
    capacity: 256,
    blend: 'add',
    shade: twinkle,
    settings: { gravity: [0, -18, 0], drag: 1.5 },
  });

  return {
    /** Kernels can `dust.emit` motes of their own. */
    dust,
    /** Add its drawing after the dust's. */
    sparks,

    spawn(pass: TgpuComputePass) {
      dust.spawn(pass);
      sparks.spawn(pass);
    },
    compact(pass: TgpuComputePass, ahead: number) {
      dust.compact(pass, ahead);
      sparks.compact(pass, ahead);
    },
    initAsync: () => pipelines.compile([dust, sparks]),

    /** Emits dust at the hero's feet. */
    puff(x: number, y: number) {
      dust.burst({
        count: 6,
        position: [x, y, 0],
        area: [0.25, 0, 0],
        scatter: 2.5,
        size: 0.22,
        life: 0.35,
        color: DUST,
      });
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
      dust.destroy();
      sparks.destroy();
    },
  };
}

export type Effects = ReturnType<typeof createEffects>;
