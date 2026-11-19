import {
  cameras,
  clock,
  Particle,
  particles,
  passes,
  pipelines,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { perlin3d, randf } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';

/** How many motes circle the eye, and how far from it they start again. */
const MOTES = 12_000;
const RIM = 3.2;
/** The kit's blue for slow motes and its ice for fast ones. */
const SLOW = d.vec3f(0.3, 0.47, 1);
const FAST = d.vec3f(0.85, 0.94, 1);

/**
 * One step of a mote: it circles the eye, faster closer in, falls slowly towards it, drifts on
 * noise that changes with its age and is pulled back to the disc. In the eye it starts again near
 * the rim, as every mote does on its first step.
 */
const swirl = (mote: Particle) => {
  'use gpu';
  const p = mote.position;
  const radius = std.max(std.length(p.xz), 0.05);
  const around = d.vec3f(-p.z, 0, p.x) / radius;
  const inward = d.vec3f(-p.x, 0, -p.z) / radius;
  const spiral = (around + inward * 0.3) * (1.6 / std.sqrt(radius));
  const at = p * 0.5 + d.vec3f(0, mote.age * 0.15, 0);
  const drift = d.vec3f(perlin3d.sample(at), perlin3d.sample(at + 31), perlin3d.sample(at + 57));
  const next = Particle(mote);
  next.velocity = spiral + drift - d.vec3f(0, p.y, 0);
  if (radius < 0.2 + mote.seed * 0.4) {
    randf.seed2(d.vec2f(mote.seed, mote.age));
    const rim = randf.onUnitCircle() * (RIM - randf.sample());
    next.position = d.vec3f(rim.x, 0, rim.y);
  }
  return next;
};

/** A soft dot whose light adds to what is behind it, blue when slow and ice when fast. */
const glow = (mote: Particle, local: d.v2f) => {
  'use gpu';
  const edge = std.saturate(1 - std.length(local));
  const color = std.mix(SLOW, FAST, std.smoothstep(0.9, 2.4, std.length(mote.velocity)));
  return d.vec4f(color * (edge * edge * 0.8), 0);
};

export async function createScene(root: TgpuRoot, surface: Surface) {
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0, 0],
    distance: 8.5,
    yaw: 0.3,
    pitch: 0.6,
  });
  const sim = clock.fixedStep({ step: 1 / 60 });
  const motes = particles.create(root, view, {
    capacity: MOTES,
    step: sim.step,
    update: swirl,
    shade: glow,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  motes.burst({ count: MOTES, position: [0, 0, 0], size: 0.05, life: 1e9 });
  const scene = passes
    .create(root, surface, { clear: [0.004, 0.006, 0.016, 1] })
    .pipe(view.inject());
  const shown = scene.add(motes.drawing);
  await pipelines.compile([scene]);
  return {
    view,
    frame(frame: Frame) {
      view.update(frame.dt);
      const compute = frame.encoder.beginComputePass();
      motes.spawn(compute);
      for (let steps = sim.update(frame.dt); steps > 0; steps--) motes.step(compute);
      motes.compact(compute, sim.ahead);
      compute.end();
      scene.record(frame, shown);
    },
    destroy() {
      motes.destroy();
    },
  };
}
