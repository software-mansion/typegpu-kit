import { clock, frames, meshPool, poses } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { resolve } from '../src/assets.ts';
import { assets } from '../src/skeletons/assets.ts';
import { createHorde, Skeleton } from '../src/skeletons/horde.ts';
import { fromDisk } from './utils/examples.ts';

const COUNT = 300;
const STEP = 1 / 60;
const FRAME = 2 * STEP;

/** Repeatable dice for where the skeletons lie. */
const dice = (seed: number) => () => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 2 ** 32;
};

it('wakes the horde, sets it fighting, and makes the skeletons struck flinch', async () => {
  const { models } = await resolve('skeletons', assets, fromDisk);
  const root = await gpuRoot();
  const pool = meshPool.create(root, { capacity: 64 });
  const sim = clock.fixedStep({ step: STEP });
  const radius = Math.sqrt(COUNT) * 2;
  const skeletons = createHorde(root, pool, {
    models,
    count: COUNT,
    radius,
    step: STEP,
    random: dice(7),
  });
  const { horde, poses: figures, doing } = skeletons;
  const { flinch } = figures.layers;

  // How far into its flinch each instance row's pose is, 0 for one never struck
  const flinched = root.createMutable(d.arrayOf(d.f32, horde.capacity));
  const reading = root
    .pipe(horde.inject())
    .pipe(figures.inject())
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
        ({ gid }) => {
          'use gpu';
          if (!Skeleton.alive(gid.x)) return;
          flinched.$[gid.x] = poses.seconds(Skeleton.pose(gid.x), flinch);
        },
      ),
    });

  let now = 0;
  const loop = frames.create(root, (frame) => {
    const steps = sim.update(FRAME);
    now += steps * STEP;
    const pass = frame.encoder.beginComputePass();
    skeletons.animate(pass, now, steps);
    pass.setPipeline(reading);
    pass.dispatchWorkgroups(Math.ceil(horde.span / 64));
    pass.end();
  });

  const seen = new Set<string>();
  for (let frame = 0; frame < 30 / FRAME; frame++) {
    loop.run(frame * FRAME);
    if (frame % 15 === 0) {
      for (const { state } of await doing.rows.read()) seen.add(doing.definition.names[state]);
    }
  }
  expect(seen).toEqual(new Set(doing.definition.names));

  // A skeleton struck flinches on every row it has, three or four
  const struck = (await flinched.read()).filter((seconds) => seconds > 0).length;
  expect(struck).toBeGreaterThan(COUNT / 4);

  skeletons.destroy();
  pool.destroy();
  root.destroy();
});
