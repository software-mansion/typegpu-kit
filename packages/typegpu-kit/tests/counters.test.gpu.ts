import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { counters, frames, type Frame } from '../src/index.ts';

it('counts what kernels add across workgroups, and reports the gains a frame or two later', async () => {
  const root = await gpuRoot();
  const tally = counters.create(root, ['spawned', 'slain', 'reward'], { hz: 120 });

  const fight = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= 1000) return;

    tally.add.spawned(1);
    if (gid.x % 4 === 0) {
      tally.add.slain(1);
      tally.add.reward(d.u32(25));
    }
  });
  const fighting = root.pipe(tally.inject()).createComputePipeline({ compute: fight });

  let clearing = false;
  const loop = frames.create(root, (frame: Frame) => {
    if (clearing) tally.clear(frame.encoder);
    clearing = false;

    const pass = frame.encoder.beginComputePass();
    pass.setPipeline(fighting);
    pass.dispatchWorkgroups(16);
    pass.end();
    tally.capture(frame);
  });

  let time = 0;
  /** Runs frames until a read lands, and takes it. */
  const next = async () => {
    for (let i = 0; i < 30; i++) {
      loop.run(time);
      time += 1 / 60;
      await root.device.queue.onSubmittedWorkDone();
      await new Promise((resolve) => setTimeout(resolve, 5));

      const gained = tally.take();
      if (gained) return { ...gained };
    }
    throw new Error('No read landed.');
  };

  // A read lands with the counts of every frame run so far, each of 1000 spawns and 250 kills.
  const first = await next();
  const frames1 = first.spawned / 1000;
  expect(Number.isInteger(frames1) && frames1 >= 1).toBe(true);
  expect(first).toEqual({ spawned: 1000 * frames1, slain: 250 * frames1, reward: 6250 * frames1 });
  expect(tally.latest).toEqual(first);

  const second = await next();
  expect(second.spawned % 1000).toBe(0);
  expect(second.slain * 4).toBe(second.spawned);

  // After a clear, the totals start again from the frame that cleared.
  clearing = true;
  const cleared = await next();
  expect(cleared.spawned % 1000).toBe(0);
  expect(tally.latest!.spawned).toBe(cleared.spawned);
  expect(tally.latest!.spawned).toBeLessThanOrEqual(5000);
  root.destroy();
});

it('reports one frame at a lower read rate, clearing even frames that are not copied', async () => {
  const root = await gpuRoot();
  const tally = counters.create(root, ['near', 'far'], { per: 'frame', hz: 15 });
  const amount = root.createUniform(d.u32, 10);
  const count = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    tally.add.near(amount.$);
    tally.add.far(20 - amount.$);
  });
  const counting = root.pipe(tally.inject()).createComputePipeline({ compute: count });
  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginComputePass();
    pass.setPipeline(counting);
    pass.dispatchWorkgroups(1);
    pass.end();
    tally.capture(frame);
    tally.capture(frame);
  });
  let time = 0;
  const next = async () => {
    for (let i = 0; i < 30; i++) {
      loop.run(time);
      time += 1 / 60;
      await root.device.queue.onSubmittedWorkDone();
      await new Promise((resolve) => setTimeout(resolve, 0));
      const read = tally.take();
      if (read) return { ...read };
    }
    throw new Error('No frame counts landed.');
  };
  expect(await next()).toEqual({ near: 10, far: 10 });
  amount.write(3);
  expect(await next()).toEqual({ near: 3, far: 17 });
  expect(tally.latest).toEqual({ near: 3, far: 17 });
  loop.destroy();
  tally.destroy();
  root.destroy();
});
