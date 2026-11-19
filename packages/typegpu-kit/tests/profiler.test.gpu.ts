import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { frames, profiler } from '../src/index.ts';
import { gpuRoot } from '@typegpu/kit/testing/dawn';

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

it('measures named passes on the GPU', async () => {
  const root = await gpuRoot({ features: ['timestamp-query'] });
  const timer = profiler.create(root, { skip: 1, samples: 3 });
  const busyPass = timer.timer('busy');
  const idle = timer.timer('idle');
  expect(timer.available).toBe(true);

  const data = root.createMutable(d.arrayOf(d.f32, 65536));
  const busy = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    let x = data.$[gid.x];
    for (let i = 0; i < 2000; i++) x = x * 0.999 + 0.001;
    data.$[gid.x] = x;
  });

  const pipeline = root.createComputePipeline({ compute: busy });
  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginComputePass({ timestampWrites: busyPass.writes() });
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(1024);
    pass.end();
    timer.capture(frame);
  });

  for (let i = 0; i < 5; i++) {
    loop.run(i);
    await settle();
  }

  expect(busyPass.ms).toBeGreaterThan(0);
  expect(idle.ms).toBeUndefined();

  root.destroy();
});

it('times the frame from its first timed pass to its last', async () => {
  const root = await gpuRoot({ features: ['timestamp-query'] });
  const timer = profiler.create(root, { skip: 1, samples: 3 });
  const first = timer.timer('first');
  const last = timer.timer('last');
  expect(timer.frame.ms).toBeUndefined();

  const data = root.createMutable(d.arrayOf(d.f32, 65536));
  const busy = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    let x = data.$[gid.x];
    for (let i = 0; i < 2000; i++) x = x * 0.999 + 0.001;
    data.$[gid.x] = x;
  });

  // The last pass reads and writes what the first one wrote, so it runs after it.
  const pipeline = root.createComputePipeline({ compute: busy });
  const loop = frames.create(root, (frame) => {
    for (const each of [first, last]) {
      const pass = frame.encoder.beginComputePass({ timestampWrites: each.writes() });
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(1024);
      pass.end();
    }
    timer.capture(frame);
  });

  // Another program's work can run between the two passes, so the tightest of a few rounds decides
  let tightest = Infinity;
  for (let round = 0; round < 8 && tightest > 3; round++) {
    for (let i = 0; i < 5; i++) {
      loop.run(round * 5 + i);
      await settle();
    }
    expect(timer.frame.ms).toBeGreaterThanOrEqual(Math.max(first.ms!, last.ms!));
    tightest = Math.min(tightest, timer.frame.ms! / (first.ms! + last.ms!));
  }

  expect(tightest).toBeLessThanOrEqual(3);

  root.destroy();
});
