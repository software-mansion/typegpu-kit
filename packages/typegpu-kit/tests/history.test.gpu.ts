import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { frames, history } from '../src/index.ts';

const COUNT = 64;
const Positions = d.arrayOf(d.f32, COUNT);
const positions = tgpu.mutableAccessor(Positions);

const positionOf = (i: number) => {
  'use gpu';
  return positions.$[i];
};
const positionBefore = history.previously(positionOf);

it('keeps last frame of state that kernels move, for code that reads both', async () => {
  const root = await gpuRoot();
  const state = history.create(root, Positions);
  const moved = root.createBuffer(Positions).$usage('storage');
  const movedBy = moved.as('mutable');

  const step = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.x >= COUNT) return;
    positions.$[gid.x] = positionOf(gid.x) + d.f32(gid.x);
  });
  const measure = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.x >= COUNT) return;
    movedBy.$[gid.x] = positionOf(gid.x) - positionBefore(gid.x);
  });
  const cfg = root.with(positions, state.reading);
  const stepping = cfg.createComputePipeline({ compute: step });
  const measuring = cfg.createComputePipeline({ compute: measure });

  const loop = frames.create(root, (frame) => {
    state.record(frame);
    const pass = frame.encoder.beginComputePass();
    pass.setPipeline(stepping);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(measuring);
    pass.dispatchWorkgroups(1);
    pass.end();
  });
  const frame = () => loop.run(0);

  frame();
  frame();
  expect((await moved.read()).slice(0, 4)).toEqual([0, 1, 2, 3]);
  expect((await state.current.read()).slice(0, 4)).toEqual([0, 2, 4, 6]);

  // A write lands at the next record, and the kernels move on from it: the jump reads as motion.
  state.write(Array.from({ length: COUNT }, () => 100));
  frame();
  expect((await moved.read()).slice(0, 4)).toEqual([100, 99, 98, 97]);
  expect((await state.current.read()).slice(0, 4)).toEqual([100, 101, 102, 103]);
  expect((await state.previous.read()).slice(0, 4)).toEqual([0, 2, 4, 6]);
  root.destroy();
});
