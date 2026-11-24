import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

it('runs ordinary TypeGPU compute pipelines and reads their output', async () => {
  const root = cpuRoot();
  const output = root.createMutable(d.arrayOf(d.u32, 16));
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    output.$[id.x] = id.x * d.u32(7);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(2);
  expect(await output.read()).toEqual(Array.from({ length: 16 }, (_, i) => i * 7));
  root.destroy();
});
