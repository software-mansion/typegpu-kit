import { counters, workgroup, sorter } from '@typegpu/kit';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

it('accumulates real Kit counters across multiple workgroups', async () => {
  const root = cpuRoot();
  const counts = counters.create(root, ['all', 'even']);
  const kernel = tgpu.computeFn({ workgroupSize: [32], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      counts.add.all(1);
      if (id.x % d.u32(2) === d.u32(0)) counts.add.even(3);
    },
  );
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(4);
  expect(await counts.buffer.read()).toEqual([128, 192]);
  counts.destroy();
  root.destroy();
});

it('reduces and scans with Kit workgroup functions', async () => {
  const root = cpuRoot();
  const total = workgroup.reduce(workgroup.sum(d.u32), 16);
  const prefix = workgroup.scan(workgroup.sum(d.u32), 16);
  const output = root.createMutable(d.arrayOf(d.u32, 32));
  const kernel = tgpu.computeFn({
    workgroupSize: [16],
    in: { id: d.builtin.localInvocationIndex },
  })(({ id }) => {
    'use gpu';
    output.$[id] = total(id + d.u32(1), id);
    output.$[id + d.u32(16)] = prefix(id + d.u32(1), id);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  expect(await output.read()).toEqual([
    ...Array(16).fill(136),
    ...Array.from({ length: 16 }, (_, i) => (i * (i + 1)) / 2),
  ]);
  root.destroy();
});

it('sorts with the Kit radix sorter and preserves the input permutation', async () => {
  const root = cpuRoot();
  const values = Array.from({ length: 257 }, (_, i) => (i * 7919) % 613);
  const keys = root.createBuffer(d.arrayOf(d.u32, values.length), values).$usage('storage');
  const sort = sorter.create(root, keys, { keyBits: 16 });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  sort.sort(pass);
  pass.end();
  encoder.submit();
  expect(await keys.read()).toEqual(values.toSorted((a, b) => a - b));
  sort.destroy();
  root.destroy();
});
