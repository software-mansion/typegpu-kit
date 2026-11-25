import { d, std, tgpu } from 'typegpu';
import { expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot } from './compare.ts';

const COUNT = 256;

test('a GPU form whose output a CPU kernel reads back moves to the CPU and stays correct', async () => {
  const root = glRoot();
  try {
    const input = root.createReadonly(d.arrayOf(d.u32, COUNT));
    const doubled = root.createMutable(d.arrayOf(d.u32, COUNT));
    const total = root.createMutable(d.atomic(d.u32));
    const double = tgpu.computeFn({
      workgroupSize: [64],
      in: { id: d.builtin.globalInvocationId },
    })(({ id }) => {
      'use gpu';
      doubled.$[id.x] = input.$[id.x]! * 2;
    });
    // A shared atomic keeps this kernel on the CPU, so it reads back what `double` wrote
    const sum = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(({
      id,
    }) => {
      'use gpu';
      std.atomicAdd(total.$, doubled.$[id.x]!);
    });
    const doubling = root.createComputePipeline({ compute: double }).$name('doubling');
    const summing = root.createComputePipeline({ compute: sum }).$name('summing');
    const totals: number[] = [];
    for (let frame = 1; frame <= 16; frame++) {
      input.write(Array.from({ length: COUNT }, () => frame));
      total.write(0);
      doubling.dispatchWorkgroups(COUNT / 64);
      summing.dispatchWorkgroups(COUNT / 64);
      totals.push(await total.read());
    }
    expect(totals).toEqual(Array.from({ length: 16 }, (_, i) => (i + 1) * 2 * COUNT));
    const { placement } = (root.device as GLDevice).stats;
    expect(placement.get('doubling')).toMatch(/^cpu:/);
  } finally {
    root.destroy();
  }
});
