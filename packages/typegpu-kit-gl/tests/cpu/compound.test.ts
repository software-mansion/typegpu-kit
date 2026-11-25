import { workgroup } from '@typegpu/kit';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

it('evaluates scalar compound destinations once for buffers and local arrays', async () => {
  const root = cpuRoot();
  const values = root.createMutable(d.arrayOf(d.u32, 4), [10, 20, 30, 40]);
  const calls = tgpu.privateVar(d.u32);
  const nextIndex = () => {
    'use gpu';
    const index = calls.$;
    calls.$++;
    return index;
  };
  const output = root.createMutable(d.arrayOf(d.u32, 4));
  const Local = d.arrayOf(d.u32, 4);
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        values.$[nextIndex()] += 1;
        output.$[0] = calls.$;
        calls.$ = 0;
        const local = Local([10, 20, 30, 40]);
        local[nextIndex()] *= 3;
        output.$[1] = calls.$;
        output.$[2] = local[0];
        output.$[3] = local[1];
      }),
    })
    .dispatchWorkgroups(1);
  expect(await values.read()).toEqual([11, 20, 30, 40]);
  expect(await output.read()).toEqual([1, 1, 30, 20]);
  root.destroy();
});

it('suspends a compound RHS without losing its cached destination', async () => {
  const root = cpuRoot();
  const values = root.createMutable(d.arrayOf(d.u32, 2), [10, 20]);
  const f = () => {
    'use gpu';
    std.workgroupBarrier();
    return d.u32(1);
  };
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.localInvocationId } })(
        ({ id }) => {
          'use gpu';
          values.$[id.x + 1] += f();
        },
      ),
    })
    .dispatchWorkgroups(1);
  expect(await values.read()).toEqual([10, 21]);
  root.destroy();
});

it('uses a workgroup scan result in an indexed compound store', async () => {
  const root = cpuRoot();
  const values = root.createMutable(d.arrayOf(d.u32, 5), [10, 20, 30, 40, 50]);
  const prefix = workgroup.scan(workgroup.sum(d.u32), 4);
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [4], in: { lid: d.builtin.localInvocationIndex } })(
        ({ lid }) => {
          'use gpu';
          values.$[lid + 1] += prefix(lid + 1, lid);
        },
      ),
    })
    .dispatchWorkgroups(1);
  expect(await values.read()).toEqual([10, 20, 31, 43, 56]);
  root.destroy();
});
