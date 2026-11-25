import { d, tgpu } from 'typegpu';
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
