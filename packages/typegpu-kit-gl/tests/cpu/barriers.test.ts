import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile } from '../../src/cpu/index.ts';

it('waits for every lane before reading shared memory and resets between groups', () => {
  const { root } = recordingRoot();
  const shared = tgpu.workgroupVar(d.arrayOf(d.u32, 8));
  const output = root.createMutable(d.arrayOf(d.u32, 16));
  const kernel = tgpu.computeFn({
    workgroupSize: [8],
    in: { local: d.builtin.localInvocationIndex, global: d.builtin.globalInvocationId },
  })(({ local, global }) => {
    'use gpu';
    shared.$[local] = global.x + d.u32(1);
    std.workgroupBarrier();
    output.$[global.x] = shared.$[(local + d.u32(1)) % d.u32(8)];
  });
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const bytes = new ArrayBuffer(64);
  module.dispatch(2, 1, 1, {
    buffer: () => ({ bytes, offset: 0, size: 64 }),
    texture: (): never => {
      throw new Error('unexpected texture');
    },
    sampler: (): never => {
      throw new Error('unexpected sampler');
    },
  });
  expect(Array.from(new Uint32Array(bytes))).toEqual([
    2, 3, 4, 5, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16, 9,
  ]);
  root.destroy();
});
