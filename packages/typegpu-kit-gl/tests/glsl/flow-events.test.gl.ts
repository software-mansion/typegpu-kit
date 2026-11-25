import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

/**
 * Adds 1 to a cell, then adds what it loads back from that cell to another buffer. Each cell of
 * either buffer takes one addition, so WebGPU's outcome is determined: the load sees its own 1.
 */
async function nested(root: TgpuRoot) {
  const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
  const other = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      const i = gid.x ^ 1;
      std.atomicAdd(sums.$[i], 1);
      std.atomicAdd(other.$[i], std.atomicLoad(sums.$[i]));
    },
  );
  root.createComputePipeline({ compute: kernel }).$name('nested').dispatchWorkgroups(1);
  return [await sums.read(), await other.read()];
}

it('loads inside a later addition after adding to that cell, as WebGPU does', async () => {
  const actual = await nested(gl);
  expect(actual[1]).toEqual(Array<number>(64).fill(1));
  expect(actual).toEqual(await nested(gpu));
});
