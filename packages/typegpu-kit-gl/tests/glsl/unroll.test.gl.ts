import { d, std, tgpu as real, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

// A runtime range spelled `tgpu.unroll`: the loop runs four times and must be counted as a loop
const tgpu = { unroll: std.range };

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

async function spelled(root: TgpuRoot) {
  const cells = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
  const kernel = real.computeFn({ workgroupSize: [4] })(() => {
    'use gpu';
    for (const v of tgpu.unroll(4)) {
      std.atomicOr(cells.$[v], 1);
    }
  });
  root.createComputePipeline({ compute: kernel }).$name('spelled').dispatchWorkgroups(1);
  return cells.read();
}

it('sets every cell a loop spelled tgpu.unroll over a runtime range reaches, as WebGPU does', async () => {
  const actual = await spelled(gl);
  expect(actual).toEqual([1, 1, 1, 1]);
  expect(actual).toEqual(await spelled(gpu));
});
