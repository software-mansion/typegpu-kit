import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

// Expressions whose GLSL must mean what WGSL defines: each kernel runs on WebGPU and as the
// fallback's GPU form, and the two must agree

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots({ placement: false }));
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

const gid = { gid: d.builtin.globalInvocationId };

/** Runs `run` on the fallback, checks it ran as a GPU form, then runs it on WebGPU. */
async function both<T>(label: string, run: (root: TgpuRoot) => Promise<T>) {
  const actual = await run(gl);
  const stats = (gl.device as unknown as GLDevice).stats;
  expect(stats.fallbacks.get(label)).toBeUndefined();
  expect([...stats.kernels.keys()]).toContain(`${label} (gpu)`);
  return { actual, expected: await run(gpu) };
}

const INTS = [-1, -2147483648, 2147483647, 7, 0, -12345];

it('bitcasts between i32 and u32, scalars and vectors, keeping the bits', async () => {
  const run = async (root: TgpuRoot) => {
    const ints = root.createReadonly(d.arrayOf(d.i32, INTS.length), INTS);
    const out = root.createMutable(d.arrayOf(d.vec4u, INTS.length));
    const back = root.createMutable(d.arrayOf(d.vec4i, INTS.length));
    const kernel = tgpu.computeFn({ workgroupSize: [INTS.length], in: gid })(({ gid: at }) => {
      'use gpu';
      const a = ints.$[at.x] as number;
      const b = ints.$[(at.x + 1) % INTS.length] as number;
      const bits = std.bitcast(d.i32, d.u32)(a);
      const pair = std.bitcast(d.vec2i, d.vec2u)(d.vec2i(a, b));
      out.$[at.x] = d.vec4u(bits, pair, bits ^ 0x80000000);
      const again = std.bitcast(d.u32, d.i32)(bits);
      const pairAgain = std.bitcast(d.vec2u, d.vec2i)(pair);
      back.$[at.x] = d.vec4i(again, pairAgain, std.bitcast(d.u32, d.i32)(bits ^ 0x80000000));
    });
    root.createComputePipeline({ compute: kernel }).$name('bitcast').dispatchWorkgroups(1);
    return {
      out: (await out.read()).map((v) => [...v]),
      back: (await back.read()).map((v) => [...v]),
    };
  };
  const { actual, expected } = await both('bitcast', run);
  expect(actual).toEqual(expected);
  expect(actual.out[1]).toEqual([0x80000000, 0x80000000, 0x7fffffff, 0]);
  expect(actual.back.map((v) => v[0])).toEqual(INTS);
});
