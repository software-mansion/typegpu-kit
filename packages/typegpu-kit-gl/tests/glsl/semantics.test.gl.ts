import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { readTexture, roots, values } from '../device/compare.ts';

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

/**
 * Draws a row of `width` pixels, each `pixel(x)`, on `root` and reads them back. Builtins the CPU
 * backend lacks are tested here, since a compute pipeline needs a CPU form as well.
 */
async function paint(root: TgpuRoot, width: number, pixel: (x: number) => d.v4f) {
  const fragment = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
    position,
  }) => {
    'use gpu';
    return pixel(d.u32(position.x));
  });
  const format = 'rgba32float';
  const target = root.device.createTexture({
    size: [width, 1],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  root
    .createRenderPipeline({ vertex: common.fullScreenTriangle, fragment, targets: { format } })
    .withColorAttachment({ view: target.createView(), loadOp: 'clear', storeOp: 'store' })
    .draw(3);
  const texels = values(await readTexture(root.device, target));
  target.destroy();
  return Array.from({ length: width }, (_, x) => [...texels.subarray(x * 4, x * 4 + 4)]);
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

const FLOATS = [1, -0.75, 3.3, 1 / 3, 65504, 1e-3, 12345.678, -2.5e-5];

function lacking(root: TgpuRoot) {
  const floats = root.createReadonly(d.arrayOf(d.f32, FLOATS.length), FLOATS);
  return paint(root, FLOATS.length, (i) => {
    'use gpu';
    const x = floats.$[i] as number;
    const e = d.i32(i) - 4;
    const scaled = std.ldexp(d.vec2f(x, -x), d.vec2i(e, -e));
    const quantized = std.quantizeToF16(d.vec3f(x, x * 3, x / 7));
    return d.vec4f(std.ldexp(x, e), scaled.y, quantized.x, quantized.z);
  });
}

it('emits ldexp and quantizeToF16, which GLSL ES 3.00 lacks', async () => {
  const actual = await lacking(gl);
  expect(actual).toEqual(await lacking(gpu));
  expect(actual[0]?.[0]).toBe(1 / 16);
});

it('names modf and frexp, which it does not emit', async () => {
  const out = gl.createReadonly(d.arrayOf(d.f32, 1), [1.5]);
  const modf = () => {
    'use gpu';
    return d.vec4f(std.modf(out.$[0] as number).fract);
  };
  const frexp = () => {
    'use gpu';
    return d.vec4f(std.frexp(out.$[0] as number).fract);
  };
  await expect(paint(gl, 1, modf)).rejects.toThrow("'modf' is not supported");
  await expect(paint(gl, 1, frexp)).rejects.toThrow("'frexp' is not supported");
});
