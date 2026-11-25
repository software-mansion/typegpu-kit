import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import { CrossShaderStageState, GlslGenerator } from '@typegpu/gl';
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

/** The GLSL `@typegpu/gl` generates for `item`. */
function glslOf(item: object) {
  return tgpu.resolve([item as never], {
    unstable_shaderGenerator: new GlslGenerator('neutral', new CrossShaderStageState()),
  });
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

const COUNTS = [0, 1, 5, 31, 32, 33, 63, 64];

async function shifts(root: TgpuRoot) {
  const counts = root.createReadonly(d.arrayOf(d.u32, COUNTS.length), COUNTS);
  const left = root.createMutable(d.arrayOf(d.vec4u, COUNTS.length));
  const right = root.createMutable(d.arrayOf(d.vec2i, COUNTS.length));
  const kernel = tgpu.computeFn({ workgroupSize: [COUNTS.length], in: gid })(({ gid: at }) => {
    'use gpu';
    const n = counts.$[at.x] as number;
    let shifted = d.u32(3);
    shifted <<= n;
    left.$[at.x] = d.vec4u(d.u32(1) << n, d.u32(0x80000000) >>> n, d.u32(5) << (n + 1), shifted);
    right.$[at.x] = d.vec2i(d.i32(-8) >> n, d.i32(1) << n);
  });
  root.createComputePipeline({ compute: kernel }).$name('shifts').dispatchWorkgroups(1);
  return {
    left: (await left.read()).map((v) => [...v]),
    right: (await right.read()).map((v) => [...v]),
  };
}

/** Vector shifts, which the CPU backend lacks, in a fragment. */
function vectorShifts(root: TgpuRoot) {
  const counts = root.createReadonly(d.arrayOf(d.u32, COUNTS.length), COUNTS);
  return paint(root, COUNTS.length, (i) => {
    'use gpu';
    const n = counts.$[i] as number;
    const wide = d.vec2u(n, n + 1);
    const pair = std.bitShiftLeft(d.vec2u(1, 0x80000000), wide);
    const signed = std.bitShiftRight(d.vec2i(-8, 1), wide);
    return d.vec4f(d.vec2f(pair), d.vec2f(signed));
  });
}

it('shifts by a runtime count modulo the bit width, as WGSL does', async () => {
  const { actual, expected } = await both('shifts', shifts);
  expect(expected.left[4]).toEqual([1, 0x80000000, 10, 3]);
  expect(actual).toEqual(expected);
  const vectors = await vectorShifts(gl);
  expect(vectors).toEqual(await vectorShifts(gpu));
  expect(vectors[4]).toEqual([1, 0, -8, 0]);
  // GLSL leaves a count of 32 undefined, so the count is masked even where a driver wraps it
  const shift = tgpu.fn(
    [d.u32, d.u32, d.vec2i, d.vec2u],
    d.vec2u,
  )((x, n, v, m) => {
    'use gpu';
    return d.vec2u((x << n) + (x >>> 3), d.vec2u(std.bitShiftRight(v, m)).x);
  });
  const code = glslOf(shift);
  expect(code).toContain('(x << (n & 31u))');
  expect(code).toContain('(x >> 3u)');
  expect(code).toContain('(v >> (m & 31u))');
});

const FLAGS = [0, 1];

/** Builtins whose argument bumps a private counter: WGSL evaluates each argument once. */
function bumped(root: TgpuRoot) {
  const flags = root.createReadonly(d.arrayOf(d.u32, FLAGS.length), FLAGS);
  const counter = tgpu.privateVar(d.u32);
  const bump = () => {
    'use gpu';
    counter.$ += 1;
    return counter.$;
  };
  return paint(root, FLAGS.length, (i) => {
    'use gpu';
    const chosen = std.select(bump(), bump() + 10, flags.$[i] !== 0);
    const selected = counter.$;
    const dotted = std.dot(d.vec3i(d.i32(bump())), d.vec3i(1, 2, 3));
    const bits = std.countOneBits(d.vec4u(bump()));
    const leading = std.firstLeadingBit(d.i32(bump()));
    const quantized = std.quantizeToF16(d.vec2f(d.f32(bump())));
    return d.vec4f(
      d.f32(chosen),
      d.f32(selected),
      d.f32(dotted) + d.f32(bits.w) * 100 + d.f32(leading) * 1000,
      d.f32(counter.$) + quantized.y,
    );
  });
}

it('evaluates each argument of a builtin once, as WGSL does', async () => {
  const actual = await bumped(gl);
  expect(actual).toEqual(await bumped(gpu));
  expect(actual).toEqual([
    [1, 2, 18 + 100 + 2000, 6 + 6],
    [12, 2, 18 + 100 + 2000, 6 + 6],
  ]);
});
