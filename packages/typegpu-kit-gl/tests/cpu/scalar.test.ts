import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile } from '../../src/cpu/index.ts';

it('executes a resolved scalar kernel over a bound subrange', () => {
  const { root } = recordingRoot();
  const data = root.createMutable(d.arrayOf(d.u32, 256));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      data.$[id.x] = data.$[id.x] * d.u32(1664525) + d.u32(1013904223);
    },
  );
  const code = tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() });
  const module = compile(code);
  const bytes = new ArrayBuffer(258 * 4);
  const values = new Uint32Array(bytes);
  values.fill(0xffffffff);
  module.dispatch(4, 1, 1, {
    buffer: () => ({ bytes, offset: 4, size: 256 * 4 }),
    texture: () => {
      throw new Error('unexpected texture');
    },
    sampler: () => {
      throw new Error('unexpected sampler');
    },
  });
  expect(values[0]).toBe(0xffffffff);
  expect(values[257]).toBe(0xffffffff);
  expect(Array.from(values.subarray(1, 257))).toEqual(
    Array(256).fill((Math.imul(0xffffffff, 1664525) + 1013904223) >>> 0),
  );
  expect(module.bindings).toEqual([
    { group: 0, binding: 0, kind: 'buffer', reads: true, writes: true },
  ]);
  root.destroy();
});

it('rejects strings without compute metadata', () => {
  expect(() => compile('fn main() {}')).toThrow('missing compute marker');
  expect(() => compile('//#kit-gl compute')).toThrow('missing entry metadata');
});

it('compiles barriers and writes individual vector buffer components', () => {
  const barrier = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    std.workgroupBarrier();
  });
  expect(
    compile(tgpu.resolve([barrier], { unstable_shaderGenerator: new CpuGenerator() }))
      .workgroupSize,
  ).toEqual([1, 1, 1]);
  const { root } = recordingRoot();
  const data = root.createMutable(d.arrayOf(d.vec2f, 1));
  const composite = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    data.$[0].x = 1;
  });
  const module = compile(
    tgpu.resolve([composite], { unstable_shaderGenerator: new CpuGenerator() }),
  );
  const bytes = new ArrayBuffer(8);
  module.dispatch(1, 1, 1, {
    buffer: () => ({ bytes, offset: 0, size: 8 }),
    texture: () => {
      throw new Error('unexpected texture');
    },
    sampler: () => {
      throw new Error('unexpected sampler');
    },
  });
  expect(Array.from(new Float32Array(bytes))).toEqual([1, 0]);
  root.destroy();
});

it('executes atomic additions with sequential return values', () => {
  const { root } = recordingRoot();
  const counter = root.createMutable(d.atomic(d.u32));
  const output = root.createMutable(d.arrayOf(d.u32, 16));
  const kernel = tgpu.computeFn({ workgroupSize: [16], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      output.$[id.x] = std.atomicAdd(counter.$, d.u32(3));
    },
  );
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const buffers = module.bindings.map(() => new ArrayBuffer(64));
  module.dispatch(1, 1, 1, {
    buffer: (_group, binding) => ({ bytes: buffers[binding]!, offset: 0, size: 64 }),
    texture: (): never => {
      throw new Error('unexpected texture');
    },
    sampler: (): never => {
      throw new Error('unexpected sampler');
    },
  });
  expect(buffers.map((buffer) => Array.from(new Uint32Array(buffer)))).toContainEqual(
    Array.from({ length: 16 }, (_, i) => i * 3),
  );
  expect(buffers.some((buffer) => new Uint32Array(buffer)[0] === 48)).toBe(true);
  root.destroy();
});

it('passes scalar function references without copying their storage', () => {
  const { root } = recordingRoot();
  const output = root.createMutable(d.u32);
  const increment = tgpu.fn([d.ptrFn(d.u32)])((value) => {
    'use gpu';
    value.$ += d.u32(7);
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const value = d.ref(d.u32(4));
    increment(value);
    output.$ = value.$;
  });
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const bytes = new ArrayBuffer(4);
  module.dispatch(1, 1, 1, {
    buffer: () => ({ bytes, offset: 0, size: 4 }),
    texture: (): never => {
      throw new Error('unexpected texture');
    },
    sampler: (): never => {
      throw new Error('unexpected sampler');
    },
  });
  expect(new Uint32Array(bytes)[0]).toBe(11);
  root.destroy();
});
