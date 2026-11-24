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

it('rejects barriers and writes individual vector buffer components', () => {
  const barrier = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    std.workgroupBarrier();
  });
  expect(() => tgpu.resolve([barrier], { unstable_shaderGenerator: new CpuGenerator() })).toThrow(
    'unsupported builtin workgroupBarrier',
  );
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
