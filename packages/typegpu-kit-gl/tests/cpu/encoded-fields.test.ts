import { d, std, tgpu } from 'typegpu';
import { recordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';
import { encode } from '../../../typegpu-kit/src/render/octahedral.ts';
import { ordered } from '../../../typegpu-kit/src/render/ordered.ts';
import { encodedFields, packedEqual } from './encoded-fields.ts';

it('infers normalized, half-float and ordered float words from typed WGSL stores', () => {
  const { root } = recordingRoot();
  try {
    const values = root.createUniform(d.vec3f, d.vec3f(0.5));
    const attributes = root.createMutable(d.arrayOf(d.vec2u, 4));
    const bounds = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
    const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      const normal = std.pack2x16snorm(encode(values.$));
      attributes.$[0] = d.vec2u(normal, std.pack2x16float(values.$.xy));
      std.atomicMin(bounds.$[0], ordered(0.5));
    });
    const code = tgpu.resolve([kernel]);
    const names = [...code.matchAll(/var<storage,\s*read_write>\s+(\w+)/g)].map(
      (match) => match[1]!,
    );
    const encodings = names.map((name) => encodedFields(code, name));
    expect(encodings, code).toContainEqual([
      { component: 0, components: 2, encoding: 'octahedral-2x16snorm' },
      { component: 1, components: 2, encoding: '2x16float' },
    ]);
    expect(encodings).toContainEqual([{ component: 0, components: 1, encoding: 'ordered-f32' }]);
  } finally {
    root.destroy();
  }
});

it('compares normalized channels independently without relaxing ordinary integers', () => {
  expect(packedEqual(0x30e48001, 0xcf1c8001, 'octahedral-2x16snorm')).toBe(true);
  expect(packedEqual(0x30e48001, 0xcf1c8001, '2x16snorm')).toBe(false);
  expect(packedEqual(0x8001406e, 0x8001bf91, 'octahedral-2x16snorm')).toBe(true);
  expect(packedEqual(0x8001406e, 0x8001bf90, 'octahedral-2x16snorm')).toBe(false);
  expect(packedEqual(0xc001c001, 0xc000c000, '2x16snorm')).toBe(true);
  expect(packedEqual(0xc002c001, 0xc000c000, '2x16snorm')).toBe(false);
  expect(packedEqual(0x01020304, 0x02030405, '4x8unorm')).toBe(true);
});
