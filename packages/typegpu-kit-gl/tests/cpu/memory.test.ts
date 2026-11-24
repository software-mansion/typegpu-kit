import { d } from 'typegpu';
import { expect, it } from 'vitest';
import { layoutOf } from '../../src/cpu/memory.ts';

it('uses WGSL struct offsets, vec3 padding, and array strides', () => {
  const Item = d.struct({ tag: d.u32, position: d.vec3f, weight: d.f32 });
  const schema = d.arrayOf(Item, 2);
  const layout = layoutOf(schema);
  expect(layout.size).toBe(d.sizeOf(schema));
  expect(layout.stride).toBe(32);
  expect(layout.element?.fields?.position?.offset).toBe(16);
});

it('keeps the stride of a runtime array', () => {
  const layout = layoutOf(d.arrayOf(d.vec3f, 0));
  expect(layout.count).toBe(0);
  expect(layout.stride).toBe(16);
});
