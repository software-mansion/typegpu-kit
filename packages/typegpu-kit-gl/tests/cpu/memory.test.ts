import { d } from 'typegpu';
import { expect, it } from 'vitest';
import { bufferValue, layoutOf } from '../../src/cpu/memory.ts';

it('uses WGSL struct offsets, vec3 padding, and array strides', () => {
  const Item = d.struct({ tag: d.u32, position: d.vec3f, weight: d.f32 });
  const schema = d.arrayOf(Item, 2);
  const layout = layoutOf(schema);
  expect(layout.size).toBe(d.sizeOf(schema));
  expect(layout.stride).toBe(32);
  expect(layout.element?.fields?.position?.offset).toBe(16);
  const bytes = new ArrayBuffer(72);
  const words = new Uint32Array(bytes);
  words.fill(0xdeadbeef);
  const value = bufferValue({ bytes, offset: 4, size: 64 }, layout) as {
    tag: number;
    position: { x: number; y: number; z: number };
    weight: number;
  }[];
  value[1]!.position.y = 3.25;
  value[0]!.tag = 42;
  expect(new Float32Array(bytes)[14]).toBe(3.25);
  expect(words[1]).toBe(42);
  expect(words[0]).toBe(0xdeadbeef);
  expect(words[13]).toBe(0xdeadbeef);
  expect(words[15]).toBe(0xdeadbeef);
  expect(words[17]).toBe(0xdeadbeef);
});

it('derives a runtime array length from its bound range', () => {
  const layout = layoutOf(d.arrayOf(d.vec3f, 0));
  const value = bufferValue({ bytes: new ArrayBuffer(128), offset: 16, size: 64 }, layout) as {
    length: number;
  };
  expect(value.length).toBe(4);
});
