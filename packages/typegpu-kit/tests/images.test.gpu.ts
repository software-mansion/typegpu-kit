import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { images } from '../src/index.ts';

const DIRECTIONS = [
  d.vec3f(1, 0, 0),
  d.vec3f(-1, 0, 0),
  d.vec3f(0, 1, 0),
  d.vec3f(0, -1, 0),
  d.vec3f(0, 0, 1),
  d.vec3f(0, 0, -1),
];

/** A 4 × 4 face of half floats, all `bits` in red. */
const face = (bits: number) => {
  const pixels = new Uint16Array(64);
  for (let i = 0; i < 16; i++) pixels.set([bits, 0, 0, 0x3c00], i * 4);
  return { width: 4, height: 4, pixels };
};

it('puts cube faces in WebGPU order, half floats as they decode, and averages colour mips in linear light', async () => {
  const root = await gpuRoot();
  // 1 to 6 in red, one per face
  const box = images.cube(root, [0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600].map(face));
  // A checkerboard of black and white
  const checker = images.create(root, {
    width: 2,
    height: 2,
    pixels: new Uint8Array([0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255]),
  });
  const cubeView = box.createView(d.textureCube(d.f32));
  const checkerView = checker.createView(d.texture2d(d.f32));
  const smooth = images.sampler(root, { wrap: 'clamp-to-edge', anisotropy: 4 });
  const out = root.createMutable(d.arrayOf(d.f32, 13));

  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    for (const i of tgpu.unroll([0, 1, 2, 3, 4, 5])) {
      out.$[i] = std.textureSampleLevel(cubeView.$, smooth.$, DIRECTIONS[i], 0).x;
      // Each face's 1 × 1 level
      out.$[i + 6] = std.textureSampleLevel(cubeView.$, smooth.$, DIRECTIONS[i], 2).x;
    }
    out.$[12] = std.textureSampleLevel(checkerView.$, smooth.$, d.vec2f(0.5), 1).x;
  });
  root.createComputePipeline({ compute: read }).dispatchWorkgroups(1);

  const values = await out.read();
  expect(values.slice(0, 12)).toEqual([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);
  // Averaging the bytes would give sRGB 128, linear 0.216
  expect(values[12]).toBeCloseTo(0.5, 2);
});

it('unpacks RGBM into light once, so each layer filters and mips as light, not as its bytes', async () => {
  const root = await gpuRoot({ features: ['rg11b10ufloat-renderable'] });
  // 1.004 then 2.008 at a range of 8, which filtering the bytes would blend to 2.82
  const pair = {
    width: 2,
    height: 1,
    pixels: new Uint8Array([255, 255, 255, 32, 64, 64, 64, 255]),
  };
  const blank = { width: 2, height: 1, pixels: new Uint8Array(8) };
  const packed = images.rgbm(8);
  const layers = images.array(root, [blank, pair], { packed });
  expect(layers.props.format).toBe('rgba16float');
  expect(() => images.create(root, pair, { packed, color: 'linear' })).toThrow(/color/);
  expect(() => images.create(root, pair, { format: 'rgba16float' })).toThrow(/packed/);

  const compact = images.create(root, pair, { packed, format: 'rg11b10ufloat', mips: false });
  // Half floats narrowed as they are, 2 in red
  const narrowed = images.create(root, face(0x4000), { format: 'rg11b10ufloat', mips: false });

  const view = layers.createView(d.texture2dArray(d.f32));
  const compactView = compact.createView(d.texture2d(d.f32));
  const narrowedView = narrowed.createView(d.texture2d(d.f32));
  const smooth = images.sampler(root, { wrap: 'clamp-to-edge' });
  const out = root.createMutable(d.arrayOf(d.f32, 6));
  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    out.$[0] = std.textureLoad(view.$, d.vec2i(0, 0), 1, 0).x;
    out.$[1] = std.textureLoad(view.$, d.vec2i(1, 0), 1, 0).x;
    out.$[2] = std.textureSampleLevel(view.$, smooth.$, d.vec2f(0.5), 1, 0).x;
    // The 1 × 1 level
    out.$[3] = std.textureSampleLevel(view.$, smooth.$, d.vec2f(0.5), 1, 1).x;
    out.$[4] = std.textureLoad(compactView.$, d.vec2i(1, 0), 0).x;
    out.$[5] = std.textureLoad(narrowedView.$, d.vec2i(3, 3), 0).x;
  });
  root.createComputePipeline({ compute: read }).dispatchWorkgroups(1);

  const [first, second, between, mip, small, half] = await out.read();
  expect(first).toBeCloseTo((8 * 32) / 255, 2);
  expect(second).toBeCloseTo((8 * 64) / 255, 2);
  expect(between).toBeCloseTo((first! + second!) / 2, 2);
  expect(mip).toBeCloseTo((first! + second!) / 2, 2);
  // Six bits of mantissa
  expect(small).toBeCloseTo(second!, 1);
  expect(half).toBe(2);
  root.destroy();
});
