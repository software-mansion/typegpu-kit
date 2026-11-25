import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile, type CpuTexture } from '../../src/cpu/index.ts';
import { decodeHalf, encodeHalf, textureOps } from '../../src/cpu/textures.ts';

it('round-trips every finite half-float bit pattern', () => {
  for (let bits = 0; bits < 65536; bits++)
    if ((bits & 0x7c00) !== 0x7c00) expect(encodeHalf(decodeHalf(bits)), `half ${bits}`).toBe(bits);
});

it('rounds every finite half-float midpoint to even for both signs', () => {
  for (let bits = 0; bits < 0x7bff; bits++) {
    const midpoint = (decodeHalf(bits) + decodeHalf(bits + 1)) / 2;
    const rounded = bits + (bits & 1);
    expect(encodeHalf(midpoint), `positive midpoint ${bits}`).toBe(rounded);
    expect(encodeHalf(-midpoint), `negative midpoint ${bits}`).toBe(rounded | 0x8000);
  }
});

it('runs texture sampling and storage from a resolved TypeGPU kernel', () => {
  const { root } = recordingRoot();
  const layout = tgpu.bindGroupLayout({
    source: { texture: d.texture2d(d.f32) },
    destination: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
    sampler: { sampler: 'filtering' },
  });
  const output = root.createMutable(d.vec4f);
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const value = std.textureSampleLevel(layout.$.source, layout.$.sampler, d.vec2f(0.5), 0);
    std.textureStore(layout.$.destination, d.vec2i(0), value);
    output.$ = d.vec4f(value);
  });
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const source: CpuTexture = {
    format: 'rgba8unorm',
    dimension: '2d',
    levels: [
      {
        width: 2,
        height: 1,
        depthOrArrayLayers: 1,
        data: new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]),
      },
    ],
  };
  const destination: CpuTexture = {
    format: 'rgba16float',
    dimension: '2d',
    levels: [{ width: 1, height: 1, depthOrArrayLayers: 1, data: new Uint16Array(4) }],
  };
  const bytes = new ArrayBuffer(16);
  module.dispatch(1, 1, 1, {
    buffer: () => ({ bytes, offset: 0, size: 16 }),
    texture: (_group, binding) => (binding === 1 ? destination : source),
    sampler: () => ({ magFilter: 'linear', minFilter: 'linear' }),
  });
  expect(Array.from(new Float32Array(bytes))).toEqual([0.5, 0, 0.5, 1]);
  expect(Array.from(destination.levels[0]!.data as Uint16Array).map(decodeHalf)).toEqual([
    0.5, 0, 0.5, 1,
  ]);
  root.destroy();
});

it('loads and stores array layers and samples 3d slices', () => {
  const array: CpuTexture = {
    format: 'r32uint',
    dimension: '2d-array',
    levels: [{ width: 2, height: 1, depthOrArrayLayers: 2, data: new Uint32Array(4) }],
  };
  textureOps.textureStore(array, [1, 0], 1, [17, 0, 0, 0]);
  expect(textureOps.textureLoad(array, [1, 0], 1, 0)).toEqual([17, 0, 0, 1]);
  const volume: CpuTexture = {
    format: 'r32float',
    dimension: '3d',
    levels: [{ width: 1, height: 1, depthOrArrayLayers: 2, data: new Float32Array([2, 6]) }],
  };
  expect(
    textureOps.textureSampleLevel(volume, { magFilter: 'linear' }, [0.5, 0.5, 0.5], 0),
  ).toEqual([4, 0, 0, 1]);
});

it('samples cube faces and compares depth before filtering', () => {
  const cube: CpuTexture = {
    format: 'r32float',
    dimension: 'cube',
    levels: [
      { width: 1, height: 1, depthOrArrayLayers: 6, data: new Float32Array([1, 2, 3, 4, 5, 6]) },
    ],
  };
  for (const [direction, expected] of [
    [[1, 0, 0], 1],
    [[-1, 0, 0], 2],
    [[0, 1, 0], 3],
    [[0, -1, 0], 4],
    [[0, 0, 1], 5],
    [[0, 0, -1], 6],
  ] as const)
    expect(textureOps.textureSampleLevel(cube, {}, [...direction], 0)).toEqual([expected, 0, 0, 1]);
  const depth: CpuTexture = {
    format: 'depth32float',
    dimension: '2d',
    levels: [{ width: 2, height: 1, depthOrArrayLayers: 1, data: new Float32Array([0.25, 0.75]) }],
  };
  expect(
    textureOps.textureSampleCompareLevel(
      depth,
      { compare: 'less', magFilter: 'linear' },
      [0.5, 0.5],
      0.5,
    ),
  ).toBe(0.5);
  expect(textureOps.textureLoad(depth, [0, 0], 0)).toBe(0.25);
});
