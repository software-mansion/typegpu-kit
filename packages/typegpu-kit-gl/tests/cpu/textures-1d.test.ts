import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu } from 'typegpu';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile } from '../../src/cpu/index.ts';
import { expect, it } from 'vitest';
import { textureOps } from '../../src/cpu/textures.ts';
import type { CpuTexture } from '../../src/cpu/types.ts';

it('loads, stores and sizes a 1D texture with scalar coordinates', () => {
  const texture: CpuTexture = {
    dimension: '1d',
    format: 'r32float',
    levels: [{ width: 2, height: 1, depthOrArrayLayers: 1, data: new Float32Array([3, 7]) }],
  };
  expect(textureOps.textureDimensions(texture)).toBe(2);
  expect(textureOps.textureLoad(texture, 1, 0)).toEqual([7, 0, 0, 1]);
  textureOps.textureStore(texture, 1, [11, 0, 0, 1]);
  expect(texture.levels[0]!.data).toEqual(new Float32Array([3, 11]));
  expect(textureOps.textureSampleLevel(texture, { magFilter: 'linear' }, 0.5, 0)).toEqual([
    7, 0, 0, 1,
  ]);
  expect(textureOps.textureSampleLevel(texture, { addressModeU: 'clamp-to-edge' }, -1, 0)).toEqual([
    3, 0, 0, 1,
  ]);
});

it('executes scalar 1D coordinates through a resolved compute module', () => {
  const { root } = recordingRoot();
  const layout = tgpu.bindGroupLayout({
    source: { texture: d.texture1d(d.f32) },
    destination: { storageTexture: d.textureStorage1d('r32float', 'write-only') },
  });
  const output = root.createMutable(d.vec2f);
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const value = std.textureLoad(layout.$.source, 1, 0);
    output.$ = d.vec2f(d.f32(std.textureDimensions(layout.$.source)), value.x);
    std.textureStore(layout.$.destination, 1, value);
  });
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const source: CpuTexture = {
    dimension: '1d',
    format: 'r32float',
    levels: [{ width: 2, height: 1, depthOrArrayLayers: 1, data: new Float32Array([3, 7]) }],
  };
  const destination: CpuTexture = {
    ...source,
    levels: [{ ...source.levels[0]!, data: new Float32Array(2) }],
  };
  const bytes = new ArrayBuffer(8);
  module.dispatch(1, 1, 1, {
    buffer: () => ({ bytes, offset: 0, size: 8 }),
    texture: (_group, binding) => (binding === 0 ? source : destination),
    sampler: () => ({}),
  });
  expect([...new Float32Array(bytes)]).toEqual([2, 7]);
  expect([...(destination.levels[0]!.data as Float32Array)]).toEqual([0, 7]);
  root.destroy();
});
