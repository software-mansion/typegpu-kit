import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { encodeHalf } from '../../src/cpu/textures.ts';
import { cpuRoot } from './device.ts';

async function run(root: TgpuRoot) {
  const image = root.createTexture({ size: [2, 1], format: 'rgba8unorm' }).$usage('sampled');
  image.write(new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]));
  const imageView = image.createView(d.texture2d(d.f32));
  const volume = root
    .createTexture({ size: [1, 1, 2], dimension: '3d', format: 'rgba16float' })
    .$usage('sampled');
  volume.write(new Uint16Array([2, 0, 0, 1, 6, 0, 0, 1].map(encodeHalf)));
  const volumeView = volume.createView(d.texture3d(d.f32));
  const layers = root.createTexture({ size: [1, 1, 2], format: 'rgba8unorm' }).$usage('sampled');
  layers.write(new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]));
  const layersView = layers.createView(d.texture2dArray(d.f32));
  const stored = root
    .createTexture({ size: [1, 1], format: 'rgba16float' })
    .$usage('storage', 'sampled');
  const target = stored.createView(d.textureStorage2d('rgba16float', 'write-only'));
  const read = stored.createView(d.texture2d(d.f32));
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const output = root.createMutable(d.arrayOf(d.vec4f, 4));
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$[0] = std.textureSampleLevel(imageView.$, sampler.$, d.vec2f(0.5), 0);
    output.$[1] = std.textureSampleLevel(volumeView.$, sampler.$, d.vec3f(0.5), 0);
    output.$[2] = std.textureLoad(layersView.$, d.vec2i(0), 1, 0);
    std.textureStore(target.$, d.vec2i(0), d.vec4f(0.333, 0.5, -2, 1));
  });
  const load = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$[3] = std.textureLoad(read.$, d.vec2i(0), 0);
  });
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  root.createComputePipeline({ compute: load }).dispatchWorkgroups(1);
  return (await output.read()).flatMap((v) => [v.x, v.y, v.z, v.w]);
}

it('matches dawn for 2d, array, 3d, filtering, and half storage', async () => {
  const gpu = await gpuRoot(),
    cpu = cpuRoot();
  try {
    const expected = await run(gpu),
      actual = await run(cpu);
    const mismatches: string[] = [];
    for (let i = 0; i < expected.length; i++) {
      const tolerance = i >= 12 ? Math.max(2 ** -24, Math.abs(expected[i]!) / 1024) : 1e-5;
      if (Math.abs(actual[i]! - expected[i]!) > tolerance)
        mismatches.push(`word ${i}: CPU=${actual[i]}, dawn=${expected[i]}, tolerance=${tolerance}`);
    }
    expect(mismatches, 'texture comparisons').toEqual([]);
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
});
