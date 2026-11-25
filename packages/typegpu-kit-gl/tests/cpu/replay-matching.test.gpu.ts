import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { encodeHalf } from '../../src/cpu/textures.ts';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { observeScene } from './scene-observer.ts';
import { replay, snapshotDispatch, type ReplaySnapshot } from './replay.ts';

async function swapped(root: TgpuRoot) {
  const source = root
    .createTexture({ size: [1, 1], format: 'rgba16float' })
    .$usage('sampled', 'storage');
  const target = root
    .createTexture({ size: [1, 1], format: 'rgba16float' })
    .$usage('sampled', 'storage');
  source.write(new Uint16Array([1, 2, 3, 4].map(encodeHalf)));
  const layout = tgpu.bindGroupLayout({
    unused: { uniform: d.u32 },
    source: { texture: d.texture2d(d.f32) },
    target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
  });
  const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    std.textureStore(
      layout.$.target,
      d.vec2u(),
      std.textureLoad(layout.$.source, d.vec2u(), 0) + d.vec4f(1),
    );
  });
  const pipeline = root.createComputePipeline({ compute: kernel });
  const unused = root.createUniform(d.u32, 7);
  const forward = root.createBindGroup(layout, {
    unused,
    source: source.createView(d.texture2d(d.f32)),
    target: target.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const backward = root.createBindGroup(layout, {
    unused,
    source: target.createView(d.texture2d(d.f32)),
    target: source.createView(d.textureStorage2d('rgba16float', 'write-only')),
  });
  const encoder = root.createCommandEncoder(),
    pass = encoder.beginComputePass();
  pipeline.with(pass).with(forward).dispatchWorkgroups(1);
  pipeline.with(pass).with(backward).dispatchWorkgroups(1);
  pass.end();
  encoder.submit();
  await root.device.queue.onSubmittedWorkDone();
}

it('matches swapped resources and omits unused explicit-layout bindings', async () => {
  const dawn = await gpuRoot(),
    gpuObserver = observeScene(),
    cpuObserver = observeScene();
  const gpu = tgpu.initFromDevice({ device: gpuObserver.device(dawn.device) });
  let count = 0;
  let snapshot: ReplaySnapshot | undefined;
  const cpu = cpuRoot({
    generator: SweepGenerator,
    observeDevice: cpuObserver.device,
    beforeDispatch(code, label, dispatch) {
      if (count++ === 1) snapshot = snapshotDispatch(code, label, dispatch, cpuObserver);
    },
  });
  try {
    await swapped(gpu);
    await swapped(cpu);
    expect(snapshot).toBeDefined();
    const result = await replay(dawn.device, gpuObserver.dispatches, snapshot!);
    expect(result.differences).toEqual([]);
  } finally {
    gpu.destroy();
    cpu.destroy();
    dawn.destroy();
  }
});
