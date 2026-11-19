import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { frames } from '../src/index.ts';
import { offscreenSurface } from '../src/testing.ts';

it('takes a frame on a real device, into pixels a test can read', async () => {
  const root = await gpuRoot();
  const surface = offscreenSurface(root, { width: 64, height: 2, format: 'rgba8unorm' });

  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: surface.view(frame), clearValue: [0.2, 0.4, 0.6, 1] },
    });
    pass.end();
  });
  loop.run(0);

  // 64 RGBA8 pixels make a 256-byte row, the alignment texture copies need.
  const { device } = root;
  const pixels = device.createBuffer({
    size: 64 * 2 * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture: surface.texture },
    { buffer: pixels, bytesPerRow: 256 },
    [64, 2],
  );
  device.queue.submit([encoder.finish()]);

  await pixels.mapAsync(GPUMapMode.READ);
  expect([...new Uint8Array(pixels.getMappedRange(), 0, 4)]).toEqual([51, 102, 153, 255]);
  root.destroy();
});
