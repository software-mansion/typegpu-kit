import { expect, test } from 'vitest';
import { glRoot, readTexture, values } from './compare.ts';

const LAYERS = 300;

test('render passes into more layers than the device keeps framebuffers for', async () => {
  const root = glRoot();
  const { device } = root;
  try {
    const texture = device.createTexture({
      size: [4, 4, LAYERS],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const clear = (layer: number, value: number) => {
      const encoder = device.createCommandEncoder();
      const view = texture.createView({
        dimension: '2d',
        baseArrayLayer: layer,
        arrayLayerCount: 1,
      });
      encoder
        .beginRenderPass({
          colorAttachments: [
            { view, loadOp: 'clear', storeOp: 'store', clearValue: [value / 255, 0, 0, 1] },
          ],
        })
        .end();
      device.queue.submit([encoder.finish()]);
    };
    for (let layer = 0; layer < LAYERS; layer++) clear(layer, layer % 251);
    // The first layer's framebuffer was evicted long ago
    clear(0, 200);
    const red = async (layer: number) =>
      Math.round(values(await readTexture(device, texture, { layer }))[0]! * 255);
    expect([await red(0), await red(1), await red(150), await red(LAYERS - 1)]).toEqual([
      200,
      1,
      150,
      (LAYERS - 1) % 251,
    ]);
  } finally {
    root.destroy();
  }
});
