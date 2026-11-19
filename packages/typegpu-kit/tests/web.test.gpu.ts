import { gpuRoot } from '@typegpu/kit/testing/dawn';
import type { TgpuRoot } from 'typegpu';
import { afterEach, expect, it, vi } from 'vitest';
import type { Frame, Surface } from '../src/index.ts';
import * as web from '../src/web.ts';

afterEach(() => {
  vi.unstubAllGlobals();
});

/** A canvas whose context hands out a texture of the device it is configured with. */
function fakeCanvas(width: number, height: number) {
  let device: GPUDevice | undefined;
  let texture: GPUTexture | undefined;

  const context = {
    configure(configuration: GPUCanvasConfiguration) {
      device = configuration.device;
    },

    unconfigure() {
      texture?.destroy();
    },

    getCurrentTexture() {
      texture ??= device!.createTexture({
        size: [canvas.width, canvas.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      return texture;
    },
  };

  const canvas = {
    width: 1,
    height: 1,
    getContext: () => context,
    getBoundingClientRect: () => ({ width, height }),
  };

  vi.stubGlobal('window', { devicePixelRatio: 1 });
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
    },
  );

  return { canvas: canvas as unknown as HTMLCanvasElement, texture: () => texture! };
}

function clearing(_root: TgpuRoot, surface: Surface) {
  return {
    frame(frame: Frame) {
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), clearValue: [0.2, 0.4, 0.6, 1] },
      });
      pass.end();
    },

    destroy() {},
  };
}

it('requests a device with the features it can get, and draws on the canvas', async () => {
  await gpuRoot(); // installs Dawn's globals, whose navigator.gpu the host requests its device from
  const display: FrameRequestCallback[] = [];
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
    display.push(callback),
  );
  vi.stubGlobal('cancelAnimationFrame', () => {});

  // 64 RGBA8 pixels make a 256-byte row, the alignment texture copies need.
  const { canvas, texture } = fakeCanvas(64, 2);
  const onError = vi.fn();

  const hosted = web.host(
    clearing,
    {},
    { canvas, format: 'rgba8unorm', optionalFeatures: ['timestamp-query'], onError },
  );
  const device = await hosted.run(({ root }) => root.device);
  display.shift()?.(0);

  const pixels = device.createBuffer({
    size: 64 * 2 * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture: texture() },
    { buffer: pixels, bytesPerRow: 256 },
    [64, 2],
  );
  device.queue.submit([encoder.finish()]);
  await pixels.mapAsync(GPUMapMode.READ);

  expect([...new Uint8Array(pixels.getMappedRange(), 0, 4)]).toEqual([51, 102, 153, 255]);
  expect(device.features.has('timestamp-query')).toBe(true);
  expect(onError).not.toHaveBeenCalled();

  hosted.destroy();
  await device.lost;
  expect(onError).not.toHaveBeenCalled();
});
