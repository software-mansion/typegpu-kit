import { finish, frames, pipelines, targets } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { createFeedback } from '../src/afterimage/feedback.ts';

it('fades an image through both feedback targets and clears it on reset', async () => {
  const root = await gpuRoot();
  const size = { width: 64, height: 1 };
  const surface = offscreenSurface(root, { ...size, format: 'rgba8unorm' });
  const screen = targets.create(root, size, { hdr: { format: 'rgba16float', sampled: true } });
  const source = () => screen.textures.hdr;
  const echo = createFeedback(root, size, { output: screen.textures.hdr }, { source: source });
  const final = finish.create(root, {
    source: echo,
    format: surface.format,
    tone: null,
    encode: false,
    settings: { exposure: 1, vignette: 0, dither: 0 },
  });
  await pipelines.compile([echo, final]);
  const pixel = root.device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let light = 1;
  const loop = frames.create(root, (frame) => {
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: screen.views.hdr, clearValue: [light, light, light, 1] },
    });
    pass.end();
    echo.record(frame, { halfLife: 0.1, bloom: 0, vertical: 0 });
    final.record(frame, surface.view(frame));
    root
      .unwrap(frame.encoder)
      .copyTextureToBuffer(
        { texture: surface.texture },
        { buffer: pixel, bytesPerRow: 256 },
        [64, 1],
      );
  });
  const shown = async (seconds: number) => {
    loop.run(seconds);
    await pixel.mapAsync(GPUMapMode.READ);
    const value = new Uint8Array(pixel.getMappedRange())[0];
    pixel.unmap();
    return value;
  };

  try {
    expect(await shown(0)).toBe(255);
    light = 0;
    expect(await shown(0.1)).toBeCloseTo(128, 0);
    expect(await shown(0.2)).toBeCloseTo(64, 0);
    echo.reset();
    expect(await shown(0.3)).toBe(0);
  } finally {
    loop.destroy();
    pixel.destroy();
    echo.destroy();
    screen.destroy();
    surface.destroy();
    root.destroy();
  }
});
