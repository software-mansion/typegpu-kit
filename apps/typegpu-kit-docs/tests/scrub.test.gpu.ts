import { cameras, passes, targets } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { createSparks } from '../src/components/viz/Scrub.scene.ts';

const size = { width: 64, height: 64 };

it('draws the same sparks at a moment however the replay reached it', async () => {
  const root = await gpuRoot();
  const screen = targets.create(root, size, { color: { format: 'rgba8unorm' } });
  const view = cameras.flat(root, { size, height: 10 });
  view.update(0);
  const canvas = passes.create(root, screen, { color: passes.color }).pipe(view.inject());
  const { since, drawing } = createSparks(root);
  const drawn = canvas.add(drawing);
  await canvas.initAsync();

  const at = async (seconds: number) => {
    since.write(seconds);
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: { view: screen.views.color } });
    drawn.draw(pass);
    pass.end();
    encoder.submit();

    const readback = root.device.createBuffer({
      size: size.width * 4 * size.height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const copy = root.device.createCommandEncoder();
    copy.copyTextureToBuffer(
      { texture: root.unwrap(screen.textures.color) },
      { buffer: readback, bytesPerRow: size.width * 4 },
      [size.width, size.height],
    );
    root.device.queue.submit([copy.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange().slice(0));
    readback.destroy();
    return pixels;
  };

  const forward = await at(0.3);
  const later = await at(0.5);
  const rewound = await at(0.3);
  const gone = await at(2);

  expect(forward.some((channel) => channel > 0)).toBe(true);
  expect(later).not.toEqual(forward);
  expect(rewound).toEqual(forward);
  expect(gone.every((channel) => channel === 0)).toBe(true);
  root.destroy();
});
