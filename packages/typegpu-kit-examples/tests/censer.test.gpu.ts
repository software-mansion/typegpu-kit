import { backends, frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { createScene } from '../src/censer/scene.ts';

const size = { width: 64, height: 36 };

/** Runs the censer for a tenth of a second and returns the last frame's pixels. */
async function run(compute: 'gpu' | 'cpu') {
  // The compile test runs every example without texture-formats-tier1
  const root = await gpuRoot({ features: ['texture-formats-tier1'] });
  // A device that says its compute runs on the CPU averages the smoke in a kernel of its own
  if (compute === 'cpu') {
    Object.assign(root.device, {
      [backends.brand]: { ...backends.of(root), compute } satisfies backends.Backend,
    });
  }
  const surface = offscreenSurface(root, { ...size, format: 'rgba8unorm' });
  const scene = await createScene(root, surface);
  const loop = frames.create(root, (frame) => scene.frame(frame));
  for (let seconds = 0; seconds < 0.1; seconds += 1 / 120) loop.run(seconds);

  const buffer = root.device.createBuffer({
    size: size.width * 4 * size.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer(
    { texture: surface.texture },
    { buffer, bytesPerRow: size.width * 4 },
    size,
  );
  root.device.queue.submit([copy.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(buffer.getMappedRange().slice(0));
  loop.destroy();
  scene.destroy();
  root.destroy();
  return pixels;
}

it('runs the censer with its smoke in two halves on a real device, and alike where compute runs on the CPU', async () => {
  const shared = await run('gpu');
  const averaged = await run('cpu');
  let differing = 0;
  for (const [i, value] of shared.entries()) {
    if (Math.abs(value - (averaged[i] as number)) > 2) differing++;
  }
  expect(shared.some((value, i) => i % 4 !== 3 && value > 0)).toBe(true);
  expect(differing / shared.length).toBeLessThan(0.01);
});
