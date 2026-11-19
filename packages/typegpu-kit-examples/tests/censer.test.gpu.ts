import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { it } from 'vitest';
import { createScene } from '../src/censer/scene.ts';

// The compile test runs every example without texture-formats-tier1
it('runs the censer with its smoke in two halves on a real device', async () => {
  const root = await gpuRoot({ features: ['texture-formats-tier1'] });
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  const scene = await createScene(root, surface);
  const loop = frames.create(root, (frame) => scene.frame(frame));
  for (let seconds = 0; seconds < 0.1; seconds += 1 / 120) loop.run(seconds);
  await root.device.queue.onSubmittedWorkDone();
  loop.destroy();
  scene.destroy();
  root.destroy();
});
