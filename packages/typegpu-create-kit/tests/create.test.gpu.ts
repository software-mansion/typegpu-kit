import { readdirSync } from 'node:fs';
import path from 'node:path';
import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { describe, it } from 'vitest';

const scenes = readdirSync(path.join(import.meta.dirname, '../scenes'));

// Instances and particles draw one way on a device with indirect-first-instance and another
// without it. Nearly every device can have the feature, but only a program that asks gets it, and
// the starter's host asks for none.
const devices: [string, GPUFeatureName[]][] = [
  ['with indirect-first-instance', ['indirect-first-instance']],
  ['without it', []],
];

describe.each(devices)('%s', (_name, features) => {
  it.each(scenes)('runs the frames of the %s scene on a real device', async (name) => {
    const root = await gpuRoot({ features });
    const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
    const { createScene } = await import(`../scenes/${name}/scene.ts`);
    const scene = await createScene(root, surface);
    const loop = frames.create(root, scene.frame);

    for (let seconds = 0; seconds < 0.16; seconds += 0.04) loop.run(seconds);
    await root.device.queue.onSubmittedWorkDone();

    scene.destroy?.();
    root.destroy();
  });
});
