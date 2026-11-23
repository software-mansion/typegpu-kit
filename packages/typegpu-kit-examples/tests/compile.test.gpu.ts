import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { it } from 'vitest';
import { examples } from '../src/index.ts';
import { create } from '../src/hosts.ts';
import { fromDisk } from './utils/examples.ts';

/** Fewer of what these make many of: still enough to cull and pose over several workgroups. */
const fewer: Record<string, object> = {
  forest: { trees: 2000 },
  'quiver-trees': { count: 400 },
  skeletons: { count: 100 },
};

type Run = [title: string, key: string, options: object, features: readonly string[]];

/**
 * Every example on a device with the features it uses where Dawn has them, and again without, as
 * a device that lacks them runs it. The jump flood runs again with a bind group per pass where the
 * device has immediates.
 */
const runs: Run[] = [
  ...examples.map(({ key, title, features = [] }): Run => [title, key, fewer[key] ?? {}, features]),
  ...examples
    .filter(({ features }) => features?.length)
    .map(({ key, title }): Run => [`${title}, without its features`, key, fewer[key] ?? {}, []]),
  ['Jump flood, with bind groups', 'jump-flood', { immediates: false }, []],
];

it.each(runs)(
  'runs the frames of %s on a real device',
  async (_, key, options, features) => {
    const root = await gpuRoot({ optionalFeatures: features as GPUFeatureName[] });
    const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
    const scene = await create(key, root, surface, fromDisk, options);
    const loop = frames.create(root, (frame) => scene.frame(frame));

    // A scene that compiles inside its frames, as behind a page's cover, gets them until it draws,
    // and fails here with the error that stops it.
    const state = { drawing: scene.ready === undefined };
    const ready = scene.ready?.then(() => (state.drawing = true));
    let seconds = 0;
    for (; !state.drawing; seconds += 0.04) {
      loop.run(seconds);
      await Promise.race([ready, new Promise((resolve) => setTimeout(resolve, 10))]);
    }

    for (const end = seconds + 0.12; seconds < end; seconds += 0.04) loop.run(seconds);
    await root.device.queue.onSubmittedWorkDone();

    scene.destroy?.();
    root.destroy();
  },
  // The quiver trees read four scans.
  20_000,
);
