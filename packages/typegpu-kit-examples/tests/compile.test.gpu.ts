import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { describe, it } from 'vitest';
import { examples } from '../src/index.ts';
import { create } from '../src/hosts.ts';
import { fromDisk } from './utils/examples.ts';

/** Fewer of what these make many of: still enough to cull and pose over several workgroups. */
const fewer: Record<string, object> = {
  forest: { trees: 2000 },
  'quiver-trees': { count: 400 },
  skeletons: { count: 100 },
};

/** Every example, and the jump flood again with a bind group per pass where the device has immediates. */
const runs: [string, string, object][] = [
  ...examples.map(({ key, title }): [string, string, object] => [title, key, fewer[key] ?? {}]),
  ['Jump flood, with bind groups', 'jump-flood', { immediates: false }],
];

// Culled instances draw through one pipeline with indirect-first-instance, which desktop browsers
// have, and through one that reads each draw's mesh from a bind group without it.
const devices: [string, GPUFeatureName[]][] = [
  ['with indirect-first-instance', ['indirect-first-instance']],
  ['without it', []],
];

describe.each(devices)('%s', (_name, features) => {
  it.each(runs)(
    'runs the frames of %s on a real device',
    async (_, key, options) => {
      const root = await gpuRoot({ features });
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
});
