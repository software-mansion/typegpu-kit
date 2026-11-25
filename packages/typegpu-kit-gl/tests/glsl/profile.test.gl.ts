import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { it } from 'vitest';
import { loading } from '../../../../apps/typegpu-kit-docs/src/examples/files.ts';
import { begin, filesOf } from '../../../typegpu-kit-examples/src/hosts.ts';
import { exampleOf } from '../../../typegpu-kit-examples/src/index.ts';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot } from '../device/compare.ts';

/**
 * Where an example's frames go on the WebGL device, kernel by kernel: which run on the GPU, which
 * on the CPU and why, and what readback and uploads cost. Set `VITE_PROFILE` to examples, comma
 * separated; the test prints one JSON line per example.
 */
const keys = (import.meta.env.VITE_PROFILE ?? '').split(',').filter(Boolean);

/** Seconds an example may spend getting ready before it is measured anyway. */
const WARM_UP = 25;

if (keys.length === 0) it.skip('profiles the examples VITE_PROFILE names', () => {});

for (const key of keys) {
  it(`profiles ${key}`, async () => {
    const root = glRoot();
    const device = root.device as unknown as GLDevice;
    const surface = offscreenSurface(root, { width: 320, height: 180, format: 'rgba8unorm' });
    const errors: string[] = [];
    root.device.addEventListener('uncapturederror', (event) =>
      errors.push((event as GPUUncapturedErrorEvent).error.message.slice(0, 300)),
    );
    const { createScene } = await exampleOf(key).scene();
    const scene = await begin(await createScene(root, surface, await filesOf(key, loading())));
    const state = { ready: scene.ready === undefined };
    void scene.ready?.then(() => (state.ready = true));
    const loop = frames.create(root, scene.frame);
    let seconds = 0;
    let warmed = 0;
    const started = performance.now();
    while (!state.ready && performance.now() - started < WARM_UP * 1000 && warmed < 100) {
      loop.run(seconds);
      seconds += 0.04;
      warmed++;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    // Frames to run once ready, so a scene that builds up, such as a filling tank, is measured full
    const settle = Number(import.meta.env.VITE_PROFILE_FRAMES ?? 0);
    for (let i = 0; i < settle; i++) {
      loop.run(seconds);
      seconds += 0.04;
      if (i % 20 === 19) await root.device.queue.onSubmittedWorkDone();
    }
    device.stats.reset();
    const steady = 3;
    const start = performance.now();
    for (let i = 0; i < steady; i++) {
      loop.run(seconds);
      seconds += 0.04;
    }
    await root.device.queue.onSubmittedWorkDone();
    const frameMs = (performance.now() - start) / steady;
    const kernels = [...device.stats.kernels]
      .map(([label, stats]) => ({
        label,
        msPerFrame: Number((stats.ms / steady).toFixed(2)),
        invocationsPerFrame: Math.round(
          ((stats as { invocations?: number }).invocations ?? 0) / steady,
        ),
        dispatches: stats.dispatches,
      }))
      .toSorted((a, b) => b.msPerFrame - a.msPerFrame);
    console.log(
      `PROFILE ${JSON.stringify({
        key,
        ready: state.ready,
        warmUpFrames: warmed,
        frameMs: Number(frameMs.toFixed(1)),
        kernels,
        readback: device.stats.readback,
        upload: device.stats.upload,
        fallbacks: Object.fromEntries(device.stats.fallbacks),
        errors: errors.slice(0, 5),
      })}`,
    );
    loop.destroy();
    scene.destroy?.();
    surface.destroy();
  }, 600_000);
}
