/**
 * Runs whole examples on a root and reads back their last frame, so a test can compare the WebGL
 * device with WebGPU on the same frames.
 */

import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import type { TgpuRoot } from 'typegpu';
import type { Loading } from '../../../typegpu-kit-examples/src/contract.ts';
import { begin, filesOf } from '../../../typegpu-kit-examples/src/hosts.ts';
import { exampleOf } from '../../../typegpu-kit-examples/src/index.ts';
import { type Readback, readTexture } from './compare.ts';

export interface SceneRun {
  readonly frame: Readback;
  /** `uncapturederror` messages and errors thrown while the scene ran. */
  readonly errors: readonly string[];
  /** Milliseconds per steady frame, recording and submitting. */
  readonly frameTime: number;
}

const unavailable = (what: string) => () => Promise.reject(new Error(`${what} is not available in this test`));

const loading: Loading = {
  read: unavailable('reading files'),
  font: unavailable('fonts'),
  decode: unavailable('image decoding'),
  draco: unavailable('Draco'),
} as unknown as Loading;

/**
 * Creates the example's scene on an offscreen surface, runs frames 1/25 s apart until it is ready
 * and then `steady` more, and reads the surface.
 */
export async function runScene(
  root: TgpuRoot,
  key: string,
  options: { width?: number; height?: number; steady?: number } = {},
): Promise<SceneRun> {
  const { width = 160, height = 120, steady = 6 } = options;
  const errors: string[] = [];
  const listener = (event: Event) => errors.push((event as GPUUncapturedErrorEvent).error.message);
  root.device.addEventListener('uncapturederror', listener);
  const surface = offscreenSurface(root, { width, height, format: 'rgba8unorm' });
  let destroy: (() => void) | undefined;
  let frameTime = 0;
  try {
    const { createScene } = await exampleOf(key).scene();
    const files = await filesOf(key, loading);
    const scene = await begin(await createScene(root, surface, files));
    destroy = () => scene.destroy?.();
    let ready = scene.ready === undefined;
    void scene.ready?.then(() => (ready = true));
    const loop = frames.create(root, scene.frame);
    let seconds = 0;
    for (let i = 0; i < 100 && !ready; i++) {
      loop.run(seconds);
      seconds += 0.04;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const start = performance.now();
    for (let i = 0; i < steady; i++) {
      loop.run(seconds);
      seconds += 0.04;
    }
    frameTime = (performance.now() - start) / steady;
    await root.device.queue.onSubmittedWorkDone();
    const frame = await readTexture(root.device, surface.texture);
    loop.destroy();
    return { frame, errors, frameTime };
  } catch (error) {
    errors.push(`thrown: ${(error as Error).message}`);
    return { frame: { width, height, format: 'rgba8unorm', bytes: new Uint8Array(width * height * 4) }, errors, frameTime };
  } finally {
    try {
      destroy?.();
    } catch (error) {
      errors.push(`destroy: ${(error as Error).message}`);
    }
    surface.destroy();
    root.device.removeEventListener('uncapturederror', listener);
  }
}
