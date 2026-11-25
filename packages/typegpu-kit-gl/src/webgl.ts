/**
 * Runs a TypeGPU Kit program on WebGL 2 where WebGPU is missing. The root it creates is TypeGPU's
 * own, on a device implemented over a WebGL 2 context, so scenes and modules run unchanged.
 * Kernels run on the GPU as fragment passes or points where their writes allow it, and on the
 * CPU otherwise; `LIMITATIONS.md` lists what differs.
 *
 * @module
 * @kind platform
 * @platform web
 * @unstable
 */

import { tgpu, type TgpuRoot } from 'typegpu';
import { createDevice, isSupported, type DeviceOptions } from './device/index.ts';
import { KitGenerator } from './generator.ts';
import { defineConstants } from './globals.ts';
import { assertPatched } from './patched.ts';

export interface InitOptions extends DeviceOptions {}

/** What `web.host` takes as `fallback`: it creates the root on the hosted canvas. */
export interface Fallback {
  readonly force: boolean;
  init(canvas: HTMLCanvasElement): Promise<TgpuRoot>;
}

export interface FallbackOptions extends Omit<InitOptions, 'canvas'> {
  /** Hosts on WebGL 2 even where WebGPU works, to see a scene on the fallback. Default false. */
  readonly force?: boolean;
}

/** Reports whether this browser can create a WebGL 2 context. */
export function supported(): boolean {
  return isSupported();
}

/**
 * Creates a root that runs on WebGL 2. Pass the canvas you draw to, so the device presents to it
 * without a copy; other canvases still work through `web.surface`. Throws if WebGL 2 is missing,
 * or if TypeGPU lacks the patch this package ships.
 *
 * @returns a root whose `destroy` also frees the WebGL device
 */
export async function init(options: InitOptions = {}): Promise<TgpuRoot> {
  assertPatched();
  defineConstants();
  const device = createDevice(options);
  const root = tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: KitGenerator });
  // TypeGPU reads WGSL language features from `navigator.gpu`, whatever the device. This device
  // compiles no WGSL, so it has none, also in a browser whose WebGPU has them but gave no adapter
  Object.defineProperty(root, 'enabledWgslLanguageFeatures', {
    value: new Set<string>(),
    configurable: true,
  });

  const destroyRoot = root.destroy.bind(root);
  root.destroy = () => {
    destroyRoot();
    device.destroy();
  };
  return root;
}

/**
 * Returns the fallback for `web.host`: `web.host(createScene, { canvas, fallback:
 * webgl.fallback() })` hosts on WebGPU where it works and on WebGL 2 where it does not.
 */
export function fallback(options: FallbackOptions = {}): Fallback {
  const { force = false, ...rest } = options;
  return { force, init: (canvas) => init({ ...rest, canvas }) };
}

export interface InitWithFallbackOptions extends InitOptions {
  /** Requests these WebGPU features when the adapter supports them. Default none. */
  readonly optionalFeatures?: Iterable<GPUFeatureName>;
}

/**
 * Creates a WebGPU root, or a WebGL 2 root where WebGPU is missing or gives no adapter. Use it in
 * a program that makes its own root instead of calling `web.host`. Read which one you got with
 * `backends.of(root).kind`.
 */
export async function initWithFallback(options: InitWithFallbackOptions = {}): Promise<TgpuRoot> {
  const { optionalFeatures = [], ...rest } = options;
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      return await tgpu.init({ device: { optionalFeatures } });
    } catch {
      // No adapter or no device: fall through to WebGL
    }
  }
  return init(rest);
}
