/**
 * Reports which graphics API a root runs on. A module reads it to choose another kernel, capacity
 * or technique on the WebGL fallback of `@typegpu/kit-gl`, where kernels that share workgroup
 * memory run on the CPU.
 *
 * @module
 * @unstable
 */

import type { TgpuRoot } from 'typegpu';

export interface Backend {
  /** Names the API: `'webgpu'` for a device from an adapter, `'webgl'` for the fallback's. */
  readonly kind: 'webgpu' | 'webgl';
  /**
   * Tells where kernels that share workgroup memory run. The fallback runs those on the CPU, row
   * by row, and most other kernels on the GPU.
   */
  readonly compute: 'gpu' | 'cpu';
  /** Gives the canvas format the backend presents without a conversion. */
  readonly preferredCanvasFormat: GPUTextureFormat;
  /**
   * Configures a canvas for the device and returns its context, as `canvas.getContext('webgpu')`
   * and `context.configure` do on WebGPU.
   */
  configure(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    configuration: Omit<GPUCanvasConfiguration, 'device'>,
  ): GPUCanvasContext;
}

/**
 * The key a device that is not WebGPU's describes itself under: `device[brand]` is its `Backend`.
 * `Symbol.for` keeps it one key across copies of the kit.
 */
export const brand: unique symbol = Symbol.for('@typegpu/kit/backend');

/**
 * Reads the backend of a root or its device.
 *
 * @returns the device's own description, or WebGPU's for a device without one
 */
export function of(root: TgpuRoot | GPUDevice): Backend {
  const device = 'device' in root ? root.device : root;
  const own = (device as { readonly [brand]?: Backend })[brand];
  if (own) return own;

  return {
    kind: 'webgpu',
    compute: 'gpu',
    get preferredCanvasFormat() {
      return navigator.gpu.getPreferredCanvasFormat();
    },
    configure(canvas, configuration) {
      const context = canvas.getContext('webgpu');
      if (!context) throw new Error('backends: the canvas has no WebGPU context.');
      context.configure({ ...configuration, device });
      return context;
    },
  };
}
