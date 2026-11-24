/**
 * A `GPUDevice` implemented on a `WebGL2RenderingContext`, for TypeGPU to run on unchanged.
 */

export interface DeviceOptions {
  /**
   * The canvas the WebGL context is created on, which the device then presents to without a copy.
   * Default an `OffscreenCanvas`; every configured canvas then receives its frame as a bitmap.
   */
  readonly canvas?: HTMLCanvasElement | OffscreenCanvas;
  /** Default `'high-performance'`. */
  readonly powerPreference?: WebGLPowerPreference;
}

/** How `@typegpu/kit` recognises the fallback: `device[BACKEND]`, read by `backends.of(root)`. */
export const BACKEND: unique symbol = Symbol.for('@typegpu/kit/backend');

/** What the device reports about itself under `BACKEND`. */
export interface BackendInfo {
  readonly kind: 'webgl';
  /** Where compute pipelines run. */
  readonly compute: 'cpu';
  /** The canvas format the device presents without conversion. */
  readonly preferredCanvasFormat: GPUTextureFormat;
  /** Stands in for `canvas.getContext('webgpu')` and `context.configure`. */
  configure(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    configuration: Omit<GPUCanvasConfiguration, 'device'>,
  ): GPUCanvasContext;
}

export interface GLDevice extends GPUDevice {
  readonly gl: WebGL2RenderingContext;
  readonly [BACKEND]: BackendInfo;
}

/** Reports whether this environment can create a WebGL 2 context. */
export function isSupported(): boolean {
  if (typeof OffscreenCanvas === 'undefined') return false;
  return new OffscreenCanvas(1, 1).getContext('webgl2') !== null;
}

/** Creates the device. Throws if WebGL 2 is unavailable. */
export function createDevice(_options: DeviceOptions = {}): GLDevice {
  throw new Error('@typegpu/kit-gl: the WebGL device is not implemented yet');
}
