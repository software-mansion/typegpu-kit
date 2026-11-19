/**
 * Provides a react-native-webgpu canvas surface that pauses in the background, and a check that
 * two threads can share a device.
 *
 * @module
 * @kind platform
 * @platform native
 * @requires implicit-device-synchronization required react-native-webgpu requests it for every
 * device; `checkDevice` throws without it. Devices used from two threads require it to avoid
 * crashes
 * @allocates each frame: a view of the canvas's texture, which is new every frame
 * @owns the canvas's configuration
 */

import type { TgpuRoot } from 'typegpu';
import { checkSize, configuration } from './canvases.ts';
import type { Frame } from './frames.ts';
import type { CanvasOptions } from './hosting.ts';
import type { Surface as Base, FixedSize } from './targets.ts';

export type { FixedSize } from './targets.ts';

/** Provides the `getContext` method of react-native-webgpu's `CanvasRef`, from a `<Canvas ref>`. */
export interface Canvas {
  getContext(type: 'webgpu'): (GPUCanvasContext & { present(): void }) | null;
}

export interface SurfaceOptions extends CanvasOptions {
  /** Sets the display pixel ratio from `PixelRatio.get()`. */
  readonly pixelRatio: number;
  /** Caps drawing-buffer pixels per point unless `size` is set. Default 2. */
  readonly maxPixelRatio?: number;
  /**
   * Fixes the drawing-buffer size. `resize` keeps this size.
   * Default the layout size in points multiplied by the capped pixel ratio.
   */
  readonly size?: FixedSize;
}

export interface Surface extends Base {
  /** Provides the canvas context, configured for the root. */
  readonly context: GPUCanvasContext;
  /**
   * Reports whether the surface is paused. True between `pause` and `resume`. While true,
   * `animate` skips frames and `view` throws.
   */
  readonly paused: boolean;
  /** Sizes the drawing buffer from the canvas's layout size in points, as `onLayout` reports it. */
  resize(width: number, height: number): void;
  /**
   * Fixes the drawing-buffer size, as `options.size` does. Pass `undefined` to follow the layout.
   */
  setSize(size: FixedSize | undefined): void;
  /** Pauses frames while the app is in the background, where it must not submit GPU work. */
  pause(): void;
  /**
   * Resumes the surface and reconfigures its context with the same options. This recovers
   * drawables that can remain unavailable after a long stay in the background.
   */
  resume(): void;
}

/**
 * Creates a react-native-webgpu canvas surface for presentation on the JS thread. Throws if the
 * canvas has no WebGPU context or the fixed size is invalid.
 *
 * @param canvas the `CanvasRef` from a `<Canvas ref>`; its context's canvas holds the size
 * @param options.pixelRatio `PixelRatio.get()`
 * @returns a surface you resize, pause, resume and destroy
 */
export function surface(root: TgpuRoot, canvas: Canvas, options: SurfaceOptions): Surface {
  const { pixelRatio, maxPixelRatio = 2 } = options;
  const format = options.format ?? navigator.gpu.getPreferredCanvasFormat();
  const maxSize = root.device.limits.maxTextureDimension2D;
  let fixed = options.size && checkSize(options.size, maxSize, 'native.surface');

  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('The canvas has no WebGPU context.');
  const buffer = context.canvas;

  const configured = { device: root.device, format, ...configuration(options) };
  const configure = () => context.configure(configured);
  configure();

  const ratio = Math.min(pixelRatio, maxPixelRatio);
  const side = (points: number) => Math.max(1, Math.min(maxSize, Math.round(points * ratio)));
  const layout = { width: buffer.width, height: buffer.height };

  const apply = () => {
    buffer.width = fixed?.width ?? layout.width;
    buffer.height = fixed?.height ?? layout.height;
  };
  apply();

  let texture: GPUTexture | undefined;
  let current: GPUTextureView | undefined;
  let acquired = false;
  let scheduled: Frame | undefined;
  let paused = false;

  // Present only after the frame has acquired the canvas texture
  const present = () => {
    if (!acquired) return;
    acquired = false;
    context.present();
  };

  const destroy = () => context.unconfigure();

  return {
    context,
    format,

    get pixelRatio() {
      return fixed?.pixelRatio ?? ratio;
    },

    get width() {
      return buffer.width;
    },

    get height() {
      return buffer.height;
    },

    get paused() {
      return paused;
    },

    resize(width, height) {
      layout.width = side(width);
      layout.height = side(height);
      apply();
    },

    setSize(next) {
      fixed = next && checkSize(next, maxSize, 'native.surface');
      apply();
    },

    view(frame) {
      if (paused) {
        throw new Error(
          'native.surface cannot call view() while paused. Give the surface to animate to skip paused frames.',
        );
      }

      const next = context.getCurrentTexture();
      if (next !== texture) {
        texture = next;
        current = next.createView();
      }

      acquired = true;
      if (scheduled !== frame) {
        scheduled = frame; // a frame that was never submitted dropped its callback
        frame.afterSubmit(present);
      }

      return current!;
    },

    pause() {
      paused = true;
    },

    resume() {
      if (!paused) return;
      paused = false;
      texture = undefined;
      configure();
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

/**
 * Throws unless the device has `implicit-device-synchronization`, which a device used from two
 * threads requires. react-native-webgpu requests it for every device unless the descriptor sets
 * `implicitDeviceSynchronization: false`.
 *
 * @param device the device to check, such as a `<GPUDeviceProvider>`'s
 * @returns the device
 */
export function checkDevice(device: GPUDevice): GPUDevice {
  if (device.features.has(sync)) return device;
  throw new Error(
    "The device lacks implicit-device-synchronization, so two threads can't share it. Request the feature in requiredFeatures, and don't set implicitDeviceSynchronization: false.",
  );
}

const sync = 'implicit-device-synchronization' as GPUFeatureName;
