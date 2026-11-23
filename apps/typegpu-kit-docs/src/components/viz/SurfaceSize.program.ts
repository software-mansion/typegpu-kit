import * as native from '@typegpu/kit/native';
import { recordingRoot } from '@typegpu/kit/testing';

// A canvas with a context that only takes configuration; nothing here draws
const stubCanvas = (): native.Canvas => {
  const context = {
    canvas: { width: 1, height: 1 },
    configure() {},
    unconfigure() {},
    present() {},
  };
  return { getContext: () => context as unknown as GPUCanvasContext & { present(): void } };
};

export interface Options {
  /** The canvas's layout, in points. Default a phone's whole screen, 390 by 844. */
  readonly width?: number;
  readonly height?: number;
  /** The screen's pixels per point. Default 3. */
  readonly pixelRatio?: number;
  /** As native.surface takes it. Default 2. */
  readonly maxPixelRatio?: number;
}

/** A surface's size in pixels, and its pixels per point. */
export interface Size {
  readonly width: number;
  readonly height: number;
  readonly pixelRatio: number;
}

/**
 * The drawing buffer as a layout callback would size the surface, and the screen's pixels under the
 * canvas, which are the buffer of a surface whose cap is the screen's own ratio.
 */
export function createProgram({
  width = 390,
  height = 844,
  pixelRatio = 3,
  maxPixelRatio = 2,
}: Options = {}): { readonly buffer: Size; readonly screen: Size } {
  const { root } = recordingRoot();
  const sized = (cap: number): Size => {
    const surface = native.surface(root, stubCanvas(), {
      pixelRatio,
      maxPixelRatio: cap,
      format: 'bgra8unorm',
    });
    surface.resize(width, height);
    const size = { width: surface.width, height: surface.height, pixelRatio: surface.pixelRatio };
    surface.destroy();
    return size;
  };
  return { buffer: sized(maxPixelRatio), screen: sized(pixelRatio) };
}
