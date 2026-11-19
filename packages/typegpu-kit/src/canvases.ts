import type { CanvasOptions } from './hosting.ts';
import type { FixedSize } from './targets.ts';

/** Returns the canvas configuration options other than the device and format. */
export function configuration(options: CanvasOptions) {
  const { alphaMode = 'opaque', usage, colorSpace, toneMapping } = options;

  return {
    alphaMode,
    ...(usage === undefined ? {} : { usage }),
    ...(colorSpace === undefined ? {} : { colorSpace }),
    ...(toneMapping === undefined ? {} : { toneMapping }),
  };
}

/**
 * Checks the fixed size and returns it with a pixel ratio. Throws with the `surface` name if
 * either dimension is not an integer from 1 to `maxSize`, or if `pixelRatio` is not above 0.
 * Default pixel ratio 1.
 */
export function checkSize(size: FixedSize, maxSize: number, surface: string): Required<FixedSize> {
  const { width, height, pixelRatio = 1 } = size;
  const whole = (side: number) => Number.isInteger(side) && side >= 1 && side <= maxSize;

  if (!whole(width) || !whole(height)) {
    throw new Error(
      `${surface} requires fixed dimensions in whole pixels from 1 to ${maxSize}, got ${width} × ${height}.`,
    );
  }
  if (!(pixelRatio > 0)) {
    throw new Error(`${surface} requires a fixed pixelRatio above 0, got ${pixelRatio}.`);
  }

  return { width, height, pixelRatio };
}
