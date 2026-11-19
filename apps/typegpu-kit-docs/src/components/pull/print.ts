// Posters printed as the site's veil prints them: 3 px cells, an 8×8 Bayer threshold, six levels a
// channel. The foil shader reads the same cells, so its print matches this one where it shows none.

import { CELL } from 'typegpu-kit-pull/dither.ts';
import { bayer, type Rgb } from '../site/dither.ts';

export { CELL };

/** A poster shrunk to a pixel a cell, cropped as `object-fit: cover` would. */
export interface Cells {
  readonly columns: number;
  readonly rows: number;
  readonly pixels: Uint8ClampedArray<ArrayBuffer>;
}

const images = new Map<string, Promise<HTMLImageElement>>();

/** The poster at `address`, decoded once for the page. */
export function posterImage(address: string) {
  let loading = images.get(address);
  if (!loading) {
    const image = new Image();
    image.src = address;
    loading = image.decode().then(() => image);
    images.set(address, loading);
  }
  return loading;
}

/** `image` in cells filling a box `width` × `height` CSS pixels. */
export function cellsOf(image: HTMLImageElement, width: number, height: number): Cells {
  const columns = Math.max(1, Math.ceil(width / CELL));
  const rows = Math.max(1, Math.ceil(height / CELL));
  const context = new OffscreenCanvas(columns, rows).getContext('2d', {
    willReadFrequently: true,
  })!;
  context.imageSmoothingQuality = 'high';
  const scale = Math.max(columns / image.naturalWidth, rows / image.naturalHeight);
  const [w, h] = [image.naturalWidth * scale, image.naturalHeight * scale];
  context.drawImage(image, (columns - w) / 2, (rows - h) / 2, w, h);
  return { columns, rows, pixels: context.getImageData(0, 0, columns, rows).data };
}

/** `value`, a byte, printed in six levels through `threshold`. */
const level = (value: number, threshold: number) => {
  const at = (value / 255) * 6;
  const low = Math.floor(at);
  return ((low + (at - low > threshold ? 1 : 0)) / 6) * 255;
};

/** Prints `cells` into `canvas`, a pixel a cell, never darker than `ground`. */
export function printInto(canvas: HTMLCanvasElement, cells: Cells, ground: Rgb) {
  const { columns, rows, pixels } = cells;
  const image = new ImageData(columns, rows);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < columns; i++) {
      const threshold = bayer(i, j);
      const at = (j * columns + i) * 4;
      image.data[at] = Math.max(ground[0], level(pixels[at], threshold));
      image.data[at + 1] = Math.max(ground[1], level(pixels[at + 1], threshold));
      image.data[at + 2] = Math.max(ground[2], level(pixels[at + 2], threshold));
      image.data[at + 3] = 255;
    }
  }
  canvas.width = columns;
  canvas.height = rows;
  canvas.style.width = `${columns * CELL}px`;
  canvas.style.height = `${rows * CELL}px`;
  canvas.getContext('2d')!.putImageData(image, 0, 0);
}
