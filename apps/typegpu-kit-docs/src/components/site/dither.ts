import type { Colors } from 'typegpu-kit-pull/dither.ts';
import { clamp } from '../home/geometry.ts';

export { bayerCell, COLORS, toneOf, type Colors } from 'typegpu-kit-pull/dither.ts';

const BAYER = [
  0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28,
  52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7,
  39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
];

/** The 8×8 ordered threshold of the cell at column `i` and row `j`, in (0, 1). */
export const bayer = (i: number, j: number) => ((BAYER[(j % 8) * 8 + (i % 8)] ?? 0) + 0.5) / 64;

export type Rgb = readonly [number, number, number];

/**
 * `colors`, any CSS colours, as sRGB bytes, resolved where `element` is. A computed colour can be in
 * any space, such as oklab from `color-mix`, so each is read back from a pixel.
 */
export function bytesOf(element: Element, colors: readonly string[]): Rgb[] {
  const probe = document.createElement('i');
  probe.style.display = 'none';
  element.append(probe);
  const context = new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true })!;
  const bytes = colors.map((color): Rgb => {
    probe.style.color = color;
    context.clearRect(0, 0, 1, 1);
    context.fillStyle = getComputedStyle(probe).color;
    context.fillRect(0, 0, 1, 1);
    const [r = 0, g = 0, b = 0] = context.getImageData(0, 0, 1, 1).data;
    return [r, g, b];
  });
  probe.remove();
  return bytes;
}

/** The tokens `Colors` are made of, greys from the ground up, then the accents. */
const TOKENS = [
  'var(--ground)',
  'var(--ground-3)',
  'var(--rule-strong)',
  'color-mix(in oklab, var(--rule-strong), var(--muted))',
  'var(--muted)',
  'var(--text-2)',
  'var(--text)',
  'var(--accent)',
  'var(--accent-high)',
];

/** The colours from the tokens where `element` is. */
export function colorsOf(element: Element): Colors {
  const [g0, g1, g2, g3, g4, g5, g6, c0, c1] = bytesOf(element, TOKENS).map(([r, g, b]): Rgb => [
    r / 255,
    g / 255,
    b / 255,
  ]) as [Rgb, Rgb, Rgb, Rgb, Rgb, Rgb, Rgb, Rgb, Rgb];
  return { grey: [g0, g1, g2, g3, g4, g5, g6], accent: [c0, c1] };
}

/** From the ground up: the raised grounds and rules, then the accent at its brightest. */
const LIGHT = [
  'var(--ground-2)',
  'var(--ground-3)',
  'var(--ground-4)',
  'var(--rule-strong)',
  'color-mix(in oklab, var(--accent) 40%, var(--rule-strong))',
  'var(--accent)',
  'var(--accent-high)',
];

/** A ramp of `colors` resolved where `element` is, with a clear step below them. */
export const rampOf = (element: Element, colors: readonly string[] = LIGHT): (Rgb | null)[] => [
  null,
  ...bytesOf(element, colors),
];

/** Fill `canvas` at its layout size with `cell`-wide CSS-pixel cells. `field` receives normalised coordinates and returns a ramp position in [0, 1] for the cell centre. */
export function paint(
  canvas: HTMLCanvasElement,
  field: (x: number, y: number) => number,
  { cell = 2, ramp = rampOf(canvas) }: { cell?: number; ramp?: readonly (Rgb | null)[] } = {},
) {
  const { width, height } = canvas.getBoundingClientRect();
  const ratio = Math.max(1, Math.round(devicePixelRatio));
  const size = Math.max(1, Math.round(cell * ratio));
  const columns = Math.max(1, Math.ceil((width * ratio) / size));
  const rows = Math.max(1, Math.ceil((height * ratio) / size));

  const cells = new ImageData(columns, rows);
  const steps = ramp.length - 1;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < columns; i++) {
      const at = clamp(field((i + 0.5) / columns, (j + 0.5) / rows)) * steps;
      const low = Math.min(Math.floor(at), steps - 1);
      const color = ramp[at - low > bayer(i, j) ? low + 1 : low];
      if (!color) continue;
      const k = (j * columns + i) * 4;
      cells.data.set(color, k);
      cells.data[k + 3] = 255;
    }
  }

  const small = new OffscreenCanvas(columns, rows);
  small.getContext('2d')!.putImageData(cells, 0, 0);
  if (canvas.width !== columns * size) canvas.width = columns * size;
  if (canvas.height !== rows * size) canvas.height = rows * size;
  const context = canvas.getContext('2d')!;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.imageSmoothingEnabled = false;
  context.drawImage(small, 0, 0, canvas.width, canvas.height);
}
