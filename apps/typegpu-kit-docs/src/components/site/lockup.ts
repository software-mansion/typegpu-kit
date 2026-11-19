import { clamp } from '../home/geometry.ts';
import { bayer, bytesOf, type Rgb } from './dither.ts';

/** Rows of cells the lockup is printed in: the logo's own pixels, one a CSS pixel in the header. */
export const ROWS = 31;

/** Seconds the light takes to cross the lockup. */
const CROSS = 1.1;

export interface PlateOptions {
  /** The pointer's lens radius, in CSS pixels. Default 18. */
  readonly lens?: number;
  /** What the pointer's lens and crossing listen on. Default the canvas's parent. */
  readonly over?: HTMLElement;
}

interface Cell {
  readonly at: number;
  readonly x: number;
  readonly y: number;
  readonly threshold: number;
  readonly ramp: readonly (Rgb | null)[];
}

const mix = (a: Rgb, b: Rgb, t: number): Rgb => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/** The SVG as an image, at its own size. */
export async function imageOf(svg: SVGSVGElement) {
  const source = new XMLSerializer().serializeToString(svg);
  const url = URL.createObjectURL(new Blob([source], { type: 'image/svg+xml' }));
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Prints `svg`, the lockup, into `canvas`, which lies over it at its size, in `ROWS` rows of cells,
 * and hides the SVG once it has; the SVG stays in the page for its size and for readers.
 */
export async function plate(
  canvas: HTMLCanvasElement,
  svg: SVGSVGElement,
  options: PlateOptions = {},
) {
  const { lens = 18 } = options;
  const over = options.over ?? canvas.parentElement ?? canvas;
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  const image = await imageOf(svg);
  const [rule, accent, bright] = bytesOf(canvas, [
    'var(--rule-strong)',
    'var(--accent)',
    'var(--accent-high)',
  ]) as [Rgb, Rgb, Rgb];

  const context = canvas.getContext('2d')!;
  let cells: Cell[] = [];
  let small = new OffscreenCanvas(1, 1);
  let printed = new ImageData(1, 1);
  let width = 1;

  const build = () => {
    const box = canvas.getBoundingClientRect();
    if (box.width === 0) return;
    const ratio = Math.max(1, Math.round(devicePixelRatio));
    const size = Math.max(1, Math.round((box.height / ROWS) * ratio));
    const columns = Math.max(1, Math.ceil((box.width * ratio) / size));
    const rows = Math.max(1, Math.ceil((box.height * ratio) / size));
    width = box.width;
    canvas.width = columns * size;
    canvas.height = rows * size;

    small = new OffscreenCanvas(columns, rows);
    const reading = small.getContext('2d', { willReadFrequently: true })!;
    reading.drawImage(image, 0, 0, columns, rows);
    const { data } = reading.getImageData(0, 0, columns, rows);
    printed = new ImageData(columns, rows);

    cells = [];
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < columns; i++) {
        const at = (j * columns + i) * 4;
        if ((data[at + 3] ?? 0) < 128) continue;
        const own: Rgb = [data[at], data[at + 1], data[at + 2]];
        cells.push({
          at,
          x: (i + 0.5) / columns,
          y: (j + 0.5) / rows,
          threshold: bayer(i, j),
          ramp: [null, mix(rule, own, 0.45), own, mix(own, accent, 0.6), bright],
        });
      }
    }
  };

  // The light's middle across the lockup, from 0 to 1, and the pointer in CSS pixels
  let band = -1;
  let pointer: { x: number; y: number } | null = null;

  const draw = () => {
    const { data } = printed;
    data.fill(0);
    const height = canvas.getBoundingClientRect().height;
    for (const c of cells) {
      const across = (c.x - band) * 4.2;
      let light = Math.exp(-across * across);
      if (pointer) {
        const dx = c.x * width - pointer.x;
        const dy = c.y * height - pointer.y;
        light += Math.exp(-(dx * dx + dy * dy) / (lens * lens)) * 0.9;
      }
      const steps = c.ramp.length - 1;
      const place = clamp(0.5 + light * 0.55, 0, 0.9999) * steps;
      const low = Math.floor(place);
      const color = c.ramp[place - low > c.threshold ? low + 1 : low];
      if (!color) continue;
      data[c.at] = color[0];
      data[c.at + 1] = color[1];
      data[c.at + 2] = color[2];
      data[c.at + 3] = 255;
    }
    small.getContext('2d')!.putImageData(printed, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = false;
    context.drawImage(small, 0, 0, canvas.width, canvas.height);
  };

  let frame = 0;
  let crossing: number | null = null;

  const tick = (now: number) => {
    frame = 0;
    if (crossing !== null) {
      const k = (now - crossing) / 1000 / CROSS;
      band = -0.25 + k * 1.5;
      if (k >= 1) {
        crossing = null;
        band = -1;
      }
    }
    draw();
    if (crossing !== null) frame = requestAnimationFrame(tick);
  };
  const run = () => {
    if (!frame) frame = requestAnimationFrame(tick);
  };
  const cross = () => {
    if (still.matches || crossing !== null) return;
    crossing = performance.now();
    run();
  };

  over.addEventListener('pointerenter', cross);
  over.addEventListener('pointermove', (event) => {
    const box = canvas.getBoundingClientRect();
    pointer = { x: event.clientX - box.left, y: event.clientY - box.top };
    run();
  });
  over.addEventListener('pointerleave', () => {
    pointer = null;
    run();
  });

  new ResizeObserver(() => {
    build();
    run();
  }).observe(canvas);
  build();
  draw();
  svg.style.visibility = 'hidden';
}
