// Lays out the profiler graph as shapes and text. The web panel paints it on a 2D canvas in
// graph.ts; the React Native panel draws its shapes with WebGPU in graphGpu.ts and its text with
// React Native.

import {
  type Bar,
  formatMs,
  perFrame,
  type Processor,
  processors,
  type Segment,
} from './recorder.ts';

export const theme = {
  font: '600 11px system-ui, -apple-system, sans-serif',
  text: '#e8e8ee',
  muted: '#8a8a99',
  faint: '#5c5c6b',
  accent: '#3498db',
};

export const palette = [
  '#1abc9c',
  '#3498db',
  '#2ecc71',
  '#e74c3c',
  '#9b59b6',
  '#f1c40f',
  '#e67e22',
  '#16a085',
  '#2980b9',
  '#27ae60',
  '#c0392b',
  '#8e44ad',
];

const colors = {
  outline: 'rgba(255, 255, 255, 0.55)',
  budget: 'rgba(255, 255, 255, 0.12)',
  peak: '#b8b0d8',
  rest: '#6b6b7b',
};
const padding = 10;
const graphWidth = { share: 0.38, max: 240 };
const column = { width: 4, gap: 1 };
const columnStep = column.width + column.gap;
const rowHeight = 15;
const legendBars = 5;
const legendOffset = 33;
const frameBudgets = [8.33, 16.67];
const scaleSteps = [0.25, 0.5, 1, 2, 4, 8.33, 16.67, 33.33, 66.67, 133.33];
const scaleHeadroom = 1.1;
const scalePercentile = 0.9;
const scaleEasing = 0.1;
const textHeight = 13;
const swatchWidth = 6;
const numberWidth = 46;
const dash = 3;
const dimmed = 0.55;
const legendAlpha = 0.85;

/**
 * Floats per shape: x, y, width and height in points, premultiplied RGBA, then how far the right
 * edge's top and bottom move down from the left edge's, and two floats of padding. Rectangles move
 * neither.
 */
export const rectFloats = 12;

/** Text placed over the graph, in points from its top left. */
export interface Label {
  readonly text: string;
  readonly x: number;
  readonly y: number;
  /** Width the text fits in; `right` labels end at `x + width`. */
  readonly width: number;
  readonly align: 'left' | 'right';
  readonly color: string;
}

export interface Layout {
  /** `count` shapes of `rectFloats` floats each. */
  readonly rects: Float32Array<ArrayBuffer>;
  readonly count: number;
  readonly labels: Label[];
  /** Whether a section's scale still eases toward its target, so a later layout differs. */
  readonly easing: boolean;
}

export interface LayoutOptions {
  /** Shows a note in the GPU section, such as a missing `timestamp-query`. */
  readonly note?: string | undefined;
  /** Highlights this column, counted from the newest, and lists its segments in the legends. */
  readonly hovered?: number | undefined;
}

export interface GraphLayout {
  /** How many columns fit in a section at this size, so a reader can ask for no more. */
  columns(width: number): number;
  /**
   * Lays out the newest bars in a graph of this size in points. `seconds` since the previous
   * layout eases each section's scale toward its target.
   */
  layout(
    bars: Bar[],
    width: number,
    height: number,
    seconds: number,
    options?: LayoutOptions,
  ): Layout;
  /** The column under a point in the last layout, counted from the newest, if any. */
  columnAt(x: number, y: number): number | undefined;
  colorOf(name: string): string;
}

type Box = { x: number; y: number; w: number; h: number };
type Span = { top: number; bottom: number };

/**
 * Titles a section with its milliseconds per frame. GPU times come from frames paced by the
 * display, where the GPU lowers its clock between frames.
 */
export function sectionTitle(processor: Processor, ms: number): string {
  return processor === 'gpu' ? `gpu ${formatMs(ms)} at vsync` : `cpu ${formatMs(ms)}`;
}

function total(segments: Segment[]): number {
  return segments.reduce((sum, segment) => sum + segment.ms, 0);
}

function scaleFor(bars: Bar[], processor: Processor): number {
  const peaks = bars.map((bar) => bar.peak[processor]);
  peaks.sort((a, b) => a - b);
  const reference = Math.max(
    0.1,
    peaks[Math.floor((peaks.length - 1) * scalePercentile)] ?? 0,
    ...bars.map((bar) => total(bar.segments[processor]) / Math.max(bar.frames[processor], 1)),
  );
  return (
    scaleSteps.find((step) => step >= reference * scaleHeadroom) ??
    Math.ceil(reference * scaleHeadroom)
  );
}

/** Parses `#rrggbb` and `rgba(r, g, b, a)` into premultiplied RGBA from 0 to 1. */
export function premultiplied(color: string): [number, number, number, number] {
  const hex = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(color);
  const rgba = /^rgba?\(([^)]+)\)$/.exec(color);
  const channels = hex
    ? [...hex.slice(1).map((part) => Number.parseInt(part, 16)), 1]
    : rgba
      ? rgba[1].split(',').map(Number)
      : [];
  const [r = 0, g = 0, b = 0, a = 1] = channels;
  return [(r / 255) * a, (g / 255) * a, (b / 255) * a, a];
}

const sectionWidth = (width: number) =>
  Math.round(Math.min(width * graphWidth.share, graphWidth.max));

/**
 * Creates a layout of GPU and CPU milliseconds per frame as stacked columns, with the newest on
 * the right, that keeps each section's eased scale and each segment's colour. The legends list
 * the largest segments of the last columns or the hovered one.
 */
export function createGraphLayout(capacity: number): GraphLayout {
  const rects = new Float32Array(capacity * rectFloats);
  const segmentColors = new Map<string, string>();
  const scales: Record<Processor, number> = { gpu: 1, cpu: 1 };
  let easing = false;
  let boxes: Box[] = [];
  let count = 0;

  function colorOf(name: string): string {
    let color = segmentColors.get(name);
    if (!color) {
      color = palette[segmentColors.size % palette.length];
      segmentColors.set(name, color);
    }
    return color;
  }

  /** Adds a shape whose right edge's top and bottom sit `top` and `bottom` below its left's. */
  function shape(
    x: number,
    y: number,
    w: number,
    h: number,
    color: string,
    alpha = 1,
    top = 0,
    bottom = 0,
  ) {
    if (count === capacity || w <= 0 || Math.max(h, h + bottom - top) <= 0) return;
    const [r, g, b, a] = premultiplied(color);
    rects.set(
      [x, y, w, h, r * alpha, g * alpha, b * alpha, a * alpha, top, bottom],
      count * rectFloats,
    );
    count++;
  }

  function section(
    processor: Processor,
    bars: Bar[],
    box: Box,
    ease: number,
    labels: Label[],
    hovered: number | undefined,
    note: string | undefined,
  ) {
    const visible = bars.slice(-Math.floor(box.w / columnStep));
    const target = scaleFor(visible, processor);
    scales[processor] += (target - scales[processor]) * ease;
    if (Math.abs(target - scales[processor]) > target * 1e-3) easing = true;
    else scales[processor] = target;
    const pixelsPerMs = box.h / scales[processor];
    const top = box.y + 1;
    const bottom = box.y + box.h;

    // Outline
    shape(box.x, box.y, box.w, 1, colors.outline);
    shape(box.x, bottom - 1, box.w, 1, colors.outline);
    shape(box.x, box.y, 1, box.h, colors.outline);
    shape(box.x + box.w - 1, box.y, 1, box.h, colors.outline);

    for (const budget of frameBudgets.filter((ms) => ms < scales[processor])) {
      const y = Math.round(bottom - budget * pixelsPerMs);
      for (let x = box.x + 1; x < box.x + box.w - 1; x += dash * 2) {
        shape(x, y, Math.min(dash, box.x + box.w - 1 - x), 1, colors.budget);
      }
    }

    visible.forEach((bar, index) => {
      const slot = visible.length - 1 - index;
      const alpha = hovered === undefined || hovered === slot ? 1 : dimmed;
      const x = box.x + box.w - (slot + 1) * columnStep;
      let y = bottom - 1;
      for (const segment of bar.segments[processor]) {
        const height = Math.min((segment.ms / bar.frames[processor]) * pixelsPerMs, y - top);
        shape(x, y - height, column.width, height, colorOf(segment.name), alpha);
        y -= height;
      }
      if (bar.frames[processor] > 0) {
        const peak = Math.max(top, Math.round(bottom - bar.peak[processor] * pixelsPerMs) - 1);
        shape(x, peak, column.width, 2, colors.peak, alpha);
      }
    });

    const focus = hovered === undefined ? undefined : visible[visible.length - 1 - hovered];
    // Draw CPU-only columns while GPU timestamp readbacks are pending
    const recent = visible.filter((bar) => bar.frames[processor] > 0).slice(-legendBars);
    const segments = perFrame(focus ? [focus] : recent, processor).filter(({ ms }) => ms > 0);
    const inner = box.w - 10;
    labels.push(
      {
        text: sectionTitle(processor, total(segments)),
        x: box.x + 5,
        y: box.y + 2,
        width: inner,
        align: 'left',
        color: theme.muted,
      },
      {
        text: formatMs(target),
        x: box.x + 5,
        y: box.y + 2,
        width: inner,
        align: 'right',
        color: theme.faint,
      },
    );
    if (note) {
      labels.push({
        text: note,
        x: box.x + 5,
        y: box.y + box.h / 2 - textHeight / 2,
        width: inner,
        align: 'left',
        color: theme.muted,
      });
    }

    legend(box, segments, pixelsPerMs, labels);
  }

  function legend(box: Box, segments: Segment[], pixelsPerMs: number, labels: Label[]) {
    // Each segment's span on the stacked column, which its legend row links to
    let stacked = box.y + box.h;
    const spans = segments.map(({ ms }) => {
      const bottom = stacked;
      stacked = Math.max(stacked - ms * pixelsPerMs, box.y);
      return { top: stacked, bottom };
    });

    const rows = Math.max(1, Math.floor(box.h / rowHeight));
    const bySize = [...segments];
    bySize.sort((a, b) => b.ms - a.ms);
    const largest = new Set(bySize.slice(0, segments.length > rows ? rows - 1 : rows));
    const entries: { color: string; ms: number; text: string; span?: Span }[] = [];
    segments.forEach((segment, index) => {
      if (!largest.has(segment)) return;
      const { name, ms, calls } = segment;
      const repeats = calls >= 1.5 ? ` ×${Math.round(calls)}` : '';
      entries.push({ color: colorOf(name), ms, text: `${name}${repeats}`, span: spans[index] });
    });
    const rest = segments.filter((segment) => !largest.has(segment));
    if (rest.length > 0) {
      entries.push({ color: colors.rest, ms: total(rest), text: `${rest.length} more` });
    }

    const edge = box.x + box.w + 1;
    const swatch = edge + legendOffset;
    entries.forEach(({ color, ms, text, span }, row) => {
      const rowBottom = box.y + box.h - row * rowHeight - 1;
      const rowTop = rowBottom - rowHeight + 3;
      if (span) {
        const h = span.bottom - span.top;
        shape(
          edge,
          span.top,
          swatch - edge,
          h,
          color,
          legendAlpha,
          rowTop - span.top,
          rowBottom - span.bottom,
        );
      }
      shape(swatch, rowTop, swatchWidth, rowBottom - rowTop, color, legendAlpha);
      const y = rowBottom - textHeight;
      labels.push(
        {
          text: formatMs(ms),
          x: swatch + swatchWidth,
          y,
          width: numberWidth,
          align: 'right',
          color: theme.text,
        },
        {
          text,
          x: swatch + swatchWidth + numberWidth + 6,
          y,
          width: Number.POSITIVE_INFINITY,
          align: 'left',
          color: theme.text,
        },
      );
    });
  }

  return {
    columns: (width) => Math.floor(sectionWidth(width) / columnStep),

    layout(bars, width, height, seconds, { note, hovered } = {}) {
      count = 0;
      easing = false;
      const labels: Label[] = [];
      const ease = 1 - (1 - scaleEasing) ** Math.max(1, seconds * 60);
      const h = (height - padding * 3) / 2;
      const w = sectionWidth(width);

      boxes = processors.map((processor, index) => {
        const box = { x: padding, y: padding + index * (h + padding), w, h };
        const sectionNote = processor === 'gpu' ? note : undefined;
        section(processor, bars, box, ease, labels, hovered, sectionNote);
        return box;
      });

      // Fit legend names to the graph's right edge
      const fitted = labels.map((label) =>
        Number.isFinite(label.width)
          ? label
          : { ...label, width: Math.max(0, width - padding - label.x) },
      );

      return { rects: rects.subarray(0, count * rectFloats), count, labels: fitted, easing };
    },

    columnAt(x, y) {
      const box = boxes.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
      return box && Math.floor((box.x + box.w - x) / columnStep);
    },

    colorOf,
  };
}
