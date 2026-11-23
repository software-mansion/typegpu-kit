import { createGraphLayout, rectFloats, theme } from './graphLayout.ts';
import type { Bar } from './recorder.ts';

export interface Graph {
  /**
   * Paints the bars when they, the pointer, the canvas size or the note changed since the last
   * paint, or while a scale eases. `revision` tells changed bars apart.
   */
  draw(bars: Bar[], revision: number, gpuNote?: string): void;
  colorOf(name: string): string;
}

/** Shapes the graph can draw: columns, peaks, budgets, outlines, swatches and their links. */
const capacity = 4096;

function fitText(context: CanvasRenderingContext2D, text: string, width: number): string {
  if (context.measureText(text).width <= width) return text;

  let low = 0;
  let high = text.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (context.measureText(`${text.slice(0, middle)}…`).width <= width) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return `${text.slice(0, low)}…`;
}

/**
 * Paints the graph laid out by `graphLayout` on a 2D canvas. Hovering a column freezes the graph
 * and shows its legend.
 */
export function createGraph(canvas: HTMLCanvasElement): Graph {
  const context = canvas.getContext('2d') as CanvasRenderingContext2D;
  const layout = createGraphLayout(capacity);
  let hovered: number | undefined;
  let frozen: Bar[] | undefined;
  let previous: number | undefined;
  let painted: unknown[] = [];
  let easing = false;

  canvas.addEventListener('pointermove', (event) => {
    const box = canvas.getBoundingClientRect();
    hovered = layout.columnAt(event.clientX - box.left, event.clientY - box.top);
  });
  canvas.addEventListener('pointerleave', () => {
    hovered = undefined;
  });

  function draw(bars: Bar[], revision: number, gpuNote?: string) {
    frozen = hovered === undefined ? undefined : (frozen ?? bars.slice());
    const now = performance.now() / 1000;
    const seconds = previous === undefined ? 1 : now - previous;
    previous = now;

    const ratio = window.devicePixelRatio;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const state = [frozen ?? revision, hovered, width, height, ratio, gpuNote];
    if (!easing && state.every((value, index) => value === painted[index])) return;
    painted = state;
    if (
      canvas.width !== Math.round(width * ratio) ||
      canvas.height !== Math.round(height * ratio)
    ) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const laid = layout.layout(frozen ?? bars, width, height, seconds, { note: gpuNote, hovered });
    const { rects, count, labels } = laid;
    easing = laid.easing;
    for (let index = 0; index < count; index++) {
      const [x, y, w, h, r, g, b, a, top, bottom] = rects.subarray(
        index * rectFloats,
        index * rectFloats + 10,
      );
      if (!a) continue;
      context.fillStyle = `rgba(${(r / a) * 255}, ${(g / a) * 255}, ${(b / a) * 255}, ${a})`;
      if (!top && !bottom) {
        context.fillRect(x, y, w, h);
        continue;
      }
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(x + w, y + top);
      context.lineTo(x + w, y + h + bottom);
      context.lineTo(x, y + h);
      context.fill();
    }

    context.font = theme.font;
    context.textBaseline = 'top';
    for (const label of labels) {
      context.fillStyle = label.color;
      context.textAlign = label.align;
      const x = label.align === 'right' ? label.x + label.width : label.x;
      context.fillText(fitText(context, label.text, label.width), x, label.y + 1);
    }
  }

  return { draw, colorOf: layout.colorOf };
}
