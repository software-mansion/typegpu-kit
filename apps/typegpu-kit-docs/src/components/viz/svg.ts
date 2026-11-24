import { bayer, plain } from '../site/dither.ts';
import { onKeys } from './figure.ts';

const NS = 'http://www.w3.org/2000/svg';
let patternId = 0;
const patterns = new WeakMap<SVGSVGElement, Map<string, SVGPatternElement>>();

/** The size of SVG text in viz.css, in CSS pixels. */
export const FONT = 11;

/** JetBrains Mono character width at FONT, using a 0.6 em advance. */
export const CHAR = FONT * 0.6;

/** Creates an SVG element with attributes, appended to `parent` when given. */
export function svg<K extends keyof SVGElementTagNameMap>(
  name: K,
  attributes: Record<string, string | number> = {},
  parent?: Element,
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  parent?.append(element);
  return element;
}

export function text(parent: Element, x: number, y: number, content: string, attributes = {}) {
  const element = svg('text', { x, y, ...attributes }, parent);
  element.textContent = content;
  return element;
}

/**
 * An 8×8 Bayer fill on the drawing's ground, in cells of `cell` SVG units, or `color` at `density`
 * with the dither off.
 */
export function stipple(
  parent: SVGElement,
  color = 'var(--viz-line)',
  density = 0.25,
  cell = 1,
): string {
  if (plain) return `color-mix(in srgb, ${color} ${Math.round(density * 100)}%, transparent)`;
  const figure = parent instanceof SVGSVGElement ? parent : parent.ownerSVGElement!;
  let fills = patterns.get(figure);
  if (!fills) patterns.set(figure, (fills = new Map()));
  const key = `${color}:${density}:${cell}`;
  let pattern = fills.get(key);
  if (!pattern || !figure.contains(pattern)) {
    let defs = figure.querySelector<SVGDefsElement>(':scope > defs[data-viz-patterns]');
    if (!defs) {
      fills.clear();
      defs = svg('defs', { 'data-viz-patterns': '' }, figure);
    }
    pattern = svg(
      'pattern',
      {
        id: `viz-stipple-${++patternId}`,
        width: 8 * cell,
        height: 8 * cell,
        patternUnits: 'userSpaceOnUse',
      },
      defs,
    );
    let cells = '';
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        if (bayer(x, y) < density) cells += `M${x * cell},${y * cell}h${cell}v${cell}h-${cell}Z`;
      }
    }
    svg('path', { d: cells, fill: color, 'shape-rendering': 'crispEdges' }, pattern);
    fills.set(key, pattern);
  }
  return `url(#${pattern.id})`;
}

/** A thin buffer or texture plate; its top keeps the caller's rectangular layout. */
export function plate(
  parent: SVGElement,
  x: number,
  y: number,
  width: number,
  height: number,
  {
    active = false,
    empty = false,
    depth = 4,
  }: { active?: boolean; empty?: boolean; depth?: number } = {},
) {
  const group = svg('g', {}, parent);
  const stroke = active ? 'var(--viz-accent)' : 'var(--viz-line)';
  svg(
    'path',
    {
      d: `M${x},${y + height}l${depth},${depth}h${width}v-${height}l-${depth},-${depth}M${x + width},${y + height}l${depth},${depth}`,
      fill: 'none',
      stroke,
    },
    group,
  );
  svg(
    'rect',
    {
      x,
      y,
      width,
      height,
      rx: 1,
      fill: empty ? 'none' : stipple(parent, active ? 'var(--viz-accent)' : 'var(--viz-line)'),
      stroke,
      ...(empty ? { 'stroke-dasharray': '3 3' } : {}),
    },
    group,
  );
  return group;
}

/** A mono number in a hairline box, centred at `x`. */
export function badge(
  parent: SVGElement,
  x: number,
  y: number,
  value: string | number,
  active = false,
) {
  const label = String(value);
  const width = Math.max(18, label.length * CHAR + 8);
  svg(
    'rect',
    {
      x: x - width / 2,
      y,
      width,
      height: 18,
      rx: 2,
      fill: 'var(--ground)',
      stroke: active ? 'var(--viz-accent)' : 'var(--viz-line)',
    },
    parent,
  );
  return text(parent, x, y + 12.5, label, {
    'text-anchor': 'middle',
    fill: active ? 'var(--accent-high)' : 'var(--text-2)',
  });
}

/** A label's hairline, with a dot at the part it names. */
export function leader(
  parent: SVGElement,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  active = false,
) {
  const color = active ? 'var(--viz-accent)' : 'var(--viz-line)';
  svg('line', { x1, y1, x2, y2, stroke: color }, parent);
  svg('circle', { cx: x2, cy: y2, r: 2, fill: color }, parent);
}

/** Fits SVG units to CSS pixels so text keeps its size on phones. `draw(width)` clears and redraws the figure, returning its height. Call the returned function once state is ready and after state changes. Later width changes redraw automatically. Below `min`, drawing uses `min` and scales down. */
export function fitWidth(
  figure: SVGSVGElement,
  draw: (width: number) => number,
  min = 0,
): () => void {
  // The authored width, for a figure that isn't laid out
  const fallback = figure.viewBox.baseVal?.width || 640;
  // Measured once, then kept by the observer, so a figure redrawn every frame forces no layout
  let laid = 0;
  let width = 0;

  const redraw = () => {
    laid ||= Math.round(figure.getBoundingClientRect().width);
    width = Math.max(laid || fallback, min);
    figure.replaceChildren();
    figure.setAttribute('viewBox', `0 0 ${width} ${draw(width)}`);
  };

  new ResizeObserver(([entry]) => {
    laid = Math.round(entry?.contentRect.width ?? 0);
    if (width !== 0 && Math.max(laid || fallback, min) !== width) redraw();
  }).observe(figure);

  return redraw;
}

/** Where a pointer event is, in `figure`'s units. */
export const pointIn = (figure: SVGSVGElement, event: MouseEvent) =>
  new DOMPoint(event.clientX, event.clientY).matrixTransform(figure.getScreenCTM()!.inverse());

/** SVG units an arrow key moves a handle by. */
const NUDGE = 5;

/**
 * Drags SVG handles: a press on a handle's element calls its `drag` with each point of the stroke,
 * in `figure`'s units, until the pointer lifts. A handle that gives `at`, where it is now, also
 * moves by arrow keys while its element has focus.
 */
export function onStroke(
  figure: SVGSVGElement,
  handles: readonly (readonly [Element, (point: DOMPoint) => void, (() => DOMPoint)?])[],
) {
  let dragging: ((point: DOMPoint) => void) | undefined;

  for (const [element, drag, at] of handles) {
    element.addEventListener('pointerdown', (event) => {
      dragging = drag;
      figure.setPointerCapture((event as PointerEvent).pointerId);
      drag(pointIn(figure, event as PointerEvent));
    });
    if (at) {
      onKeys(element, {
        arrow(dx, dy) {
          const { x, y } = at();
          drag(new DOMPoint(x + dx * NUDGE, y + dy * NUDGE));
        },
      });
    }
  }
  figure.addEventListener('pointermove', (event) => dragging?.(pointIn(figure, event)));
  figure.addEventListener('pointerup', () => (dragging = undefined));
  figure.addEventListener('pointercancel', () => (dragging = undefined));
}

/** Seeded random values keep diagrams reproducible across loads. */
export function seeded(seed: number) {
  let state = seed >>> 0;

  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
