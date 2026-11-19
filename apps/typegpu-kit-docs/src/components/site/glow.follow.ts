import * as frames from '@typegpu/kit/frames';
import * as web from '@typegpu/kit/web';
import { d, type TgpuRoot } from 'typegpu';
import { pageRoot } from '../viz/gpu.ts';
import { bytesOf } from './dither.ts';
import type { LightOptions } from './glow.ts';
import {
  band,
  createGlow,
  rulesOf,
  SETTLE,
  strengthAt,
  type Glow,
  type Marker,
} from './glow.scene.ts';

const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * The glow's seconds, which pass only while a light moves. Under reduced motion they don't pass:
 * the smoke holds still and the light jumps.
 */
let clock = 4;
const now = () => clock;

interface Followed {
  readonly canvas: HTMLCanvasElement;
  readonly surface: web.Surface;
  readonly marker: Marker;
  readonly color: [number, number, number, number];
  readonly ruleColor: [number, number, number, number];
  /** The move the box holds, kept to start the next one from wherever the light is. */
  from: d.v4f;
  to: d.v4f;
  strength: d.v2f;
  start: number;
  /** Until the list's first frame, which starts the move set up before it. */
  waiting: boolean;
  /** Until a frame redraws the canvas, which resizing cleared. */
  stale: boolean;
  /** The canvas's width in CSS pixels, and the pixel ratio the box was last written with. */
  width: number;
  ratio: number;
  written: boolean;
}

const followed: Followed[] = [];
let ready: Promise<{ root: TgpuRoot; glow: Glow; draw: () => void } | undefined> | undefined;
let pending = false;
let ticking = false;

/** Whether a list needs a frame: its first, a redraw, or while its light moves. */
const drawing = (entry: Followed) =>
  entry.waiting || entry.stale || now() < entry.start + SETTLE + 0.1;

/** A colour of the theme, as sRGB from 0 to 1 with alpha 1. */
function colorOf(list: HTMLElement, value: string): [number, number, number, number] {
  const [r, g, b] = bytesOf(list, [value])[0];
  return [r / 255, g / 255, b / 255, 1];
}

/** Request a redraw after a resize or an immediate light move. */
function drawSoon() {
  if (pending || ticking) return;
  pending = true;

  requestAnimationFrame(() => {
    pending = false;
    void ready?.then((gpu) => gpu?.draw());
  });
}

/** Frames until every light has settled, its clock passing at the display's rate. */
function animate() {
  if (still) return drawSoon();
  if (ticking) return;
  ticking = true;

  let last: number | undefined;
  const tick = (milliseconds: number) => {
    // A frame that comes late, after a stall or a hidden tab, moves the light on by one step
    if (last !== undefined) clock += Math.min(milliseconds - last, 50) / 1000;
    last = milliseconds;

    void ready?.then((gpu) => gpu?.draw());
    if (followed.some(drawing)) requestAnimationFrame(tick);
    else ticking = false;
  };
  requestAnimationFrame(tick);
}

function setUp() {
  ready ??= pageRoot().then(async (root) => {
    if (!root) return undefined;

    // Compiled before the first frame, which then costs no more than the next
    const glow = createGlow(root, navigator.gpu.getPreferredCanvasFormat());
    await glow.initAsync();

    const loop = frames.create(root, (frame) => {
      const t = now();
      glow.time.write(t);

      for (const entry of followed) {
        if (!drawing(entry)) continue;

        // A busy start delays the light's move rather than skipping part of it
        if (entry.waiting) {
          entry.start = t;
          entry.waiting = false;
          entry.written = false;
        }
        entry.stale = false;

        // Use the drawing-buffer ratio because devicePixelRatio can disagree under zoom and emulation
        const view = entry.surface.view(frame);
        const ratio = entry.surface.width / entry.width;
        if (!entry.written || ratio !== entry.ratio) {
          const { color, from, to, strength, start } = entry;
          entry.marker.box.write({ color, before: from, after: to, strength, start, ratio });
          entry.ratio = ratio;
          entry.written = true;
        }

        const pass = frame.encoder.beginRenderPass({
          colorAttachments: { view, clearValue: [0, 0, 0, 0] },
        });
        entry.marker.draw(pass);
        pass.end();
      }
    });

    // The glow reads its own clock, not the frame's time
    const draw = () => loop.run(performance.now() / 1000);

    return { root, glow, draw };
  });

  return ready;
}

/**
 * Sends the light toward `next`, an entry's edges in CSS pixels of its list, or puts it out. It
 * starts from wherever it is now, so a move can turn mid-way; a light that is out comes up in place.
 */
function move(entry: Followed, next: d.v4f | undefined) {
  const t = now();
  const here = band(entry.from, entry.to, t - entry.start);
  const lit = strengthAt(entry.strength, t - entry.start);
  const there = next ?? here;
  const target = next ? 1 : 0;
  const jump = still || lit <= 0;

  entry.from = jump ? there : here;
  entry.to = there;
  entry.strength = d.vec2f(still ? target : jump ? 0 : lit, target);
  entry.start = t;
  entry.written = false;
}

const edgesKey = (edges: d.v4f | undefined) =>
  edges ? `${edges.x} ${edges.y} ${edges.z} ${edges.w}` : '';

/**
 * Lights `current()` inside `list`, which must not scroll itself, and moves the light whenever the
 * entry changes, the list's layout changes or an attribute in it does; the rules move with the
 * layout. False without a WebGPU device; throws where one fails to set up.
 */
export async function follow(
  list: HTMLElement,
  current: () => HTMLElement | null,
  { before, rules }: LightOptions,
) {
  const gpu = await setUp();
  if (!gpu) return false;

  const canvas = document.createElement('canvas');
  canvas.className = 'glow';
  canvas.setAttribute('aria-hidden', 'true');
  list.prepend(canvas);

  const entry: Followed = {
    canvas,
    surface: web.surface(gpu.root, canvas, { alphaMode: 'premultiplied', maxPixelRatio: 1 }),
    marker: gpu.glow.marker(),
    color: colorOf(list, 'var(--accent)'),
    ruleColor: colorOf(list, 'var(--rule-strong)'),
    from: d.vec4f(),
    to: d.vec4f(),
    strength: d.vec2f(),
    start: -SETTLE,
    waiting: true,
    stale: true,
    width: canvas.clientWidth || 1,
    ratio: 1,
    written: false,
  };
  followed.push(entry);

  // For an entry in a folded group, or a list that isn't showing, the light goes out. A folded
  // group's content keeps its boxes where the browser only skips drawing it
  const edgesOf = (target: Element | null | undefined) => {
    if (target?.closest('details:not([open])')) return undefined;

    const box = target?.getBoundingClientRect();
    if (!box || box.width === 0 || box.height === 0) return undefined;

    const corner = list.getBoundingClientRect();
    return d.vec4f(
      box.left - corner.left,
      box.top - corner.top,
      box.right - corner.left,
      box.bottom - corner.top,
    );
  };

  // A rule is the middle of its element's left border, as tall as the element
  let ruled = '';
  const rule = () => {
    const lines = [...(rules?.() ?? [])].flatMap((element) => {
      const edges = edgesOf(element);
      return edges ? [d.vec4f(edges.x + 0.5, edges.y, edges.w, 0)] : [];
    });
    const key = lines.map(edgesKey).join(',');
    if (key === ruled) return;
    ruled = key;

    entry.marker.rules.write(rulesOf(entry.ruleColor, lines));
    entry.stale = true;
    drawSoon();
  };

  let placed = '';
  const place = () => {
    const next = edgesOf(current());
    const edges = edgesKey(next);
    if (edges === placed) return;
    placed = edges;

    move(entry, next);
    animate();
  };

  new ResizeObserver(() => {
    entry.width = canvas.clientWidth || 1;
    entry.stale = true;
    rule();
    place();
    drawSoon();
  }).observe(list);
  new MutationObserver(() => {
    rule();
    place();
  }).observe(list, {
    subtree: true,
    attributes: true,
    attributeFilter: ['aria-current', 'open'],
  });

  const setOut = edgesOf(before);
  if (setOut) {
    entry.from = entry.to = setOut;
    entry.strength = d.vec2f(1, 1);
  }
  rule();
  place();
  return true;
}
