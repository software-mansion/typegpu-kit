// The strip of frames the two clock figures share: boxes as wide as the frames took, scrolling
// left as time passes
import { action, animation } from './figure.ts';
import { seeded, stipple, svg, text } from './svg.ts';

/** Pixels a second takes, so a pixel is a millisecond. */
export const PER_SECOND = 1000;

/** How many times slower than real time a strip runs. */
const SLOWDOWN = 5;

/** Seconds a stall adds to the running frame. */
export const STALL = 0.15;

/** Pixels right of the present, so the running frame's edge and the axis's last label show. */
export const PRESENT = 4;

/** Seconds a paused strip passes to show a change. */
const SHOW = 0.12;

/** Seconds of finished frames a strip keeps, more than the widest figure shows. */
const KEPT = 1.6;

/** Seconds between a strip's own hitches. */
const HITCH_EVERY = 0.3;

/** How far a frame's length strays from its display's, as a fraction. */
const JITTER = 0.15;

/** Seconds between the axis's marks. */
const MARK = 0.2;

interface Frame {
  /** How many frames came before it, which keeps its band as the strip scrolls. */
  readonly index: number;
  readonly start: number;
  /** When it ends. A stall moves the end of the running frame. */
  end: number;
}

interface StripOptions<R> {
  readonly seed: number;
  /** The display rate a frame starts at, in frames a second. */
  readonly fps: () => number;
  /** Seconds the strip's own hitches add to the running frame, in turn, 0 for none. */
  readonly hitches: readonly number[];
  /** When the first of them comes, in seconds. */
  readonly first: number;
  /** Runs the figure's clock over a finished frame, and gives what the figure draws of it. */
  readonly run: (frame: Frame) => R;
}

interface Strip<R> {
  readonly now: number;
  /** The finished frames, oldest first, each with what `run` gave for it. */
  readonly frames: readonly (Frame & { readonly ran: R })[];
  /** The frame that has not ended yet. */
  readonly running: Frame;
  /** Passes `seconds`, finishing the frames that end within them. */
  advance(seconds: number): void;
  /** Passes time until the running frame and `more` seconds after it have ended. */
  finish(more?: number): void;
  /** Lengthens the running frame by `seconds`. */
  stall(seconds: number): void;
  /** Runs every kept frame again, after the figure's clock changed. */
  replay(): void;
}

/** Frames of a display's length, give or take, one after another from time 0. */
export function frameStrip<R>(options: StripOptions<R>): Strip<R> {
  const { fps, hitches, run } = options;
  const random = seeded(options.seed);
  const length = () => (1 - JITTER + random() * 2 * JITTER) / fps();
  const frames: (Frame & { ran: R })[] = [];
  let running: Frame = { index: 0, start: 0, end: length() };
  let now = 0;
  let hitch = options.first;
  let hitched = 0;

  const strip: Strip<R> = {
    get now() {
      return now;
    },

    frames,

    get running() {
      return running;
    },

    advance(seconds) {
      now += seconds;
      for (;;) {
        if (hitch <= Math.min(now, running.end)) {
          running.end += hitches[hitched++ % hitches.length] ?? 0;
          hitch += HITCH_EVERY;
        } else if (running.end <= now) {
          frames.push({ ...running, ran: run(running) });
          running = { index: running.index + 1, start: running.end, end: running.end + length() };
        } else break;
      }
      while (frames.length > 0 && frames[0].end < now - KEPT) frames.shift();
    },

    finish(more = 0) {
      strip.advance(Math.max(running.end - now, 0) + more);
    },

    stall(seconds) {
      running.end += seconds;
    },

    replay() {
      for (const frame of frames) frame.ran = run(frame);
    },
  };
  return strip;
}

/**
 * Plays `strip` at SLOWDOWN with the figure's play button, drawing it with `draw`, and stalls its
 * running frame by STALL with the stall button. `show` passes enough time for a change to show
 * while paused, and draws.
 */
export function playStrip<R>(root: HTMLElement, strip: Strip<R>, draw: () => void) {
  const shown = animation(root, {
    step: (dt) => strip.advance(dt / SLOWDOWN),
    draw,
    play: action(root, 'play'),
  });

  const show = () => {
    if (shown.playing) return;
    strip.finish(SHOW);
    shown.redraw();
  };

  action(root, 'stall', () => {
    strip.stall(STALL);
    show();
  });
  return { show, redraw: shown.redraw };
}

const clips = new WeakMap<SVGSVGElement, string>();
let clipped = 0;

/** A strip's place in a figure: `left` to `right`, with the strip's present at `right`. */
interface Scroll {
  /** What the strip draws into, in pixels since time 0, which keeps a dither still on its box. */
  readonly group: SVGGElement;
  /** The earliest time in view. */
  readonly from: number;
}

/** Where a time sits in a scroll's group. */
export const x = (time: number) => time * PER_SECOND;

/** The same, on a pixel's centre, for a hairline. */
export const edge = (time: number) => Math.round(time * PER_SECOND) + 0.5;

/** Starts a strip's drawing, clipped to `left`…`right` over `height`. */
export function scroll(
  figure: SVGSVGElement,
  now: number,
  left: number,
  right: number,
  height: number,
): Scroll {
  let id = clips.get(figure);
  if (!id) clips.set(figure, (id = `viz-strip-${++clipped}`));
  const clip = svg('clipPath', { id }, figure);
  svg('rect', { x: left, y: 0, width: right - left, height }, clip);
  const inside = svg('g', { 'clip-path': `url(#${id})` }, figure);
  // Whole pixels, so hairlines and dither stay sharp as they move
  const group = svg('g', { transform: `translate(${right - Math.round(x(now))})` }, inside);
  return { group, from: now - (right - left) / PER_SECOND };
}

/**
 * Draws a lane of frame boxes, each as wide as its frame took, every other one banded, and the
 * running frame open at the present. `lit` gives the colour of a frame the figure lights.
 */
export function frameBoxes<R>(
  figure: SVGSVGElement,
  { group, from }: Scroll,
  strip: Strip<R>,
  top: number,
  height: number,
  lit: (ran: R) => string | undefined = () => undefined,
) {
  const shown = strip.frames.filter((frame) => frame.end > from);
  const box = (start: number, end: number, attributes: Record<string, string | number>) =>
    svg(
      'rect',
      {
        x: edge(start),
        y: top + 0.5,
        width: edge(end) - edge(start),
        height: height - 1,
        ...attributes,
      },
      group,
    );

  for (const frame of shown) {
    if (frame.index % 2 === 1 && lit(frame.ran) === undefined) {
      box(frame.start, frame.end, {
        fill: stipple(figure, 'var(--viz-line)', 0.25),
        stroke: 'none',
      });
    }
  }
  // Lit boxes last, so a neighbour's hairline never crosses their edge
  for (const lighting of [false, true]) {
    for (const frame of shown) {
      const colour = lit(frame.ran);
      if ((colour !== undefined) !== lighting) continue;
      box(frame.start, frame.end, {
        fill: colour ? stipple(figure, colour, 0.5) : 'none',
        stroke: colour ?? 'var(--viz-line)',
      });
    }
  }

  const open = Math.round(x(strip.now));
  svg(
    'path',
    {
      d: `M${open},${top + 0.5}H${edge(strip.running.start)}V${top + height - 0.5}H${open}`,
      fill: 'none',
      stroke: 'var(--viz-accent)',
    },
    group,
  );
}

/** Marks the present and each fifth of a second before it, under a strip whose lanes end at `y`. */
export function axis(figure: SVGSVGElement, left: number, right: number, y: number) {
  for (let mark = 0; right - x(mark * MARK) >= left; mark++) {
    const at = right - Math.round(x(mark * MARK)) - 0.5;
    svg('line', { x1: at, x2: at, y1: y, y2: y + 4, stroke: 'var(--viz-line)' }, figure);
    // The last label needs room left of its mark
    if (mark > 0 && at - left < 28) continue;
    text(
      figure,
      mark === 0 ? right : at,
      y + 16,
      mark === 0 ? 'now' : `−${Math.round(mark * MARK * 1000)} ms`,
      {
        'text-anchor': mark === 0 ? 'end' : 'middle',
      },
    );
  }
}
