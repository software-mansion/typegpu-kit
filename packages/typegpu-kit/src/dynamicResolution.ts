/**
 * Picks a resolution scale that keeps a GPU timing under a budget.
 *
 * @module
 * @records `update(frame)` once a frame, before the targets that read `scale` fit
 */

import type { Frame } from './frames.ts';
import type { Timing } from './profiler.ts';

export interface Options {
  /**
   * Sets the milliseconds the timing should stay under, such as 6 for a frame's GPU time, or a
   * function of the display's frame interval in milliseconds, called at every judgement:
   * `(interval) => 0.75 * interval`. The interval is measured between calls of `update` on the
   * `now` clock, 1000 / 60 until two have come.
   * @cpu
   */
  readonly budget: number | ((interval: number) => number);
  /** Bounds the scale, lowest first, at most 1. Default [0.5, 1]. */
  readonly range?: readonly [number, number];
  /**
   * Reads the clock that frames arrive on, in milliseconds, to measure the display's interval. A
   * frame's `time` is the program's clock, which it may slow or pause. Default
   * `performance.now()`.
   * @cpu
   */
  readonly now?: () => number;
}

export interface DynamicResolution {
  /**
   * Reports the scale for a target set's `resolution`, the top of `range` until a move or a set.
   * Moves keep it on steps of 0.05 within `range`. Setting it clamps to `range`, ignores samples
   * of earlier frames and throws a RangeError unless the value is finite.
   */
  scale: number;
  /**
   * Reads the timing's newest sample, if one arrived, and moves `scale` when the samples call for
   * it. Call once a frame, before the targets that read the scale fit.
   */
  update(frame: Frame): void;
}

/** Scales are multiples of 1 / STEPS. */
const STEPS = 20;
/** Samples in the median a move is judged on. */
const WINDOW = 9;
/** Seconds at one scale before a move up. */
const RISE = 1;
/**
 * Seconds for which a scale that ran over budget stays out of reach of moves up. Running over
 * again within twice that doubles it, up to four times.
 */
const HOLD = 10;
/** Fraction of the budget a move aims for. */
const AIM = 0.85;
/** How far the measured interval rises toward a longer frame, per frame. */
const CREEP = 0.002;

/** The middle of three values. */
const middle = (a: number, b: number, c: number) =>
  Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));

const snap = (scale: number) => Math.floor(scale * STEPS + 1e-6) / STEPS;

/**
 * Picks a resolution scale that keeps `timing` under `budget`. It assumes time grows with pixels,
 * the square of the scale, to step close to the budget in one move: down as soon as a median of
 * recent samples runs over, up after a second at one scale with room to spare, at least one step
 * while the median is under 85% of the budget. It judges only samples of frames recorded at the
 * current scale. It measures the display's interval as the middle of the last three, dropping to
 * a shorter one at once and rising slowly toward a longer one, so frames that miss the display
 * keep it near the display's. Without samples, as without `timestamp-query`, the scale stays where it is: set
 * it after `create` to start lower. Throws unless `budget` is finite and above 0, when created or
 * computed, and `range` runs from above 0 up to at most 1.
 *
 * @param timing what to keep under budget, such as a profiler's `frame` or a pass's timer
 * @returns a resolution controller you update every frame and read the scale of
 */
export function create(timing: Timing, options: Options): DynamicResolution {
  const { range = [0.5, 1], now = () => performance.now() } = options;
  const [low, high] = range;
  let interval = 1000 / 60;
  // The two previous times between frames. A middle of three ignores one early frame in a burst
  const recent = [interval, interval];
  let arrived: number | undefined;
  const measure = () => {
    const clock = now();
    const elapsed = clock - (arrived ?? clock);
    arrived = clock;
    if (!(elapsed > 0)) return;
    const typical = middle(elapsed, recent[0], recent[1]);
    interval = typical < interval ? typical : interval + (typical - interval) * CREEP;
    recent[1] = recent[0];
    recent[0] = elapsed;
  };
  const budgetOf = () => {
    const ms = typeof options.budget === 'function' ? options.budget(interval) : options.budget;
    if (!(ms > 0) || !Number.isFinite(ms)) {
      throw new RangeError(`budget must be finite and above 0 milliseconds, got ${ms}.`);
    }
    return ms;
  };
  if (
    typeof options.budget !== 'function' &&
    !(options.budget > 0 && Number.isFinite(options.budget))
  ) {
    throw new RangeError(`budget must be finite and above 0 milliseconds, got ${options.budget}.`);
  }
  if (!(low > 0 && high >= low && high <= 1)) {
    throw new RangeError(`range must run from above 0 up to at most 1, got [${low}, ${high}].`);
  }

  const window = new Float64Array(WINDOW);
  const sorted = new Float64Array(WINDOW);

  let scale = high;
  let seen = timing.count;
  let fresh = 0;
  /** The first frame recorded at the current scale. */
  let since = -1;
  /** The latest frame `update` saw, so a scale set between frames applies from the next. */
  let latest = -1;
  let latestTime = 0;
  let movedAt = 0;
  let ceiling = Infinity;
  let heldUntil = -Infinity;
  let holding = HOLD;
  let overAt = -Infinity;

  const move = (to: number, frame: number, time: number) => {
    scale = Math.min(high, Math.max(low, to));
    since = frame;
    movedAt = time;
    fresh = 0;
  };

  return {
    get scale() {
      return scale;
    },

    set scale(value) {
      if (!Number.isFinite(value)) throw new RangeError(`scale must be finite, got ${value}.`);
      move(value, latest + 1, latestTime);
      heldUntil = -Infinity;
    },

    update(frame) {
      measure();
      latest = frame.index;
      latestTime = frame.time;
      if (timing.count === seen) return;
      seen = timing.count;
      if (timing.lastFrame < since) return;

      window[fresh % WINDOW] = timing.last!;
      fresh += 1;
      if (fresh < WINDOW) return;

      sorted.set(window);
      sorted.sort();
      const ms = sorted[WINDOW >> 1];
      const budget = budgetOf();
      // Pixels go with the square of the scale. Time that doesn't scale makes this guess move
      // too little down, which the next median corrects, and never too far up
      const ideal = snap(scale * Math.sqrt((AIM * budget) / ms));
      const time = frame.time;

      if (ms > budget) {
        if (scale <= low) return;
        holding = time - overAt < holding * 2 ? Math.min(holding * 2, HOLD * 4) : HOLD;
        overAt = time;
        ceiling = scale;
        heldUntil = time + holding;
        move(Math.min(ideal, snap(scale - 1 / STEPS)), frame.index, time);
        return;
      }

      if (time - movedAt < RISE || scale >= high) return;
      // A GPU lowers its clock under light work, so time at a low scale overstates what a step
      // up costs: a median under the aim steps up at least once
      const up = ms < AIM * budget ? Math.max(ideal, snap(scale + 1 / STEPS)) : ideal;
      const to = time < heldUntil ? Math.min(up, snap(ceiling - 1 / STEPS)) : up;
      if (Math.min(to, high) > scale) move(to, frame.index, time);
    },
  };
}
