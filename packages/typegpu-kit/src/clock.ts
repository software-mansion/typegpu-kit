/**
 * Counts fixed steps and cadences, and eases and springs values at the same pace at any frame
 * rate.
 *
 * @module
 * @records `update(dt)` once a frame and run that many steps; a cadence's `due(dt)` says whether
 * its work runs in the frame.
 */

export interface FixedStepOptions {
  /** Sets the step duration in seconds, such as 1 / 120. */
  readonly step: number;
  /**
   * Limits the steps run per frame. Carries unused time forward, up to this many steps. Default 4.
   */
  readonly maxSteps?: number;
  /**
   * Counts a frame longer than both frames before it as the longer of the two. A frame that misses
   * the display then runs no catch-up steps, while a slower display keeps time. Default false.
   */
  readonly steady?: boolean;
}

export interface FixedStep {
  /**
   * Sets the step duration in seconds. A new value applies from the next `update`: unstepped time
   * carries over, capped at `maxSteps` new steps, and `time` keeps the seconds already stepped.
   * Particles and machines take its value when created and keep it. Throws unless the value is
   * finite and above 0.
   */
  step: number;
  /** Reports the simulated seconds: the sum of every step `update` has returned. */
  readonly time: number;
  /** Reports the unstepped time in seconds, capped at `maxSteps * step`. */
  readonly debt: number;
  /**
   * Reports the unstepped fraction of a step, capped at 1, for interpolation between two states.
   */
  readonly alpha: number;
  /**
   * Reports the unstepped time in seconds, capped at one step. Equals `alpha * step`. Use it to
   * draw objects that far along their velocity between simulation steps.
   */
  readonly ahead: number;
  /** Pauses the clock while true: `update` returns 0 and keeps the unstepped time. */
  paused: boolean;
  /**
   * Adds `dt` seconds and returns the number of steps to run. Each step lasts `step` seconds, so a
   * GPU kernel can use a fixed duration. All steps in a frame see the same queued CPU writes. Keep
   * a step counter on the GPU for values that must change per step.
   */
  update(dt: number): number;
  /** Discards unstepped time after a reload or a long pause. */
  reset(): void;
}

const checkedStep = (value: number) => {
  if (!(value > 0) || !Number.isFinite(value))
    throw new RangeError(`step must be finite and above 0 seconds, got ${value}.`);
  return value;
};

/**
 * Accumulates elapsed time for fixed simulation steps. Accumulates nothing while paused. Throws
 * unless `step` is finite and above 0 and `maxSteps` is an integer above 0.
 *
 * @returns a clock whose `update` says how many steps to run each frame
 */
export function fixedStep(options: FixedStepOptions): FixedStep {
  const { maxSteps = 4, steady = false } = options;
  let step = checkedStep(options.step);
  if (!Number.isInteger(maxSteps) || maxSteps < 1) {
    throw new RangeError(`maxSteps must be an integer above 0, got ${maxSteps}.`);
  }

  let debt = 0;
  let simulated = 0;
  let paused = false;
  // The last two frames' lengths, for `steady`
  let before = Infinity;
  let earlier = Infinity;

  return {
    get step() {
      return step;
    },

    set step(value) {
      step = checkedStep(value);
      debt = Math.min(debt, maxSteps * step);
    },

    get paused() {
      return paused;
    },

    set paused(value) {
      paused = value;
    },

    get time() {
      return simulated;
    },

    get debt() {
      return debt;
    },

    get alpha() {
      return Math.min(1, debt / step);
    },

    get ahead() {
      return Math.min(debt, step);
    },

    update(dt) {
      if (paused || !(dt > 0)) return 0;
      debt += steady ? Math.min(dt, Math.max(before, earlier)) : dt;
      earlier = before;
      before = dt;
      // Allow rounding error when one frame lasts exactly one step, such as `run(i * step)`
      const steps = Math.min(maxSteps, Math.floor(debt / step + 1e-6));
      debt = Math.max(0, Math.min(debt - steps * step, maxSteps * step));
      simulated += steps * step;

      return steps;
    },

    reset() {
      debt = 0;
    },
  };
}

export interface Cadence {
  /** Reports the seconds since the last run. Valid when `due` returns true. */
  readonly dt: number;
  /** Adds `dt` seconds and returns whether work is due this frame, always on the first call. */
  due(dt: number): boolean;
  /** Restarts the schedule so the next `due` returns true. */
  reset(): void;
}

/** Sets a cadence by frame count or frequency. Specify exactly one of `every` and `hz`. */
export type CadenceOptions =
  | {
      /** Runs work every `every` frames. Default none: set this or `hz`. */
      readonly every: number;
      readonly hz?: never;
    }
  | {
      /** Runs work `hz` times per second on average. Default none: set this or `every`. */
      readonly hz: number;
      readonly every?: never;
    };

/**
 * Schedules work every `every` frames or `hz` times per second on average. Reports the time since
 * its last run. With `hz`, it runs in the frame closest to each scheduled tick. At 25 Hz and 60
 * fps, gaps alternate between two and three frames. Throws unless exactly one of `every` and `hz`
 * is set. `every` must be an integer above 0, and `hz` must be finite and above 0.
 *
 * @returns a cadence whose `due` says whether the work runs this frame
 */
export function cadence(options: CadenceOptions): Cadence {
  const { every = 0, hz } = options;
  if ((options.every === undefined) === (hz === undefined)) {
    throw new TypeError('A cadence requires exactly one of every and hz.');
  }
  if (options.every !== undefined && !(Number.isInteger(every) && every > 0)) {
    throw new RangeError(`every must be an integer above 0 frames, got ${every}.`);
  }
  if (hz !== undefined && !(hz > 0 && Number.isFinite(hz))) {
    throw new RangeError(`hz must be finite and above 0 cycles per second, got ${hz}.`);
  }

  const interval = hz === undefined ? 0 : 1 / hz;

  let frames = 0;
  let clock = 0;
  let last = 0;
  let next = 0;
  let dt = 0;
  let first = true;

  return {
    get dt() {
      return dt;
    },

    due(frameDt) {
      clock += frameDt;
      frames += 1;

      // Choose the frame closest to the tick by comparing with the expected next frame
      const run = first || (every > 0 ? frames >= every : clock > next - frameDt / 2);
      if (!run) return false;

      // Restart the schedule after a stall or reset instead of catching up missed ticks
      const onSchedule = !first && Math.abs(clock - next) < interval / 2;
      next = onSchedule ? next + interval : clock + interval;

      first = false;
      dt = clock - last;
      last = clock;
      frames = 0;

      return true;
    },

    reset() {
      first = true;
      last = clock;
      frames = 0;
    },
  };
}

/**
 * Returns the fraction of the way to a goal to move in a frame `dt` seconds long,
 * `1 - Math.exp(-dt * rate)`, so a value eases at the same pace at any frame rate. Move by it each
 * frame, as in `at += (goal - at) * clock.ease(frame.dt, 8)`. At `rate` 8, half the gap closes in
 * 87 ms.
 *
 * @param dt the frame's length in seconds
 * @param rate how fast the gap closes, per second
 * @returns a fraction from 0 to 1
 */
export function ease(dt: number, rate: number): number {
  return 1 - Math.exp(-dt * rate);
}

/** A mutable list of numbers a vector spring moves in place, such as `[0, 0, 0]` or a `d.vec3f()`. */
export interface Vector {
  [index: number]: number;
  readonly length: number;
}

export interface SpringOptions {
  /**
   * Sets how hard the spring pulls toward its target, per second squared for each unit away.
   * Higher values settle faster.
   */
  readonly stiffness: number;
  /**
   * Sets how much speed the spring loses, per second. `2 * Math.sqrt(stiffness)` settles fastest
   * without overshooting; lower values bounce.
   */
  readonly damping: number;
  /** Sets the starting position. Default 0. */
  readonly value?: number;
  /** Sets the starting speed in units per second. Default 0. */
  readonly velocity?: number;
}

export interface VectorSpringOptions<V extends Vector> extends Omit<
  SpringOptions,
  'value' | 'velocity'
> {
  /** Sets the vector the spring moves in place, and its starting position. */
  readonly value: V;
  /** Sets the starting speed of each component in units per second. Default zeros. */
  readonly velocity?: ArrayLike<number>;
}

export interface Spring {
  /** Holds the spring's position. Set it to jump. */
  value: number;
  /** Holds its speed in units per second. Add to it to kick the spring. */
  velocity: number;
  /** Moves the spring `dt` seconds toward `target` and returns the new `value`. */
  step(target: number, dt: number): number;
}

export interface VectorSpring<V extends Vector> {
  /** Holds the spring's position, the vector given as `value`, which `step` writes in place. */
  readonly value: V;
  /** Holds its speed per component, in units per second. Add to it to kick the spring. */
  readonly velocity: number[];
  /** Moves each component `dt` seconds toward `target`'s and returns `value`. */
  step(target: ArrayLike<number>, dt: number): V;
}

/**
 * Creates a damped spring that moves a value toward a target. It steps the exact motion, so it
 * settles the same at any frame rate and stays stable for long frames. Pass a vector as `value`
 * to move each of its components in place, without allocating. Throws unless `stiffness` and
 * `damping` are finite and not negative.
 *
 * @returns a spring you `step` toward a target each frame
 */
export function spring(options: SpringOptions): Spring;
export function spring<V extends Vector>(options: VectorSpringOptions<V>): VectorSpring<V>;
export function spring(
  options: SpringOptions | VectorSpringOptions<Vector>,
): Spring | VectorSpring<Vector> {
  const { stiffness, damping, value = 0 } = options;
  for (const [name, given] of [
    ['stiffness', stiffness],
    ['damping', damping],
  ] as const) {
    if (!(given >= 0) || !Number.isFinite(given)) {
      throw new RangeError(`${name} must be finite and not negative, got ${given}.`);
    }
  }

  // Offset y = value - target and speed v evolve as [y, v] <- [[c + a s, s], [-k s, c - a s]] [y, v]
  // over dt, for the roots -a +- sqrt(a^2 - k) of y'' + 2a y' + k y = 0
  const a = damping / 2;
  const discriminant = a * a - stiffness;
  let lastDt = Number.NaN;
  let c = 1;
  let s = 0;
  const solve = (dt: number) => {
    if (dt === lastDt) return;
    lastDt = dt;
    if (discriminant > 1e-9) {
      const b = Math.sqrt(discriminant);
      // e^-at cosh(bt) and e^-at sinh(bt) without overflowing cosh for heavy damping
      const slow = Math.exp((b - a) * dt);
      const fast = Math.exp(-(a + b) * dt);
      c = (slow + fast) / 2;
      s = (slow - fast) / 2 / b;
    } else if (discriminant < -1e-9) {
      const w = Math.sqrt(-discriminant);
      const decay = Math.exp(-a * dt);
      c = decay * Math.cos(w * dt);
      s = (decay * Math.sin(w * dt)) / w;
    } else {
      const decay = Math.exp(-a * dt);
      c = decay;
      s = decay * dt;
    }
  };

  if (typeof value === 'number') {
    const made: Spring = {
      value,
      velocity: typeof options.velocity === 'number' ? options.velocity : 0,
      step(target, dt) {
        if (dt > 0) {
          solve(dt);
          const y = made.value - target;
          const v = made.velocity;
          made.value = target + (c + a * s) * y + s * v;
          made.velocity = -stiffness * s * y + (c - a * s) * v;
        }
        return made.value;
      },
    };
    return made;
  }

  const initial = options.velocity as ArrayLike<number> | undefined;
  const velocity = Array.from({ length: value.length }, (_, i) => initial?.[i] ?? 0);
  const made: VectorSpring<Vector> = {
    value,
    velocity,
    step(target, dt) {
      if (dt > 0) {
        solve(dt);
        for (let i = 0; i < value.length; i++) {
          const y = value[i] - target[i];
          const v = velocity[i];
          value[i] = target[i] + (c + a * s) * y + s * v;
          velocity[i] = -stiffness * s * y + (c - a * s) * v;
        }
      }
      return value;
    },
  };
  return made;
}
