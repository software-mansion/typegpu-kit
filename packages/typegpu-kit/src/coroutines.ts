/**
 * Runs generators that pause at `yield*` and resume in a later frame. Supports waits, typed joins
 * and cancellation.
 *
 * @module
 * @unstable
 * @records Call `run` once per frame while recording to resume coroutines whose waits are complete.
 */

import type { Frame } from './frames.ts';

/**
 * Records when a coroutine suspended. Each coroutine keeps its own record, so coroutines can share
 * a wait.
 */
export interface Suspension {
  /** Identifies the frame in which the coroutine yielded the wait. */
  readonly index: number;
  /** Records the scheduler's `time` in seconds when the coroutine yielded the wait. */
  readonly time: number;
  /** Records when the current resume began, in milliseconds on the scheduler clock. */
  readonly resumedAt: number;
  /** Reads the scheduler clock in milliseconds, using its `now` option or `performance.now()`. */
  now(): number;
  /** Reads the scheduler's `time` in seconds at `frame`, the clock timed waits count. */
  clock(frame: Frame): number;
}

/**
 * Pauses a coroutine at `yield*` until it can resume with `T`. If `begin`, `ready` or `value`
 * throws, the coroutine receives that error at its `yield*`.
 */
export interface Wait<T> {
  [Symbol.iterator](): Iterator<Wait<unknown>, T, unknown>;
  /** Runs once per coroutine in the frame where it yields this wait. */
  begin?(frame: Frame): void;
  /** Tests readiness at each `run` until it returns true. */
  ready(frame: Frame, since: Suspension): boolean;
  /** Returns the value passed to the resumed coroutine. */
  value(frame: Frame): T;
  /** Runs once per coroutine when it is cancelled while waiting. */
  leave?(): void;
}

/** The members `wait` takes: a wait's, with `value` optional. */
type Parts = Omit<Wait<unknown>, typeof Symbol.iterator | 'value'> & {
  value?(frame: Frame): unknown;
};

/** What a coroutine resumes with from a wait made of `P`: its `value`, or else the frame. */
type Resumed<P> = P extends { value(frame: Frame): infer T } ? T : Frame;

const byFrame = (frame: Frame) => frame;

/**
 * Makes a new wait with the members of `parts`, which it leaves unchanged. The wait yields itself
 * once, then returns the resume value, the frame without `value`. Many coroutines can share it,
 * and resuming through it allocates nothing.
 *
 * @param parts `ready`, and optionally `value`, `begin` and `leave`
 * @returns a wait to `yield*`
 */
export function wait<P extends Parts>(parts: P): NoInfer<P & Wait<Resumed<P>>> {
  type T = Resumed<P>;
  // `yield*` passes the iterator result through unchanged, so one object serves every resume.
  // Readiness uses each coroutine's suspension, and the result is written and read within one
  // synchronous `yield*`
  let yielded = false;
  const step = { done: false, value: undefined as unknown };
  const iterator: Iterator<Wait<unknown>, T, unknown> = {
    next(resumed?: unknown) {
      step.done = yielded;
      step.value = yielded ? resumed : waiting;
      yielded = true;
      return step as IteratorResult<Wait<unknown>, T>;
    },
    throw(error?: unknown): never {
      throw error;
    },
  };
  const waiting = {
    ...parts,
    value: parts.value ?? byFrame,
    [Symbol.iterator]() {
      yielded = false;
      return iterator;
    },
  };
  return waiting as P & Wait<T>;
}

/**
 * Waits until `run` in the next frame. Coroutines share this wait, and resuming through it
 * allocates nothing.
 */
export const nextFrame: Wait<Frame> = frames(1);

/**
 * Waits `count` frames. A count of 0 resumes at once. Creates one wait object, so reuse it outside
 * hot loops. Throws unless `count` is a nonnegative integer.
 *
 * @param count how many frames to wait
 * @returns a wait to `yield*`
 */
export function frames(count: number): Wait<Frame> {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError(`frames() takes a whole number of frames, got ${count}.`);
  }

  return wait({ ready: (frame, since) => frame.index >= since.index + count });
}

/**
 * Waits until the scheduler's `time` has advanced by `duration` seconds, frame time by default.
 * Throws unless `duration` is finite and nonnegative.
 *
 * @param duration seconds to wait
 * @returns a wait to `yield*`
 */
export function seconds(duration: number): Wait<Frame> {
  if (!Number.isFinite(duration) || duration < 0) {
    throw new RangeError(`seconds() takes a finite duration of zero or more, got ${duration}.`);
  }

  // Allow one microsecond of rounding error in accumulated frame times
  return wait({ ready: (frame, since) => since.clock(frame) - since.time >= duration - 1e-6 });
}

/**
 * Resumes in the same frame if the coroutine has run for less than `ms` milliseconds since its
 * last resume, using the scheduler clock. Otherwise, waits for the next frame. Use it to spread
 * work over frames. Throws if `ms` is negative or NaN.
 *
 * @param ms milliseconds a coroutine may run each frame
 * @returns a wait to `yield*` between pieces of work
 */
export function budget(ms: number): Wait<Frame> {
  if (!(ms >= 0)) throw new RangeError(`budget() takes zero or more milliseconds, got ${ms}.`);
  return wait({
    ready: (frame, since) => frame.index > since.index || since.now() - since.resumedAt < ms,
  });
}

/**
 * Waits for another caller to resolve or reject it, such as a callback that receives a result.
 */
export interface Deferred<T> extends Wait<T> {
  /** Starts as `pending` and records the first `resolve` or `reject`. */
  readonly status: 'pending' | 'resolved' | 'rejected';
  /** Resumes waiting coroutines with `value` at the next `run`. Later calls do nothing. */
  resolve(value: T): void;
  /** Throws `error` at waiting coroutines' `yield*` at the next `run`. Later calls do nothing. */
  reject(error: unknown): void;
}

/**
 * Creates a wait whose coroutines resume at the first `run` after `resolve` or `reject`.
 *
 * @returns a wait you resolve or reject
 */
export function deferred<T>(): Deferred<T> {
  let result: T | undefined;
  let failure: unknown;
  const settled: Deferred<T> & { status: Deferred<T>['status'] } = wait({
    status: 'pending' as Deferred<T>['status'],
    resolve(value: T) {
      if (settled.status !== 'pending') return;
      settled.status = 'resolved';
      result = value;
    },
    reject(error: unknown) {
      if (settled.status !== 'pending') return;
      settled.status = 'rejected';
      failure = error;
    },
    ready: () => settled.status !== 'pending',
    value() {
      if (settled.status === 'rejected') throw failure;
      return result as T;
    },
  });
  return settled;
}

/**
 * Waits for a promise, such as a readback's `next()` or `pipelines.compile`. Resumes at the first
 * `run` after it settles, with its value or its error.
 *
 * @param promise the promise to wait for
 * @returns a wait to `yield*`
 */
export function until<T>(promise: PromiseLike<T>): Wait<T> {
  const settled = deferred<T>();
  promise.then(
    (value) => settled.resolve(value),
    (error: unknown) => settled.reject(error),
  );
  return settled;
}

/** A coroutine from `start`. Yielding it waits until it finishes. */
export interface Coroutine<T> extends Wait<T>, Disposable {
  /** Reports whether it is running, done, failed or cancelled. */
  readonly status: 'running' | 'done' | 'failed' | 'cancelled';
  /** Holds the return value once `status` is `done`. */
  readonly result: T | undefined;
  /** Holds the thrown error once `status` is `failed`. */
  readonly error: unknown;
  /**
   * Calls its wait's `leave`, or cancels the child coroutine it owns and waits on, then runs this
   * coroutine's `finally` blocks and `using` disposals. Other coroutines waiting on it receive an
   * error.
   */
  cancel(): void;
}

export interface Options {
  /**
   * Receives errors from failed coroutines with no waiters, and from cleanup during cancellation.
   * Default `reportError` where available, otherwise `console.error`.
   * @cpu
   */
  onError?: (error: unknown) => void;
  /**
   * Supplies the wall clock in milliseconds for resume timestamps and `budget`. Default
   * `performance.now()`.
   * @cpu
   */
  now?: () => number;
  /**
   * Supplies the clock in seconds that `seconds` waits count, such as a fixed step's `time`, so
   * timed waits stop while a game is paused. Default `frame.time`.
   * @cpu
   */
  time?: (frame: Frame) => number;
}

export interface Coroutines extends Disposable {
  /**
   * Schedules a coroutine for the next `run`, or later in the current run if called from a
   * coroutine. If the caller first yields it with `yield* scheduler.start(child())`, the child
   * starts at once and the caller owns it. Cancelling the owner cancels the child. Cancelling
   * other waiters only stops their waits. After `destroy`, returns a cancelled coroutine without
   * running it.
   */
  start<T>(process: Generator<Wait<unknown>, T, unknown>): Coroutine<T>;
  /**
   * Resumes every coroutine whose wait is complete. Call while a frame is recording. Coroutine and
   * wait errors go to the coroutine's `yield*` or to `onError`, rather than escaping `run`.
   */
  run(frame: Frame): void;
  /** Counts the coroutines the scheduler holds. `run` drops those that finished. */
  readonly size: number;
  /** Cancels every coroutine. Also `[Symbol.dispose]`. */
  destroy(): void;
}

/** A coroutine as its scheduler keeps it. */
interface Scheduled<T> extends Coroutine<T> {
  status: Coroutine<T>['status'];
  result: T | undefined;
  error: unknown;
  readonly process: Generator<Wait<unknown>, T, unknown>;
  started: boolean;
  executing: boolean;
  wait: Wait<unknown> | undefined;
  joiners: Scheduled<unknown>[] | undefined;
  // Keep the caller as owner only if it also starts the child by yielding it
  owner: Scheduled<unknown> | undefined;
  readonly since: { index: number; time: number; resumedAt: number } & Suspension;
}

function report(error: unknown) {
  if (typeof globalThis.reportError === 'function') globalThis.reportError(error);
  else console.error(error);
}

/**
 * Creates a scheduler for coroutines that advance when you call `run` with a frame.
 *
 * @returns a scheduler you start coroutines on and destroy
 */
export function create(options: Options = {}): Coroutines {
  const onError = options.onError ?? report;
  const now = options.now ?? (() => performance.now());
  const clock = options.time ?? ((frame: Frame) => frame.time);
  const live: Scheduled<unknown>[] = [];
  const scheduled = new WeakSet<object>();
  const isScheduled = (awaited: unknown): awaited is Scheduled<unknown> =>
    scheduled.has(awaited as object);
  let running = false;
  let destroyed = false;
  // Track the caller so yielding a child can establish ownership
  let current: Scheduled<unknown> | undefined;

  function schedule<T>(process: Generator<Wait<unknown>, T, unknown>): Scheduled<T> {
    const coroutine: Scheduled<T> = wait({
      status: 'running' as Coroutine<T>['status'],
      result: undefined as T | undefined,
      error: undefined as unknown,
      process,
      started: false,
      executing: false,
      wait: undefined as Wait<unknown> | undefined,
      joiners: undefined as Scheduled<unknown>[] | undefined,
      owner: current,
      since: { index: 0, time: 0, resumedAt: 0, now, clock },
      ready: () => coroutine.status !== 'running',
      value() {
        if (coroutine.status === 'failed') throw coroutine.error;
        if (coroutine.status === 'cancelled')
          throw new Error('The coroutine being waited on was cancelled.');
        return coroutine.result as T;
      },
      cancel() {
        if (coroutine.status !== 'running') return;
        coroutine.status = 'cancelled';
        if (coroutine.executing) return; // `advance` finishes the cancel when it regains control

        stop(coroutine as Scheduled<unknown>);
      },
      [Symbol.dispose]: () => coroutine.cancel(),
    });
    scheduled.add(coroutine);
    return coroutine;
  }

  function stop(coroutine: Scheduled<unknown>) {
    const awaited = coroutine.wait;

    try {
      if (!isScheduled(awaited)) awaited?.leave?.();
      else if (awaited.owner === coroutine) awaited.cancel();
      else if (awaited.joiners) {
        const index = awaited.joiners.indexOf(coroutine);
        if (index >= 0) awaited.joiners.splice(index, 1);
      }
    } catch (error) {
      onError(error);
    }

    close(coroutine);
  }

  function close(coroutine: Scheduled<unknown>) {
    const caller = current;
    current = coroutine;

    try {
      let step = coroutine.process.return(undefined);
      if (step.done) return;

      // Closing again skips the block that yielded but still runs the surrounding cleanup
      onError(
        new Error(
          'A coroutine yielded in a finally block while cancelled; the rest of it was skipped.',
        ),
      );
      while (!step.done) step = coroutine.process.return(undefined);
    } catch (error) {
      onError(error);
    } finally {
      current = caller;
    }
  }

  function settle(coroutine: Scheduled<unknown>, frame: Frame) {
    const joiners = coroutine.joiners;
    coroutine.joiners = undefined;
    if (coroutine.status === 'failed' && !joiners?.length) onError(coroutine.error);
    if (!joiners) return;

    for (let i = 0; i < joiners.length; i++) {
      const joiner = joiners[i];
      if (joiner.status === 'running' && !joiner.executing && joiner.wait === coroutine) {
        advance(joiner, frame);
      }
    }
  }

  /**
   * The joiner is running, so if the child finishes at once, `settle` leaves the joiner to read
   * the result itself.
   */
  function join(joiner: Scheduled<unknown>, coroutine: Scheduled<unknown>, frame: Frame) {
    if (coroutine.status !== 'running') return;
    (coroutine.joiners ??= []).push(joiner);
    if (coroutine.started) return;

    coroutine.started = true;
    if (coroutine.owner !== joiner) coroutine.owner = undefined;
    advance(coroutine, frame);
  }

  function advance(coroutine: Scheduled<unknown>, frame: Frame, throwing = false, input?: unknown) {
    coroutine.since.resumedAt = now();

    if (!throwing && coroutine.wait) {
      try {
        input = coroutine.wait.value(frame);
      } catch (error) {
        throwing = true;
        input = error;
      }
    }

    for (;;) {
      let step: IteratorResult<Wait<unknown>, unknown>;
      const caller = current;
      current = coroutine;
      coroutine.executing = true;

      try {
        step = throwing ? coroutine.process.throw(input) : coroutine.process.next(input);
      } catch (error) {
        current = caller;
        coroutine.executing = false;
        if (coroutine.status === 'cancelled') return;

        coroutine.status = 'failed';
        coroutine.error = error;
        settle(coroutine, frame);
        return;
      }

      current = caller;
      coroutine.executing = false;
      if (coroutine.status === 'cancelled') {
        close(coroutine);
        return;
      }

      if (step.done) {
        coroutine.status = 'done';
        coroutine.result = step.value;
        settle(coroutine, frame);
        return;
      }

      const awaited = step.value;
      coroutine.wait = awaited;
      coroutine.since.index = frame.index;
      coroutine.since.time = clock(frame);

      // Delay cancellation during `begin` until `stop` can clean up the wait
      let begun = false;
      coroutine.executing = true;

      try {
        if (typeof (awaited as Partial<Wait<unknown>> | null)?.ready !== 'function') {
          throw new TypeError(
            'A coroutine yielded something that is not a wait. Use yield* with a wait.',
          );
        }

        if (isScheduled(awaited)) join(coroutine, awaited, frame);
        else awaited.begin?.(frame);
        begun = true;
      } catch (error) {
        input = error;
      } finally {
        coroutine.executing = false;
      }

      if ((coroutine.status as Coroutine<unknown>['status']) === 'cancelled') {
        stop(coroutine);
        return;
      }

      if (!begun) {
        throwing = true;
        continue;
      }

      try {
        if (!awaited.ready(frame, coroutine.since)) return;
        input = awaited.value(frame);
        throwing = false;
      } catch (error) {
        input = error;
        throwing = true;
      }
    }
  }

  function resume(coroutine: Scheduled<unknown>, frame: Frame) {
    let ready: boolean;

    try {
      ready = coroutine.wait!.ready(frame, coroutine.since);
    } catch (error) {
      advance(coroutine, frame, true, error);
      return;
    }

    if (ready) advance(coroutine, frame);
  }

  const destroy = () => {
    destroyed = true;
    for (let i = 0; i < live.length; i++) live[i].cancel();
    live.length = 0;
  };

  return {
    start<T>(process: Generator<Wait<unknown>, T, unknown>): Coroutine<T> {
      const coroutine = schedule(process);
      if (destroyed) coroutine.cancel();
      else live.push(coroutine as Scheduled<unknown>);

      return coroutine;
    },

    run(frame) {
      if (running) throw new Error('Coroutines cannot run inside their own run.');
      running = true;

      try {
        for (let i = 0; i < live.length; i++) {
          const coroutine = live[i];
          if (coroutine.status !== 'running') continue;

          if (coroutine.started) {
            resume(coroutine, frame);
            continue;
          }

          coroutine.started = true;
          coroutine.owner = undefined;
          advance(coroutine, frame);
        }
      } finally {
        running = false;

        let kept = 0;
        for (let i = 0; i < live.length; i++) {
          if (live[i].status === 'running') live[kept++] = live[i];
        }

        live.length = kept;
      }
    },

    get size() {
      return live.length;
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
