import type { Callback, Owned } from './hosting.ts';

const STARTING = 0;
const READY = 1;
const ENDED = 2;

interface Holder<T> {
  value: T | undefined;
  state: number;
  /** Queues calls sent while `create` is pending. Each call checks the state when it runs. */
  waiting: (() => void)[];
}

/**
 * Holds the value in a Shareable on its owning runtime, or in a local holder whose `isHost` is
 * true.
 */
export type Held<T> =
  | { readonly isHost: true; readonly value: Holder<T> }
  | { readonly isHost: false };

/** Locates an owned value and schedules calls and results between runtimes. */
export interface Place<T> {
  readonly holder: Held<T>;
  /** Calls `fn(...args)` on the value's runtime after earlier scheduled calls. */
  schedule<Args extends unknown[]>(fn: (...args: Args) => void, ...args: Args): void;
  /** Returns `value` to `settle` on the runtime that called `run`. */
  back<V>(settle: (value: V) => void, value: V): void;
  /** Names the owning runtime in errors. Undefined on the calling runtime. */
  readonly runtime: string | undefined;
}

/** Creates a holder for a value whose creation has not finished. */
export function emptyHolder<T>(): Holder<T> {
  return { value: undefined, state: STARTING, waiting: [] };
}

export function owned<T>(
  create: () => T | Promise<T>,
  onError: Callback<[unknown]> | undefined,
  place: Place<T>,
): Owned<T> {
  const { holder, schedule, back, runtime } = place;
  const report = reporter(onError, runtime);

  schedule(start, holder, create, report);

  const destroy = () => {
    'worklet';
    schedule(end, holder, report);
  };

  return {
    send(fn, ...args) {
      'worklet';
      try {
        schedule(deliver, holder, fn, args, report, undefined);
      } catch (error) {
        report(error);
      }
    },

    run(fn, ...args) {
      return new Promise((resolve, reject) => {
        const fail = (error: unknown) => {
          'worklet';
          back(reject, asError(error, runtime));
        };
        const answer = (value: T) => {
          'worklet';
          const result = fn(value, ...args);

          // Promises cannot cross runtimes, so return the result after it settles
          if (!settles(result)) return back(resolve, result as never);
          return result.then((settled) => back(resolve, settled as never), fail);
        };
        const gone = () => {
          'worklet';
          fail(new Error('The value was destroyed.'));
        };

        schedule(deliver, holder, answer, [], fail, gone);
      });
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

export function settles(value: unknown): value is PromiseLike<unknown> {
  'worklet';
  return typeof (value as PromiseLike<unknown> | null | undefined)?.then === 'function';
}

function start<T>(holder: Held<T>, create: () => T | Promise<T>, report: (error: unknown) => void) {
  'worklet';
  if (!holder.isHost) return;
  const state = holder.value;

  const ready = (value: T) => {
    state.value = value;
    state.state = READY;
    for (const call of state.waiting.splice(0)) call();
  };

  const failed = (error: unknown) => {
    state.state = ENDED;
    report(error);
    for (const call of state.waiting.splice(0)) call();
  };

  try {
    Promise.resolve(create()).then(ready, failed);
  } catch (error) {
    failed(error);
  }
}

function destroyValue(value: unknown, report: (error: unknown) => void) {
  'worklet';
  try {
    (value as { destroy?: () => void } | undefined)?.destroy?.();
  } catch (error) {
    report(error);
  }
}

function deliver<T, Args extends unknown[]>(
  holder: Held<T>,
  fn: (value: T, ...args: Args) => unknown,
  args: Args,
  report: (error: unknown) => void,
  gone: (() => void) | undefined,
) {
  'worklet';
  if (!holder.isHost) return;
  const state = holder.value;

  const call = () => {
    if (state.state !== READY) {
      gone?.();
      return;
    }

    try {
      const result = fn(state.value as T, ...args);
      if (!settles(result)) return;

      result.then(undefined, report);
    } catch (error) {
      report(error);
    }
  };

  if (state.state === STARTING) state.waiting.push(call);
  else call();
}

function end<T>(holder: Held<T>, report: (error: unknown) => void) {
  'worklet';
  if (!holder.isHost) return;
  const state = holder.value;

  const finish = () => {
    if (state.state !== READY) return;
    const { value } = state;
    state.state = ENDED;
    state.value = undefined;
    destroyValue(value, report);
  };

  if (state.state === STARTING) state.waiting.push(finish);
  else finish();
}

function reporter(onError: Callback<[unknown]> | undefined, runtime: string | undefined) {
  return (error: unknown) => {
    'worklet';
    const plain = asError(error, runtime);
    try {
      if (onError) onError(plain);
      else console.error(plain);
    } catch (failure) {
      console.error(plain, failure);
    }
  };
}

export function cloneError(source: Error, suffix = ''): Error {
  'worklet';
  const clone = new Error(source.message + suffix);
  clone.name = source.name;
  if (source.stack !== undefined) clone.stack = source.stack;
  return clone;
}

/**
 * Converts `error` to an `Error`. For another runtime, copies the error so it can cross back and
 * adds the runtime name to its message.
 */
export function asError(error: unknown, runtime?: string): Error {
  'worklet';
  const source = error instanceof Error ? error : new Error(String(error));
  return runtime === undefined ? source : cloneError(source, ` (on ${runtime})`);
}
