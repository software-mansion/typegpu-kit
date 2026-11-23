import type { Callback, Owned } from './hosting.ts';

type Phase<T> =
  /** Queues calls sent while `create` is pending. Each call checks the phase when it runs. */
  | { readonly state: 'pending'; readonly waiting: (() => void)[] }
  | { readonly state: 'delivered'; readonly value: T }
  | { readonly state: 'disposed' };

interface Holder<T> {
  phase: Phase<T>;
}

const DISPOSED = { state: 'disposed' } as const;

// Keys the teardown that `teardown` waits for, which no public type names
const acknowledged = Symbol('acknowledged');

interface Acknowledged {
  readonly [acknowledged]: () => Promise<void>;
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
  return { phase: { state: 'pending', waiting: [] } };
}

export function owned<T>(
  create: () => T | Promise<T>,
  onError: Callback<[unknown]> | undefined,
  place: Place<T>,
): Owned<T> & Acknowledged {
  const { holder, schedule, back, runtime } = place;
  const report = reporter(onError, runtime);

  schedule(start, holder, create, report);

  const destroy = () => {
    'worklet';
    schedule(end, holder, report, undefined);
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

    [acknowledged]() {
      return new Promise((resolve) => {
        const done = () => {
          'worklet';
          back(resolve, undefined);
        };
        schedule(end, holder, report, done);
      });
    },
  };
}

/**
 * Destroys an owned value as `destroy` does, and resolves once its `destroy` has run on its
 * runtime. Rejects when the teardown can't be scheduled, which leaves the value alive.
 */
export function teardown(handle: Owned<unknown>): Promise<void> {
  return (handle as Owned<unknown> & Acknowledged)[acknowledged]();
}

export function settles(value: unknown): value is PromiseLike<unknown> {
  'worklet';
  return typeof (value as PromiseLike<unknown> | null | undefined)?.then === 'function';
}

function start<T>(holder: Held<T>, create: () => T | Promise<T>, report: (error: unknown) => void) {
  'worklet';
  if (!holder.isHost) return;
  const state = holder.value;
  const settle = (next: Phase<T>) => {
    const { phase } = state;
    state.phase = next;
    return phase.state === 'pending' ? phase.waiting : [];
  };

  const ready = (value: T) => {
    for (const call of settle({ state: 'delivered', value })) call();
  };

  const failed = (error: unknown) => {
    const waiting = settle(DISPOSED);
    report(error);
    for (const call of waiting) call();
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
    const { phase } = state;
    if (phase.state !== 'delivered') {
      gone?.();
      return;
    }

    try {
      const result = fn(phase.value, ...args);
      if (!settles(result)) return;

      result.then(undefined, report);
    } catch (error) {
      report(error);
    }
  };

  if (state.phase.state === 'pending') state.phase.waiting.push(call);
  else call();
}

function end<T>(holder: Held<T>, report: (error: unknown) => void, done: (() => void) | undefined) {
  'worklet';
  if (!holder.isHost) return;
  const state = holder.value;

  const finish = () => {
    const { phase } = state;
    if (phase.state === 'delivered') {
      state.phase = DISPOSED;
      destroyValue(phase.value, report);
    }
    done?.();
  };

  if (state.phase.state === 'pending') state.phase.waiting.push(finish);
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

/**
 * Runs every cleanup, also after one throws. Then throws the first error, or, with several, an
 * `AggregateError` with the first one's message that holds them all in order.
 *
 * @param cleanups the steps to run, in order
 * @param failed errors from before the cleanups, which come first
 */
export function freeEach(cleanups: readonly (() => void)[], failed: [unknown, ...unknown[]]): never;
export function freeEach(cleanups: readonly (() => void)[], failed?: unknown[]): void;
export function freeEach(cleanups: readonly (() => void)[], failed: unknown[] = []): void {
  'worklet';
  for (const cleanup of cleanups) {
    try {
      cleanup();
    } catch (error) {
      failed.push(error);
    }
  }
  if (failed.length === 0) return;
  if (failed.length === 1) throw failed[0];
  const [first] = failed;
  const message = first instanceof Error ? first.message : String(first);
  throw new AggregateError(failed, `${message} (and ${failed.length - 1} more)`);
}
