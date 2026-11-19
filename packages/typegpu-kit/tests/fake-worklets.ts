// react-native-worklets on one thread, for tests. Each runtime has a queue that `settle` drains,
// the current runtime follows promise continuations, a Shareable's value is reachable only on its
// host, and animation frames remember the runtime that asked for them. Plain objects and arrays
// sent to another runtime arrive as copies, and the same object sent again arrives as its first
// copy, as worklets caches what it sends by identity. Worklets are the functions passed to
// `worklet`; the Babel plugin, which marks them in an app, doesn't run in tests. It can't see what
// a worklet captures, so captured values are shared.

import { AsyncLocalStorage } from 'node:async_hooks';

export const RuntimeKind = { ReactNative: 1, UI: 2, Worker: 3 } as const;
type RuntimeKind = (typeof RuntimeKind)[keyof typeof RuntimeKind];

export const UIRuntimeId = RuntimeKind.UI;

interface Task {
  readonly runtime: number;
  readonly run: () => void;
}

const current = new AsyncLocalStorage<number>();

const state = {
  get current(): number {
    return current.getStore() ?? RuntimeKind.ReactNative;
  },
  nextRuntime: 10,
  tasks: [] as Task[],
  frames: [] as { id: number; runtime: number; callback: (milliseconds: number) => void }[],
  nextFrame: 1,
  worklets: new WeakSet<object>(),
  names: new Map<number, string>(),
  initializers: new Map<number, (() => void) | undefined>(),
  /** What crosses by reference, as worklets' host objects do. */
  handles: new WeakSet<object>(),
  /** The first copy of each plain object or array sent to another runtime. */
  sent: new WeakMap<object, unknown>(),
};

export function reset() {
  state.tasks.length = 0;
  state.frames.length = 0;
  state.worklets = new WeakSet();
  state.names.clear();
  state.initializers.clear();
  state.handles = new WeakSet();
  state.sent = new WeakMap();
}

/** Marks functions as worklets, as the Babel plugin would. */
export function worklet<F extends (...args: never[]) => unknown>(fn: F): F {
  state.worklets.add(fn);
  return fn;
}

export const isWorkletFunction = (value: unknown) =>
  typeof value === 'function' && state.worklets.has(value);

export function getRuntimeKind(): RuntimeKind {
  if (state.current === RuntimeKind.ReactNative) return RuntimeKind.ReactNative;
  return state.current === UIRuntimeId ? RuntimeKind.UI : RuntimeKind.Worker;
}

/** The runtime code is running on now. */
export const currentRuntime = () => state.current;

/** The name a worklet runtime was made with. */
export const runtimeName = (runtimeId: number) => state.names.get(runtimeId);

function on<R>(runtime: number, run: () => R): R {
  return current.run(runtime, run);
}

function handle<T extends object>(value: T): T {
  state.handles.add(value);
  return value;
}

function isPlain(value: unknown): value is object {
  if (value === null || typeof value !== 'object' || state.handles.has(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return Array.isArray(value) || prototype === Object.prototype || prototype === null;
}

/** A new copy of plain objects and arrays, all the way down. */
function copied(value: unknown): unknown {
  if (!isPlain(value)) return value;
  if (Array.isArray(value)) return value.map(copied);
  return Object.fromEntries(Object.entries(value).map(([key, field]) => [key, copied(field)]));
}

/** What arrives on another runtime: a copy of what the value held when it was first sent. */
function crossed(value: unknown): unknown {
  if (value instanceof Promise) throw new Error('[Worklets] Promises cannot be serialized.');
  if (!isPlain(value)) return value;

  let first = state.sent.get(value);
  if (first === undefined) {
    first = Array.isArray(value)
      ? value.map(crossed)
      : Object.fromEntries(Object.entries(value).map(([key, field]) => [key, crossed(field)]));
    state.sent.set(value, first);
  }

  return copied(first);
}

function schedule(runtime: number, fn: (...args: never[]) => unknown, args: unknown[]) {
  const sent = runtime === state.current ? args : args.map(crossed);
  state.tasks.push({ runtime, run: () => fn(...(sent as never[])) });
}

export function createWorkletRuntime(config: { name: string; initializer?: () => void }) {
  const runtimeId = state.nextRuntime++;
  state.names.set(runtimeId, config.name);
  state.initializers.set(runtimeId, config.initializer);

  return handle({ runtimeId, name: config.name });
}

/** Runs a runtime's initializer on it; a test stubs what it patches first. */
export const initialize = (runtimeId: number) =>
  on(runtimeId, () => state.initializers.get(runtimeId)?.());

export const scheduleOnRuntimeWithId = (
  runtimeId: number,
  fn: (...args: never[]) => unknown,
  ...args: unknown[]
) => schedule(runtimeId, fn, args);

export const scheduleOnUI = (fn: (...args: never[]) => unknown, ...args: unknown[]) =>
  schedule(UIRuntimeId, fn, args);

export const scheduleOnRN = (fn: (...args: never[]) => unknown, ...args: unknown[]) =>
  schedule(RuntimeKind.ReactNative, fn, args);

export function createShareable<T>(hostRuntimeId: number, initial: T) {
  const value = (hostRuntimeId === state.current ? initial : crossed(initial)) as T;

  return handle({
    get isHost() {
      return state.current === hostRuntimeId;
    },

    get value() {
      if (state.current !== hostRuntimeId) throw new Error('A guest read a host value.');
      return value;
    },
  });
}

export function createSynchronizable<T>(initial: T) {
  let value = initial;

  return handle({
    getDirty: () => value,
    setDirty: (next: T) => {
      value = next;
    },
  });
}

/** Runs every scheduled task and every promise they start, until nothing is left. */
export async function settle() {
  for (let round = 0; round < 100; round++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (state.tasks.length === 0) return;

    for (const task of state.tasks.splice(0)) on(task.runtime, task.run);
  }

  throw new Error('Tasks kept scheduling tasks.');
}

/** `requestAnimationFrame` for every runtime, as a stub of the global. */
export function requestAnimationFrame(callback: (milliseconds: number) => void) {
  const id = state.nextFrame++;
  state.frames.push({ id, runtime: state.current, callback });
  return id;
}

export function cancelAnimationFrame(id: number) {
  const index = state.frames.findIndex((frame) => frame.id === id);
  if (index >= 0) state.frames.splice(index, 1);
}

/** Animation frames asked for and not run yet, on every runtime. */
export const requestedFrames = () => state.frames.length;

/** One display frame: runs the frame callbacks every runtime asked for, each on its runtime. */
export function display(milliseconds: number) {
  for (const { runtime, callback } of state.frames.splice(0)) {
    on(runtime, () => callback(milliseconds));
  }
}
