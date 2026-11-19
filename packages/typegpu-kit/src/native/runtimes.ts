/**
 * Runs GPU work on React Native worklet runtimes. Provides workers, display-linked frames,
 * runtime-owned values, shared records, callbacks to the JS thread and camera frames for a video
 * feed. Owns no loop, frame or scene. Requires react-native-worklets 0.13 in Bundle Mode. Add
 * `@typegpu/kit` and `typegpu` to `importForwarding` module names, and the kit to its relative
 * paths: `@typegpu/kit` when installed from npm, or the kit's source folder when linked from a
 * workspace.
 *
 * @module
 * @unstable
 * @kind platform
 * @platform native
 */

import { installWebGPU, type NativeVideoFrame } from 'react-native-webgpu';
import {
  createShareable,
  createSynchronizable,
  createWorkletRuntime,
  getRuntimeKind,
  isWorkletFunction,
  RuntimeKind,
  scheduleOnRN,
  scheduleOnRuntimeWithId,
  scheduleOnUI,
  type FixedSynchronizable,
  type Shareable,
  type WorkletRuntime,
} from 'react-native-worklets';
import type { Callback, OwnOptions, Owned } from '../hosting.ts';
import { cloneError, emptyHolder, owned } from '../owned.ts';
import type { Schedule } from '../animate.ts';
import type { Rotation, Source } from '../video.ts';

export type { Callback, OwnOptions, Owned } from '../hosting.ts';
export type { Schedule } from '../animate.ts';

/** Identifies a worklet runtime or the UI runtime as `{ runtimeId: UIRuntimeId, name: 'UI' }`. */
export type Runtime = Pick<WorkletRuntime, 'runtimeId' | 'name'>;

export interface WorkerOptions {
  /** Labels the runtime in errors and the debugger. Default `'TypeGPU Kit'`. */
  readonly name?: string;
  /**
   * Sets the minimum delay of `setTimeout(0)`, in milliseconds. While mapping is pending, the
   * react-native-webgpu promise pump reschedules itself with zero-delay timers. Without a floor, a
   * worker runs them at once and keeps a core busy. Use 0 to disable the floor. Default 2.
   */
  readonly timerFloor?: number;
}

/** A runtime from `worker`, which you destroy when you no longer run code on it. */
export interface WorkerRuntime extends Runtime, Disposable {
  /**
   * Restores the runtime's own `setTimeout`. Worklets cannot end a runtime, so it lives until
   * nothing references it. Also `[Symbol.dispose]`.
   */
  destroy(): void;
}

/**
 * Creates a worker runtime with WebGPU installed and a minimum delay for zero-delay timers.
 *
 * @returns a runtime to `own` values on, which you destroy
 */
export function worker(options: WorkerOptions = {}): WorkerRuntime {
  const { name = 'TypeGPU Kit', timerFloor = 2 } = options;
  const runtime = createWorkletRuntime({
    name,
    initializer: () => {
      'worklet';
      installWebGPU();
      if (timerFloor > 0) floorTimers(timerFloor);
    },
  });
  const { runtimeId } = runtime;

  const destroy = () => {
    'worklet';
    scheduleOnRuntimeWithId(runtimeId, unfloorTimers);
  };

  // Worklets holds a runtime weakly, so the worker keeps its object alive
  const held = { runtimeId, name, destroy, [Symbol.dispose]: destroy };
  return Object.defineProperty(held, 'runtime', { value: runtime });
}

type Floored = typeof setTimeout & { readonly unfloored?: typeof setTimeout };

function floorTimers(milliseconds: number) {
  'worklet';
  const timeout = globalThis.setTimeout;
  const floored = (callback: (...args: unknown[]) => void, delay?: number, ...args: unknown[]) =>
    timeout(callback, delay || milliseconds, ...args);
  globalThis.setTimeout = Object.assign(floored, { unfloored: timeout }) as Floored;
}

function unfloorTimers() {
  'worklet';
  const { unfloored } = globalThis.setTimeout as Floored;
  if (unfloored) globalThis.setTimeout = unfloored;
}

export interface Vsync extends Schedule, Disposable {
  /** Returns the number of display frames skipped while the runtime was running the prior frame. */
  missed(): number;
  /** Stops the relay. Also `[Symbol.dispose]`. */
  destroy(): void;
}

interface Waiting {
  callback: ((milliseconds: number) => void) | undefined;
  id: number;
}

/**
 * Relays display-linked frames from the UI runtime to a worker. A worker's own
 * `requestAnimationFrame` uses a 16 ms timer. Sends one frame at a time, skipping and counting
 * display frames that arrive while it runs. Pass it as `schedule` to `frames.animate` on that
 * runtime.
 *
 * @param runtime the only runtime allowed to request these frames
 * @returns a schedule you destroy to stop the relay
 */
export function vsync(runtime: Runtime): Vsync {
  const { runtimeId, name } = runtime;
  const wanted = createSynchronizable(false, { fixedType: true });
  const running = createSynchronizable(false, { fixedType: true });
  const stopped = createSynchronizable(false, { fixedType: true });
  // The worklets Babel plugin rejects a method capturing its own name, so avoid naming this
  // `missed`
  const misses = createSynchronizable(0, { fixedType: true });
  const waiting = createShareable<Waiting>(runtimeId, { callback: undefined, id: 0 });

  scheduleOnUI(relay, runtimeId, wanted, running, stopped, misses, waiting);

  const destroy = () => {
    'worklet';
    stopped.setDirty(true);
  };

  return {
    request(callback) {
      'worklet';
      if (!waiting.isHost) throw new Error(`Request frames from vsync(${name}) on ${name} only.`);
      waiting.value.callback = callback;
      waiting.value.id += 1;
      wanted.setDirty(true);
      return waiting.value.id;
    },

    cancel(request) {
      'worklet';
      if (!waiting.isHost || waiting.value.id !== request) return;
      waiting.value.callback = undefined;
      wanted.setDirty(false);
    },

    missed() {
      'worklet';
      return misses.getDirty();
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

function fire(waiting: Shareable<Waiting>, running: FixedSynchronizable<boolean>, ms: number) {
  'worklet';
  if (!waiting.isHost) return;
  const { callback } = waiting.value;
  waiting.value.callback = undefined;

  try {
    callback?.(ms);
  } catch (error) {
    console.error(
      'A frame failed on a worker runtime. Pass onError to animate to handle it.',
      error,
    );
  } finally {
    running.setDirty(false);
  }
}

function relay(
  runtimeId: number,
  wanted: FixedSynchronizable<boolean>,
  running: FixedSynchronizable<boolean>,
  stopped: FixedSynchronizable<boolean>,
  misses: FixedSynchronizable<number>,
  waiting: Shareable<Waiting>,
) {
  'worklet';
  const tick = (milliseconds: number) => {
    if (stopped.getDirty()) return;

    if (wanted.getDirty()) {
      if (running.getDirty()) {
        misses.setDirty(misses.getDirty() + 1);
      } else {
        wanted.setDirty(false);
        running.setDirty(true);
        scheduleOnRuntimeWithId(runtimeId, fire, waiting, running, milliseconds);
      }
    }

    requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
}

/**
 * Creates and keeps a value on `runtime`. `create` must be a worklet and may be asynchronous.
 * Calls sent before it resolves wait in order. A worklets Shareable holds the value on that
 * runtime. Nothing registers or looks it up by name.
 *
 * @param runtime where the value lives
 * @param create a worklet that makes the value, there
 * @returns a handle you send closures to and destroy
 */
export function own<T>(
  runtime: Runtime,
  create: () => T | Promise<T>,
  options: OwnOptions = {},
): Owned<T> {
  const { runtimeId, name } = runtime;
  const holder = createShareable(runtimeId, emptyHolder<T>());

  return owned(create, options.onError && callable(options.onError), {
    holder,
    runtime: name,

    schedule(fn, ...args) {
      'worklet';
      scheduleOnRuntimeWithId(runtimeId, fn, ...args);
    },

    back(settle, value) {
      'worklet';
      scheduleOnRN(settle, copy(value) as typeof value);
    },
  });
}

/**
 * Makes a function callable from any runtime. Returns worklets unchanged and wraps other functions
 * with `post`.
 *
 * @param fn a worklet, or a React Native function
 * @returns a worklet that calls it
 */
export function callable<Args extends unknown[]>(fn: Callback<Args>): Callback<Args> {
  return isWorkletFunction(fn) ? fn : post(fn);
}

/**
 * Calls `fn` on the React Native JS thread from any runtime. Copies arguments first, because
 * worklets caches sent objects by identity. Otherwise, changing an object in place and sending it
 * again delivers its first copy. Release builds omit the warning for this case.
 *
 * @param fn a React Native function
 * @returns a worklet that calls it
 */
export function post<Args extends unknown[]>(fn: (...args: Args) => void): Callback<Args> {
  return (...args: Args) => {
    'worklet';
    if (getRuntimeKind() === RuntimeKind.ReactNative) {
      fn(...args);
      return;
    }

    try {
      scheduleOnRN(fn, ...(copy(args) as Args));
    } catch (error) {
      console.error('Could not post to React Native.', error);
    }
  };
}

function copy(value: unknown): unknown {
  'worklet';
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(copy);
  if (value instanceof ArrayBuffer) return value.slice(0);
  if (value instanceof DataView) {
    const { buffer, byteOffset, byteLength } = value;
    return new DataView(buffer.slice(byteOffset, byteOffset + byteLength));
  }
  if (ArrayBuffer.isView(value)) return (value as Uint8Array).slice();
  if (value instanceof Date) return new Date(value.getTime());
  if (value instanceof Error) return cloneError(value);
  if (value instanceof Map) return new Map([...value].map(([key, field]) => [key, copy(field)]));
  if (value instanceof Set) return new Set([...value].map(copy));

  const result: Record<string, unknown> = {};
  for (const key in value) result[key] = copy((value as Record<string, unknown>)[key]);
  return result;
}

type Fields = Record<string, number | boolean>;

/** Widening lets a field initialized as `{ down: false }` later hold true. */
type Widened<T extends Fields> = {
  -readonly [K in keyof T]: T[K] extends boolean ? boolean : number;
};

/**
 * Shares a flat record of numbers and flags between runtimes without locking or allocating. Reads
 * fields separately, so a read can combine two writes. Stores only the latest values. Use
 * `Owned.send` for events that must not be lost.
 */
export interface Cell<T extends Fields> {
  /** Copies the latest fields into `into` and returns it. Reuse an object created by the reader. */
  read(into: T): T;
  /** Writes the fields in `values` and keeps the others. */
  write(values: Partial<T>): void;
}

/**
 * Creates one fixed-type synchronizable per field for a shared record.
 *
 * @param initial its fields and their first values
 * @returns a cell any runtime reads and writes
 */
export function cell<T extends Fields>(initial: T): Cell<Widened<T>> {
  const keys = Object.keys(initial) as (keyof T & string)[];
  const fields = {} as Record<keyof T, FixedSynchronizable<number | boolean>>;

  for (const key of keys) {
    const value: number | boolean = initial[key];
    fields[key] = createSynchronizable(value, { fixedType: true });
  }

  return {
    read(into) {
      'worklet';
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        (into as Fields)[key] = fields[key].getDirty();
      }
      return into;
    },

    write(values) {
      'worklet';
      for (const key in values) {
        const value = values[key];
        if (value !== undefined) fields[key].setDirty(value);
      }
    },
  };
}

/** Tells how a camera frame's pixels are turned from upright, as VisionCamera reports it. */
export type Orientation = 'up' | 'right' | 'down' | 'left';

/** The parts of a VisionCamera 5 `Frame` that `camera` reads. */
export interface CameraFrame {
  readonly width: number;
  readonly height: number;
  readonly orientation: Orientation;
  /** Reports whether the pixels must be read mirrored to match the camera output. */
  readonly isMirrored: boolean;
  /** When the frame was captured, in seconds. */
  readonly timestamp: number;
  getNativeBuffer(): { readonly pointer: bigint; release(): void };
}

/** Hands camera frames from the camera thread to a feed or live view on another runtime. */
export interface CameraSource extends Source, Disposable {
  /**
   * Passes `frame` to the feed's runtime. Call it from VisionCamera's `onFrame` on the camera
   * thread, then dispose the frame. Keeps the newest frame's pixels until a newer one replaces it,
   * and releases a frame that arrived and was replaced between two reads.
   */
  write(frame: CameraFrame): void;
  /** Releases held frames and ignores later ones. Also `[Symbol.dispose]`. */
  destroy(): void;
}

interface Arrived {
  readonly pixels: NativeVideoFrame;
  readonly width: number;
  readonly height: number;
  readonly rotation: Rotation;
  readonly mirrored: boolean;
  readonly time: number;
  readonly serial: number;
}

interface Shown extends Arrived {
  readonly texture: GPUExternalTexture;
}

interface Held {
  /** The newest frame, not yet imported. */
  arrived: Arrived | undefined;
  /** The frame imported for reading, kept until a newer one is imported. */
  shown: Shown | undefined;
  /** Counts frames that arrived. */
  arrivals: number;
  stopped: boolean;
}

/** Clockwise turns that show each VisionCamera orientation upright. */
const UPRIGHT = { up: 0, right: 90, down: 180, left: 270 } as const;

/**
 * Creates a video source fed from VisionCamera on the camera thread. The camera thread wraps each
 * frame's pixels and sends them to `runtime`, where feeds and live views import them as an
 * external texture. Attach the camera to them on that runtime. The newest frame is imported once
 * and kept until a newer one replaces it, so every feed and live view reads the same frame.
 *
 * @param runtime the runtime that hosts the feed or live view
 * @returns a source you attach there, write to from `onFrame` and destroy
 */
export function camera(runtime: Runtime): CameraSource {
  const { runtimeId, name } = runtime;
  const held = createShareable<Held>(runtimeId, {
    arrived: undefined,
    shown: undefined,
    arrivals: 0,
    stopped: false,
  });
  const webgpu = RNWebGPU;

  const destroy = () => {
    'worklet';
    scheduleOnRuntimeWithId(runtimeId, stopCamera, held);
  };

  return {
    write(frame) {
      'worklet';
      const buffer = frame.getNativeBuffer();
      let pixels: NativeVideoFrame;
      try {
        pixels = webgpu.createVideoFrameFromNativeBuffer(buffer.pointer);
      } finally {
        buffer.release();
      }
      const { width, height, isMirrored, timestamp } = frame;
      const rotation = UPRIGHT[frame.orientation];
      scheduleOnRuntimeWithId(
        runtimeId,
        hold,
        held,
        pixels,
        width,
        height,
        rotation,
        isMirrored,
        timestamp,
      );
    },

    latest(device, seen) {
      'worklet';
      if (!held.isHost) throw new Error(`Read camera frames on ${name} only.`);
      const state = held.value;
      if (state.arrivals === seen) return undefined;
      const shown = showNewest(state, device);
      if (!shown) return undefined;
      const { texture, width, height, rotation, mirrored, time, serial } = shown;
      return { texture, width, height, rotation, mirrored, time, serial };
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

function hold(
  held: Shareable<Held>,
  pixels: NativeVideoFrame,
  width: number,
  height: number,
  rotation: Rotation,
  mirrored: boolean,
  time: number,
) {
  'worklet';
  if (!held.isHost) return;
  const state = held.value;
  if (state.stopped) {
    pixels.release();
    return;
  }
  state.arrived?.pixels.release();
  state.arrivals += 1;
  const serial = state.arrivals;
  state.arrived = { pixels, width, height, rotation, mirrored, time, serial };
}

function freeShown(state: Held) {
  'worklet';
  state.shown?.texture.destroy();
  state.shown?.pixels.release();
  state.shown = undefined;
}

/** Imports the newest frame in place of the one shown, once. */
function showNewest(state: Held, device: GPUDevice): Shown | undefined {
  'worklet';
  const arrived = state.arrived;
  if (arrived) {
    state.arrived = undefined;
    freeShown(state);
    state.shown = { ...arrived, texture: device.importExternalTexture({ source: arrived.pixels }) };
  }
  return state.shown;
}

function stopCamera(held: Shareable<Held>) {
  'worklet';
  if (!held.isHost) return;
  const state = held.value;
  state.stopped = true;
  freeShown(state);
  state.arrived?.pixels.release();
  state.arrived = undefined;
}
