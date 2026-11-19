/**
 * Provides shared hosting code without importing React Native. Includes runtime-owned values,
 * scene contracts, host options, `stage` to run a scene on a root and surface, and `watch` to
 * report device errors. The React Native `host` and `web.host` use it. The web and tests can
 * import it without react-native-worklets.
 *
 * @module
 * @kind platform
 * @owns a stage's frames and their animation, what its `attach` returned, and the scene, which it
 * destroys
 */

import type { TgpuRoot } from 'typegpu';
// Use named imports because worklets forwards them to the scene runtime. It does not forward
// namespace imports
import { animate, type Animation, type Schedule } from './animate.ts';
import { create as createFrames, type Frame, type Frames } from './frames.ts';
import { emptyHolder, owned, settles } from './owned.ts';
import type { Surface } from './targets.ts';

/**
 * Defines a callback invoked where a value lives. On React Native, worklets run on the calling
 * runtime. Other functions run on the React Native JS thread with copied arguments. Elsewhere,
 * callbacks are plain functions.
 */
export type Callback<Args extends unknown[]> = (...args: Args) => void;

export interface OwnOptions {
  /**
   * Receives errors from `create`, sent closures and calls that worklets cannot schedule. An
   * uncaught error in a worklet otherwise aborts a release build. Default `console.error` on the
   * runtime that owns the value.
   * @cpu
   */
  readonly onError?: Callback<[unknown]>;
}

/** Keeps a value alive on one runtime and gives closures access to it there. */
export interface Owned<T> extends Disposable {
  /**
   * Runs `fn` with the value on its runtime, after earlier sent calls. Any runtime can call it, so
   * a gesture on the UI runtime can control a scene on a worker. Worklets copies `args` and caches
   * objects by identity. If you change an object in place and send it again, the runtime receives
   * the first copy. Send a new object after a change. If `fn` returns a promise, a rejection
   * goes to `onError`.
   */
  send<Args extends unknown[]>(fn: (value: T, ...args: Args) => void, ...args: Args): void;
  /**
   * Runs `fn` from React Native or the page, as `send` does, and copies its result back as `post`
   * does. Awaits asynchronous functions on the runtime that owns the value. Rejects if `fn` throws
   * or its promise rejects, and once the value is destroyed.
   */
  run<R, Args extends unknown[]>(
    fn: (value: T, ...args: Args) => R,
    ...args: Args
  ): Promise<Awaited<R>>;
  /**
   * Calls `destroy` on the value's runtime after earlier sent calls, then releases the value
   * without waiting for their promises. A run still in flight meets destroyed resources. Drops
   * calls sent after it. Safe to call twice.
   */
  destroy(): void;
}

/**
 * Creates an owned value on the calling runtime, such as the JS thread, the web or a test.
 * Provides the same `Owned` interface as `runtimes.own` without worklets. Closures run in order in
 * microtasks.
 *
 * @param create makes the value, here
 * @returns a handle you send closures to and destroy
 */
export function local<T>(create: () => T | Promise<T>, options: OwnOptions = {}): Owned<T> {
  return owned(create, options.onError, {
    holder: { isHost: true, value: emptyHolder<T>() },
    runtime: undefined,
    schedule: (fn, ...args) => queueMicrotask(() => fn(...args)),
    back: (settle, value) => settle(value),
  });
}

/**
 * Defines the scene returned by `createScene`. A scene can return additional members for its
 * caller.
 */
export interface Scene {
  /**
   * Records the frame. Must be synchronous: the host submits when it returns, so it rejects a
   * `frame` that returns a promise as a scene error. Load and compile before, or in `ready`.
   */
  frame(frame: Frame): void;
  /** Frees the scene's resources, such as textures and non-uniform buffers. Default none. */
  destroy?(): void;
  /**
   * Resolves when the scene can draw, such as after compiling pipelines inside its frames. Keep
   * the loading screen visible until `stage.ready`. The host reports a rejection as a scene error.
   * Without this promise, the scene is ready at once. Default none.
   */
  readonly ready?: PromiseLike<unknown>;
}

export type CreateScene<Options, S extends Scene> = (
  root: TgpuRoot,
  surface: Surface,
  options: Options,
) => S | Promise<S>;

/** Provides access to a hosted scene on the runtime where it runs. */
export interface Stage<S extends Scene, Canvas extends Surface = Surface> {
  /** Holds the scene the host created. */
  readonly scene: S;
  /** Holds the root the scene was created on. */
  readonly root: TgpuRoot;
  /** Holds the canvas surface the scene draws into. */
  readonly surface: Canvas;
  /** Holds the frame loop that records the scene. */
  readonly loop: Frames;
  /**
   * Reports whether the scene's `ready` promise has resolved. True from the start if the scene has
   * no promise.
   */
  readonly ready: boolean;
}

/** Hosts a scene on a canvas with frames you can pause and resume. */
export interface Hosted<S extends Scene, Canvas extends Surface = Surface> extends Owned<
  Stage<S, Canvas>
> {
  /**
   * Stops frames and elapsed time for an app in the background, a canvas outside the viewport or a
   * pause menu. The first frame after `resume` has a `dt` of 0.
   */
  pause(): void;
  /** Resumes the frames after `pause`. */
  resume(): void;
}

/**
 * What a host's `attach` may return to free with the scene: a cleanup function or an object
 * with `destroy()`.
 */
export type Detach = (() => void) | { destroy(): void };

/** Adds a paused state to a surface. `animate` runs no frames while it is paused. */
export type Pausing = Surface & { readonly paused: boolean };

/** Configures a canvas on the web or React Native. */
export interface CanvasOptions {
  /** Sets the canvas texture format. Default `navigator.gpu.getPreferredCanvasFormat()`. */
  readonly format?: GPUTextureFormat;
  /**
   * Controls canvas transparency. With `premultiplied`, alpha below 1 reveals what is behind the
   * canvas, and colours must be premultiplied by alpha. Default `opaque`.
   */
  readonly alphaMode?: GPUCanvasAlphaMode;
  /**
   * Sets the canvas texture usage flags. Use `RENDER_ATTACHMENT | COPY_SRC` for a loupe that
   * copies from the canvas. Default `GPUTextureUsage.RENDER_ATTACHMENT`.
   */
  readonly usage?: GPUTextureUsageFlags;
  /** Sets the colour space of canvas values. Default `'srgb'`. */
  readonly colorSpace?: PredefinedColorSpace;
  /**
   * Controls the range of canvas values for display. Use `{ mode: 'extended' }` with a float
   * `format`, such as `rgba16float`, to show values above 1 on an HDR screen. Default `{ mode:
   * 'standard' }`.
   */
  readonly toneMapping?: GPUCanvasToneMapping;
}

/**
 * Configures `web.host` and the React Native `host` beyond their canvas options. On React Native,
 * these functions run on the scene runtime and must be worklets when the scene runs off the JS
 * thread.
 */
export interface HostOptions<
  S extends Scene,
  Canvas extends Surface = Surface,
> extends CanvasOptions {
  /**
   * Supplies a device that the host keeps alive. Tests can pass the device from a recording root.
   * React Native requires this or `root`. Default a device that the web host requests and destroys
   * with the scene.
   */
  readonly device?: GPUDevice;
  /**
   * Supplies a root that the host keeps alive, so scenes can share it. On React Native, requires
   * `on: 'js'`. Default a root that the host creates on `device` and destroys with the scene.
   */
  readonly root?: TgpuRoot;
  /**
   * Supplies animation frames. With false, call `stage.loop.run(seconds)` to draw on demand.
   * Default `requestAnimationFrame`, or display vsync on a React Native worker.
   */
  readonly schedule?: Schedule | false;
  /**
   * Runs after the scene's `frame` each frame, for page work such as a HUD that reads scene state.
   * An error stops the frames. Default none.
   * @cpu
   */
  readonly onFrame?: (stage: Stage<S, Canvas>, frame: Frame) => void;
  /**
   * Connects page code, such as orbit controls, to the stage after scene creation. The host frees
   * what it returns with the scene. Default none.
   * @cpu
   */
  readonly attach?: (stage: Stage<S, Canvas>) => Detach | void;
  /**
   * Receives the first scene error, from `createScene`, a frame, `onFrame`, `ready` or the device,
   * which stops the frames. Also receives each error a sent closure throws, which leaves the
   * frames running. Default `console.error`.
   * @cpu
   */
  readonly onError?: Callback<[unknown]>;
}

export interface StageOptions<S extends Scene, Canvas extends Surface = Surface> {
  /**
   * Supplies animation frames to `animate`. With false, call `staged.loop.run(seconds)` to draw on
   * demand. Default `requestAnimationFrame`.
   */
  readonly schedule?: Schedule | false;
  /**
   * Receives the first scene error once, after the frames stop: from a frame, `onFrame`, a
   * rejected `ready`, the watched device or `staged.fail`. `stage` rethrows what `createScene`
   * throws instead.
   * @cpu
   */
  readonly onError: (error: unknown) => void;
  /**
   * Watches the device from scene creation until `destroy`, and fails the stage on its first
   * uncaptured error or unexpected loss. Device events reach only the React Native JS thread, so
   * pass it there or on the web. Default none.
   */
  readonly watch?: GPUDevice;
  /** Runs after the scene's `frame` in every frame, with the stage. Default none. @cpu */
  readonly onFrame?: (stage: Stage<S, Canvas>, frame: Frame) => void;
  /**
   * Runs with the stage once it is up; `destroy` frees what it returns before the scene. Default
   * none.
   * @cpu
   */
  readonly attach?: (stage: Stage<S, Canvas>) => Detach | void;
  /**
   * Releases host resources, such as the surface and the root, at the end of `destroy`, also when
   * a cleanup throws. Does not run if `stage` throws. Default none.
   * @cpu
   */
  readonly release?: () => void;
}

/** Gives a host access to a stage and its lifecycle methods. */
export interface Staged<S extends Scene, Canvas extends Surface = Surface>
  extends Stage<S, Canvas>, Disposable {
  /**
   * Stops the frames and reports `error` to `onError`, unless the stage has failed or been
   * destroyed. Use it for errors the host finds itself, such as a validation error scope around a
   * frame.
   */
  fail(error: unknown): void;
  /**
   * Stops the frames, frees what `attach` returned, destroys the loop and the scene, then runs
   * `release`. When one of them throws, runs the rest, then rethrows the error, or an
   * `AggregateError` of several. Also `[Symbol.dispose]`.
   */
  destroy(): void;
}

/**
 * Creates a scene on the host's root and surface, then creates its frames and animation. Destroys
 * them in reverse order. Owns the rule every host follows: the first error from a frame,
 * `onFrame`, `ready` or the watched device stops the frames and reaches `onError` once. If setup
 * fails after scene creation, destroys those resources and rethrows the error. The host keeps its
 * root and surface and destroys them in `release`.
 *
 * @param surface the surface the scene draws into, which the host keeps
 * @param createScene the scene file's `createScene`
 * @param options the scene's third argument
 * @param hosting where frames come from, what takes their errors and what the host runs beside
 * @returns the stage, which you destroy before the root and the surface
 */
export async function stage<Options, S extends Scene, Canvas extends Pausing>(
  root: TgpuRoot,
  surface: Canvas,
  createScene: CreateScene<Options, S>,
  options: Options,
  hosting: StageOptions<S, Canvas>,
): Promise<Staged<S, Canvas>> {
  const { schedule, onError, onFrame, attach, release } = hosting;

  let animation: Animation | undefined;
  let status: 'running' | 'failed' | 'destroyed' = 'running';

  const fail = (error: unknown) => {
    if (status !== 'running') return;
    status = 'failed';
    animation?.destroy();
    onError(error);
  };

  let scene: S | undefined;
  let loop: Frames | undefined;
  let detach: (() => void) | undefined;
  const watching = hosting.watch && watch(hosting.watch, fail);

  // Frees the animation, what `attach` returned, the loop, the scene, then the watch, and the
  // rest when one throws
  const free = () => {
    const errors: unknown[] = [];
    for (const cleanup of [
      () => animation?.destroy(),
      () => detach?.(),
      () => loop?.destroy(),
      () => scene?.destroy?.(),
      () => watching?.[Symbol.dispose](),
    ]) {
      try {
        cleanup();
      } catch (error) {
        errors.push(error);
      }
    }
    if (errors.length > 1) throw new AggregateError(errors, 'Several resources failed to free.');
    if (errors.length === 1) throw errors[0];
  };

  const destroy = () => {
    if (status === 'destroyed') return;
    status = 'destroyed';
    try {
      free();
    } finally {
      release?.();
    }
  };

  try {
    const made = await createScene(root, surface, options);
    scene = made;
    let ready = made.ready === undefined;

    // Call `frame` on the scene, so a class scene keeps its `this`. Return an async frame's
    // promise, which the frames reject before submitting
    const frames = createFrames(root, (frame) => {
      const recorded: unknown = made.frame(frame);
      return settles(recorded) ? recorded : onFrame?.(staged, frame);
    });
    loop = frames;

    const staged: Staged<S, Canvas> = {
      scene: made,
      root,
      surface,
      loop: frames,

      get ready() {
        return ready;
      },

      fail,
      destroy,
      [Symbol.dispose]: destroy,
    };

    detach = detachOf(attach?.(staged));
    if (schedule !== false && status === 'running') {
      animation = animate(frames, {
        surface,
        onError: fail,
        ...(schedule ? { schedule } : {}),
      });
    }

    made.ready?.then(() => {
      ready = true;
    }, fail);
    return staged;
  } catch (error) {
    status = 'destroyed';
    let thrown = error;
    try {
      free();
    } catch (cleanup) {
      thrown = new AggregateError([error, cleanup], 'The stage failed, and so did its cleanup.');
    }
    throw thrown;
  }
}

function detachOf(held: Detach | void): (() => void) | undefined {
  if (held === undefined) return undefined;
  if (typeof held === 'function') return held;
  if (held !== null && typeof held === 'object' && typeof held.destroy === 'function') {
    return () => held.destroy();
  }
  throw new Error('`attach` must return a cleanup function or an object with `destroy()`.');
}

/**
 * Reports the device's first uncaptured error or unexpected loss until disposed. Ignores
 * deliberate device destruction. WebGPU does not identify which scene caused an uncaptured error,
 * so every watcher of the device receives it. On React Native, these events reach only the JS
 * thread.
 *
 * @param device the device it listens to
 * @param onError takes the first error, once
 * @returns a watch you dispose to stop listening
 */
export function watch(device: GPUDevice, onError: (error: Error) => void): Disposable {
  let watching = true;

  const report = (error: Error) => {
    if (!watching) return;
    watching = false;
    onError(error);
  };

  const uncaptured = (event: Event) =>
    report(new Error((event as GPUUncapturedErrorEvent).error.message));

  device.addEventListener('uncapturederror', uncaptured);
  void device.lost.then((info) => {
    if (info.reason !== 'destroyed') report(new Error(`The GPU device was lost: ${info.message}`));
  });

  return {
    [Symbol.dispose]() {
      // react-native-webgpu does not remove listeners, so also disable this callback with
      // `watching`
      watching = false;
      device.removeEventListener('uncapturederror', uncaptured);
    },
  };
}
