/**
 * Hosts a scene on a React Native canvas on a worker, the UI runtime or the JS thread. The scene
 * supplies `createScene(root, surface, options)` and returns `frame` and an optional `destroy`, as
 * examples do. Uses `runtimes` and the kit, with `stage` to start the scene as `web.host` does.
 *
 * Off the JS thread, requires the `@typegpu/kit/native/runtimes` setup. Pass a `createScene`
 * worklet that calls the scene file's function through a named import. Worklets then loads that
 * file, its shaders and its modules on the scene runtime.
 *
 * @module
 * @kind platform
 * @platform native
 * @flat
 * @owns a root on the device it is given unless it was given a root, the canvas's configuration, a
 * worker runtime unless it is given one, the scene's stage and what its `attach` returned
 */

import { installWebGPU } from 'react-native-webgpu';
import { isWorkletFunction, UIRuntimeId } from 'react-native-worklets';
import { tgpu, type TgpuRoot } from 'typegpu';
// Use named imports because worklets does not forward namespace imports
import {
  local,
  stage,
  type CreateScene,
  type Hosted as Base,
  type HostOptions as BaseOptions,
  type Scene,
  type Stage as BaseStage,
} from '../hosting.ts';
import { checkDevice, surface as canvasSurface, type Surface as CanvasSurface } from '../native.ts';
import type { FixedSize } from '../targets.ts';
import * as runtimes from './runtimes.ts';

export type { CreateScene, Scene } from '../hosting.ts';

/** Selects where a scene runs. Pass your own runtime to share it between scenes. */
export type SceneRuntime = 'worker' | 'ui' | 'js' | runtimes.Runtime;

/** Provides a sent closure with the stage on the scene runtime. */
export type Stage<S extends Scene> = BaseStage<S, CanvasSurface>;

/** Provides a hosted scene with worklet methods, so a gesture on the UI runtime can call them. */
export interface Hosted<S extends Scene> extends Base<S, CanvasSurface> {
  /** Updates the canvas layout size in points, as reported by `onLayout`. */
  resize(width: number, height: number): void;
}

export interface HostOptions<S extends Scene = Scene> extends BaseOptions<S, CanvasSurface> {
  /**
   * Selects the scene runtime. Default `'worker'`, a runtime named after `createScene` and paced
   * by the display.
   */
  readonly on?: SceneRuntime;
  /** Supplies the canvas context. After hosting, only the scene runtime uses it. */
  readonly context: GPUCanvasContext & { present(): void };
  /** Sets the canvas layout width in points. */
  readonly width: number;
  /** Sets the canvas layout height in points. */
  readonly height: number;
  /** Sets the display pixel ratio from `PixelRatio.get()`. */
  readonly pixelRatio: number;
  /** Caps drawing-buffer pixels per point unless `size` is set. Default 2. */
  readonly maxPixelRatio?: number;
  /**
   * Fixes the drawing-buffer size, as in `native.surface`. Default the layout size multiplied by
   * the capped pixel ratio.
   */
  readonly size?: FixedSize;
}

/**
 * Hosts a scene on the runtime selected by `on`, with display-linked frames. Top-level functions
 * in `options` that are not worklets run on the React Native JS thread with copied arguments, so
 * `onScore: setScore` works from a worker. Device errors reach only the JS thread: with `on: 'js'`
 * the host watches the device, and elsewhere you listen with `hosting.watch(device, onError)`, as
 * `useScene` does. Throws unless exactly one of `device` and `root` is given. Off the JS thread,
 * throws for a root, for a device without `implicit-device-synchronization`, and for `createScene`,
 * `onFrame` or `attach` functions that are not worklets.
 *
 * @param createScene a worklet unless `on` is `'js'`, which calls the scene file's `createScene`
 * @param options the scene's third argument
 * @param hosting the device, the canvas's context and size, and where the scene runs
 * @returns a hosted scene you resize, send closures to and destroy
 */
export function host<Options, S extends Scene>(
  createScene: CreateScene<Options, S>,
  options: Options,
  hosting: HostOptions<S>,
): Hosted<S> {
  const { device, root: given, context, width, height, schedule, onFrame, attach } = hosting;
  const { onError } = hosting;
  const on = hosting.on ?? 'worker';
  const where = typeof on === 'string' ? `the ${on} runtime` : on.name;

  if (on !== 'js' && !isWorkletFunction(createScene)) {
    throw new Error(
      `createScene must be a worklet to run on ${where}. Pass one that calls it, ` +
        "(root, surface, options) => { 'worklet'; return createScene(root, surface, options); }, " +
        "written where worklets forwards imports, or host the scene with on: 'js'.",
    );
  }
  if (!device === !given) {
    throw new Error('host requires exactly one of device and root.');
  }
  if (given && on !== 'js') {
    throw new Error(
      `host: a root can't leave the JS thread for ${where}. Give its device instead, or host the scene with on: 'js'.`,
    );
  }
  if (on !== 'js') checkDevice(device!);
  for (const [name, fn] of [
    ['onFrame', onFrame],
    ['attach', attach],
  ] as const) {
    if (on !== 'js' && fn && !isWorkletFunction(fn)) {
      throw new Error(
        `host: ${name} runs on ${where} with the scene, so it must be a worklet, or host the scene with on: 'js'.`,
      );
    }
  }

  const made =
    on === 'worker'
      ? runtimes.worker({ name: createScene.name || 'TypeGPU Kit scene' })
      : undefined;
  const runtime = on === 'worker' ? made : placed(on);
  const elsewhere = runtime !== undefined;
  const sceneOptions = runtime ? crossing(options) : options;
  const vsync =
    schedule === undefined && (on === 'worker' || typeof on === 'object')
      ? runtimes.vsync(runtime!)
      : undefined;
  const frames = schedule ?? vsync;
  const report = onError
    ? runtime
      ? runtimes.callable(onError)
      : onError
    : (error: unknown) => {
        'worklet';
        console.error(error);
      };
  const { pixelRatio, maxPixelRatio, size, format, alphaMode, usage, colorSpace, toneMapping } =
    hosting;
  const surfaceOptions = {
    pixelRatio,
    ...(maxPixelRatio === undefined ? {} : { maxPixelRatio }),
    ...(size === undefined ? {} : { size }),
    ...(format === undefined ? {} : { format }),
    ...(alphaMode === undefined ? {} : { alphaMode }),
    ...(usage === undefined ? {} : { usage }),
    ...(colorSpace === undefined ? {} : { colorSpace }),
    ...(toneMapping === undefined ? {} : { toneMapping }),
  };
  const staging = {
    ...(frames === undefined ? {} : { schedule: frames }),
    ...(onFrame ? { onFrame } : {}),
    ...(attach ? { attach } : {}),
  };

  const fail = (error: unknown) => {
    'worklet';
    report(error instanceof Error ? error : new Error(String(error)));
  };

  const create = async () => {
    'worklet';
    let root: TgpuRoot | undefined = given;
    let surface: CanvasSurface | undefined;

    // Stop the relay even if scene setup fails
    const release = () => {
      vsync?.destroy();
      made?.destroy();
      surface?.destroy();
      if (!given) root?.destroy();
    };

    try {
      if (elsewhere) installWebGPU(); // the UI runtime and runtimes of your own may not have it yet

      root ??= tgpu.initFromDevice({ device: device! });
      const canvas = { getContext: () => context };
      surface = canvasSurface(root, canvas, surfaceOptions);
      surface.resize(width, height);

      return await stage(root, surface, createScene, sceneOptions, {
        ...staging,
        // Device events reach only the JS thread
        ...(elsewhere ? {} : { watch: root.device }),
        onError: fail,
        release,
      });
    } catch (error) {
      release();
      throw error;
    }
  };

  const owning = onError ? { onError } : {};
  const owned = runtime ? runtimes.own(runtime, create, owning) : local(create, owning);
  const { send } = owned;

  return {
    ...owned,

    resize(nextWidth, nextHeight) {
      'worklet';
      send(resizeStage, nextWidth, nextHeight);
    },

    pause() {
      'worklet';
      send(pauseStage);
    },

    resume() {
      'worklet';
      send(resumeStage);
    },
  };
}

function placed(on: Exclude<SceneRuntime, 'worker'>): runtimes.Runtime | undefined {
  if (on === 'js') return undefined;
  if (on === 'ui') return { runtimeId: UIRuntimeId, name: 'UI' };
  return on;
}

function resizeStage({ surface }: Stage<Scene>, width: number, height: number) {
  'worklet';
  surface.resize(width, height);
}

function pauseStage({ surface }: Stage<Scene>) {
  'worklet';
  surface.pause();
}

function resumeStage({ surface }: Stage<Scene>) {
  'worklet';
  surface.resume();
}

function crossing<Options>(options: Options): Options {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) return options;

  const result = { ...options } as Record<string, unknown>;
  for (const key in result) {
    const value = result[key];
    if (typeof value === 'function' && !isWorkletFunction(value)) {
      result[key] = runtimes.post(value as (...args: unknown[]) => void);
    }
  }

  return result as Options;
}
