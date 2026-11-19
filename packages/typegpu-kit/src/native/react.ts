/**
 * Hosts a scene on a react-native-webgpu `<Canvas>` through `useScene`. Connects the device,
 * canvas layout, `AppState`, device errors and teardown. Has the same requirements as
 * `@typegpu/kit/native/host`.
 *
 * @module
 * @kind platform
 * @platform native
 * @flat
 * @owns what `host` owns, plus a requested device: releases them when the component unmounts
 */

import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import { AppState, PixelRatio, type LayoutChangeEvent } from 'react-native';
import { useMainDevice, type CanvasRef } from 'react-native-webgpu';
import { isWorkletFunction } from 'react-native-worklets';
import { watch } from '../hosting.ts';
import * as runtimes from './runtimes.ts';
import { host, type CreateScene, type Hosted, type SceneRuntime, type Scene } from './host.ts';

export type { CreateScene, Hosted, SceneRuntime, Scene, Stage } from './host.ts';

export interface UseSceneOptions {
  /**
   * Selects the scene runtime. Reuses the runtime when the scene restarts. Changing it restarts
   * the scene. Default `'worker'`, one per component named after `createScene`.
   */
  readonly on?: SceneRuntime;
  /**
   * Pauses frames while true, such as for a canvas outside the viewport. Frames also pause when
   * the app leaves the foreground. Default false.
   */
  readonly paused?: boolean;
  /**
   * Supplies a device shared by scenes. Give each scene in an app the same device through this
   * option or a provider. Uncaptured errors do not identify their scene, so an error on a shared
   * device stops every scene. Default the nearest `<GPUDeviceProvider>` device, or a device
   * requested for this scene alone.
   */
  readonly device?: GPUDevice;
  /**
   * Requests these features when the adapter supports them, such as `indirect-first-instance`.
   * Read only when requesting a device. Default none.
   */
  readonly optionalFeatures?: Iterable<GPUFeatureName>;
  /** Caps drawing-buffer pixels per point. Changing it restarts the scene. Default 2. */
  readonly maxPixelRatio?: number;
  /**
   * Controls canvas transparency. Use `premultiplied` to show content behind the canvas. Changing
   * it restarts the scene. Default `opaque`.
   */
  readonly alphaMode?: GPUCanvasAlphaMode;
}

export interface SceneHandle<S extends Scene> {
  /**
   * Provides props to spread onto react-native-webgpu's `<Canvas>`. A new canvas, such as one
   * remounted under a new `key`, restarts the scene on it.
   */
  readonly canvas: {
    readonly ref: RefObject<CanvasRef | null>;
    readonly onLayout: (event: LayoutChangeEvent) => void;
  };
  /** Provides the hosted scene after the canvas has a size and the device is ready. */
  readonly scene: Hosted<S> | undefined;
  /**
   * Reports the first error: from the device, setup, a frame or a sent closure. Every error but a
   * sent closure's stops the scene. Remount the component with a new `key` to restart it.
   */
  readonly error: Error | undefined;
}

/**
 * Hosts a scene and returns props for its canvas. Reads `options` from the render that starts the
 * scene. Functions always call their versions from the latest render, so inline functions do not
 * restart the scene. Send changing values from an effect. For example, use `useEffect(() =>
 * send?.(setWeather, weather), [send, weather])` with `const send = scene?.send`. This also sends
 * the value when the scene restarts.
 *
 * @param createScene a worklet unless `on` is `'js'`
 * @param options the scene's third argument
 * @param hosting where the scene runs, its device and whether it is paused
 * @returns props for the `<Canvas>`, the hosted scene once it runs, and its first error
 */
export function useScene<Options extends object, S extends Scene>(
  createScene: CreateScene<Options, S>,
  options: Options,
  hosting: UseSceneOptions = {},
): SceneHandle<S> {
  const { on = 'worker', paused = false, optionalFeatures, maxPixelRatio, alphaMode } = hosting;
  const [canvas, setCanvas] = useState<CanvasRef | null>(null);
  const [ref] = useState(() => tracked<CanvasRef>(setCanvas));
  const [size, setSize] = useState<{ width: number; height: number }>();
  const [scene, setScene] = useState<Hosted<S>>();
  const [error, setError] = useState<Error>();
  const [requested, setRequested] = useState<GPUDevice>();
  const [active, setActive] = useState(AppState.currentState === 'active');
  const worker = useRef<runtimes.WorkerRuntime>(undefined);

  const provided = useProvidedDevice();
  const given = hosting.device ?? provided;
  const device = given ?? requested;

  const latest = useRef(options);
  latest.current = options;

  const report = useCallback(
    (next: unknown) =>
      setError((first) => first ?? (next instanceof Error ? next : new Error(String(next)))),
    [],
  );

  // Keep the previous teardown promise so a replacement host can wait before using the canvas
  const released = useRef<Promise<unknown>>(Promise.resolve());

  // Destroy the component's worker after the last teardown, read after this commit's cleanups
  useEffect(
    () => () => {
      const made = worker.current;
      worker.current = undefined;
      void Promise.resolve()
        .then(() => released.current)
        .then(() => made?.destroy());
    },
    [],
  );

  useEffect(() => {
    if (given) return;

    let live = true;
    let made: GPUDevice | undefined;

    requestDevice(optionalFeatures).then((next) => {
      if (!live) return next.destroy();
      made = next;
      setRequested(next);
    }, report);

    return () => {
      live = false;
      setRequested(undefined);
      if (!made) return;

      // Read after this commit's cleanups so the latest scene teardown is included
      const own = made;
      void Promise.resolve()
        .then(() => released.current)
        .then(() => own.destroy());
    };
    // Ignore feature array identity because features are read only when requesting the device
  }, [given, report]);

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      setActive(state === 'active');
    });

    return () => subscription.remove();
  }, []);

  const sized = size !== undefined;

  useEffect(() => {
    const context = canvas?.getContext('webgpu');
    if (!device || !size || !context) return;

    const name = createScene.name || 'TypeGPU Kit scene';
    const placement =
      on === 'worker' && isWorkletFunction(createScene)
        ? (worker.current ??= runtimes.worker({ name }))
        : on;

    let live = true;
    let hosted: Hosted<S> | undefined;

    // Device error events reach only the JS thread
    const watching = watch(device, (next) => {
      report(next);
      hosted?.destroy();
    });

    // Wait for the prior host to unconfigure this canvas before starting another
    void released.current
      .then(() => {
        if (!live) return;

        hosted = host(createScene, latestFunctions(latest), {
          on: placement,
          device,
          context,
          width: size.width,
          height: size.height,
          pixelRatio: PixelRatio.get(),
          ...(maxPixelRatio === undefined ? {} : { maxPixelRatio }),
          ...(alphaMode === undefined ? {} : { alphaMode }),
          onError: (next) => live && report(next),
        });
        setScene(hosted);
      })
      .catch((thrown: unknown) => {
        if (live) report(thrown);
      });

    return () => {
      live = false;
      watching[Symbol.dispose]();
      setScene(undefined);
      if (!hosted) return;

      // A call sent after destruction waits until teardown completes
      hosted.destroy();
      released.current = hosted.run(ignore).catch(ignore);
    };
    // Layout changes resize the surface without restarting the scene
  }, [device, sized, canvas, createScene, on, maxPixelRatio, alphaMode, report]);

  useEffect(() => {
    if (scene && size) scene.resize(size.width, size.height);
  }, [scene, size]);

  const stopped = paused || !active;

  useEffect(() => {
    if (!scene) return;
    if (stopped) scene.pause();
    else scene.resume();
  }, [scene, stopped]);

  const onLayout = useCallback((event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setSize({ width, height });
  }, []);

  return { canvas: { ref, onLayout }, scene, error };
}

/** A ref that reports each value React gives it, so a new canvas renders again. */
function tracked<T>(onChange: (value: T | null) => void): RefObject<T | null> {
  let current: T | null = null;
  return {
    get current() {
      return current;
    },
    set current(next) {
      current = next;
      onChange(next);
    },
  };
}

function useProvidedDevice(): GPUDevice | undefined {
  try {
    return useMainDevice().device ?? undefined;
  } catch {
    return undefined; // react-native-webgpu throws outside a provider
  }
}

function ignore() {
  'worklet';
}

async function requestDevice(optionalFeatures: Iterable<GPUFeatureName> = []) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter.');
  // react-native-webgpu adds implicit-device-synchronization to every device it requests
  const requiredFeatures = [...optionalFeatures].filter((feature) => adapter.features.has(feature));
  return adapter.requestDevice({ requiredFeatures });
}

function latestFunctions<Options extends object>(latest: { readonly current: Options }): Options {
  const result = { ...latest.current } as Record<string, unknown>;

  for (const key in result) {
    const value = result[key];
    if (typeof value !== 'function' || isWorkletFunction(value)) continue;

    result[key] = (...args: unknown[]) => {
      const current = (latest.current as Record<string, unknown>)[key];
      if (typeof current === 'function') current(...args);
    };
  }

  return result as Options;
}
