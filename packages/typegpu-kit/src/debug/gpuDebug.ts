import { type Interfaces, instrument } from './instrument.ts';
import { measure, type Measurement, type MeasureOptions } from './measure.ts';
import { mountPanel } from './panel.ts';
import { createRecorder, type Detail, type FrameCapture, type Recorder } from './recorder.ts';

export { measureText, type Measurement, type MeasureOptions, type Spread } from './measure.ts';
export type { Detail } from './recorder.ts';
export type { Interfaces } from './instrument.ts';

export interface GpuDebug {
  /** Mounts the panel and starts timing frames. */
  start(): void;
  /** Removes the panel and stops recording. The page keeps running. */
  stop(): void;
  /** Reports whether it is recording, between `start` and `stop`. */
  readonly running: boolean;
  /**
   * Captures commands, resource reads and writes, and timings for the next frame. The panel
   * capture button does the same. Throws synchronously if recording has not started, and rejects
   * if it stops first.
   */
  capture(): Promise<FrameCapture>;
  /**
   * Returns the last two seconds of timed commands in Chrome trace format, as saved by the trace
   * button. Times start at the first timestamp after `start()`. Throws while stopped.
   */
  trace(): object;
  /**
   * Returns mean frame times over the last second as text, with GPU then CPU sections sorted by
   * time. The panel log button prints the same text. Times are in milliseconds per frame. Throws
   * while stopped.
   */
  summary(): string;
  /**
   * Runs the page's frames back to back for about a second, unpaced by the display, so the GPU
   * keeps its full clock, and times them. The panel measure button does the same and prints
   * `measureText` of the result. Throws synchronously while stopped.
   */
  measure(options?: MeasureOptions): Promise<Measurement>;
  /**
   * Sets what the graph and measurements split GPU time into, as the panel's detail button does.
   * Default `'dispatch'`, or the panel's saved choice.
   */
  detail: Detail;
  /** Stops recording and removes patches. `[Symbol.dispose]` does the same. */
  destroy(): void;
  [Symbol.dispose](): void;
}

export interface InstallOptions {
  /**
   * Interface objects of a device whose objects are not WebGPU's, patched beside WebGPU's own, such
   * as `webgl.interfaces` from `@typegpu/kit-gl` to profile a scene on WebGL 2. Default none.
   */
  readonly interfaces?: Interfaces;
}

let installed: GpuDebug | undefined;

/**
 * Installs WebGPU profiling patches without starting recording. Call this before requesting a
 * device. It requests `timestamp-query` and records pipeline names, bind groups, texture views
 * and shader code so recording can start later without reloading.
 *
 * When stopped, `finish` and `submit` still look up command buffers for passes timed before the
 * stop. Other patched calls check the recording flag and run the original method. Animation and
 * video frame callbacks define frames. Installing again destroys the preceding installation.
 * Without WebGPU or `interfaces`, it installs no patches and the panel remains empty.
 *
 * @unstable
 * @kind tool
 * @platform web
 * @requires timestamp-query optional it asks for it on devices requested after it installs;
 * without it, no timings
 * @owns up to 16 query sets, 32 buffers, a texture per canvas: per device, while it times passes:
 * timestamps, a buffer they resolve into and one to map; while it measures, a texture each canvas's
 * frames draw into
 * @returns a profiler you start and stop, and destroy to remove the patches
 */
export function install(options: InstallOptions = {}): GpuDebug {
  installed?.destroy();

  const instrumentation = instrument(options.interfaces ? { interfaces: options.interfaces } : {});
  let unmount: (() => void) | undefined;
  let recorder: Recorder | undefined;

  const started = () => {
    if (!recorder) {
      throw new Error('Call gpuDebug.start() before capture(), trace(), summary() or measure().');
    }
    return recorder;
  };

  const stop = () => {
    unmount?.();
    unmount = undefined;
    recorder?.stop();
    recorder = undefined;
    instrumentation.stop();
  };

  const destroy = () => {
    if (installed !== debug) return;

    installed = undefined;
    stop();
    instrumentation[Symbol.dispose]();
  };

  const debug: GpuDebug = {
    start() {
      if (recorder) return;

      recorder = createRecorder();
      instrumentation.start(recorder);
      unmount = mountPanel(recorder, (given) => debug.measure(given));
    },
    stop,
    get running() {
      return recorder !== undefined;
    },
    capture: () => started().capture(),
    trace: () => started().trace(),
    summary: () => started().summary(),
    measure(given = {}) {
      const current = started();
      const fps = Math.round(current.fps());
      return measure(instrumentation, {
        seconds: given.seconds ?? 1,
        dt: given.dt ?? 1 / (fps > 0 ? fps : 60),
        detail: given.detail ?? current.detail(),
      });
    },
    get detail() {
      return started().detail();
    },
    set detail(value) {
      started().setDetail(value);
    },
    destroy,
    [Symbol.dispose]: destroy,
  };

  installed = debug;
  return debug;
}
