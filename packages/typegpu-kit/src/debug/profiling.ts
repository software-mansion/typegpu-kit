import { instrument, type Instrumentation } from './instrument.ts';
import { measure, type MeasureOptions, measureText } from './measure.ts';
import { captureText, createRecorder, type Bar, type Detail, type Recorder } from './recorder.ts';

/** What a panel on another runtime reads from a recording, as data that copies between runtimes. */
export interface Profile {
  /** Frames per second over the last second, or 0 before two frames. */
  readonly fps: number;
  /** The newest graph columns, oldest first, each averaging the frames of 100 ms. */
  readonly bars: Bar[];
  /** Whether the device times passes. Undefined until the first encoder. */
  readonly timestampQuery: boolean | undefined;
  /** What the graph splits GPU time into. */
  readonly detail: Detail;
}

// One instrumentation per runtime, kept installed after the first start so passes timed before a
// stop still return their query sets
let instrumentation: Instrumentation | undefined;
let recorder: Recorder | undefined;

const started = () => {
  if (!recorder) throw new Error('Start profiling on this runtime before reading it.');
  return recorder;
};

/**
 * Starts recording frames on the calling runtime, installing WebGPU patches on its first call.
 * Frames come from the kit's `animate`, so call it on the runtime that runs the scene. Patches
 * name pipelines created after the first call by their entry points, and earlier ones by label.
 */
export function startProfiling() {
  if (recorder) return;
  instrumentation ??= instrument({ frames: 'animate' });
  recorder = createRecorder();
  instrumentation.start(recorder);
}

/**
 * Stops recording on the calling runtime, rejecting a pending capture. Keeps the patches for the
 * next start.
 */
export function stopProfiling() {
  instrumentation?.stop();
  recorder?.stop();
  recorder = undefined;
}

/** Reads the last `columns` graph columns and the frame rate. */
export function readProfile(columns: number): Profile {
  const current = started();
  return {
    fps: current.fps(),
    bars: columns > 0 ? current.bars().slice(-columns) : [],
    timestampQuery: current.timestampQuery(),
    detail: current.detail(),
  };
}

/** Sets what the graph splits GPU time into. */
export function detailProfile(detail: Detail) {
  started().setDetail(detail);
}

/**
 * Runs the scene's frames back to back for about a second, unpaced by the display, and times
 * them, as the web panel's measure does. Resolves with the measurement as text.
 */
export async function measureProfile(options: MeasureOptions = {}): Promise<string> {
  const current = started();
  const fps = Math.round(current.fps());
  return measureText(
    await measure(instrumentation!, {
      seconds: options.seconds ?? 1,
      dt: options.dt ?? 1 / (fps > 0 ? fps : 60),
      detail: options.detail ?? current.detail(),
    }),
  );
}

/** Mean frame times over the last second as text, GPU before CPU, as the web panel logs. */
export function profileSummary(): string {
  return started().summary();
}

/** Captures the next frame's commands with their times, as text. */
export async function captureProfile(): Promise<string> {
  return captureText(await started().capture());
}

/** The last two seconds of timed commands in Chrome trace format, as JSON for Perfetto. */
export function profileTrace(): string {
  return JSON.stringify(started().trace());
}
