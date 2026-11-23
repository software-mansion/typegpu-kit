import type { GpuCommand, InstrumentHooks, Instrumentation } from './instrument.ts';
import {
  addSegment,
  createBreakdown,
  type Detail,
  displayName,
  formatMs,
  isTimed,
  ownTimes,
  passIndices,
  type Segment,
  type TimedCommand,
  untimedLine,
} from './recorder.ts';

export interface MeasureOptions {
  /** Sets how long to measure, in seconds, after a quarter second of warm-up. Default 1. */
  readonly seconds?: number;
  /**
   * Sets the scene time between measured frames, in seconds, so each does a displayed frame's
   * work. Default the live frame interval, or 1/60 before two frames.
   */
  readonly dt?: number;
  /** Sets what GPU time is split into in `parts`. Default the panel's. */
  readonly detail?: Detail;
}

/**
 * The median of some milliseconds per frame, the 10th and 90th percentiles around it, and the
 * largest.
 */
export interface Spread {
  readonly median: number;
  readonly low: number;
  readonly high: number;
  readonly max: number;
}

/** Frames measured back to back, with the GPU at full clock. */
export interface Measurement {
  /** Counts the frames measured after warm-up. */
  readonly frames: number;
  /** Reports the scene seconds between frames. */
  readonly dt: number;
  /**
   * Reports the GPU's busy milliseconds per frame, from the first timed pass's start to the last
   * one's end without gaps, or undefined without timestamps.
   */
  readonly gpu: Spread | undefined;
  /** Reports the frame callbacks' mean milliseconds per frame. */
  readonly cpu: number;
  /**
   * Reports the milliseconds between frames finishing on the GPU, the frame rate's interval with
   * no display to wait for. Above `gpu` when the CPU or gaps between passes hold the GPU back.
   * `NaN` when the frames measured finished in one burst, which leaves no interval to time.
   */
  readonly wall: number;
  readonly detail: Detail;
  /** Splits `gpu`'s median frame into mean milliseconds per frame at `detail`, largest first. */
  readonly parts: Segment[];
  /** Names the passes that write their own timestamps, which the measurement leaves untimed. */
  readonly untimed: readonly string[];
}

interface Collected {
  readonly commands: GpuCommand[];
  cpu: number;
  awaiting: number;
  dropped: boolean;
}

/** Waits this long for a measurement's last timestamps. */
const lateAfter = 2000;
const warmup = 250;
/** Milliseconds of frames split per dispatch, for the breakdown. */
const splitDuration = 250;

function collect(split: boolean) {
  const frames: Collected[] = [];
  let awaiting = 0;
  let last: number | undefined;
  let settle: (() => void) | undefined;

  const hooks: InstrumentHooks = {
    capturing: () => false,
    splitComputePasses: () => split,
    deviceRequested() {},
    // Every callback of an animation frame reports the frame's timestamp
    frameStarted(timestamp) {
      if (timestamp === last) return;
      last = timestamp;
      frames.push({ commands: [], cpu: 0, awaiting: 0, dropped: false });
    },
    callbackTimed(_name, ms) {
      const frame = frames.at(-1);
      if (frame) frame.cpu += ms;
    },
    submitted(commands) {
      const frame = frames.at(-1);
      if (!frame) return;
      for (const command of commands) {
        command.frame = frames.length - 1;
        frame.commands.push(command);
        if (command.timestamps === 'written') {
          frame.awaiting++;
          awaiting++;
        }
        if (command.timestamps === 'dropped') frame.dropped = true;
      }
    },
    timed(commands) {
      for (const command of commands) {
        const frame = frames[command.frame];
        if (!frame || command.timestamps !== 'written') continue;
        frame.awaiting--;
        awaiting--;
        if (command.start === undefined) frame.dropped = true;
      }
      if (awaiting === 0) settle?.();
    },
  };

  const settled = () =>
    new Promise<void>((resolve) => {
      if (awaiting === 0) return resolve();
      const timeout = setTimeout(resolve, lateAfter);
      settle = () => {
        clearTimeout(timeout);
        resolve();
      };
    });

  return { hooks, frames, settled };
}

/** Times each frame's commands, each starting no earlier than the frame before it ended. */
function timeFrames(frames: Collected[]) {
  const timings: { frame: Collected; timed: TimedCommand[]; own: number[] }[] = [];
  let floor = 0n;
  for (const frame of frames) {
    if (frame.dropped || frame.awaiting > 0) continue;
    const timed = frame.commands.filter(isTimed);
    if (timed.length === 0) continue;
    timings.push({ frame, timed, own: ownTimes(timed, floor) });
    for (const { end } of timed) if (end > floor) floor = end;
  }
  return timings;
}

function spread(values: number[]): Spread {
  const sorted = values.slice();
  sorted.sort((a, b) => a - b);
  const at = (fraction: number) =>
    sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))] ?? Number.NaN;
  return { median: at(0.5), low: at(0.1), high: at(0.9), max: at(1) };
}

/**
 * Runs the page's frames back to back for `seconds` after a quarter second of warm-up, so the GPU
 * keeps its full clock, and times them. Under a detail finer than `'pass'`, it then runs a quarter
 * second of frames split per dispatch and divides the whole frames' passes as those split.
 *
 * @param instrumentation started instrumentation, whose frames it drives
 * @returns the measurement, once the last timestamps are back. Rejects if the GPU reports no frame
 * finished before the measurement ends
 */
export async function measure(
  instrumentation: Instrumentation,
  options: Required<MeasureOptions>,
): Promise<Measurement> {
  const { seconds, dt, detail } = options;
  if (!(seconds > 0) || !(dt > 0)) {
    throw new RangeError(`Measuring needs seconds and dt above 0, got ${seconds} and ${dt}.`);
  }

  const interval = dt * 1000;
  const whole = collect(false);
  const wall = await instrumentation.drive(whole.hooks, {
    interval,
    warmup,
    duration: seconds * 1000,
  });
  // The drive waits for each burst to finish on the GPU before the next, so a GPU that reports
  // finishing late leaves every frame in the warm-up
  if (whole.frames.length === 0) {
    throw new Error(
      `No frame was measured. The GPU did not report completing the warm-up frames within ${warmup + seconds * 1000} ms.`,
    );
  }

  const breakdown = createBreakdown(detail);
  if (detail !== 'pass') {
    const split = collect(true);
    await instrumentation.drive(split.hooks, { interval, warmup: 0, duration: splitDuration });
    await split.settled();
    timeFrames(split.frames).forEach(({ frame, timed, own }, i) => {
      breakdown.learn(timed, own, passIndices(frame.commands), 1 / (i + 1));
    });
  }

  await whole.settled();
  const timings = timeFrames(whole.frames);
  const parts: Segment[] = [];
  const add = (name: string, ms: number, calls: number) =>
    addSegment(parts, name, ms / timings.length, calls / timings.length);
  for (const { frame, timed, own } of timings) {
    const indices = passIndices(frame.commands);
    timed.forEach((command, i) => breakdown.divide(command, indices.get(command), own[i], add));
  }
  parts.sort((a, b) => b.ms - a.ms);

  return {
    frames: whole.frames.length,
    dt,
    gpu:
      timings.length > 0
        ? spread(timings.map(({ own }) => own.reduce((sum, ms) => sum + ms, 0)))
        : undefined,
    cpu: whole.frames.reduce((sum, { cpu }) => sum + cpu, 0) / whole.frames.length,
    wall,
    detail,
    parts: parts.filter(({ ms }) => ms > 0),
    untimed: [
      ...new Set(
        whole.frames.flatMap(({ commands }) =>
          commands.filter(({ timestamps }) => timestamps === 'own').map(displayName),
        ),
      ),
    ],
  };
}

const range = ({ median, low, high, max }: Spread) =>
  `${formatMs(median)} (${low.toFixed(2)} to ${high.toFixed(2)}, max ${max.toFixed(2)})`;

/**
 * Formats a measurement as the panel shows it: totals, then GPU parts with calls per frame.
 *
 * @param measurement what `debug.measure()` resolved with
 * @returns lines of plain text
 */
export function measureText(measurement: Measurement): string {
  const { frames, dt, gpu, cpu, wall, parts, untimed } = measurement;
  const lines = [
    `measured ${frames} frames back to back, ${formatMs(dt * 1000)} of scene time apart`,
    [
      gpu ? `gpu ${range(gpu)}` : 'gpu untimed',
      ...(Number.isNaN(wall) ? [] : [`a frame every ${formatMs(wall)}`]),
      `cpu ${formatMs(cpu)}`,
    ].join(' · '),
  ];
  for (const { name, ms, calls } of parts) {
    const times = Math.round(calls) > 1 ? ` ×${Math.round(calls)}` : '';
    lines.push(`gpu ${formatMs(ms).padStart(8)}  ${name}${times}`);
  }
  lines.push(...untimed.map(untimedLine));
  return lines.join('\n');
}
