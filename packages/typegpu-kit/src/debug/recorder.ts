import type { GpuCommand, InstrumentHooks } from './instrument.ts';
import type { GpuResource } from './metadata.ts';

export type Processor = 'gpu' | 'cpu';

/**
 * Names what GPU time is split into. `'pass'` times each pass whole. `'dispatch'` splits compute
 * passes by dispatch, and `'group'` by the debug groups pushed around dispatches, by dispatch
 * outside them. Both learn the split from occasional frames timed per dispatch and divide whole
 * frames' passes the same way, since timing every dispatch apart adds time to each.
 */
export type Detail = 'pass' | 'group' | 'dispatch';

export const details = ['pass', 'group', 'dispatch'] as const satisfies Detail[];

/** Labels the panels' detail button with what the graph splits GPU time into. */
export const detailLabels: Record<Detail, string> = {
  pass: 'Per pass',
  group: 'Per group',
  dispatch: 'Per dispatch',
};

export interface Segment {
  name: string;
  ms: number;
  calls: number;
}

/** This type holds frames averaged into one graph column. */
export interface Bar {
  time: number;
  /**
   * Counts frames included in each average. GPU frames count only after all timestamps arrive.
   * Dropped timestamps or late readbacks leave a frame in the CPU average only.
   */
  frames: Record<Processor, number>;
  segments: Record<Processor, Segment[]>;
  peak: Record<Processor, number>;
}

export interface Dependency {
  /**
   * Identifies the last command that wrote the resource in this frame. Undefined means the write
   * occurred in an earlier frame.
   */
  from: number | undefined;
  resource: GpuResource;
}

export interface CapturedCommand extends Omit<GpuCommand, 'start' | 'end'> {
  index: number;
  displayName: string;
  /** Gives start and end times in milliseconds from the first timed command in the frame. */
  start: number | undefined;
  end: number | undefined;
  /** Gives command time in milliseconds excluding overlap with earlier commands in the frame. */
  ms: number | undefined;
  after: Dependency[];
}

/**
 * This status describes captured timings. They can be complete, dropped because query sets
 * awaited readback, late after two seconds or absent because the device lacks timestamp queries.
 */
export type Timings = 'complete' | 'dropped' | 'late' | 'none';

export interface FrameCapture {
  frame: number;
  timings: Timings;
  commands: CapturedCommand[];
}

export interface Recorder extends InstrumentHooks {
  /** Returns the graph columns, one per 100 ms, oldest first, up to 200. */
  bars(): Bar[];
  /** Counts the changes to the bars so far, so a graph draws again only after one. */
  revision(): number;
  /** Returns the frame rate over the last 60 frames, or 0 before two frames. */
  fps(): number;
  /**
   * Reports whether the device has `timestamp-query`, or undefined before a device is requested.
   */
  timestampQuery(): boolean | undefined;
  /** Returns what GPU time is split into. */
  detail(): Detail;
  /** Sets what GPU time is split into. */
  setDetail(detail: Detail): void;
  /**
   * Captures commands and resource use for the next frame, resolving when its timestamps arrive.
   * Calls made before resolution share the same capture.
   */
  capture(): Promise<FrameCapture>;
  /** Rejects a pending capture. Call it when recording stops. */
  stop(): void;
  /** Returns recent timed commands in Chrome trace format for Perfetto. */
  trace(): object;
  /** Returns mean frame times over the last second by part, formatted by `summarize`. */
  summary(): string;
}

interface FrameRecord {
  time: number;
  /** Whether its compute passes are timed per dispatch, to learn a breakdown. */
  split: boolean;
  commands: GpuCommand[];
  segments: Record<Processor, Segment[]>;
  total: Record<Processor, number>;
  awaiting: number;
  dropped: boolean;
  bar?: Bar;
}

export type TimedCommand = GpuCommand & { start: bigint; end: bigint };

export const processors = ['gpu', 'cpu'] as const satisfies Processor[];

/** Retain two seconds of frames to allow GPU timestamp readbacks to arrive. */
const historyDuration = 2000;
const barDuration = 100;
const barCount = 200;
const summaryBars = 1000 / barDuration;
const fpsWindow = 60;
/** A gap between frames that restarts the frame rate, as after a measurement. */
const fpsGap = 1000;
/** Frames between those split per dispatch, under a detail finer than `'pass'`. */
const splitEvery = 8;

export function isTimed(command: GpuCommand): command is TimedCommand {
  return command.start !== undefined && command.end !== undefined && command.end > command.start;
}

const earlier = (a: bigint, b: bigint) => (a < b ? a : b);
const later = (a: bigint, b: bigint) => (a > b ? a : b);

/**
 * Subtract overlap with earlier commands so command times sum to total GPU frame time.
 * Tile-based GPUs can overlap vertex work with fragments from earlier passes. A command starts no
 * earlier than the one before it or `floor`, because some drivers write a pass's first timestamp
 * before it can run, as Dawn on Metal does for the canvas pass by frames without vsync.
 */
export function ownTimes(commands: TimedCommand[], floor = 0n): number[] {
  let covered = floor;
  let previous = floor;

  return commands.map((command) => {
    const start = later(command.start, previous);
    const { end } = command;
    previous = start;
    // Starts never go backward, so earlier spans cover at most one interval after this start
    const uncovered = later(start, covered);
    covered = later(covered, end);
    return end > uncovered ? Number(end - uncovered) / 1e6 : 0;
  });
}

/** Joins names into a path, skipping missing and repeated ones. */
function path(names: (string | undefined)[]): string {
  const kept: string[] = [];
  for (const name of names) if (name && name !== kept.at(-1)) kept.push(name);
  return kept.join(' › ');
}

export function displayName(command: GpuCommand): string {
  const own = command.label ?? (command.pipelines.join(' + ') || command.name);
  return path([command.group, ...command.groups, own]);
}

/**
 * Names a command at a detail: under `'group'`, by the debug groups around it where it has any,
 * else by its own name.
 */
export function nameAt(command: GpuCommand, detail: Detail): string {
  if (detail === 'group' && command.groups.length > 0) {
    return path([command.group, ...command.groups]);
  }
  return displayName(command);
}

/** Numbers each pass in a frame's commands in order, dispatches split from one pass alike. */
export function passIndices(commands: GpuCommand[]): Map<GpuCommand, number> {
  const indices = new Map<GpuCommand, number>();
  let index = -1;
  let previous: number | undefined;
  for (const command of commands) {
    if (command.pass === undefined) continue;
    if (command.pass !== previous) index++;
    previous = command.pass;
    indices.set(command, index);
  }
  return indices;
}

interface Part {
  ms: number;
  calls: number;
}

export interface Breakdown {
  /**
   * Learns how each pass divides from a frame split per dispatch and its commands' own times,
   * moving `weight` of the way toward it. Default 0.25.
   */
  learn(
    timed: TimedCommand[],
    own: number[],
    indices: Map<GpuCommand, number>,
    weight?: number,
  ): void;
  /** Divides a whole pass's own time as learned, or names it whole until a split frame arrives. */
  divide(
    command: GpuCommand,
    index: number | undefined,
    ms: number,
    add: (name: string, ms: number, calls: number) => void,
  ): void;
}

/**
 * Creates a breakdown that divides passes into dispatches or groups, by pass position in the
 * frame. It eases toward each split frame, so a dispatch that runs in some frames keeps a share.
 */
export function createBreakdown(detail: Detail): Breakdown {
  const learned = new Map<number, Map<string, Part>>();

  return {
    learn(timed, own, indices, weight = 0.25) {
      const frame = new Map<number, Map<string, Part>>();
      timed.forEach((command, i) => {
        const index = indices.get(command);
        if (index === undefined) return;
        let parts = frame.get(index);
        if (!parts) frame.set(index, (parts = new Map()));
        const name = nameAt(command, detail);
        const part = parts.get(name);
        if (part) {
          part.ms += own[i];
          part.calls++;
        } else {
          parts.set(name, { ms: own[i], calls: 1 });
        }
      });

      for (const [index, parts] of frame) {
        const known = learned.get(index);
        if (!known) {
          learned.set(index, parts);
          continue;
        }
        for (const [name, part] of known) {
          if (parts.has(name)) continue;
          part.ms *= 1 - weight;
          part.calls *= 1 - weight;
        }
        for (const [name, part] of parts) {
          const before = known.get(name) ?? { ms: 0, calls: 0 };
          before.ms += (part.ms - before.ms) * weight;
          before.calls += (part.calls - before.calls) * weight;
          known.set(name, before);
        }
      }
    },

    divide(command, index, ms, add) {
      const parts = index === undefined || detail === 'pass' ? undefined : learned.get(index);
      let sum = 0;
      for (const part of parts?.values() ?? []) sum += part.ms;
      if (!parts || !(sum > 0)) return add(nameAt(command, detail), ms, 1);
      for (const [name, part] of parts) {
        if (part.ms > 0) add(name, (ms * part.ms) / sum, part.calls);
      }
    },
  };
}

export function addSegment(segments: Segment[], name: string, ms: number, calls = 1) {
  const segment = segments.find((existing) => existing.name === name);
  if (segment) {
    segment.ms += ms;
    segment.calls += calls;
  } else {
    segments.push({ name, ms, calls });
  }
}

/** Returns mean segment times and call counts per frame over `bars`. */
export function perFrame(bars: Bar[], processor: Processor): Segment[] {
  const frames = bars.reduce((sum, bar) => sum + bar.frames[processor], 0);
  const segments: Segment[] = [];

  for (const bar of bars) {
    for (const { name, ms, calls } of bar.segments[processor]) {
      addSegment(segments, name, ms / frames, calls / frames);
    }
  }

  return segments;
}

export const formatMs = (ms: number) => `${ms.toFixed(2)}ms`;

/** Formats a pass that writes its own timestamps as a GPU part without a time. */
export const untimedLine = (name: string) => `gpu ${'untimed'.padStart(8)}  ${name}`;

/**
 * Formats mean frame times over the last second in milliseconds. GPU totals and parts precede
 * CPU totals and parts. Parts are sorted by time and include calls per frame, and `untimed`
 * passes follow the GPU parts. Each processor uses the latest bars with available timings.
 */
export function summarize(bars: Bar[], fps: number, untimed: Iterable<string> = []): string {
  const totals = [`${fps.toFixed(0)} fps`];
  const parts: string[] = [];
  for (const processor of processors) {
    const recent = bars.filter((bar) => bar.frames[processor] > 0).slice(-summaryBars);
    const frames = recent.reduce((sum, bar) => sum + bar.frames[processor], 0);
    const segments = perFrame(recent, processor).filter(({ ms }) => ms > 0);
    segments.sort((a, b) => b.ms - a.ms);
    const sum = segments.reduce((all, { ms }) => all + ms, 0);
    totals.push(`${processor} ${formatMs(sum)} over ${frames} frames`);
    for (const { name, ms, calls } of segments) {
      const times = Math.round(calls) > 1 ? ` ×${Math.round(calls)}` : '';
      parts.push(`${processor} ${formatMs(ms).padStart(8)}  ${name}${times}`);
    }
    if (processor === 'gpu') for (const name of untimed) parts.push(untimedLine(name));
  }
  return [totals.join(' · '), ...parts].join('\n');
}

/**
 * Formats one line per captured command in submission order, with its time, name and call count.
 */
export function captureText({ frame, timings, commands }: FrameCapture): string {
  const lines = commands.map(({ index, ms, displayName: name, calls }) => {
    const time = ms === undefined ? 'untimed' : formatMs(ms);
    return `#${index} ${time.padStart(8)}  ${name}${calls > 1 ? ` ×${calls}` : ''}`;
  });
  return [`frame ${frame} · timings ${timings}`, ...lines].join('\n');
}

function captureOf(frame: number, record: FrameRecord | undefined, late: boolean): FrameCapture {
  const commands = record?.commands ?? [];
  const lastWriter = new Map<GpuResource, number>();
  const timed = commands.filter(isTimed);
  const own = ownTimes(timed);
  let next = 0;
  const origin = timed.reduce((min, { start }) => earlier(min, start), timed[0]?.start ?? 0n);
  const sinceOrigin = (time: bigint | undefined) =>
    time === undefined ? undefined : Number(time - origin) / 1e6;

  const timings: Timings = record?.dropped
    ? 'dropped'
    : late
      ? 'late'
      : commands.some((command) => command.timestamps === 'written')
        ? 'complete'
        : 'none';

  return {
    frame,
    timings,
    commands: commands.map((command, index) => {
      const after: Dependency[] = [];
      for (const { resource, access } of command.uses) {
        if (access !== 'write' && !after.some((dependency) => dependency.resource === resource)) {
          after.push({ from: lastWriter.get(resource), resource });
        }
      }

      for (const { resource, access } of command.uses) {
        if (access !== 'read') lastWriter.set(resource, index);
      }

      return {
        ...command,
        index,
        displayName: displayName(command),
        start: sinceOrigin(command.start),
        end: sinceOrigin(command.end),
        ms: isTimed(command) ? own[next++] : undefined,
        after,
      };
    }),
  };
}

/** Use one trace origin so captures taken at different times line up. */
function chromeTrace(commands: GpuCommand[], origin: bigint) {
  const timed = commands.filter(isTimed);

  return {
    traceEvents: [
      { name: 'process_name', ph: 'M', pid: 1, args: { name: 'GPU' } },
      { name: 'thread_name', ph: 'M', pid: 1, tid: 1, args: { name: 'compute' } },
      { name: 'thread_name', ph: 'M', pid: 1, tid: 2, args: { name: 'render' } },
      ...timed.map((command) => ({
        name: displayName(command),
        cat: command.kind,
        ph: 'X',
        pid: 1,
        tid: command.kind === 'compute' ? 1 : 2,
        ts: Number(command.start - origin) / 1000,
        dur: Number(command.end - command.start) / 1000,
        args: { frame: command.frame, calls: command.calls, pipelines: command.pipelines },
      })),
    ],
  };
}

/**
 * Creates a recorder that averages frame timings into graph columns and captures frames on
 * request.
 *
 * @returns hooks for `instrument`'s `start`, which the panel reads
 */
export function createRecorder(): Recorder {
  const frameTimes: number[] = [];
  const history = new Map<number, FrameRecord>();
  const timeline: Bar[] = [];
  let frame = 0;
  let detail: Detail = 'dispatch';
  let breakdown = createBreakdown(detail);
  let timestampQuery: boolean | undefined;
  let origin: bigint | undefined;
  let revision = 0;

  // React Native's Hermes lacks Promise.withResolvers
  type PendingCapture = {
    promise: Promise<FrameCapture>;
    resolve: (capture: FrameCapture) => void;
    reject: (error: Error) => void;
  } & (
    | { kind: 'requested' }
    | { kind: 'recording'; frame: number }
    | {
        kind: 'awaiting';
        frame: number;
        record: FrameRecord;
        timeout: ReturnType<typeof setTimeout>;
      }
  );
  let pending: PendingCapture | undefined;

  const capturing = () => pending?.kind === 'recording' && pending.frame === frame;

  function finishCapture(timedOut = false) {
    if (pending === undefined || pending.kind === 'requested' || pending.frame >= frame) return;

    const record = pending.kind === 'awaiting' ? pending.record : history.get(pending.frame);
    const waiting = record !== undefined && !record.dropped && record.awaiting > 0;
    if (waiting && !timedOut) {
      if (pending.kind === 'recording') {
        const timeout = setTimeout(() => finishCapture(true), historyDuration);
        pending = { ...pending, kind: 'awaiting', record, timeout };
      }
      return;
    }

    if (pending.kind === 'awaiting') clearTimeout(pending.timeout);
    const { frame: captured, resolve } = pending;
    pending = undefined;
    resolve(captureOf(captured, record, waiting));
  }

  function addTime(record: FrameRecord, processor: Processor, name: string, ms: number, calls = 1) {
    record.total[processor] += ms;
    addSegment(record.segments[processor], name, ms, calls);
  }

  function addFrame(bar: Bar, record: FrameRecord, processor: Processor) {
    revision++;
    bar.frames[processor]++;
    for (const { name, ms, calls } of record.segments[processor]) {
      addSegment(bar.segments[processor], name, ms, calls);
    }
    bar.peak[processor] = Math.max(bar.peak[processor], record.total[processor]);
  }

  /**
   * Count GPU time only after all timestamps arrive. A frame that has left retained history
   * cannot contribute to graph averages. A frame split per dispatch only teaches the breakdown,
   * since splitting adds time.
   */
  function settle(record: FrameRecord) {
    if (!record.bar || record.dropped || record.awaiting > 0) return;

    const timed = record.commands.filter(isTimed);
    const own = ownTimes(timed);
    const indices = passIndices(record.commands);
    if (record.split) {
      breakdown.learn(timed, own, indices);
      return;
    }

    const add = (name: string, ms: number, calls: number) =>
      addTime(record, 'gpu', name, ms, calls);
    timed.forEach((command, i) => breakdown.divide(command, indices.get(command), own[i], add));
    addFrame(record.bar, record, 'gpu');
  }

  function fold(record: FrameRecord) {
    const time = Math.floor(record.time / barDuration) * barDuration;
    let bar = timeline.at(-1);
    if (bar?.time !== time) {
      bar = {
        time,
        frames: { gpu: 0, cpu: 0 },
        segments: { gpu: [], cpu: [] },
        peak: { gpu: 0, cpu: 0 },
      };
      timeline.push(bar);
      if (timeline.length > barCount) timeline.shift();
    }

    addFrame(bar, record, 'cpu');
    record.bar = bar;
    settle(record);
  }

  return {
    capturing,
    splitComputePasses: () => history.get(frame)?.split ?? false,
    detail: () => detail,
    setDetail(value) {
      if (value === detail) return;
      detail = value;
      breakdown = createBreakdown(detail);
    },
    deviceRequested(supported) {
      timestampQuery = supported;
    },
    timestampQuery: () => timestampQuery,

    frameStarted(timestamp) {
      if (timestamp === frameTimes.at(-1)) return;

      const previous = history.get(frame);
      if (previous) fold(previous);

      if (timestamp - (frameTimes.at(-1) ?? timestamp) > fpsGap) frameTimes.length = 0;
      frameTimes.push(timestamp);
      if (frameTimes.length > fpsWindow) frameTimes.shift();

      frame++;
      for (const [key, record] of history) {
        if (record.time >= timestamp - historyDuration) break;
        history.delete(key);
      }
      const record: FrameRecord = {
        time: timestamp,
        split: false,
        commands: [],
        segments: { gpu: [], cpu: [] },
        total: { gpu: 0, cpu: 0 },
        awaiting: 0,
        dropped: false,
      };
      history.set(frame, record);

      if (pending?.kind === 'requested') {
        pending = { ...pending, kind: 'recording', frame };
      } else {
        finishCapture();
      }
      // A capture shows the frame as finely as the graph does
      record.split = detail !== 'pass' && (frame % splitEvery === 0 || capturing());
    },

    callbackTimed(name, ms) {
      const record = history.get(frame);
      if (record) addTime(record, 'cpu', name, ms);
    },

    submitted(commands) {
      const record = history.get(frame);
      for (const command of commands) {
        command.frame = frame;
        if (!record) continue;

        record.commands.push(command);
        if (command.timestamps === 'written') record.awaiting++;
        if (command.timestamps === 'dropped') record.dropped = true;
      }
    },

    timed(commands) {
      for (const command of commands) {
        if (command.start !== undefined) origin ??= command.start;

        const record = history.get(command.frame);
        if (!record || command.timestamps !== 'written') continue;

        record.awaiting--;
        if (command.start === undefined) record.dropped = true;
        settle(record);
      }
      finishCapture();
    },

    bars: () => timeline,
    revision: () => revision,

    fps() {
      const span = (frameTimes.at(-1) ?? 0) - (frameTimes[0] ?? 0);
      return span > 0 ? ((frameTimes.length - 1) * 1000) / span : 0;
    },

    capture() {
      if (!pending) {
        let resolve!: (capture: FrameCapture) => void;
        let reject!: (error: Error) => void;
        const promise = new Promise<FrameCapture>((onResolve, onReject) => {
          resolve = onResolve;
          reject = onReject;
        });
        pending = { kind: 'requested', promise, resolve, reject };
      }
      return pending.promise;
    },

    stop() {
      if (!pending) return;
      if (pending.kind === 'awaiting') clearTimeout(pending.timeout);
      pending.reject(new Error('Recording stopped before the capture finished.'));
      pending = undefined;
    },

    summary() {
      const since = (frameTimes.at(-1) ?? 0) - summaryBars * barDuration;
      const untimed = new Set<string>();
      for (const record of history.values()) {
        if (record.time < since) continue;
        for (const command of record.commands) {
          if (command.timestamps === 'own') untimed.add(displayName(command));
        }
      }
      return summarize(timeline, this.fps(), untimed);
    },

    trace: () =>
      chromeTrace(
        [...history.values()].flatMap((record) => record.commands),
        origin ?? 0n,
      ),
  };
}
