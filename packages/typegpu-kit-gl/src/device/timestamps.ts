/**
 * Timestamp queries, from `EXT_disjoint_timer_query_webgl2`. WebGL gives the GPU time between a
 * query's begin and end, not the GPU's clock, which `TIMESTAMP_EXT` would read but few contexts
 * support. So from the first pass of a task that writes a timestamp, the device keeps one
 * `TIME_ELAPSED_EXT` query running, ending it and beginning the next at each timestamp a pass
 * writes, until the task ends. A timestamp is a running total of those spans: the difference of two
 * is the GPU time between them, whatever ran in between, timed passes or not. Between tasks the
 * total stands still, so a frame's timestamps follow the last frame's without the idle time.
 *
 * Results arrive frames later. A resolve leaves a read on its destination that `mapAsync` waits
 * for, as for a copy from GL, and a copy of the destination carries it along. A span the GPU
 * reports disjoint makes every timestamp after it in its task zero, WebGPU's value for a timestamp
 * it could not take.
 */

import { type BufferImpl, type PendingRead, writing, written } from './buffer.ts';
import { type Ctx, IMPL, implOf } from './context.ts';

/** A point in the running total: the end of a span, or the start of a task's spans. */
interface Stamp {
  /** Whether a span ends here; a task's first point adds no time. */
  readonly span: boolean;
  query: WebGLQuery | undefined;
  ended: boolean;
  /** When the span ended, for giving up on a result that never arrives. */
  endedAt: number;
  /** The disjoint count when the span began. */
  readonly epoch: number;
  /** The point before; dropped once this one has a total. */
  previous: Stamp | undefined;
  /** Nanoseconds of the running total here, once every span up to here has its result. */
  total: bigint | undefined;
  /** A span of this task up to here was disjoint, so the total is not the GPU's time. */
  invalid: boolean;
}

interface Timeline {
  /** The latest point, which the next one follows. */
  last: Stamp;
  /** The span running, if a task's spans have begun. */
  open: Stamp | undefined;
  /** Points whose totals are not known yet, in order. */
  readonly waiting: Stamp[];
  /** Times GL reported a disjoint operation, and whether this task has read the flag. */
  disjoints: number;
  read: boolean;
  polled: boolean;
}

const timelines = new WeakMap<Ctx, Timeline>();

/**
 * Milliseconds a span's result may take before its timestamps count as not taken, as placement's
 * probes do, so a map never waits for a query GL never answers.
 */
const LIFETIME_MS = 2000;

function timelineOf(ctx: Ctx): Timeline {
  let timeline = timelines.get(ctx);
  if (!timeline) {
    // Start near the page's clock, so no timestamp is zero
    const origin: Stamp = {
      span: false,
      query: undefined,
      ended: true,
      endedAt: 0,
      epoch: 0,
      previous: undefined,
      total: BigInt(Math.round(performance.now() * 1e6)),
      invalid: false,
    };
    timeline = {
      last: origin,
      open: undefined,
      waiting: [],
      disjoints: 0,
      read: false,
      polled: false,
    };
    timelines.set(ctx, timeline);
    const made = timeline;
    ctx.track(() => release(ctx, made));
  }
  return timeline;
}

/** Ends the running span and deletes every query, once, when the device is destroyed. */
function release(ctx: Ctx, timeline: Timeline): void {
  if (timeline.open && !ctx.lost) ctx.gl.endQuery(ctx.ext.timerQuery!.TIME_ELAPSED_EXT);
  timeline.open = undefined;
  for (const point of timeline.waiting) {
    if (point.query) ctx.gl.deleteQuery(point.query);
    point.query = undefined;
    point.ended = true;
  }
}

/** Whether the context can time spans, which `timestamp-query` needs. */
export function canTime(gl: WebGL2RenderingContext, timer: Ctx['ext']['timerQuery']): boolean {
  if (!timer) return false;
  return (gl.getQuery(timer.TIME_ELAPSED_EXT, timer.QUERY_COUNTER_BITS_EXT) as number) > 0;
}

/**
 * The number of times GL has reported a disjoint operation, which voids every timer query in
 * flight across it. Reading the flag clears it and may cost a round trip, so a task reads it once,
 * and each user of timer queries compares the count with the one when its query began.
 */
function disjoints(ctx: Ctx): number {
  const timer = ctx.ext.timerQuery;
  if (!timer || ctx.lost) return 0;
  const timeline = timelineOf(ctx);
  if (!timeline.read) {
    timeline.read = true;
    queueMicrotask(() => {
      timeline.read = false;
    });
    if (ctx.gl.getParameter(timer.GPU_DISJOINT_EXT)) timeline.disjoints++;
  }
  return timeline.disjoints;
}

const seen = new WeakMap<object, number>();

/**
 * Whether GL reported a disjoint operation since `reader` last asked, which voids its timer queries
 * in flight. Every user of timer queries asks here: reading GL's flag clears it for all of them.
 */
export function disjointFor(ctx: Ctx, reader: object): boolean {
  const count = disjoints(ctx);
  const last = seen.get(reader) ?? 0;
  seen.set(reader, count);
  return count !== last;
}

/**
 * Whether a `TIME_ELAPSED_EXT` query may begin, which only one may be at a time. False while a
 * task's timestamps are being taken; placement then times its forms with fences.
 */
export function mayTime(ctx: Ctx): boolean {
  return timelines.get(ctx)?.open === undefined;
}

function begin(ctx: Ctx, timeline: Timeline, previous: Stamp): Stamp | undefined {
  const timer = ctx.ext.timerQuery!;
  const query = ctx.gl.createQuery();
  if (!query) return undefined;
  ctx.gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
  const span: Stamp = {
    span: true,
    query,
    ended: false,
    endedAt: 0,
    epoch: disjoints(ctx),
    previous,
    total: undefined,
    invalid: false,
  };
  timeline.waiting.push(span);
  timeline.open = span;
  timeline.last = span;
  return span;
}

/** Ends the running span at the end of the task, so the next task starts where this one stopped. */
function close(ctx: Ctx, timeline: Timeline): void {
  const span = timeline.open;
  if (!span) return;
  timeline.open = undefined;
  span.ended = true;
  span.endedAt = performance.now();
  if (ctx.lost) return;
  ctx.gl.endQuery(ctx.ext.timerQuery!.TIME_ELAPSED_EXT);
  // A query's result arrives only once the commands before its end reach the GPU
  ctx.gl.flush();
}

/** Begins a task's spans, if they have not begun, before a pass that writes a timestamp. */
export function startTiming(ctx: Ctx): void {
  if (ctx.lost || !ctx.ext.timerQuery) return;
  const timeline = timelineOf(ctx);
  if (timeline.open) return;
  // Reads what arrived, so spans nothing resolves do not pile up
  poll(ctx);
  // A task's first point adds no time to the last, and nothing before it voids it
  const start: Stamp = {
    span: false,
    query: undefined,
    ended: true,
    endedAt: 0,
    epoch: 0,
    previous: timeline.last,
    total: undefined,
    invalid: false,
  };
  timeline.waiting.push(start);
  timeline.last = start;
  if (!begin(ctx, timeline, start)) return;
  setTimeout(() => close(ctx, timeline), 0);
}

/** Takes a timestamp: ends the running span there and begins the next. */
export function stamp(ctx: Ctx): Stamp | undefined {
  if (ctx.lost || !ctx.ext.timerQuery) return undefined;
  startTiming(ctx);
  const timeline = timelineOf(ctx);
  const span = timeline.open;
  if (!span) return undefined;
  ctx.gl.endQuery(ctx.ext.timerQuery.TIME_ELAPSED_EXT);
  span.ended = true;
  span.endedAt = performance.now();
  timeline.open = undefined;
  begin(ctx, timeline, span);
  return span;
}

/** Reads the results that have arrived, in order, into running totals. */
function poll(ctx: Ctx): void {
  const timeline = timelines.get(ctx);
  if (!timeline || timeline.polled) return;
  // Results change only between tasks
  timeline.polled = true;
  queueMicrotask(() => {
    timeline.polled = false;
  });
  const { gl } = ctx;
  const count = timeline.waiting.length > 0 ? disjoints(ctx) : 0;
  while (timeline.waiting.length > 0) {
    const point = timeline.waiting[0]!;
    let ns = 0;
    let disjoint = false;
    if (point.query) {
      if (!point.ended) break;
      if (ctx.lost || count !== point.epoch) disjoint = true;
      else if (gl.getQueryParameter(point.query, gl.QUERY_RESULT_AVAILABLE)) {
        ns = gl.getQueryParameter(point.query, gl.QUERY_RESULT) as number;
      } else if (performance.now() - point.endedAt > LIFETIME_MS) disjoint = true;
      else break;
      gl.deleteQuery(point.query);
      point.query = undefined;
    }
    const previous = point.previous!;
    point.total = previous.total! + BigInt(ns);
    // A task's first point starts afresh; a span after a disjoint one in its task stays void
    point.invalid = point.span && (disjoint || previous.invalid);
    point.previous = undefined;
    timeline.waiting.shift();
  }
}

/** A query set's slots, each the timestamp a pass last wrote there. */
export interface QuerySetImpl {
  readonly type: GPUQueryType;
  readonly count: number;
  readonly slots: (Stamp | undefined)[];
  /** Made against a missing feature, so every use of it is an error, as in WebGPU. */
  readonly invalid: boolean;
  destroyed: boolean;
}

export function querySetImpl(set: GPUQuerySet): QuerySetImpl {
  return implOf<QuerySetImpl>(set, 'query set');
}

export function createQuerySet(
  ctx: Ctx,
  descriptor: GPUQuerySetDescriptor,
  timed: boolean,
): GPUQuerySet {
  const { type, count } = descriptor;
  const invalid = type === 'timestamp' && !timed;
  if (invalid) {
    ctx.errors.report(
      'validation',
      "createQuerySet of type 'timestamp' on a device without 'timestamp-query'.",
    );
  }
  const impl: QuerySetImpl = {
    type,
    count,
    slots: Array.from({ length: count }, () => undefined),
    invalid,
    destroyed: false,
  };
  return {
    [IMPL]: impl,
    __brand: 'GPUQuerySet',
    label: descriptor.label ?? '',
    type,
    count,
    destroy() {
      impl.destroyed = true;
      impl.slots.fill(undefined);
    },
  } as unknown as GPUQuerySet;
}

/** A pass's timestamp writes, checked when the pass is encoded. */
export interface Writes {
  readonly set: QuerySetImpl;
  readonly beginning: number | undefined;
  readonly end: number | undefined;
}

export function writesOf(
  ctx: Ctx,
  writes: GPURenderPassTimestampWrites | GPUComputePassTimestampWrites | undefined,
): Writes | undefined {
  if (!writes) return undefined;
  const set = querySetImpl(writes.querySet);
  if (set.invalid) {
    ctx.errors.report('validation', 'A pass writes timestamps to an invalid query set.');
    return undefined;
  }
  const { beginningOfPassWriteIndex: beginning, endOfPassWriteIndex: end } = writes;
  const inRange = (index: number | undefined) =>
    index === undefined || (index >= 0 && index < set.count);
  if (set.type !== 'timestamp' || !inRange(beginning) || !inRange(end) || beginning === end) {
    ctx.errors.report('validation', 'A pass has invalid timestampWrites.');
    return undefined;
  }
  return { set, beginning, end };
}

/** Writes a pass's timestamp into its slot, at the pass's beginning or end. */
export function write(ctx: Ctx, writes: Writes | undefined, at: 'beginning' | 'end'): void {
  if (!writes) return;
  if (at === 'beginning') startTiming(ctx);
  const index = writes[at];
  if (index === undefined || writes.set.destroyed) return;
  writes.set.slots[index] = stamp(ctx);
}

/**
 * A resolve's read: timestamps whose eight bytes each start at `offset` in its buffer, of which it
 * writes bytes `start` to `end`. A copy of part of the buffer carries that part along, and a write
 * over part of it leaves the rest waiting.
 */
interface TimestampRead extends PendingRead {
  readonly stamps: readonly (Stamp | undefined)[];
  readonly offset: number;
  readonly start: number;
  readonly end: number;
}

const timestampReads = new WeakSet<PendingRead>();

function timestampRead(
  ctx: Ctx,
  buffer: BufferImpl,
  stamps: readonly (Stamp | undefined)[],
  offset: number,
  start: number,
  end: number,
): TimestampRead {
  const read: TimestampRead = {
    stamps,
    offset,
    start,
    end,
    ready() {
      poll(ctx);
      return ctx.lost || stamps.every((each) => each === undefined || each.total !== undefined);
    },
    finish() {
      poll(ctx);
      if (buffer.destroyed) return;
      const bytes = new Uint8Array(stamps.length * 8);
      const view = new DataView(bytes.buffer);
      stamps.forEach((each, i) => {
        const value = each?.total !== undefined && !each.invalid ? each.total : 0n;
        view.setBigUint64(i * 8, value, true);
      });
      buffer.bytes.set(bytes.subarray(start - offset, end - offset), start);
      written(buffer, start, end);
    },
    cancel() {},
    without(from, to) {
      if (to <= start || from >= end) return [read];
      const parts: TimestampRead[] = [];
      if (from > start) parts.push(timestampRead(ctx, buffer, stamps, offset, start, from));
      if (to < end) parts.push(timestampRead(ctx, buffer, stamps, offset, to, end));
      return parts;
    },
  };
  timestampReads.add(read);
  return read;
}

/**
 * Resolves `count` slots from `first` into `destination` at `offset`, as their results arrive.
 * Until then the bytes hold zeros, which a draw or kernel that reads them early sees.
 */
export function resolve(
  ctx: Ctx,
  set: QuerySetImpl,
  first: number,
  count: number,
  destination: BufferImpl,
  offset: number,
): void {
  if (set.invalid) {
    ctx.errors.report('validation', 'resolveQuerySet of an invalid query set.');
    return;
  }
  if (set.destroyed || first + count > set.count || offset + count * 8 > destination.size) {
    ctx.errors.report('validation', 'resolveQuerySet out of range, or of a destroyed query set.');
    return;
  }
  const end = offset + count * 8;
  const done = writing(destination, offset, end);
  destination.bytes.fill(0, offset, end);
  written(destination, offset, end);
  done();
  const stamps = set.slots.slice(first, first + count);
  destination.reads.push(timestampRead(ctx, destination, stamps, offset, offset, end));
}

/**
 * Takes the source's unfinished resolves out of its reads, so a copy can sync the rest, and
 * returns a function that puts them back and gives the copy's destination the bytes it copied.
 */
export function carry(
  ctx: Ctx,
  source: BufferImpl,
  sourceOffset: number,
  destination: BufferImpl,
  destinationOffset: number,
  size: number,
): () => void {
  const reads = source.reads.filter((read) => timestampReads.has(read)) as TimestampRead[];
  if (reads.length === 0) return () => {};
  const rest = source.reads.filter((read) => !timestampReads.has(read));
  source.reads.splice(0, source.reads.length, ...rest);
  return () => {
    source.reads.push(...reads);
    const shift = destinationOffset - sourceOffset;
    for (const read of reads) {
      const start = Math.max(read.start, sourceOffset);
      const end = Math.min(read.end, sourceOffset + size);
      if (end <= start) continue;
      destination.reads.push(
        timestampRead(
          ctx,
          destination,
          read.stamps,
          read.offset + shift,
          start + shift,
          end + shift,
        ),
      );
    }
  };
}
