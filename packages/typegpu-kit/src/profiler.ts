import { d, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';
import { encoderOf, type Frame } from './frames.ts';
import { staging } from './staging.ts';

/** Specifies timestamp writes for a pass descriptor or `pipeline.withTimestampWrites`. */
export interface TimestampWrites {
  readonly querySet: GPUQuerySet;
  readonly beginningOfPassWriteIndex: number;
  readonly endOfPassWriteIndex: number;
}

/** Holds a timer's writes split between the first and the last of several passes. */
export interface Split {
  /** Writes the timer's start, for the first pass. */
  readonly opening: Omit<TimestampWrites, 'endOfPassWriteIndex'> | undefined;
  /** Writes the timer's end, for the last pass. */
  readonly closing: Omit<TimestampWrites, 'beginningOfPassWriteIndex'> | undefined;
}

const untimed: Split = { opening: undefined, closing: undefined };
const splits = new WeakMap<TimestampWrites, Split>();

/**
 * Splits a timer's writes so the first of several passes starts the measurement and the last
 * ends it. Returns the same objects for the same writes, so a timed frame allocates nothing.
 *
 * @param writes a timer's writes, or none for untimed passes
 * @returns the writes for the first pass and for the last
 */
export function split(writes: TimestampWrites | undefined): Split {
  if (!writes) return untimed;
  let parts = splits.get(writes);
  if (!parts) {
    const { querySet, beginningOfPassWriteIndex, endOfPassWriteIndex } = writes;
    parts = {
      opening: { querySet, beginningOfPassWriteIndex },
      closing: { querySet, endOfPassWriteIndex },
    };
    splits.set(writes, parts);
  }
  return parts;
}

/**
 * Keeps a timer's first write on the first of several passes and its last on the last, and
 * leaves the passes between untimed.
 *
 * @hidden
 */
export const opening = (writes: TimestampWrites, first: boolean, last: boolean) => {
  if (first && last) return writes;
  if (first) return split(writes).opening;
  if (last) return split(writes).closing;
  return undefined;
};

/**
 * Records an empty pass that writes both timestamps of a timer, so a frame that draws nothing
 * still times as zero.
 *
 * @hidden
 */
export const untimedFrame = (frame: Frame | TgpuCommandEncoder, writes: TimestampWrites) => {
  encoderOf(frame).beginComputePass({ timestampWrites: writes }).end();
};

/** Adds timing to a recording method through `{ timestampWrites: timer.writes() }`. */
export interface Timed {
  /** Supplies a timer's writes for the passes. Without them, passes run untimed. Default none. */
  readonly timestampWrites?: TimestampWrites | undefined;
}

export interface Options {
  /**
   * Limits the number of timers, with two timestamps each. Must be an integer above 0. Default 16.
   */
  readonly capacity?: number;
  /** Sets the number of samples per median. Must be an integer above 0. Default 30. */
  readonly samples?: number;
  /**
   * Ignores the first this many frames read back while pipelines and caches warm up. A timer whose
   * pass runs only now and then, such as a level load, still reports after them. Default 30.
   */
  readonly skip?: number;
  /**
   * Sets the number of staging buffers. Frames run untimed while all buffers are mapping. Must be
   * an integer above 0. Default 3.
   */
  readonly staging?: number;
  /**
   * Receives samples in milliseconds after the first `skip` frames, as they arrive a frame or two
   * later.
   * Use for a maximum, percentile or plot. `timing` identifies a timer or the profiler's `frame`.
   * Default none.
   * @cpu
   */
  readonly onSample?: (timing: Timing, ms: number) => void;
}

/** Reports a GPU duration tracked by the profiler. */
export interface Timing {
  /** Gives the display label set at creation. Timers can share a label. */
  readonly name: string;
  /**
   * Reports the median GPU duration in milliseconds from the last `samples` samples, or undefined
   * before any samples. A duration runs from the pass's first timestamp to its last, so it includes
   * time spent waiting on earlier passes and time shared with passes running beside it. It starts
   * no earlier than the pass timed before it in the frame, since some drivers write a first
   * timestamp before the pass can run. Passes overlap on the GPU, so their durations do not sum to
   * the frame duration.
   */
  readonly ms: number | undefined;
  /**
   * Counts samples kept after the first `skip` frames. Between two polls, a change means new
   * samples arrived; `last` is the newest.
   */
  readonly count: number;
  /** Reports the newest sample in milliseconds, or undefined before any samples. */
  readonly last: number | undefined;
  /** Identifies the frame that `last` timed, or -1 before any samples. */
  readonly lastFrame: number;
}

/** Times one GPU pass. */
export interface Timer extends Timing {
  /**
   * Reports the median of this pass's share of the frame in milliseconds, or undefined before any
   * samples. Each stretch of GPU time goes to the earliest recorded timed pass running in it, as
   * later passes usually wait on it. A pass that waits for earlier ones, such as the canvas pass,
   * has a long duration but a short share: the tail it adds to the frame. Shares add up to the time
   * any timed pass was running, which is the frame without its gaps.
   */
  readonly share: number | undefined;
  /**
   * Marks the pass for timing in this frame and returns timestamp writes, or undefined without
   * `timestamp-query`. Only marked passes are read back, because skipped passes keep stale
   * timestamps. Call in every frame that records the pass, as you record it: a timing starts no
   * earlier than the one marked before it.
   */
  writes(): TimestampWrites | undefined;
}

export interface Profiler extends Disposable {
  /**
   * Reports whether the device supports `timestamp-query`. Without it, timers return no writes or
   * timings.
   */
  readonly available: boolean;
  /** Creates a timer for one pass during setup. Throws when the timer count exceeds `capacity`. */
  timer(name: string): Timer;
  /** Lists timers in creation order. */
  readonly timers: readonly Timer[];
  /**
   * Reports the duration from the earliest timed pass start to the latest timed pass end in each
   * frame. Covers the frame's GPU work when the first and last passes are timed, including gaps
   * between passes. The final pass, such as the canvas pass, waits for work it draws.
   */
  readonly frame: Timing;
  /**
   * Resolves the frame's timestamps and maps them after the submit. Call it after the timed passes.
   * A second call in the same frame records nothing.
   */
  capture(frame: Frame): void;
  /** Destroys the query set and buffers. */
  destroy(): void;
}

const inert = () => {};

const counted = (name: string, value: number) => {
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(`${name} must be an integer above 0, got ${value}.`);
  }
  return value;
};

/**
 * Times GPU passes with one shared query set and reads results without stalling. Throws unless
 * `capacity`, `samples` and `staging` are integers above 0.
 *
 * @requires timestamp-query optional without it the profiler is inert and its timings stay
 * undefined
 * @owns a query set, 4 buffers: two timestamps per timer, a buffer they resolve into, and staging
 * buffers mapped in turn, as many as `staging`
 * @records `capture` after the passes it times; a timer's `writes` in every frame that records its
 * pass
 * @returns a profiler you create timers on, capture each frame and destroy
 */
export function create(root: TgpuRoot, options: Options = {}): Profiler {
  const { skip = 30, onSample } = options;
  const capacity = counted('capacity', options.capacity ?? 16);
  const samples = counted('samples', options.samples ?? 30);
  const slots = counted('staging', options.staging ?? 3);
  const device = root.device;
  const available = device.features.has('timestamp-query');
  const count = capacity * 2;

  const timers: Timer[] = [];
  // Series: one per timer, then the frame's, then each timer's share
  const series = capacity * 2 + 1;
  const seen = new Uint32Array(series);
  const history = Array.from({ length: series }, () => new Float64Array(samples));
  const medians = new Float64Array(series).fill(Number.NaN);
  const lasts = new Float64Array(series).fill(Number.NaN);
  const lastFrames = new Float64Array(series).fill(-1);
  const scratch = new Float64Array(samples);
  // The timers marked this frame, in the order their passes were recorded
  const used = new Uint8Array(capacity);
  const sequence = new Uint16Array(capacity);
  let marks = 0;
  const median = (i: number) => (Number.isNaN(medians[i]) ? undefined : medians[i]);
  const latest = (i: number) => (Number.isNaN(lasts[i]) ? undefined : lasts[i]);
  const kept = (i: number) => seen[i];
  let warming = skip;

  // A frame's timestamps in GPU nanoseconds by recording rank, and their order in time, kept to
  // allocate nothing per frame
  const stamps = new Float64Array(count);
  const order = new Uint16Array(count);
  const shares = new Float64Array(capacity);
  const running = new Uint8Array(capacity);

  const whole: Timing = {
    name: 'frame',
    get ms() {
      return median(capacity);
    },
    get count() {
      return kept(capacity);
    },
    get last() {
      return latest(capacity);
    },
    get lastFrame() {
      return lastFrames[capacity];
    },
  };

  const add = (i: number, ms: number, frame: number) => {
    seen[i] += 1;
    const n = seen[i];
    history[i][(n - 1) % samples] = ms;
    lasts[i] = ms;
    lastFrames[i] = frame;

    const filled = Math.min(n, samples);
    const sorted = scratch.subarray(0, filled);
    sorted.set(history[i].subarray(0, filled));
    sorted.sort();

    medians[i] =
      filled % 2 === 1
        ? sorted[filled >> 1]
        : (sorted[(filled >> 1) - 1] + sorted[filled >> 1]) / 2;
    if (i <= capacity) onSample?.(i < capacity ? timers[i] : whole, ms);
  };

  // Sweeps the frame's timestamps in time order and gives each stretch to the running pass with
  // the lowest recording rank. An event is a rank times two, plus one for an end
  const attribute = (events: number) => {
    for (let k = 1; k < events; k++) {
      const event = order[k];
      let j = k - 1;
      for (; j >= 0 && stamps[order[j]] > stamps[event]; j--) order[j + 1] = order[j];
      order[j + 1] = event;
    }

    shares.fill(0);
    running.fill(0);
    let previous = 0;
    for (let k = 0; k < events; k++) {
      const event = order[k];
      const at = stamps[event];
      const earliest = running.indexOf(1);
      if (earliest >= 0) shares[earliest] += at - previous;
      previous = at;
      running[event >> 1] = (event & 1) === 0 ? 1 : 0;
    }
  };

  // Use a raw query set to avoid TypeGPU resolving and submitting through its own encoder
  const querySet = available ? device.createQuerySet({ type: 'timestamp', count }) : undefined;
  const Words = d.arrayOf(d.u32, count * 2);
  const resolved = available
    ? root.createBuffer(Words).$addFlags(GPUBufferUsage.QUERY_RESOLVE)
    : undefined;
  const resolvedGpu = resolved && root.unwrap(resolved);

  const timed = Array.from({ length: slots }, () => new Uint16Array(capacity));
  const timedCount = new Uint16Array(slots);
  const timedFrame = new Float64Array(slots);

  const ring = available
    ? staging(root, {
        schema: Words,
        slots,

        // Timestamps use u64 nanoseconds. Doubles represent them exactly up to 2^53 ns, about 104
        // days of GPU clock time, and round later values
        read(slot, words) {
          if (warming > 0) {
            warming -= 1;
            return;
          }

          const index = timedFrame[slot];
          let first = Infinity;
          let last = 0;
          let events = 0;
          let floor = 0;

          for (let k = 0; k < timedCount[slot]; k++) {
            const i = timed[slot][k];
            // Some drivers write a pass's first timestamp before the passes recorded ahead of it
            // start, as Dawn on Metal does for the canvas pass, by frames without vsync. A pass
            // starts no earlier than the one timed before it
            const begin = Math.max(words[i * 4 + 1] * 2 ** 32 + words[i * 4], floor);
            const end = words[i * 4 + 3] * 2 ** 32 + words[i * 4 + 2];
            if (end <= begin) continue;
            floor = begin;

            add(i, (end - begin) / 1e6, index);
            stamps[k * 2] = begin;
            stamps[k * 2 + 1] = end;
            order[events++] = k * 2;
            order[events++] = k * 2 + 1;
            if (begin < first) first = begin;
            if (end > last) last = end;
          }

          if (last > first) add(capacity, (last - first) / 1e6, index);

          attribute(events);
          for (let k = 0; k < events; k++) {
            const event = order[k];
            if ((event & 1) === 0) {
              add(capacity + 1 + timed[slot][event >> 1], shares[event >> 1] / 1e6, index);
            }
          }
        },

        failed: inert, // device loss; nothing to report
      })
    : undefined;
  const destroy = () => {
    querySet?.destroy();
    resolved?.destroy();
    ring?.destroy();
  };

  return {
    available,
    timers,
    frame: whole,

    timer(name) {
      const i = timers.length;
      if (i >= capacity) {
        throw new RangeError(
          `This profiler has reached its limit of ${capacity} timers. Increase \`capacity\` in its options.`,
        );
      }

      const writes: TimestampWrites | undefined = querySet && {
        querySet,
        beginningOfPassWriteIndex: i * 2,
        endOfPassWriteIndex: i * 2 + 1,
      };
      const made: Timer = {
        name,
        get ms() {
          return median(i);
        },
        get count() {
          return kept(i);
        },
        get last() {
          return latest(i);
        },
        get lastFrame() {
          return lastFrames[i];
        },
        get share() {
          return median(capacity + 1 + i);
        },
        writes() {
          if (!used[i]) {
            used[i] = 1;
            sequence[marks++] = i;
          }
          return writes;
        },
      };

      timers.push(made);
      return made;
    },

    capture(frame) {
      if (!ring) return;
      const slot = ring.take(frame);

      if (slot >= 0 && marks > 0) {
        root
          .unwrap(frame.encoder)
          .resolveQuerySet(querySet!, 0, timers.length * 2, resolvedGpu!, 0);
        ring.copy(frame, slot, resolved!);
        timed[slot].set(sequence);
        timedCount[slot] = marks;
        timedFrame[slot] = frame.index;
      }

      used.fill(0);
      marks = 0;
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
