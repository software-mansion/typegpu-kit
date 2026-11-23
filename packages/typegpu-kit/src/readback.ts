import { d, type TgpuBuffer, type TgpuRoot } from 'typegpu';
import { cadence } from './clock.ts';
import type { Frame } from './frames.ts';
import { staging } from './staging.ts';

export interface Options {
  /**
   * Sets the number of staging buffers. Must be an integer above 0. Captures are skipped while all
   * buffers are mapping. Default 3.
   */
  readonly staging?: number;
  /**
   * Also captures for `latest` this many times per second on average, in the frame closest to each
   * tick. Without a rate, only `next()` and `request` cause captures. Default none.
   */
  readonly hz?: number;
}

export interface DecodeOptions<S extends d.AnyWgslData, T> extends Options {
  /**
   * Converts the source buffer value, read by its schema, to the value returned by `next` and
   * `latest`, such as a candidate index or `null`. If it throws, reads for that copy fail with the
   * error.
   * @cpu
   */
  readonly decode: (value: d.Infer<S>) => T;
}

export interface BytesOptions<T = ArrayBuffer> extends Options {
  /** Reads the copied bytes without decoding them through the buffer's schema. */
  readonly bytes: true;
  /** Copies from this many bytes into the buffer, a multiple of 4. Default 0. */
  readonly offset?: number;
  /** Copies this many bytes, a multiple of 4. Default the rest of the buffer after `offset`. */
  readonly size?: number;
  /**
   * Converts the copied bytes to the value returned by `next` and `latest`, such as typed arrays of
   * your own. The bytes are valid only during the call, so copy what you keep. If it throws, reads
   * for that copy fail with the error. Required for a result other than an `ArrayBuffer`. Default a
   * copy of the bytes.
   * @cpu
   */
  readonly decode?: (bytes: ArrayBuffer) => T;
}

export interface Readback<T> extends Disposable {
  /** Reports the latest value read, or undefined before the first read and after `reset`. */
  readonly latest: T | undefined;
  /** Identifies the frame copied for `latest`, or -1 if there is no value. */
  readonly latestFrame: number;
  /**
   * Returns `latest` once per completed copy, then undefined until the next copy completes. Each
   * value goes to one caller. Callers sharing a readback split the values between them.
   */
  take(): T | undefined;
  /**
   * Returns a promise for the value from the next capture, and requests that capture. Promises
   * asked for before one capture share its copy. Rejects if mapping fails or the reader is
   * destroyed first. Waits indefinitely without a capture. In a coroutine, wait with
   * `yield* coroutines.until(probe.next())`.
   */
  next(): Promise<T>;
  /**
   * Requests a copy at the next capture for `latest` and `take()`, without a promise. Use for
   * buffers written on demand, such as a tap pick. If all staging buffers are mapping, keeps the
   * request for the next capture.
   */
  request(): void;
  /**
   * Records a copy when a `next()` is waiting, a request is pending or the capture rate is due. Maps
   * the copy after submission. Call after the pass that writes the buffer. A second call in the
   * same frame does nothing.
   */
  capture(frame: Frame): void;
  /**
   * Clears `latest` and discards results from copies already recorded, such as after resetting the
   * source buffer. Reads waiting on those copies move to the next capture.
   */
  reset(): void;
  /** Destroys the staging buffers and rejects the reads still waiting. */
  destroy(): void;
}

interface Reader<T> {
  resolve(value: T): void;
  reject(error: unknown): void;
}

interface Copy<T> {
  epoch: number;
  sequence: number;
  frame: number;
  /** Keeps a request pending if the frame that captured it never submits. */
  requested: boolean;
  readonly readers: Reader<T>[];
}

const destroyedError = () => new Error('The readback was destroyed.');

/**
 * Copies a buffer, or a range of its bytes, to the CPU without stalling, using a ring of staging
 * buffers. Records one copy per capture. Throws unless `staging` is an integer above 0, `hz`, when
 * set, is finite and above 0, and `offset` and `size` are multiples of 4 that fit the buffer.
 *
 * @owns 3 buffers: staging buffers mapped in turn, as many as `staging`, each the size of the copy
 * @records `capture` after the pass that writes the buffer; a second capture in a frame does
 * nothing.
 * @param source the buffer it copies
 * @returns a reader you capture into and destroy
 */
export function create<S extends d.AnyWgslData>(
  root: TgpuRoot,
  source: TgpuBuffer<S>,
  options: BytesOptions,
): Readback<ArrayBuffer>;
export function create<S extends d.AnyWgslData, T>(
  root: TgpuRoot,
  source: TgpuBuffer<S>,
  options: BytesOptions<T> & { readonly decode: (bytes: ArrayBuffer) => T },
): Readback<T>;
export function create<S extends d.AnyWgslData, T>(
  root: TgpuRoot,
  source: TgpuBuffer<S>,
  options: DecodeOptions<S, T>,
): Readback<T>;
export function create<S extends d.AnyWgslData>(
  root: TgpuRoot,
  source: TgpuBuffer<S>,
  options?: Options & { readonly bytes?: false },
): Readback<d.Infer<S>>;
export function create<S extends d.AnyWgslData, T>(
  root: TgpuRoot,
  source: TgpuBuffer<S>,
  options: Options &
    Partial<Pick<BytesOptions, 'offset' | 'size'>> & {
      readonly bytes?: boolean;
      readonly decode?: (value: never) => T;
    } = {},
): Readback<T> {
  const { staging: slots = 3, hz, bytes = false } = options;
  if (!Number.isInteger(slots) || slots < 1) {
    throw new RangeError(`staging must be an integer above 0, got ${slots}.`);
  }
  let range: { offset: number; size: number } | undefined;
  if (bytes && (options.offset !== undefined || options.size !== undefined)) {
    const whole = d.sizeOf(source.dataType);
    const { offset = 0, size = whole - offset } = options;
    if (offset % 4 !== 0 || size % 4 !== 0 || offset < 0 || size <= 0 || offset + size > whole) {
      throw new RangeError(
        `The readback's \`offset\` and \`size\` must be multiples of 4 bytes and fit in the ${whole}-byte buffer. Use an offset of at least 0 and a size above 0. Got offset ${offset} and size ${size}.`,
      );
    }
    range = { offset, size };
  }

  const decode = (options.decode ??
    (bytes ? (copied: ArrayBuffer) => copied.slice(0) : (value: unknown) => value)) as (
    stored: unknown,
  ) => T;
  const rate = hz === undefined ? undefined : cadence({ hz });

  let latest: T | undefined;
  let latestFrame = -1;
  let latestSequence = -1;
  let takenSequence = -1;
  let sequence = 0;
  let epoch = 0;
  let requested = false;
  let destroyed = false;
  // Count frame time even when captures are skipped
  let counted = 0;

  // Promises wait here until a capture. Unsubmitted copies return their readers here
  const waiting: Reader<T>[] = [];

  const copies = Array.from({ length: slots }, (): Copy<T> => ({
    epoch: 0,
    sequence: 0,
    frame: -1,
    requested: false,
    readers: [],
  }));

  const handBack = (copy: Copy<T>) => {
    for (let i = 0; i < copy.readers.length; i++) waiting.push(copy.readers[i]);
    copy.readers.length = 0;
  };

  const ring = staging<S, unknown>(root, {
    schema: source.dataType,
    slots,
    bytes,
    range,

    read(i, stored) {
      const copy = copies[i];
      if (copy.epoch !== epoch) return;

      const value = decode(stored);
      if (destroyed || copy.epoch !== epoch) return;
      if (copy.sequence > latestSequence) {
        latest = value;
        latestFrame = copy.frame;
        latestSequence = copy.sequence;
      }

      for (let k = 0; k < copy.readers.length; k++) copy.readers[k].resolve(value);
      copy.readers.length = 0;
    },

    failed(i, error) {
      const copy = copies[i];
      for (let k = 0; k < copy.readers.length; k++) copy.readers[k].reject(error);
      copy.readers.length = 0;
    },

    dropped(i) {
      const copy = copies[i];
      handBack(copy);
      // A copy from before a reset was not going to count, so there is nothing to recapture
      if (copy.epoch !== epoch) return;
      if (copy.requested) requested = true;
      rate?.reset();
    },
  });

  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    ring.destroy();
    const error = destroyedError();
    for (const reader of waiting) reader.reject(error);
    waiting.length = 0;
    for (const copy of copies) {
      for (const reader of copy.readers) reader.reject(error);
      copy.readers.length = 0;
    }
  };

  return {
    get latest() {
      return latest;
    },

    get latestFrame() {
      return latestFrame;
    },

    take() {
      if (latest === undefined || takenSequence === latestSequence) return undefined;
      takenSequence = latestSequence;
      return latest;
    },

    next() {
      if (destroyed) return Promise.reject(destroyedError());
      return new Promise<T>((resolve, reject) => {
        waiting.push({ resolve, reject });
      });
    },

    request() {
      requested = true;
    },

    capture(frame) {
      if (destroyed) return;
      const i = ring.take(frame);
      if (i < 0) return;

      const scheduled = rate !== undefined && rate.due(frame.time - counted);
      counted = frame.time;
      if (!scheduled && !requested && waiting.length === 0) return;

      const copy = copies[i];
      copy.epoch = epoch;
      copy.sequence = sequence++;
      copy.frame = frame.index;
      copy.requested = requested;
      requested = false;

      for (let k = 0; k < waiting.length; k++) copy.readers.push(waiting[k]);
      waiting.length = 0;

      ring.copy(frame, i, source);
    },

    reset() {
      epoch += 1;
      latest = undefined;
      latestFrame = -1;
      for (const copy of copies) handBack(copy);
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
