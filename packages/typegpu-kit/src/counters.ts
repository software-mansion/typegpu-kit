import {
  d,
  std,
  tgpu,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuRoot,
} from 'typegpu';
import { encoderOf, type Frame } from './frames.ts';
import type { Inject } from './inject.ts';
import { create as createReadback } from './readback.ts';

export interface Options {
  /** Selects accumulated counters or counters for individual frames. Default `'total'`. */
  readonly per?: 'total' | 'frame';
  /** Reads counters for `take` and `latest` this many times per second on average. Default 30. */
  readonly hz?: number;
  /** Sets the number of readback staging buffers. Default 3. */
  readonly staging?: number;
}

/** Maps each counter name to its count. */
export type Counts<K extends string> = { readonly [name in K]: number };

export interface Counters<K extends string> extends Disposable {
  /** Lists the counter names in the order they were given. */
  readonly names: readonly K[];
  /** Fills `add` with these counters, as its default does. */
  inject(): Inject;
  /**
   * Adds `amount` to a counter atomically in any shader, such as `contacts.add.stomps(1)`.
   */
  readonly add: { readonly [name in K]: (amount: number) => void };
  /** Stores one `u32` per counter, in the order of `names`. */
  readonly buffer: TgpuBuffer<d.WgslArray<d.Atomic<d.U32>>> & StorageFlag;
  /**
   * Reports the latest counts read back. In total mode, these are totals since the last `clear`.
   * In frame mode, they are counts from one captured frame. Undefined before the first read.
   * Reuses one object and updates it in place.
   */
  readonly latest: Counts<K> | undefined;
  /**
   * Copies counters when a read is due, then clears them in frame mode. Record after the passes
   * that add.
   */
  capture(frame: Frame): void;
  /**
   * Returns the increase in each counter since the last `take`, or the captured counts in frame
   * mode. Returns each read once, then undefined until the next read arrives. Reuses one object
   * and updates it in place.
   */
  take(): Counts<K> | undefined;
  /**
   * Resets all counters to 0, recorded into `target`, a frame or an encoder. Record before the
   * passes that add.
   */
  clear(target: Frame | TgpuCommandEncoder): void;
  /** Destroys the counters and readback. */
  destroy(): void;
}

/**
 * Keeps a counter for each named kind of event, and no details of one. Kernels add hits, kills,
 * pickups or other events, and the CPU receives the increases a frame or two later. Requires at
 * least one name, with no duplicates. Throws if names are invalid, `hz` is not finite and above 0, or
 * `staging` is not an integer above 0.
 *
 * @injects member `add`, which adds to its own counters in any pipeline
 * @binds 1 storage: `add`, the counters
 * @owns a buffer, a readback: the counters; the readback copies them at `hz`
 * @records `capture` after the passes that add; `clear` before them, whenever the counting starts
 * again; in frame mode, `capture` clears after copying.
 * @param names one counter each
 * @returns a set of counters you destroy
 */
export function create<const K extends string>(
  root: TgpuRoot,
  names: readonly K[],
  options: Options = {},
): Counters<K> {
  const { hz = 30, staging = 3, per = 'total' } = options;
  if (names.length === 0 || new Set(names).size !== names.length) {
    throw new RangeError(
      `counters.create: the names must be one or more, each once, got ${names.join(', ') || 'none'}.`,
    );
  }

  const countersBuffer = root
    .createBuffer(d.arrayOf(d.atomic(d.u32), names.length))
    .$usage('storage');

  // An accessor per set, filled with its own counters, so one kernel can add to several sets
  const counting = countersBuffer.as('mutable');
  const counts = tgpu.mutableAccessor(d.arrayOf(d.atomic(d.u32)), counting);
  const inject: Inject = (cfg) => cfg.with(counts, counting);

  const adder = (index: number) => {
    const add = (amount: number) => {
      'use gpu';
      std.atomicAdd(counts.$[index], d.u32(amount));
    };

    return add;
  };
  const add = Object.fromEntries(names.map((name, i) => [name, adder(i)])) as Counters<K>['add'];

  const result = createReadback(root, countersBuffer, { hz, staging });

  const seen = new Uint32Array(names.length);
  const gains = Object.fromEntries(names.map((name) => [name, 0])) as Record<K, number>;
  const totals = Object.fromEntries(names.map((name) => [name, 0])) as Record<K, number>;
  let cleared = -1;

  const destroy = () => {
    countersBuffer.destroy();
    result.destroy();
  };

  return {
    names,
    inject: () => inject,
    add,
    buffer: countersBuffer,

    get latest() {
      const read = result.latest;
      if (!read) return undefined;

      for (let i = 0; i < names.length; i++) totals[names[i]] = read[i];
      return totals;
    },

    capture(frame) {
      result.capture(frame);
      if (per === 'frame' && cleared !== frame.index) {
        countersBuffer.clear(frame.encoder);
        cleared = frame.index;
      }
    },

    take() {
      const latest = result.take();
      if (!latest) return undefined;

      // Counter differences must wrap like the u32 totals
      for (let i = 0; i < names.length; i++) {
        gains[names[i]] = per === 'frame' ? latest[i] : (latest[i] - seen[i]) >>> 0;
        seen[i] = latest[i];
      }

      return gains;
    },

    clear(target) {
      countersBuffer.clear(encoderOf(target));
      result.reset();
      seen.fill(0);
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
