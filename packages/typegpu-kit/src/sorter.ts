import type { d, StorageFlag, TgpuBuffer, TgpuRoot } from 'typegpu';
import type { AnyComputePass } from './passes.ts';
import { createRadixSorter } from './sort/index.ts';

/** This type is the schema of a key: `u32`, `i32` or `f32`. */
export type Key = d.U32 | d.I32 | d.F32;

type Storage<T extends d.AnyWgslData> = TgpuBuffer<d.WgslArray<T>> & StorageFlag;

export interface Options<K extends Key = Key, V extends d.AnyWgslData = d.AnyWgslData> {
  /**
   * Holds a payload, such as indices into your own data, that moves with its key. It has as many
   * elements as `keys`. With `out`, it requires `out.values`. Default no payload.
   */
  values?: Storage<V>;
  /**
   * Receives the sorted keys and payload, and leaves `keys` and `values` as they were. Default
   * sorting `keys` and `values` in place.
   */
  out?: { keys: Storage<K>; values?: Storage<V> };
  /** Puts the largest key first with `'descending'`. Default `'ascending'`. */
  direction?: 'ascending' | 'descending';
  /**
   * Clamps each key into `[min, max]`. Integer keys sort from `min`, in the bits `max - min`
   * needs; `f32` keys sort in `2^keyBits` even steps, where keys in one step keep their order.
   * Default the whole range of the key's type.
   */
  range?: [min: number, max: number];
  /**
   * Sorts by this many low bits of the key that `range` or `key` makes, eight a pass. Below 32 on
   * `i32` and `f32` keys requires `range` or `key`. Default what `range` needs, or 32.
   */
  keyBits?: number;
  /**
   * Maps a key to a `u32` whose order is the sort order, in place of the map for the key's type.
   * It is a `'use gpu'` function, and `range` cannot be set with it. Default the key's own order.
   * @gpu
   */
  key?: (key: number) => number;
}

/** This type sorts a key buffer, with its payload, each time it records. */
export interface Sorter {
  /** Records the sort into a compute pass. */
  sort(pass: AnyComputePass): void;
  /** Compiles the sort's pipelines. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Destroys the buffers the sort made and keeps `keys` and its payload. */
  destroy(): void;
  [Symbol.dispose](): void;
}

/**
 * Creates a stable radix sort of a buffer of keys, eight bits a pass, which carries a payload
 * along. Keys that compare equal keep their order, so a sort repeats exactly. Its cost follows the
 * buffer's length and `keyBits`, not the keys' order. An `f32` sort ranks -0 and +0 equal.
 *
 * @owns buffers, pipelines, bind groups: counts per 2 048 keys, a copy of the keys and payload to
 * sort through, and the scan of the counts
 * @records `sort` in a compute pass after whatever writes the keys
 * @param keys the buffer to sort, with `storage` usage
 * @returns a sorter you destroy
 */
export function create<K extends Key, V extends d.AnyWgslData = d.AnyWgslData>(
  root: TgpuRoot,
  keys: Storage<K>,
  options: Options<K, V> = {},
): Sorter {
  const radix = createRadixSorter(root, keys, options);
  const destroy = () => radix.destroy();
  return {
    sort: (pass) => radix.run({ pass }),
    initAsync: () => radix.initAsync(),
    destroy,
    [Symbol.dispose]: destroy,
  };
}
