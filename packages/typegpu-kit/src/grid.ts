/**
 * Sorts items into a neighbour grid each step. Kernels find neighbours by visiting nearby cells
 * instead of every item. Import it from `@typegpu/kit/grid`.
 *
 * @module
 * @unstable
 */

import { randf } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuComputePass,
  type TgpuComputePipeline,
  type TgpuRoot,
} from 'typegpu';
import * as backends from './backends.ts';
import * as pipelines from './pipelines.ts';
import { checkDispatch } from './dispatchLimit.ts';
import type { Inject } from './inject.ts';
import type { IndirectDispatcher } from './liveLists.ts';
import * as sorter from './sorter.ts';
import * as workgroup from './workgroup.ts';

/** Sets the threads per workgroup of the kernels `dispatch` runs, and of the grid's own. */
export const WORKGROUP = 64;

/** Counts the cells along each edge of a block, the unit `dispatchBlocks` and `holds` work in. */
export const BLOCK = 4;

/**
 * Records the home cell of a walk and its direction along each axis, either 1 or -1. An x direction
 * of 0 reads each range in sorted order.
 */
export const Walk = d.struct({ home: d.vec3i, turn: d.vec3i });
export type Walk = d.Infer<typeof Walk>;

/**
 * Describes three adjacent cells along x as `count` items beginning at `first` in sorted order. A
 * walk starts at one end of the middle cell, given by `start`. It visits items forwards or `back`,
 * wrapping around the range.
 */
export const Range = d.struct({ first: d.u32, count: d.u32, start: d.u32, back: d.u32 });
export type Range = d.Infer<typeof Range>;

/** Counts the ranges in a walk, the home range and its eight neighbours. */
export const RANGES = 9;

/**
 * Names the items near a point as ranges of indices, and maps each index to a slot. A grid's
 * `candidates()` gives one, and a lighting's `points.candidates` takes one, a grid's or your
 * own. The ranges must hold every item that reaches the point. A slot that appears twice counts
 * twice.
 */
export interface Candidates {
  /** Gives the number of ranges per point, an integer of at least 1. Shaders fix it as they compile. */
  readonly ranges: number;
  /**
   * Returns range `r` around `position` from a `'use gpu'` function: `count` indices from
   * `first`. Attach the fills of your own with `pipe`; a grid's has them piped in.
   */
  readonly range: (
    position: d.v3f,
    r: number,
  ) => { readonly first: number; readonly count: number };
  /**
   * Maps an index to the slot it names, from a `'use gpu'` function. Attach the fills of your
   * own with `pipe`; a grid's has them piped in.
   */
  readonly slot: (index: number) => number;
}

/** Describes what a grid asks of its binning. */
export interface BinningOptions {
  /** Sets the number of slots to sort, including empty slots. */
  readonly capacity: number;
  /** Sets the number of keys, a whole number of lines. An empty slot's key equals it. */
  readonly keys: number;
  /** Sets the keys in a line, one per cell along x. Line `l` holds keys `l * line` onwards. */
  readonly line: number;
  /**
   * Returns the key of slot `i`, below `keys`, or `keys` for an empty slot. It carries the grid's
   * fills.
   * @gpu
   */
  readonly key: (i: number) => number;
}

/**
 * Orders a grid's slots by key. After `sort`, `keys` holds every slot's key in sorted order and
 * `slots` the slot it came from. The keys of one line sit together in ascending order, and empty
 * slots come last. Lines may come in any order.
 */
export interface Binning extends Disposable {
  /** Stores the keys in sorted order, one per slot. */
  readonly keys: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  /** Stores the slot of each sorted key. */
  readonly slots: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  /**
   * Records the sort into a compute pass. Slots at `span` and above hold no item, so a binning may
   * skip them.
   */
  sort(pass: TgpuComputePass, span: number): void;
  /** Compiles the binning pipelines. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Destroys the buffers the binning made. A grid destroys its own binning. */
  destroy(): void;
}

const NONE = 0xffffffff;

const sortedIndex = (of: Range, j: number) => {
  'use gpu';
  // start < count and j < count, so one subtraction wraps it
  let k = of.start + d.u32(j);
  if (of.back === 1) k = of.start + of.count - d.u32(j);
  if (k >= of.count) k -= of.count;
  return of.first + k;
};

const Tally = (keys: number, lines: number) =>
  d
    .struct({
      /** Lines listed this sort. */
      listed: d.atomic(d.u32),
      /** Items reserved this sort. */
      total: d.atomic(d.u32),
      /** Items the last sort placed, whose keys `scatter` empties when fewer remain. */
      previous: d.u32,
      /** Sorts so far. A line stamped with the coming sort is listed. */
      step: d.u32,
      stamps: d.arrayOf(d.atomic(d.u32), lines),
      counts: d.arrayOf(d.atomic(d.u32), keys),
    })
    .$name('Tally');

/**
 * Creates a counting sort for a grid: it counts the items of each cell with atomics, scans each
 * line of cells holding items in one workgroup and reserves one range of sorted indices per line.
 * Its cost follows the item count and the lines holding items. Items of one cell land in any
 * order, and lines in any order.
 *
 * @param options.key returns each slot's key
 * @returns a binning the grid owns and destroys
 */
export function counting(root: TgpuRoot, options: BinningOptions): Binning {
  const { capacity, keys: nowhere, line, key } = options;
  const lines = nowhere / line;

  const tally = root.createBuffer(Tally(nowhere, lines)).$usage('storage');
  const starts = root.createBuffer(d.arrayOf(d.u32, nowhere)).$usage('storage');
  const list = root.createBuffer(d.arrayOf(d.u32, lines)).$usage('storage');
  const args = root.createBuffer(d.arrayOf(d.u32, 3)).$usage('storage', 'indirect');
  const ranked = root.createBuffer(d.arrayOf(d.vec2u, capacity)).$usage('storage');
  const keys = root
    .createBuffer(
      d.arrayOf(d.u32, capacity),
      Array.from({ length: capacity }, () => nowhere),
    )
    .$usage('storage');
  const slots = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');

  const tallying = tally.as('mutable');
  const startsRW = starts.as('mutable');
  const startsIn = starts.as('readonly');
  const listRW = list.as('mutable');
  const listIn = list.as('readonly');
  const argsRW = args.as('mutable');
  const rankedRW = ranked.as('mutable');
  const rankedIn = ranked.as('readonly');
  const keysRW = keys.as('mutable');
  const slotsRW = slots.as('mutable');

  // Ranks each item in its cell, and lists the line on its first item
  const count = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= capacity) return;

    const k = d.u32(key(i));
    if (k >= nowhere) {
      rankedRW.$[i] = d.vec2u(nowhere, 0);
      return;
    }
    rankedRW.$[i] = d.vec2u(k, std.atomicAdd(tallying.$.counts[k], 1));

    const at = std.intdiv(k, d.u32(line));
    const stamp = tallying.$.step + 1;
    if (std.atomicLoad(tallying.$.stamps[at]) !== stamp) {
      if (std.atomicExchange(tallying.$.stamps[at], stamp) !== stamp) {
        listRW.$[std.atomicAdd(tallying.$.listed, 1)] = at;
      }
    }
  });

  const size = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    argsRW.$[0] = std.atomicLoad(tallying.$.listed);
    argsRW.$[1] = 1;
    argsRW.$[2] = 1;
    std.atomicStore(tallying.$.listed, 0);
    tallying.$.previous = std.atomicLoad(tallying.$.total);
    std.atomicStore(tallying.$.total, 0);
    tallying.$.step += 1;
  });

  const prefix = workgroup.scan(workgroup.sum(d.u32), WORKGROUP);
  const base = tgpu.workgroupVar(d.u32);
  const per = Math.ceil(line / WORKGROUP);

  // One workgroup per listed line reserves one range for it. Each thread takes consecutive cells of
  // the line and places them after the threads before it. Counts go back to zero here, so no pass
  // clears them
  const reserve = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
  })(({ wid, lid }) => {
    'use gpu';
    const from = listIn.$[wid.x] * line;
    const own = lid * per;
    let mine = d.u32(0);
    for (let c = d.u32(0); c < per; c++) {
      if (own + c < line) mine += std.atomicLoad(tallying.$.counts[from + own + c]);
    }
    const before = prefix(mine, lid);
    if (lid === WORKGROUP - 1) base.$ = std.atomicAdd(tallying.$.total, before + mine);
    let at = std.workgroupUniformLoad(base.$) + before;
    for (let c = d.u32(0); c < per; c++) {
      if (own + c < line) {
        const cell = from + own + c;
        startsRW.$[cell] = at;
        at += std.atomicExchange(tallying.$.counts[cell], 0);
      }
    }
  });

  // Places each item, and empties the keys the last sort placed beyond this one's
  const scatter = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= capacity) return;

    const total = std.atomicLoad(tallying.$.total);
    if (i >= total && i < tallying.$.previous) keysRW.$[i] = nowhere;

    const placed = rankedIn.$[i];
    if (placed.x >= nowhere) return;
    const to = startsIn.$[placed.x] + placed.y;
    keysRW.$[to] = placed.x;
    slotsRW.$[to] = i;
  });

  const counter = root.createComputePipeline({ compute: count });
  const sizer = root.createComputePipeline({ compute: size });
  const reserver = root.createComputePipeline({ compute: reserve });
  const scatterer = root.createComputePipeline({ compute: scatter });
  // Slots below the last sort's span still hold its items, so a shrinking span ranks them empty
  // and empties the keys they held
  let lastSpan = capacity;

  const destroy = () => {
    for (const buffer of [tally, starts, list, args, ranked, keys, slots]) buffer.destroy();
  };

  return {
    keys,
    slots,

    sort(pass, span) {
      const groups = Math.ceil(Math.max(span, lastSpan) / WORKGROUP);
      lastSpan = span;
      pass.setPipeline(counter);
      pass.dispatchWorkgroups(groups);
      pass.setPipeline(sizer);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(reserver);
      pass.dispatchWorkgroupsIndirect(args);
      pass.setPipeline(scatterer);
      pass.dispatchWorkgroups(groups);
    },

    initAsync: () => pipelines.compile([counter, sizer, reserver, scatterer]),
    destroy,
    [Symbol.dispose]: destroy,
  };
}

/**
 * Creates a radix sort for a grid: it sorts every slot by key, eight bits a pass. Its cost follows
 * the capacity and the number of key bits, not how items crowd into cells or the span, and equal
 * keys keep their slot order, so a sort repeats exactly.
 *
 * @param options.key returns each slot's key
 * @returns a binning the grid owns and destroys
 */
export function radix(root: TgpuRoot, options: BinningOptions): Binning {
  const { capacity, keys: nowhere, key } = options;

  const unsorted = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  const order = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  const keys = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  const slots = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  // Sorting into other buffers saves the copy back after an odd number of passes
  const byKey = sorter.create(root, unsorted, {
    range: [0, nowhere],
    values: order,
    out: { keys, values: slots },
  });

  const keying = unsorted.as('mutable');
  const ordering = order.as('mutable');
  const keyKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= capacity) return;
    keying.$[i] = d.u32(key(i));
    ordering.$[i] = i;
  });
  const keyPipeline = root.createComputePipeline({ compute: keyKernel });
  const groups = Math.ceil(capacity / WORKGROUP);

  const destroy = () => {
    byKey.destroy();
    for (const buffer of [unsorted, order, keys, slots]) buffer.destroy();
  };

  return {
    keys,
    slots,

    sort(pass) {
      pass.setPipeline(keyPipeline);
      pass.dispatchWorkgroups(groups);
      byKey.sort(pass);
    },

    initAsync: () => pipelines.compile([keyPipeline, byKey]),
    destroy,
    [Symbol.dispose]: destroy,
  };
}

/** Steps of the bitonic network one pass of `bitonic` takes, comparing 2^STEPS elements. */
const STEPS = 4;

const Stage = d.struct({ k: d.u32, j: d.u32, steps: d.u32 });
const Gathered = d.arrayOf(d.vec2u, 2 ** STEPS);

/** Whether pair `a`, a key and a slot, sorts before pair `b`: by key, then by slot. */
const precedes = (a: d.v2u, b: d.v2u) => {
  'use gpu';
  return a.x < b.x || (a.x === b.x && a.y < b.y);
};

/**
 * Creates a bitonic sort for a grid, for backends where compute runs on the CPU: every pass writes
 * its own elements and reads others, with no atomics or workgroup memory, so a WebGL 2 device runs
 * each pass on the GPU. It sorts the slots below the span, rounded up to a power of two, by key
 * and then slot, so a sort repeats exactly.
 */
function bitonic(root: TgpuRoot, options: BinningOptions): Binning {
  const { capacity, keys: nowhere, key } = options;
  const n = 2 ** Math.ceil(Math.log2(Math.max(capacity, 2)));

  const pairs = [0, 1].map(() => root.createBuffer(d.arrayOf(d.vec2u, n)).$usage('storage'));
  const keys = root
    .createBuffer(
      d.arrayOf(d.u32, capacity),
      Array.from({ length: capacity }, () => nowhere),
    )
    .$usage('storage');
  const slots = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');

  const layout = tgpu.bindGroupLayout({
    from: { storage: d.arrayOf(d.vec2u, n), access: 'readonly' },
    into: { storage: d.arrayOf(d.vec2u, n), access: 'mutable' },
    stage: { uniform: Stage },
  });
  const seeding = pairs[0]!.as('mutable');
  const keysRW = keys.as('mutable');
  const slotsRW = slots.as('mutable');

  const seed = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= n) return;
    let k = d.u32(nowhere);
    if (i < capacity) k = d.u32(key(i));
    seeding.$[i] = d.vec2u(k, i);
  });

  // Each thread gathers the 2^steps elements its steps mix, runs those steps and keeps its own
  const merge = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= n) return;
    const k = layout.$.stage.k;
    const j = layout.$.stage.j;
    const steps = layout.$.stage.steps;
    const low = j >>> (steps - 1);
    const base = i - (i & ((j << 1) - low));
    const count = d.u32(1) << steps;
    const group = Gathered();
    for (let e = d.u32(0); e < count; e++) group[e] = d.vec2u(layout.$.from[base + e * low]);
    const up = (base & k) === 0;
    for (let s = d.u32(1); s <= steps; s++) {
      const apart = count >>> s;
      for (let e = d.u32(0); e < count; e++) {
        if ((e & apart) !== 0) continue;
        const a = d.vec2u(group[e]);
        const b = d.vec2u(group[e | apart]);
        if (precedes(b, a) === up) {
          group[e] = d.vec2u(b);
          group[e | apart] = d.vec2u(a);
        }
      }
    }
    layout.$.into[i] = d.vec2u(group[(i - base) / low]);
  });

  const unpack = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= capacity) return;
    // `stage.k` holds the size this sort covered; slots past it hold no item
    let pair = d.vec2u(nowhere, i);
    if (i < layout.$.stage.k) pair = d.vec2u(layout.$.from[i]);
    keysRW.$[i] = pair.x;
    slotsRW.$[i] = pair.y;
  });

  const seeder = root.createComputePipeline({ compute: seed });
  const merger = root.createComputePipeline({ compute: merge });
  const unpacker = root.createComputePipeline({ compute: unpack });

  // After `s` stages, the sorted pairs are in `pairs[s % 2]`
  const bind = (s: number, stage: d.Infer<typeof Stage>) =>
    root.createBindGroup(layout, {
      from: pairs[s % 2]!,
      into: pairs[(s + 1) % 2]!,
      stage: root.createUniform(Stage, stage).buffer,
    });
  // The stages of a sort of `size` are the first stages of every larger one, so one list serves
  // every span
  const stages: ReturnType<typeof bind>[] = [];
  const ends = new Map<number, number>();
  for (let k = 2; k <= n; k <<= 1) {
    for (let j = k >> 1; j > 0;) {
      const steps = Math.min(STEPS, Math.log2(j) + 1);
      stages.push(bind(stages.length, { k, j, steps }));
      j >>= steps;
    }
    ends.set(k, stages.length);
  }
  const unpacking = new Map(
    [...ends].map(([size, count]) => [size, bind(count, { k: size, j: 0, steps: 0 })]),
  );

  const destroy = () => {
    for (const buffer of [...pairs, keys, slots]) buffer.destroy();
  };

  return {
    keys,
    slots,

    sort(pass, span) {
      const size = Math.max(2, 2 ** Math.ceil(Math.log2(Math.max(span, 1))));
      pass.setPipeline(seeder);
      pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP));
      pass.setPipeline(merger);
      const count = ends.get(size) ?? 0;
      for (let s = 0; s < count; s++) {
        pass.setBindGroup(layout, stages[s]!);
        pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP));
      }
      pass.setPipeline(unpacker);
      pass.setBindGroup(layout, unpacking.get(size)!);
      pass.dispatchWorkgroups(Math.ceil(capacity / WORKGROUP));
    },

    initAsync: () => pipelines.compile([seeder, merger, unpacker]),
    destroy,
    [Symbol.dispose]: destroy,
  };
}

const Box = d.struct({ low: d.vec3f });

const noItems = (): never => {
  throw new Error('This grid keeps no items to read. Create it with `item` and `itemOf`.');
};

/**
 * Takes `span` only beside `alive`, as a table's `live()` gives them, and `itemOf` only beside
 * `item`.
 */
export type Options<T extends d.AnyWgslData> = GridOptions &
  (
    | {
        /**
         * Defines the data a walk reads for each item. `sort` copies it so kernels can move the
         * original items while walking. Default none: the grid keeps no copies, and a walk reads
         * slots with `slot`.
         */
        readonly item: T;
        /**
         * Returns what a walk reads of item `i`, as a value of `item`. Needs `item`. Default none.
         * @gpu
         */
        readonly itemOf: (i: number) => d.Infer<T>;
      }
    | { readonly item?: never; readonly itemOf?: never }
  ) &
  (
    | {
        /** Tests whether slot `i` holds an item. Default every slot. @gpu */
        readonly alive: (i: number) => boolean;
        /**
         * Returns the highest live slot plus one, read when `sort` records, such as a table's from
         * `live()`. Skips the slots above it, so `alive` must be false there. Needs `alive`.
         * Default `capacity`.
         * @cpu
         */
        readonly span?: () => number;
      }
    | { readonly alive?: never; readonly span?: never }
  );

interface GridOptions {
  /** Sets the number of slots, including empty slots. Indices range from `0` to `capacity - 1`. */
  readonly capacity: number;
  /**
   * Sets the initial lower corner of the grid box. `move` changes it. Items outside the box belong
   * to the nearest cell.
   */
  readonly low: d.v3f;
  /**
   * Sets the box extent along x, y and z for the lifetime of the grid. A side within a thousandth
   * of a cell of a whole number of cells takes that number.
   */
  readonly size: d.v3f;
  /** Sets the cell side length. A walk finds every item within this distance. */
  readonly sight: number;
  /** Returns the position of item `i`. @gpu */
  readonly position: (i: number) => d.v3f;
  /**
   * Orders the items by cell each step: `counting` or `radix`. Default `counting`, or where
   * compute runs on the CPU, a bitonic sort whose passes a WebGL 2 device runs on the GPU. @cpu
   */
  readonly binning?: (root: TgpuRoot, options: BinningOptions) => Binning;
  /**
   * Widens `dispatchBlocks` to every block within this many blocks of one holding items, for work
   * that reaches past the items, such as a surface around them. A whole number. Default 0.
   */
  readonly margin?: number;
}

export interface Grid<T extends d.AnyWgslData> extends Disposable {
  /** Reports the number of slots the grid sorts. */
  readonly capacity: number;
  /** Starts a walk at `point` with random directions from `randf`. Seed `randf` first. */
  readonly walk: (point: d.v3f) => Walk;
  /**
   * Starts a walk at `point` with fixed directions and no random samples, reading each range in
   * sorted order. Use it when a shader visits every item in the nearby ranges, such as for point
   * lights in a fragment shader or a fluid's density. Neighbouring items then read nearby memory.
   */
  readonly around: (point: d.v3f) => Walk;
  /**
   * Returns range `r` of `RANGES`. Range 0 is the home range. Other ranges offset y and z by 0, 1
   * or -1 cells, using the directions chosen by the walk. Ranges outside the grid are empty.
   */
  readonly range: (from: Walk, r: number) => Range;
  /** Returns the sorted index of item `j` in a range. `j` must be below the range's `count`. */
  readonly index: (of: Range, j: number) => number;
  /**
   * Returns item `j` in a range as a value of `item`. `j` must be below the range's `count`. A
   * grid without `item` has none to read.
   */
  readonly neighbour: (of: Range, j: number) => d.Infer<T>;
  /**
   * Returns the original slot of item `j` in a range, whose data came from `itemOf(i)`. Use it to
   * act on that item or skip the walker itself. Binds a third storage buffer in addition to the
   * two used by a walk.
   */
  readonly slot: (of: Range, j: number) => number;
  /** Returns the original slot of the item at sorted index `index`. */
  readonly slotAt: (index: number) => number;
  /** Returns the item at sorted index `index` as a value of `item`. */
  readonly neighbourAt: (index: number) => d.Infer<T>;
  /** Returns the number of items the last sort placed, at sorted indices `0` to `count() - 1`. */
  readonly count: () => number;
  /**
   * Tests whether `block`, counted in blocks along x, y and z, holds items after the last sort.
   * Blocks outside the grid hold none.
   */
  readonly holds: (block: d.v3i) => boolean;
  /**
   * Returns the block that workgroup `w` of `dispatchBlocks` covers, counted in blocks. With a
   * `margin`, `holds` tells the blocks holding items from those around them.
   */
  readonly block: (w: number) => d.v3i;
  /**
   * Returns nearby ranges, for a lighting's
   * `points.candidates` when the items are lights and `sight` covers every light range.
   */
  candidates(): Candidates;
  /** Fills the grid's accessors with its own buffers, as their defaults already do. */
  inject(): Inject;
  /** Compiles the grid pipelines and its binning. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Sorts live items into cells and copies their `itemOf` data, if any. Record before a walk. */
  sort(pass: TgpuComputePass): void;
  /**
   * Dispatches the pipeline set on the compute pass over the sorted items after `sort`, in
   * workgroups of `WORKGROUP`, enough for `count()` and at least one. Each thread's global index is
   * a sorted index; threads at or beyond `count()` must return.
   */
  dispatch(pass: IndirectDispatcher): void;
  /**
   * Dispatches one workgroup of the pipeline set on the compute pass per block holding items after
   * `sort`, or within `margin` blocks of one, in any order. Workgroup `w` finds its block with
   * `block(w)`.
   */
  dispatchBlocks(pass: IndirectDispatcher): void;
  /**
   * Moves the box origin to `low` without changing size or sight. The next submission sees this
   * origin throughout, so sort again before the next walk.
   */
  move(low: d.v3f): void;
  /** Destroys its buffers and binning. */
  destroy(): void;
}

/**
 * Creates a grid that sorts items by cell so kernels can find nearby items. Walks visit the home
 * cell first and choose directions at random, for boids, crowds or collisions. Throws unless
 * `sight` is above 0, and when the capacity, 64 to a workgroup, needs more than the device's
 * `maxComputeWorkgroupsPerDimension` workgroups.
 *
 * @injects member `walk`, `around`, `range`, `neighbour`, `slot`, `slotAt`, `neighbourAt`,
 * `count`, `holds` and `block`, which read its own buffers in any pipeline
 * @binds 1 uniform: `walk` or `around`, the box
 * @binds 2 storage: `range` and `neighbour` or `neighbourAt`, each cell's sorted range and each item's copy
 * @binds 1 storage: `slot` or `slotAt`, the sorted slots
 * @binds 1 storage: `count`, the sorted count
 * @binds 3 storage: `holds` and `block`, which blocks hold items, the sort count and their list
 * @owns buffers, a uniform, pipelines, bind groups, a sorter with `radix`: each cell's sorted
 * range, the blocks holding items, with `item` each item's copy in sorted order, the dispatches,
 * the box, the binning's resources
 * @records `sort` in a compute pass each step, before the kernels that walk
 * @param options.sight the cell side length. A walk finds every item within this distance
 * @param options.alive tests whether a slot holds an item. It, `position` and `itemOf` need fills for
 * the resources they read. Spread a table's `rows.live()` into the options to include its
 * live test, capacity and span
 * @returns a grid you sort each step and destroy
 */
export function create<T extends d.AnyWgslData = never>(
  root: TgpuRoot,
  options: Options<T>,
): Grid<T> {
  const { capacity, low, size, sight, item, alive, span, position, itemOf } = options;
  // Where compute runs on the CPU, the grid sorts and lists with passes that write their own
  // elements or scatter whole ones, which a WebGL 2 device runs on the GPU
  const alone = backends.of(root).compute === 'cpu';
  const { binning = alone ? bitonic : counting, margin = 0 } = options;
  if (!(sight > 0)) throw new RangeError(`grid.create: \`sight\` must be above 0, got ${sight}.`);
  if (!(Number.isInteger(margin) && margin >= 0)) {
    throw new RangeError(
      `grid.create: \`margin\` must be a whole number of blocks, got ${margin}.`,
    );
  }

  // `size` arrives as f32, so 25.6 / 0.2 lands just above 128
  const across = (side: number) => Math.max(1, Math.ceil(side / sight - 1e-3));
  const dims = d.vec3u(across(size.x), across(size.y), across(size.z));
  const blocked = d.vec3u(
    Math.ceil(dims.x / BLOCK),
    Math.ceil(dims.y / BLOCK),
    Math.ceil(dims.z / BLOCK),
  );
  const blocks = blocked.x * blocked.y * blocked.z;
  const nowhere = dims.x * dims.y * dims.z;
  checkDispatch(root, 'grid.create', 'capacity', capacity, WORKGROUP);

  let sorted = false;
  const binned = () => {
    if (!sorted) throw new Error('The grid was never sorted. Record sort(pass) before dispatch.');
  };
  const box = root.createUniform(Box, { low });
  const boxed = tgpu.accessor(Box, box);

  const cellOf = (point: d.v3f) => {
    'use gpu';
    const cell = std.floor((point - boxed.$.low) / sight);
    return d.vec3i(std.clamp(cell, d.vec3f(), d.vec3f(dims) - 1));
  };

  // Keys run along x, then y, then z, so a row of cells along x is one line
  const keyOf = (cell: d.v3i) => {
    'use gpu';
    const at = d.vec3u(cell);
    return at.x + dims.x * (at.y + dims.y * at.z);
  };

  const blockOf = (key: number) => {
    'use gpu';
    const x = key % dims.x;
    const y = std.intdiv(key, d.u32(dims.x)) % dims.y;
    const z = std.intdiv(key, d.u32(dims.x * dims.y));
    return (x >>> 2) + blocked.x * ((y >>> 2) + blocked.y * (z >>> 2));
  };

  const blockAt = (index: number) => {
    'use gpu';
    return d.vec3i(
      d.i32(index % blocked.x),
      d.i32(std.intdiv(index, d.u32(blocked.x)) % blocked.y),
      d.i32(std.intdiv(index, d.u32(blocked.x * blocked.y))),
    );
  };

  const key = (i: number) => {
    'use gpu';
    if (alive !== undefined && !alive(i)) return d.u32(nowhere);
    return keyOf(cellOf(position(i)));
  };

  const bins = binning(root, { capacity, keys: nowhere, line: dims.x, key });

  // Each cell's range: its first sorted index and one past its last, (0, 0) when empty. The first
  // item's thread writes the start and a later pass the end, as two threads writing parts of one
  // vector race. Each sort empties the last sort's before writing its own, so no pass covers every
  // cell
  const ranges = root.createBuffer(d.arrayOf(d.vec2u, nowhere)).$usage('storage');
  // Which blocks hold items, as the number of the last sort that found them: an atomic claim that
  // lists each once, and a word shaders read. Stamps go stale by themselves, so no pass clears them
  const claims = root.createBuffer(d.arrayOf(d.atomic(d.u32), blocks)).$usage('storage');
  const held = root.createBuffer(d.arrayOf(d.u32, blocks)).$usage('storage');
  const list = root.createBuffer(d.arrayOf(d.u32, blocks)).$usage('storage');
  // The dispatch over listed blocks, then the number of sorts so far
  const blockArgs = root.createBuffer(d.arrayOf(d.atomic(d.u32), 4)).$usage('storage', 'indirect');
  // The dispatch over sorted items, their count, and the number of the sort that placed them. The
  // dispatch is at least one workgroup, so the next sort's clear always runs, and no block holds
  // items before the first sort
  const header = root
    .createBuffer(d.arrayOf(d.u32, 5), [1, 1, 1, 0, NONE])
    .$usage('storage', 'indirect');
  // With a margin, the blocks around those holding items, claimed and listed the same way
  const nearClaims =
    margin > 0
      ? root.createBuffer(d.arrayOf(d.atomic(d.u32), blocks)).$usage('storage')
      : undefined;
  const nearList =
    margin > 0 ? root.createBuffer(d.arrayOf(d.u32, blocks)).$usage('storage') : undefined;
  const nearArgs =
    margin > 0
      ? root.createBuffer(d.arrayOf(d.atomic(d.u32), 3), [0, 1, 1]).$usage('storage', 'indirect')
      : undefined;
  // TypeGPU cannot check a caller-selected schema here, so expose the copied data as `T` only at
  // the return. Without `item`, nothing reads copies, so the grid keeps none
  const shown: d.AnyWgslData = item ?? d.u32;
  const seen = item && root.createBuffer(d.arrayOf(shown, capacity)).$usage('storage');

  const destroy = () => {
    bins.destroy();
    lone?.destroy();
    for (const buffer of [ranges, claims, held, list, blockArgs, header]) buffer.destroy();
    seen?.destroy();
    nearClaims?.destroy();
    nearList?.destroy();
    nearArgs?.destroy();
  };

  const sortedKeys = bins.keys.as('readonly');
  const sortedSlots = bins.slots.as('readonly');
  const ranging = ranges.as('mutable');
  const claiming = claims.as('mutable');
  const holding = held.as('mutable');
  const listing = list.as('mutable');
  const blockArging = blockArgs.as('mutable');
  const heading = header.as('mutable');
  const nearClaiming = nearClaims?.as('mutable');
  const nearListing = nearList?.as('mutable');
  const nearArging = nearArgs?.as('mutable');
  const listIn = list.as('readonly');
  const headerIn = header.as('readonly');
  const copying = seen?.as('mutable');

  // Accessors per instance, filled with its own buffers, so one kernel can walk several grids
  const cellRanges = tgpu.accessor(d.arrayOf(d.vec2u), ranges.as('readonly'));
  const neighbours = tgpu.accessor(d.arrayOf(shown), seen?.as('readonly') ?? noItems);
  const neighbourSlots = tgpu.accessor(d.arrayOf(d.u32), sortedSlots);
  const counted = tgpu.accessor(d.arrayOf(d.u32), headerIn);
  const heldBlocks = tgpu.accessor(d.arrayOf(d.u32), held.as('readonly'));
  const listed = tgpu.accessor(d.arrayOf(d.u32), (nearList ?? list).as('readonly'));
  const fill: Inject = (cfg) =>
    cfg
      .with(boxed, box)
      .with(cellRanges, ranges.as('readonly'))
      .with(neighbours, seen?.as('readonly') ?? noItems)
      .with(neighbourSlots, sortedSlots)
      .with(counted, headerIn)
      .with(heldBlocks, held.as('readonly'))
      .with(listed, (nearList ?? list).as('readonly'));

  // Empties the ranges the last sort wrote, before the binning overwrites its keys
  const clearKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const k = gid.x;
    if (k === 0) {
      std.atomicStore(blockArging.$[0], 0);
      std.atomicAdd(blockArging.$[3], 1);
      if (nearArging !== undefined) std.atomicStore(nearArging.$[0], 0);
    }
    if (k >= headerIn.$[3]) return;

    const cell = sortedKeys.$[k];
    let before = d.u32(NONE);
    if (k > 0) before = sortedKeys.$[k - 1];
    if (before !== cell) ranging.$[cell] = d.vec2u();
  });

  // Writes where cell ranges open, the blocks and the count from where sorted keys change
  const boundKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const k = gid.x;
    if (k >= capacity) return;

    const cell = sortedKeys.$[k];
    let before = d.u32(NONE);
    if (k > 0) before = sortedKeys.$[k - 1];

    let end = d.u32(NONE);
    if (cell === nowhere && before !== nowhere) end = k;
    if (cell !== nowhere && k + 1 === capacity) end = capacity;
    if (end !== NONE) {
      heading.$[0] = std.max((end + WORKGROUP - 1) >>> 6, 1);
      heading.$[1] = 1;
      heading.$[2] = 1;
      heading.$[3] = end;
      heading.$[4] = std.atomicLoad(blockArging.$[3]);
      std.atomicStore(blockArging.$[1], 1);
      std.atomicStore(blockArging.$[2], 1);
    }
    if (cell === nowhere || before === cell) return;
    ranging.$[cell] = d.vec2u(k, k);
    // The first cell of a block to claim it for this sort lists it
    const block = blockOf(cell);
    const stamp = std.atomicLoad(blockArging.$[3]);
    if (
      std.atomicLoad(claiming.$[block]) !== stamp &&
      std.atomicExchange(claiming.$[block], stamp) !== stamp
    ) {
      holding.$[block] = stamp;
      listing.$[std.atomicAdd(blockArging.$[0], 1)] = block;
    }
  });

  // Copies each item, if the grid keeps copies, and closes each cell's range at its last item
  const copyKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const k = gid.x;
    const count = headerIn.$[3];
    if (k >= count) return;
    if (copying !== undefined && itemOf !== undefined) copying.$[k] = itemOf(sortedSlots.$[k]);
    const cell = sortedKeys.$[k];
    if (k + 1 === count || sortedKeys.$[k + 1] !== cell) {
      ranging.$[cell] = d.vec2u(ranging.$[cell].x, k + 1);
    }
  });

  // One workgroup per block holding items lists the blocks within the margin, once each
  const reach = 2 * margin + 1;
  const nearby = reach * reach * reach;
  let nearPipeline: TgpuComputePipeline | undefined;
  if (nearClaiming && nearListing && nearArging) {
    const nearKernel = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
    })(({ wid, lid }) => {
      'use gpu';
      const home = blockAt(listIn.$[wid.x]);
      const stamp = headerIn.$[4]; // this sort's number, as the bound kernel wrote it
      for (let n = d.u32(lid); n < nearby; n += WORKGROUP) {
        const step = d.vec3u(
          n % reach,
          std.intdiv(n, d.u32(reach)) % reach,
          std.intdiv(n, d.u32(reach * reach)),
        );
        const o = home + d.vec3i(step) - margin;
        if (std.any(std.lt(o, d.vec3i())) || std.any(std.ge(o, d.vec3i(blocked)))) continue;
        const index = d.u32(o.x) + blocked.x * (d.u32(o.y) + blocked.y * d.u32(o.z));
        if (
          std.atomicLoad(nearClaiming.$[index]) !== stamp &&
          std.atomicExchange(nearClaiming.$[index], stamp) !== stamp
        ) {
          nearListing.$[std.atomicAdd(nearArging.$[0], 1)] = index;
        }
      }
    });

    nearPipeline = root.createComputePipeline({ compute: nearKernel });
  }
  const clearPipeline = root.createComputePipeline({ compute: clearKernel });
  const boundPipeline = root.createComputePipeline({ compute: boundKernel });
  const copyPipeline = root.createComputePipeline({ compute: copyKernel });

  const lone = alone ? alongside() : undefined;
  // The rows the last sort covered, whose ranges the next one empties
  let lastRows = 0;
  function alongside() {
    const GROUP = 128;
    const groups = Math.ceil(blocks / GROUP);
    const flags = root.createBuffer(d.arrayOf(d.u32, blocks)).$usage('storage');
    const totals = root.createBuffer(d.arrayOf(d.u32, groups)).$usage('storage');
    const listArgs = root
      .createBuffer(d.arrayOf(d.u32, 3), [0, 1, 1])
      .$usage('storage', 'indirect');
    const flagging = flags.as('mutable');
    const flagsIn = flags.as('readonly');
    const totalling = totals.as('mutable');
    const totalsIn = totals.as('readonly');
    const listArging = listArgs.as('mutable');
    const listInto = (nearList ?? list).as('mutable');
    const heldIn = held.as('readonly');

    // Empties the ranges the last sort wrote, as `clearKernel` does, without the counters
    const clear = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const k = gid.x;
      if (k >= headerIn.$[3]) return;
      const cell = sortedKeys.$[k];
      let before = d.u32(NONE);
      if (k > 0) before = sortedKeys.$[k - 1];
      if (before !== cell) ranging.$[cell] = d.vec2u();
    });

    // Counts the sorted items and numbers the sort, from one thread
    const tally = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      let first = d.u32(0);
      let past = d.u32(capacity);
      while (first < past) {
        const middle = (first + past) >>> 1;
        if (sortedKeys.$[middle] < nowhere) first = middle + 1;
        else past = middle;
      }
      // The header starts at NONE, and no sort is numbered 0, which blocks start at
      let stamp = heading.$[4] + 1;
      if (stamp === 0) stamp = 1;
      heading.$[0] = std.max((first + WORKGROUP - 1) >>> 6, 1);
      heading.$[1] = 1;
      heading.$[2] = 1;
      heading.$[3] = first;
      heading.$[4] = stamp;
    });

    // The first item of each cell writes its whole range and stamps its block
    const bound = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const k = gid.x;
      const count = headerIn.$[3];
      if (k >= count) return;
      const cell = sortedKeys.$[k];
      if (k > 0 && sortedKeys.$[k - 1] === cell) return;
      let end = k + 1;
      while (end < count && sortedKeys.$[end] === cell) end += 1;
      ranging.$[cell] = d.vec2u(k, end);
      holding.$[blockOf(cell)] = headerIn.$[4];
    });

    const copy = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const k = gid.x;
      if (k >= headerIn.$[3]) return;
      if (copying !== undefined && itemOf !== undefined) copying.$[k] = itemOf(sortedSlots.$[k]);
    });

    // Flags the blocks to list: those holding items, or within the margin of one
    const flag = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const b = gid.x;
      if (b >= blocks) return;
      const home = blockAt(b);
      const stamp = headerIn.$[4];
      let found = d.u32(0);
      for (let n = d.u32(0); n < nearby; n++) {
        const step = d.vec3u(
          n % reach,
          std.intdiv(n, d.u32(reach)) % reach,
          std.intdiv(n, d.u32(reach * reach)),
        );
        const o = home + d.vec3i(step) - margin;
        if (std.any(std.lt(o, d.vec3i())) || std.any(std.ge(o, d.vec3i(blocked)))) continue;
        const index = d.u32(o.x) + blocked.x * (d.u32(o.y) + blocked.y * d.u32(o.z));
        if (heldIn.$[index] === stamp) {
          found = 1;
          break;
        }
      }
      flagging.$[b] = found;
    });

    const total = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const g = gid.x;
      if (g >= groups) return;
      let sum = d.u32(0);
      for (let b = g * GROUP; b < std.min((g + 1) * GROUP, blocks); b++) sum += flagsIn.$[b];
      totalling.$[g] = sum;
    });

    const listCount = tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      let sum = d.u32(0);
      for (let g = d.u32(0); g < groups; g++) sum += totalsIn.$[g];
      listArging.$[0] = sum;
      listArging.$[1] = 1;
      listArging.$[2] = 1;
    });

    // Each flagged block counts the flagged blocks before it and lists itself there
    const place = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const b = gid.x;
      if (b >= blocks || flagsIn.$[b] === 0) return;
      const g = std.intdiv(b, d.u32(GROUP));
      let at = d.u32(0);
      for (let before = d.u32(0); before < g; before++) at += totalsIn.$[before];
      for (let before = g * GROUP; before < b; before++) at += flagsIn.$[before];
      listInto.$[at] = b;
    });

    const clearing = root.createComputePipeline({ compute: clear });
    const tallying = root.createComputePipeline({ compute: tally });
    const bounding = root.createComputePipeline({ compute: bound });
    const copyingPipeline = root.createComputePipeline({ compute: copy });
    const flaggingPipeline = root.createComputePipeline({ compute: flag });
    const totalPipeline = root.createComputePipeline({ compute: total });
    const countPipeline = root.createComputePipeline({ compute: listCount });
    const placing = root.createComputePipeline({ compute: place });
    const run = [clearing, tallying, bounding, flaggingPipeline, totalPipeline, countPipeline];
    return {
      pipelines: [...run, placing, ...(copying ? [copyingPipeline] : [])],
      listArgs,
      destroy: () => {
        for (const buffer of [flags, totals, listArgs]) buffer.destroy();
      },
      /** Empties the last sort's ranges, before the binning sorts again. */
      clear(pass: TgpuComputePass, rows: number) {
        pass.setPipeline(clearing);
        pass.dispatchWorkgroups(Math.ceil(rows / WORKGROUP));
      },
      /** Counts, bounds, copies and lists the sorted items. */
      after(pass: TgpuComputePass, rows: number) {
        const over = Math.max(1, Math.ceil(rows / WORKGROUP));
        pass.setPipeline(tallying);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(bounding);
        pass.dispatchWorkgroups(over);
        if (copying) {
          pass.setPipeline(copyingPipeline);
          pass.dispatchWorkgroups(over);
        }
        pass.setPipeline(flaggingPipeline);
        pass.dispatchWorkgroups(Math.ceil(blocks / WORKGROUP));
        pass.setPipeline(totalPipeline);
        pass.dispatchWorkgroups(Math.ceil(groups / WORKGROUP));
        pass.setPipeline(countPipeline);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(placing);
        pass.dispatchWorkgroups(Math.ceil(blocks / WORKGROUP));
      },
    };
  }

  const walk = (point: d.v3f) => {
    'use gpu';
    let turn = d.vec3i(1);
    if (randf.sample() < 0.5) turn.x = -1;
    if (randf.sample() < 0.5) turn.y = -1;
    if (randf.sample() < 0.5) turn.z = -1;

    return Walk({ home: cellOf(point), turn });
  };

  const around = (point: d.v3f) => {
    'use gpu';
    return Walk({ home: cellOf(point), turn: d.vec3i(0, 1, 1) });
  };

  const range = (from: Walk, r: number) => {
    'use gpu';
    const index = d.u32(r);
    const y = from.home.y + from.turn.y * (d.i32((index + 1) % 3) - 1);
    const z = from.home.z + from.turn.z * (d.i32((std.intdiv(index, d.u32(3)) + 1) % 3) - 1);
    if (y < 0 || z < 0 || y >= d.i32(dims.y) || z >= d.i32(dims.z)) return Range();

    // Cells of a line sort together in x order, so three adjacent cells form one sorted range
    const left = std.max(from.home.x - 1, 0);
    const right = std.min(from.home.x + 1, d.i32(dims.x) - 1);
    const a = cellRanges.$[keyOf(d.vec3i(left, y, z))];
    const b = cellRanges.$[keyOf(d.vec3i(from.home.x, y, z))];
    const c = cellRanges.$[keyOf(d.vec3i(right, y, z))];
    // Cells of a line sort in x order: the range opens at its first cell holding items and closes at
    // its last
    const first = std.select(std.select(c.x, b.x, b.y > b.x), a.x, a.y > a.x);
    const end = std.select(std.select(a.y, b.y, b.y > b.x), c.y, c.y > c.x);
    const count = end - std.min(first, end);
    if (count === 0) return Range();
    let before = d.u32(0);
    if (left < from.home.x) before = a.y - a.x;
    const home = b.y - b.x;

    // Visit the middle cell first, then follow the walk's x direction and wrap. A walk without an
    // x direction reads the range in sorted order
    let start = d.u32(0);
    let back = d.u32(0);
    if (from.turn.x > 0) start = before % count;
    if (from.turn.x < 0) {
      start = (before + home + count - 1) % count;
      back = 1;
    }

    return Range({ first, count, start, back });
  };

  const neighbourAt = (at: number) => {
    'use gpu';
    return std.copy(neighbours.$[at]) as d.Infer<T>;
  };

  const neighbour = (of: Range, j: number) => {
    'use gpu';
    return neighbourAt(sortedIndex(of, j));
  };

  const slotAt = (at: number) => {
    'use gpu';
    return neighbourSlots.$[at];
  };

  const slot = (of: Range, j: number) => {
    'use gpu';
    return slotAt(sortedIndex(of, j));
  };

  const count = () => {
    'use gpu';
    return counted.$[3];
  };

  const holds = (at: d.v3i) => {
    'use gpu';
    if (std.any(std.lt(at, d.vec3i())) || std.any(std.ge(at, d.vec3i(blocked)))) return false;
    const index = d.u32(at.x) + blocked.x * (d.u32(at.y) + blocked.y * d.u32(at.z));
    return heldBlocks.$[index] === counted.$[4];
  };

  const block = (w: number) => {
    'use gpu';
    return blockAt(listed.$[w]);
  };

  const rangeAround = (point: d.v3f, r: number) => {
    'use gpu';
    return range(around(point), r);
  };
  const candidates: Candidates = {
    ranges: RANGES,
    range: tgpu.fn(rangeAround),
    slot: tgpu.fn(slotAt),
  };

  const queue = root.device.queue;
  const boxGpu = root.unwrap(box.buffer);
  const moved = new Float32Array(3);

  return {
    capacity,
    walk,
    around,
    range,
    index: sortedIndex,
    neighbour,
    slot,
    slotAt,
    neighbourAt,
    count,
    holds,
    block,

    inject: () => fill,

    candidates: () => candidates,

    async initAsync() {
      await Promise.all([
        bins.initAsync(),
        pipelines.compile(
          lone
            ? lone.pipelines
            : [clearPipeline, boundPipeline, copyPipeline, nearPipeline].filter((each) => !!each),
        ),
      ]);
    },

    sort(pass: TgpuComputePass) {
      sorted = true;

      const rows = span ? Math.min(span(), capacity) : capacity;
      if (lone) {
        lone.clear(pass, lastRows);
        bins.sort(pass, rows);
        lone.after(pass, rows);
        lastRows = rows;
        return;
      }
      pass.setPipeline(clearPipeline);
      pass.dispatchWorkgroupsIndirect(header);
      bins.sort(pass, rows);
      // Sorted items end by `rows`, and the key after the last says where
      pass.setPipeline(boundPipeline);
      pass.dispatchWorkgroups(Math.ceil(Math.min(rows + 1, capacity) / WORKGROUP));
      pass.setPipeline(copyPipeline);
      pass.dispatchWorkgroupsIndirect(header);
      if (nearPipeline) {
        pass.setPipeline(nearPipeline);
        pass.dispatchWorkgroupsIndirect(blockArgs, 0);
      }
    },

    dispatch(pass) {
      binned();
      pass.dispatchWorkgroupsIndirect(header, 0);
    },

    dispatchBlocks(pass) {
      binned();
      pass.dispatchWorkgroupsIndirect(lone?.listArgs ?? nearArgs ?? blockArgs, 0);
    },

    move(to) {
      moved[0] = to.x;
      moved[1] = to.y;
      moved[2] = to.z;
      queue.writeBuffer(boxGpu, 0, moved);
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
