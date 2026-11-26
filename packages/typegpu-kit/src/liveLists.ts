/**
 * Lists live slots by group on the GPU. Each frame, it writes one indirect draw and one indirect
 * dispatch per group. It accepts a table or any set of slots with a live test, or kernels that
 * list the rows they visit.
 *
 * @module
 * @injects member `slot`, `instanceSlot` and `count`, which read its own buffers in any
 * pipeline
 * @binds 1 storage: `slot` or `instanceSlot`, the live slots
 * @binds 1 storage: `count`, the indirect draws
 * @binds 2 storage: `list`, the live slots and their counts
 * @requires indirect-first-instance option `firstInstance: true` needs it; without it the
 * live lists throw
 * @owns 4 buffers, 2 pipelines: the live slots per group, their counts, the indirect draws and
 * the indirect dispatches
 * @records `compact` after the table's flush, or `finish` after the kernels that `list`, then
 * `draw` or `dispatch` each group from it.
 */

import {
  d,
  std,
  tgpu,
  type IndirectFlag,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuRoot,
} from 'typegpu';
import * as pipelines from './pipelines.ts';
import { checkDispatch } from './dispatchLimit.ts';
import type { Inject } from './inject.ts';
import { adopt, type AnyComputePass } from './passes.ts';

export type IndirectDraw =
  | { readonly vertexCount: number; readonly firstVertex?: number }
  | { readonly indexCount: number; readonly firstIndex?: number; readonly baseVertex?: number };

/** Takes `span` only beside `alive`, as a table's `live()` gives them. */
export type Options = ListOptions &
  (
    | {
        /**
         * Tests whether slot `i` holds a live row. A table supplies this through `live()`. Default
         * every slot is live.
         * @gpu
         */
        readonly alive: (i: number) => boolean;
        /**
         * Returns the highest live slot plus one, read when `compact` records, such as a table's
         * from `live()`. Skips the slots above it, so `alive` must be false there. Needs `alive`.
         * Default `capacity`.
         * @cpu
         */
        readonly span?: () => number;
      }
    | { readonly alive?: never; readonly span?: never }
  );

interface ListOptions {
  /** Sets the number of slots, including dead slots. Indices range from `0` to `capacity - 1`. */
  readonly capacity: number;
  /**
   * Provides one draw per group. Requires 1 to 256 draws. Default an empty draw for each of
   * `groups`, for lists that only dispatch.
   */
  readonly draws?: readonly IndirectDraw[];
  /**
   * Sets the number of groups, from 1 to 256. With `draws`, it must equal their number. Default
   * the number of `draws`, or 1.
   */
  readonly groups?: number;
  /**
   * Maps a live slot to its group, such as with a table field's getter. Skips rows whose group is
   * out of range. Default every live row belongs to group 0.
   * @gpu
   */
  readonly group?: (i: number) => number;
  /**
   * Uses `firstInstance` to start each draw at its group's region. Shaders can then find rows with
   * `live.instanceSlot(instanceIndex)`, and one pipeline can draw every group. Throws without the
   * device feature `indirect-first-instance`, because those draws otherwise draw nothing without
   * reporting an error. Default false.
   */
  readonly firstInstance?: boolean;
  /**
   * Sets the threads per workgroup for kernels run by `dispatch`. Each dispatch includes enough
   * workgroups for its live rows. Must be an integer above 0. Default 64.
   */
  readonly workgroupSize?: number;
  /** Names the buffers and pipelines in captures and the GPU debug panel. Default `live`. */
  readonly label?: string;
}

/** Dispatches indirectly through a compute pass with a pipeline set on it. */
export interface IndirectDispatcher {
  dispatchWorkgroupsIndirect(
    buffer: TgpuBuffer<d.AnyWgslData> & IndirectFlag,
    offset: number,
  ): void;
}

/** Draws indirectly through a pass or bundle with a pipeline set on it. */
export interface IndirectDrawer {
  drawIndirect(buffer: TgpuBuffer<d.AnyWgslData> & IndirectFlag, offset: number): void;
  drawIndexedIndirect(buffer: TgpuBuffer<d.AnyWgslData> & IndirectFlag, offset: number): void;
}

export interface LiveLists extends Disposable {
  /** Reports the number of slots reserved for each group. */
  readonly capacity: number;
  /** Reports the number of groups, one per draw. */
  readonly groups: number;
  /**
   * Lists row `i` in `group` from a kernel that already visits the rows, in place of `compact`.
   * Every invocation of the workgroup must call it once, outside any branch, with a group of
   * `groups` or above for a row it doesn't list, because it synchronises the workgroup. Record
   * `finish` after the kernel.
   */
  readonly list: (group: number, i: number) => void;
  /** Reports whether each group's draw starts at its own region, as `firstInstance` asked. */
  readonly firstInstance: boolean;
  /** Fills `slot`, `instanceSlot` and `count` with these lists' buffers, as their defaults do. */
  inject(): Inject;
  /** Returns the slot of live row `i` in `group`. Pass `group` as a captured constant. */
  readonly slot: (group: number, i: number) => number;
  /**
   * Returns the slot drawn by `instanceIndex`. Requires `firstInstance: true`. A shader that calls
   * it without that option fails to resolve.
   */
  readonly instanceSlot: (instanceIndex: number) => number;
  /** Stores `capacity` live slots per group. `compact` writes them. */
  readonly slots: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  /**
   * Returns the live row count for `group` in a shader. Kernels run by
   * `dispatch` must return for threads at or beyond this count. Pass `group` as a captured
   * constant.
   */
  readonly count: (group: number) => number;
  /** Reports the threads per workgroup for kernels run by `dispatch`. */
  readonly workgroupSize: number;
  /** Stores one indirect draw record of 5 words per group. */
  readonly args: TgpuBuffer<d.WgslArray<d.U32>> & IndirectFlag & StorageFlag;
  /**
   * Stores one indirect dispatch record of 3 words per group. Each workgroup has `workgroupSize`
   * threads.
   */
  readonly dispatches: TgpuBuffer<d.WgslArray<d.U32>> & IndirectFlag & StorageFlag;
  /** Updates a group's draw while keeping its instance count. */
  setDraw(group: number, draw: IndirectDraw): void;
  /** Compiles the compaction and counting pipelines. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Rewrites the live slots and instance counts. Record it after the table's flush. */
  compact(pass: AnyComputePass): void;
  /**
   * Writes the counts of the rows listed with `list` into the draws and dispatches, and empties
   * the lists for the next listing. Record it after the kernels that list.
   */
  finish(pass: AnyComputePass): void;
  /**
   * Draws `group` into the pass with the pipeline set on it. Uses an indexed draw when the group's
   * draw is indexed.
   */
  draw(pass: IndirectDrawer, group: number): void;
  /**
   * Dispatches the pipeline set on the compute pass over the live rows of `group`. Record it after
   * `compact`. Dispatches enough workgroups of `workgroupSize` for `count(group)` threads. Each
   * thread finds its row with `slot(group, i)`.
   */
  dispatch(pass: IndirectDispatcher, group: number): void;
  /** Destroys the buffers. */
  destroy(): void;
}

const ARGS_WORDS = 5; // a drawIndexedIndirect record; a drawIndirect record reads the first 4
const WORKGROUP = 256;
/** Each group takes two words of the listing kernel's workgroup memory. */
const MAX_GROUPS = 256;

/**
 * Creates a list of live slots and an indirect draw per group. Each group reserves `capacity`
 * slots. A shader finds rows with `live.slot(group, i)` using a captured group index. With
 * `firstInstance`, it can use `live.instanceSlot(instanceIndex)` instead. Throws if the groups,
 * draws or workgroup size are invalid, the slot buffer exceeds `maxStorageBufferBindingSize`,
 * `compact` over the capacity needs more than `maxComputeWorkgroupsPerDimension` workgroups, or
 * `firstInstance` lacks its required feature.
 *
 * @param options.alive whether a slot holds a live row, carrying the fills of what it reads, such
 * as a table's from `rows.live()`, spread into the options with its capacity
 * @param options.draws one draw per group
 * @returns live lists you run every frame and destroy
 */
export function create(root: TgpuRoot, options: Options): LiveLists {
  const {
    capacity,
    alive,
    group: groupOf,
    span,
    firstInstance = false,
    workgroupSize = 64,
    label = 'live',
  } = options;
  const groups = options.groups ?? options.draws?.length ?? 1;
  if (!Number.isInteger(groups) || groups < 1 || groups > MAX_GROUPS) {
    throw new RangeError(`Live lists need 1 to ${MAX_GROUPS} groups, got ${groups}.`);
  }
  if (options.draws && options.draws.length !== groups) {
    throw new RangeError(
      `Live lists have ${groups} groups but ${options.draws.length} draws; give one draw per group.`,
    );
  }
  const draws = options.draws ?? Array.from({ length: groups }, () => ({ vertexCount: 0 }));
  const slotBytes = groups * capacity * 4;
  const limit = root.device.limits.maxStorageBufferBindingSize;
  if (slotBytes > limit) {
    throw new RangeError(
      `The live lists need ${slotBytes} bytes to hold ${capacity} rows in each of ${groups} groups. ` +
        `The device limit \`maxStorageBufferBindingSize\` is ${limit} bytes. ` +
        'Use fewer groups, a smaller table, or request a higher limit.',
    );
  }
  if (!Number.isInteger(workgroupSize) || workgroupSize < 1) {
    throw new RangeError(
      `Live lists require workgroupSize to be an integer above 0, got ${workgroupSize}.`,
    );
  }
  // `compact` covers the capacity; `dispatch` covers only the rows a group lists
  checkDispatch(root, 'liveLists.create', 'capacity', capacity, WORKGROUP);
  if (firstInstance && !root.device.features.has('indirect-first-instance')) {
    throw new Error(
      "Live lists with firstInstance need the device feature 'indirect-first-instance'. " +
        'Request it in requiredFeatures, or capture each group with live.slot(group, i).',
    );
  }

  const slotsBuffer = root
    .createBuffer(d.arrayOf(d.u32, groups * capacity))
    .$usage('storage')
    .$name(`${label}_slots`);
  const countsBuffer = root
    .createBuffer(d.arrayOf(d.atomic(d.u32), groups))
    .$usage('storage')
    .$name(`${label}_counts`);
  const argsBuffer = root
    .createBuffer(d.arrayOf(d.u32, groups * ARGS_WORDS))
    .$usage('storage', 'indirect')
    .$name(`${label}_draws`);
  const dispatchBuffer = root
    .createBuffer(d.arrayOf(d.u32, groups * 3))
    .$usage('storage', 'indirect')
    .$name(`${label}_dispatches`);
  const destroy = () => {
    for (const buffer of [slotsBuffer, countsBuffer, argsBuffer, dispatchBuffer]) buffer.destroy();
  };

  const written = slotsBuffer.as('mutable');
  const counts = countsBuffer.as('mutable');
  const args = argsBuffer.as('mutable');
  const dispatching = dispatchBuffer.as('mutable');
  const read = slotsBuffer.as('readonly');

  // Accessors per instance, filled with its own buffers, so one shader can read several lists
  const drawn = argsBuffer.as('readonly');
  const slots = tgpu.accessor(d.arrayOf(d.u32), read);
  const records = tgpu.accessor(d.arrayOf(d.u32), drawn);
  const inject: Inject = (cfg) => cfg.with(slots, read).with(records, drawn);

  const count = (group: number) => {
    'use gpu';
    return records.$[d.u32(group) * ARGS_WORDS + 1];
  };

  const slot = (group: number, i: number) => {
    'use gpu';
    return slots.$[d.u32(group) * capacity + i];
  };

  // Without `firstInstance`, every draw starts at instance 0. The instance index cannot identify
  // the group, so reject this accessor instead of reading group 0
  const refuse = tgpu.comptime((): number => {
    throw new Error(
      'liveLists: `instanceSlot` needs live lists created with `firstInstance: true`; read `slot(group, instanceIndex)` instead.',
    );
  });
  const instanceSlot = (instanceIndex: number) => {
    'use gpu';
    if (!firstInstance) return refuse();
    return slots.$[instanceIndex];
  };

  // Workgroup memory starts at zero, and the first row listed in a group reserves the whole
  // workgroup's range with one global atomic
  const localCounts = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), groups));
  const localBase = tgpu.workgroupVar(d.arrayOf(d.u32, groups));
  const list = tgpu
    .fn([d.u32, d.u32])((group, i) => {
      'use gpu';
      const listed = group < groups;
      let local = d.u32(0);
      if (listed) local = std.atomicAdd(localCounts.$[group], 1);
      std.workgroupBarrier();
      if (listed && local === 0) {
        localBase.$[group] = std.atomicAdd(counts.$[group], std.atomicLoad(localCounts.$[group]));
      }
      std.workgroupBarrier();
      if (listed) written.$[group * capacity + localBase.$[group] + local] = i;
    })
    .$name(`${label}_list`);

  const compact = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    let group = d.u32(groups);
    if (i < capacity) {
      if (alive === undefined || alive(i)) {
        group = groupOf === undefined ? d.u32(0) : d.u32(groupOf(i));
      }
    }
    list(group, i);
  });

  const finishing = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const g = gid.x;
    if (g >= groups) return;

    const n = std.atomicExchange(counts.$[g], 0);
    args.$[g * ARGS_WORDS + 1] = n;
    dispatching.$[g * 3] = std.intdiv(n + (workgroupSize - 1), workgroupSize);
    dispatching.$[g * 3 + 1] = 1;
    dispatching.$[g * 3 + 2] = 1;
  });

  const compactPipeline = root
    .createComputePipeline({ compute: compact })
    .$name(`${label}_compact`);
  const finishPipeline = root
    .createComputePipeline({ compute: finishing })
    .$name(`${label}_finish`);

  const indexed: boolean[] = [];
  const record = new Uint32Array(ARGS_WORDS);
  const queue = root.device.queue;
  const argsGpu = root.unwrap(argsBuffer);

  let listed = false;
  const counted = () => {
    if (listed) return;
    throw new Error(
      `Live lists '${label}' have no recorded list yet. Record \`compact(pass)\` after flushing the table, or \`finish(pass)\` after the kernels that build the lists. Do this before drawing or dispatching with the lists.`,
    );
  };

  const finish = (pass: AnyComputePass) => {
    const into = adopt(pass);
    listed = true;
    into.setPipeline(finishPipeline);
    into.dispatchWorkgroups(Math.ceil(groups / 64));
  };

  const setDraw = (group: number, draw: IndirectDraw) => {
    record.fill(0);
    const first = firstInstance ? group * capacity : 0;
    const hasIndices = 'indexCount' in draw;
    if (hasIndices) {
      record[0] = draw.indexCount;
      record[2] = draw.firstIndex ?? 0;
      record[3] = draw.baseVertex ?? 0;
      record[4] = first;
    } else {
      record[0] = draw.vertexCount;
      record[2] = draw.firstVertex ?? 0;
      record[3] = first;
    }
    indexed[group] = hasIndices;

    const offset = group * ARGS_WORDS * 4;
    queue.writeBuffer(argsGpu, offset, record.buffer, 0, 4);
    queue.writeBuffer(argsGpu, offset + 8, record.buffer, 8, 12);
  };

  draws.forEach((draw, group) => setDraw(group, draw));

  return {
    capacity,
    groups,
    firstInstance,

    inject: () => inject,

    slot,
    instanceSlot,
    count,
    workgroupSize,
    slots: slotsBuffer,
    args: argsBuffer,
    dispatches: dispatchBuffer,
    setDraw,

    initAsync: () => pipelines.compile([compactPipeline, finishPipeline]),

    list,

    compact(pass) {
      const into = adopt(pass);
      const rows = span ? span() : capacity;
      if (rows > 0) {
        into.setPipeline(compactPipeline);
        into.dispatchWorkgroups(Math.ceil(rows / WORKGROUP));
      }
      finish(pass);
    },

    finish,

    draw(pass, group) {
      // A render bundle records its draws once, before the first frame lists anything
      if (!('finish' in pass)) counted();
      const offset = group * ARGS_WORDS * 4;
      if (indexed[group]) pass.drawIndexedIndirect(argsBuffer, offset);
      else pass.drawIndirect(argsBuffer, offset);
    },

    dispatch(pass, group) {
      counted();
      pass.dispatchWorkgroupsIndirect(dispatchBuffer, group * 12);
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
