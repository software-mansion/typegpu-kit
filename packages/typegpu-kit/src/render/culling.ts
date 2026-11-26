import {
  d,
  std,
  tgpu,
  type IndirectFlag,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuReadonly,
  type TgpuRoot,
  type TgpuUniform,
  type WithBinding,
} from 'typegpu';
import { adopt, layer, type AnyComputePass } from '../passes.ts';
import * as pipelines from '../pipelines.ts';
import * as workgroup from '../workgroup.ts';

const WORKGROUP = 256;
/** Words in a `drawIndexedIndirect` record. */
export const RECORD = 5;

export interface CullingOptions {
  /**
   * Gives the mesh count for the listing header and draw records. It must fit in one workgroup.
   */
  readonly meshes: number;
  /** Sets the maximum number of rows. */
  readonly capacity: number;
  /** Gives the number of layers, a visible list and draws for each. */
  readonly layers: number;
  /** Controls whether each draw uses `firstInstance` to start at its mesh rows. */
  readonly firstInstance: boolean;
  /** Lists mesh start offsets followed by the live rows for each mesh, as written by `flush`. */
  readonly listed: TgpuReadonly<d.WgslArray<d.U32>>;
  /** Holds the number of rows in the listing. */
  readonly entries: TgpuUniform<d.U32>;
  /** Returns the mesh index for a row. */
  readonly meshOf: (row: number) => number;
  /** Returns whether the camera injected into the configuration can see a row. */
  readonly seen: (row: number) => boolean;
  /** Reports whether a layer draws this frame, so `record` culls only those. */
  readonly active: (layer: number) => boolean;
}

/** This culling result lists visible rows and provides one draw per mesh, for each layer. */
export interface Culling {
  /** Reports the number of layers. */
  readonly layers: number;
  /**
   * Holds, for each layer, mesh start offsets followed by visible rows for each mesh. Layer `k`
   * starts at `k * (meshes + capacity)`, and its start offsets index the whole buffer.
   */
  readonly culled: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
  /**
   * Holds one `drawIndexedIndirect` record per mesh for each layer, layer by layer. Culling writes
   * the instance count and start.
   */
  readonly draws: TgpuBuffer<d.WgslArray<d.U32>> & IndirectFlag & StorageFlag;
  /** Records culling of every active layer for the first `rows` entries in the listing. */
  record(pass: AnyComputePass, rows: number): void;
  initAsync(): Promise<void>;
  destroy(): void;
}

/**
 * Creates two kernels that cull rows against the camera in `cfg`. They preserve listing order to
 * keep draw order stable across frames. The first marks visible rows and counts them by block
 * and mesh. The second uses a block scan to place each visible row after the visible rows that
 * precede it. Both dispatch over every active layer at once, y indexing the active layers' list,
 * and the kernels read the layer through `passes.layer`.
 */
export function createCulling(root: TgpuRoot, cfg: WithBinding, options: CullingOptions): Culling {
  const { meshes, capacity, layers, firstInstance, listed, entries, meshOf, seen, active } =
    options;
  const blocks = Math.ceil(capacity / WORKGROUP);
  const span = meshes + capacity;

  // Share one binding to leave storage slots for kind callbacks
  const scratchBuffer = root
    .createBuffer(
      d
        .struct({
          marks: d.arrayOf(d.u32, capacity * layers),
          sums: d.arrayOf(d.u32, blocks * layers),
          counts: d.arrayOf(d.atomic(d.u32), meshes * layers),
          /** The active layers, in order, that each row of workgroups culls. */
          order: d.arrayOf(d.u32, layers),
        })
        .$name('Scratch'),
    )
    .$usage('storage');
  const culledBuffer = root.createBuffer(d.arrayOf(d.u32, span * layers)).$usage('storage');
  const drawBuffer = root
    .createBuffer(d.arrayOf(d.u32, meshes * RECORD * layers))
    .$usage('storage', 'indirect');

  const scratch = scratchBuffer.as('mutable');
  const orderAt = d.memoryLayoutOf(scratchBuffer.dataType, (each) => each.order).offset;
  const order = new Uint32Array(layers);
  // No layer index, so the first `record` writes the order
  const written = new Uint32Array(layers).fill(0xffffffff);
  const layerAt =
    layers > 1
      ? (y: number) => {
          'use gpu';
          return scratch.$.order[y];
        }
      : (y: number) => {
          'use gpu';
          return y;
        };
  const culled = culledBuffer.as('mutable');
  const draws = drawBuffer.as('mutable');

  const seenInBlock = tgpu.workgroupVar(d.atomic(d.u32));
  const seenOfMesh = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), meshes));
  const before = tgpu.workgroupVar(d.atomic(d.u32));
  const prefix = workgroup.scan(workgroup.sum(d.u32), WORKGROUP);
  const culledLayer = tgpu.privateVar(d.u32);
  const layerOf = tgpu
    .fn(
      [],
      d.u32,
    )(() => culledLayer.$)
    .$name('culled layer');

  const builtins = {
    gid: d.builtin.globalInvocationId,
    lid: d.builtin.localInvocationIndex,
    wid: d.builtin.workgroupId,
  };

  // Store the mesh index above bit 0, which marks visibility
  const mark = tgpu.computeFn({ workgroupSize: [WORKGROUP], in: builtins })(({ gid, lid, wid }) => {
    'use gpu';
    if (lid === 0) std.atomicStore(seenInBlock.$, 0);
    if (lid < meshes) std.atomicStore(seenOfMesh.$[lid], 0);
    std.workgroupBarrier();

    const k = layerAt(gid.y);
    culledLayer.$ = k;
    const t = gid.x;
    if (t < entries.$) {
      const row = listed.$[meshes + t];
      const mesh = d.u32(meshOf(row));
      let visible = d.u32(0);
      if (seen(row)) {
        visible = 1;
        std.atomicAdd(seenInBlock.$, 1);
        std.atomicAdd(seenOfMesh.$[mesh], 1);
      }
      scratch.$.marks[k * capacity + t] = (mesh << 1) | visible;
    }
    std.workgroupBarrier();

    if (lid === 0) scratch.$.sums[k * blocks + wid.x] = std.atomicLoad(seenInBlock.$);
    if (lid < meshes) {
      const here = std.atomicLoad(seenOfMesh.$[lid]);
      if (here > 0) std.atomicAdd(scratch.$.counts[k * meshes + lid], here);
    }
  });

  const place = tgpu.computeFn({ workgroupSize: [WORKGROUP], in: builtins })(({
    gid,
    lid,
    wid,
  }) => {
    'use gpu';
    if (lid === 0) std.atomicStore(before.$, 0);
    std.workgroupBarrier();

    const k = layerAt(gid.y);
    let earlier = d.u32(0);
    for (let block = lid; block < wid.x; block += WORKGROUP) {
      earlier += scratch.$.sums[k * blocks + block];
    }
    std.atomicAdd(before.$, earlier);

    const t = gid.x;
    const listedHere = t < entries.$;
    let marked = d.u32(0);
    if (listedHere) marked = scratch.$.marks[k * capacity + t];
    const visible = marked & 1;

    const ahead = prefix(visible, lid);
    if (!listedHere) return;

    const base = k * span;
    const at = base + meshes + std.atomicLoad(before.$) + ahead;
    if (visible === 1) culled.$[at] = listed.$[meshes + t];

    // Only the first row of each mesh writes and resets its draw count
    const mesh = marked >>> 1;
    if (listed.$[mesh] === meshes + t) {
      const record = (k * meshes + mesh) * RECORD;
      culled.$[base + mesh] = at;
      draws.$[record + 1] = std.atomicExchange(scratch.$.counts[k * meshes + mesh], 0);
      if (firstInstance) draws.$[record + 4] = at;
    }
  });

  // Each invocation reads the layer it culls for, not the layer of a pass
  const culling = cfg.with(layer, layerOf);
  const marking = culling.createComputePipeline({ compute: mark });
  const placing = culling.createComputePipeline({ compute: place });

  return {
    layers,
    culled: culledBuffer,
    draws: drawBuffer,

    record(pass, rows) {
      if (rows === 0) return;
      let count = 0;
      for (let k = 0; k < layers; k++) if (active(k)) order[count++] = k;
      if (count === 0) return;
      let same = true;
      for (let i = 0; i < count; i++) same &&= order[i] === written[i];
      if (!same && layers > 1) {
        for (let i = 0; i < count; i++) written[i] = order[i];
        root.device.queue.writeBuffer(root.unwrap(scratchBuffer), orderAt, order, 0, count);
      }
      const groups = Math.ceil(rows / WORKGROUP);
      const into = adopt(pass);
      into.setPipeline(marking);
      into.dispatchWorkgroups(groups, count);
      into.setPipeline(placing);
      into.dispatchWorkgroups(groups, count);
    },

    async initAsync() {
      await pipelines.compile([marking, placing]);
    },

    destroy() {
      scratchBuffer.destroy();
      culledBuffer.destroy();
      drawBuffer.destroy();
    },
  };
}
