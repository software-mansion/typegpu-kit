import {
  d,
  std,
  tgpu,
  type IndirectFlag,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuComputePass,
  type TgpuReadonly,
  type TgpuRoot,
  type TgpuUniform,
  type WithBinding,
} from 'typegpu';
import { layer } from '../passes.ts';
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
  /** Records culling of every layer for the first `rows` entries in the listing. */
  record(pass: TgpuComputePass, rows: number): void;
  initAsync(): Promise<void>;
  destroy(): void;
}

/**
 * Creates two kernels that cull rows against the camera in `cfg`. They preserve listing order to
 * keep draw order stable across frames. The first marks visible rows and counts them by block
 * and mesh. The second uses a block scan to place each visible row after the visible rows that
 * precede it. Both dispatch over every layer at once, the layer as y, which the kernels read
 * through `passes.layer`.
 */
export function createCulling(root: TgpuRoot, cfg: WithBinding, options: CullingOptions): Culling {
  const { meshes, capacity, layers, firstInstance, listed, entries, meshOf, seen } = options;
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
        })
        .$name('Scratch'),
    )
    .$usage('storage');
  const culledBuffer = root.createBuffer(d.arrayOf(d.u32, span * layers)).$usage('storage');
  const drawBuffer = root
    .createBuffer(d.arrayOf(d.u32, meshes * RECORD * layers))
    .$usage('storage', 'indirect');

  const scratch = scratchBuffer.as('mutable');
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

    const k = gid.y;
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

    const k = gid.y;
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
      const groups = Math.ceil(rows / WORKGROUP);
      pass.setPipeline(marking);
      pass.dispatchWorkgroups(groups, layers);
      pass.setPipeline(placing);
      pass.dispatchWorkgroups(groups, layers);
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
