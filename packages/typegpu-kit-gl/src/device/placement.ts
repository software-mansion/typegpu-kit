/**
 * Where each compute pipeline with a GPU form runs, learned from its last dispatches. A GPU form
 * whose outputs the CPU reads back costs a wait for the GPU each time; a kernel whose inputs are
 * already on the CPU can run there instead and leave GL an upload, which costs no wait. The device
 * cannot see ahead within a frame, so it watches a window of dispatches and moves a small kernel to
 * the CPU when most of its outputs were read back. A readback's wait is mostly the GPU finishing
 * earlier work, which moves to the next readback when one goes, so the move is judged by all the
 * device's readbacks: the kernel goes back for good when its CPU run costs more than the waiting
 * it saved, unless that cost is too small to tell apart from how waits vary, and goes back when
 * its inputs stop being on the CPU.
 */

import type { CpuModule } from '../cpu/types.ts';
import { type Bound, bufferEntry } from './binding.ts';
import type { BufferImpl } from './buffer.ts';
import type { Ctx } from './context.ts';
import type { GpuCompute } from './gpuCompute.ts';
import type { TextureImpl } from './texture.ts';

/** Dispatches watched before a decision. */
const WINDOW = 6;
/**
 * Milliseconds a CPU run may cost over the GPU form without paying for itself: readback waits
 * swing by more than this from window to window, so cheaper kernels stay where they save syncs.
 */
const CHEAP_MS = 0.5;
/** Times a pipeline may go back to the GPU before it stays there, so placement settles. */
const MOST_RETURNS = 2;
/** The most invocations a kernel may have to move to the CPU, so a trial never stalls a frame. */
const MOST_INVOCATIONS = 1 << 16;

interface Record {
  readonly label: string;
  onCpu: boolean;
  /** Just moved: the first window on the CPU reads back what GL held, so it is not judged. */
  settling: boolean;
  /** Stays on the GPU: the CPU run cost more than it saved, or it came back too often. */
  pinned: boolean;
  /** Times it went back to the GPU because its inputs moved there. */
  returns: number;
  dispatches: number;
  /** GPU dispatches whose outputs were read back before the pipeline wrote them again. */
  readBack: number;
  /** Milliseconds of those readbacks, and of the GPU dispatches. */
  waitMs: number;
  gpuMs: number;
  /** Dispatches whose every input was current on the CPU. */
  cpuInputs: number;
  cpuMs: number;
  /** The device's readback wait when the window began. */
  waitAt: number;
  /** When it moved: a GPU dispatch's own cost, and the device's readback wait per dispatch. */
  gpuIssue: number;
  waitBefore: number;
  /** The last input that GL alone held, for the stats. */
  held: string;
}

const records = new WeakMap<object, Record>();
/** Milliseconds every readback of a device has waited. */
const waits = new WeakMap<Ctx, { ms: number }>();

function waitedOf(ctx: Ctx): { ms: number } {
  let waited = waits.get(ctx);
  if (!waited) waits.set(ctx, (waited = { ms: 0 }));
  return waited;
}
const writers = new WeakMap<BufferImpl | TextureImpl, Record>();

function recordOf(ctx: Ctx, pipeline: object, label: string): Record {
  let record = records.get(pipeline);
  if (!record) {
    record = {
      label,
      onCpu: false,
      settling: false,
      pinned: false,
      returns: 0,
      dispatches: 0,
      readBack: 0,
      waitMs: 0,
      gpuMs: 0,
      cpuInputs: 0,
      cpuMs: 0,
      waitAt: waitedOf(ctx).ms,
      gpuIssue: 0,
      waitBefore: 0,
      held: '',
    };
    records.set(pipeline, record);
    ctx.stats.placement.set(label, 'gpu: no history yet');
  }
  return record;
}

function restart(ctx: Ctx, record: Record): void {
  record.waitAt = waitedOf(ctx).ms;
  record.dispatches = 0;
  record.readBack = 0;
  record.waitMs = 0;
  record.gpuMs = 0;
  record.cpuInputs = 0;
  record.cpuMs = 0;
}

/**
 * The label of a resource the dispatch binds that only GL holds current, which a CPU run would
 * read back, or undefined when every one is current on the CPU. What the pipeline's own GPU form
 * wrote last does not count: on the CPU, the pipeline would have written it there.
 */
function heldOnGl(
  module: CpuModule,
  groups: readonly (Bound | undefined)[],
  record: Record,
): string | undefined {
  for (const use of module.bindings) {
    if (use.kind === 'sampler') continue;
    if (use.kind === 'buffer') {
      const buffer = bufferEntry(groups[use.group], use.binding)?.buffer;
      if (buffer?.gpuNewer && writers.get(buffer) !== record) {
        return buffer.label || `@group(${use.group}) @binding(${use.binding})`;
      }
      continue;
    }
    const entry = groups[use.group]?.group.entries.get(use.binding);
    if (entry?.kind === 'texture') {
      const { texture } = entry.view;
      if (texture.written && texture.home === 'gl' && writers.get(texture) !== record) {
        return texture.label || 'a texture';
      }
    }
  }
  return undefined;
}

/** Whether the dispatch should run its GPU form. */
export function runsOnGpu(
  ctx: Ctx,
  pipeline: object,
  label: string,
  module: CpuModule,
  groups: readonly (Bound | undefined)[],
): boolean {
  if (!ctx.placement) return true;
  const record = recordOf(ctx, pipeline, label);
  record.dispatches++;
  const held = heldOnGl(module, groups, record);
  if (held === undefined) record.cpuInputs++;
  else record.held = held;
  return !record.onCpu;
}

/** Notes a GPU form's dispatch: its targets now hold this pipeline's output. */
export function ranOnGpu(
  ctx: Ctx,
  pipeline: object,
  label: string,
  gpu: GpuCompute,
  groups: readonly (Bound | undefined)[],
  ms: number,
  invocations: number,
): void {
  const record = recordOf(ctx, pipeline, label);
  record.gpuMs += ms;
  for (const { target } of gpu.passes) {
    if (target.kind === 'buffer') {
      const buffer = bufferEntry(groups[target.group], target.binding)?.buffer;
      if (buffer) writers.set(buffer, record);
    } else {
      const entry = groups[target.group]?.group.entries.get(target.binding);
      if (entry?.kind === 'texture') writers.set(entry.view.texture, record);
    }
  }
  if (record.dispatches < WINDOW) return;
  const { dispatches, readBack: reads, cpuInputs } = record;
  const moves =
    !record.pinned &&
    reads * 2 >= dispatches &&
    cpuInputs * 2 >= dispatches &&
    invocations <= MOST_INVOCATIONS;
  if (moves) {
    record.onCpu = true;
    record.settling = true;
    record.gpuIssue = record.gpuMs / dispatches;
    record.waitBefore = (waitedOf(ctx).ms - record.waitAt) / dispatches;
    ctx.stats.placement.set(
      label,
      `cpu: the CPU read back its outputs after ${reads} of ${dispatches} dispatches, waiting ${(record.waitMs / dispatches).toFixed(2)} ms each, and its inputs were on the CPU`,
    );
  } else if (!record.pinned) {
    const why =
      reads * 2 < dispatches
        ? `its outputs were read back after ${reads} of ${dispatches} dispatches`
        : cpuInputs * 2 < dispatches
          ? `GL alone held its input '${record.held}' in ${dispatches - cpuInputs} of ${dispatches} dispatches`
          : `${invocations} invocations are too many to try on the CPU`;
    ctx.stats.placement.set(label, `gpu: ${why}`);
  }
  restart(ctx, record);
}

/** Notes a CPU run of a pipeline that has a GPU form, and moves it back if the CPU costs more. */
export function ranOnCpu(ctx: Ctx, pipeline: object, ms: number): void {
  const record = records.get(pipeline);
  if (!record?.onCpu) return;
  record.cpuMs += ms;
  if (record.dispatches < WINDOW) return;
  if (record.settling) {
    record.settling = false;
    restart(ctx, record);
    return;
  }
  const { dispatches, cpuInputs } = record;
  const extra = record.cpuMs / dispatches - record.gpuIssue;
  const saved = record.waitBefore - (waitedOf(ctx).ms - record.waitAt) / dispatches;
  if (extra > CHEAP_MS && extra > saved) {
    record.onCpu = false;
    record.pinned = true;
    ctx.stats.placement.set(
      record.label,
      `gpu: a CPU run cost ${extra.toFixed(2)} ms more than the GPU form, and saved ${saved.toFixed(2)} ms of readback waits`,
    );
  } else if (cpuInputs * 2 < dispatches) {
    record.onCpu = false;
    record.returns++;
    record.pinned = record.returns >= MOST_RETURNS;
    ctx.stats.placement.set(
      record.label,
      `gpu: its inputs moved to the GPU in ${dispatches - cpuInputs} of ${dispatches} dispatches${record.pinned ? ', and it stays there' : ''}`,
    );
  } else {
    ctx.stats.placement.set(
      record.label,
      `cpu: a CPU run cost ${extra.toFixed(2)} ms more than the GPU form, and saved ${saved.toFixed(2)} ms of readback waits`,
    );
  }
  restart(ctx, record);
}

/** Notes that the CPU read `resource` back from GL, which charges the GPU form that wrote it. */
export function readBack(ctx: Ctx, resource: BufferImpl | TextureImpl, ms: number): void {
  waitedOf(ctx).ms += ms;
  const record = writers.get(resource);
  if (!record) return;
  writers.delete(resource);
  record.readBack++;
  record.waitMs += ms;
}
