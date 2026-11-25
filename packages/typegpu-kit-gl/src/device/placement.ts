/**
 * Where each compute pipeline with a GPU form runs, learned from its last dispatches.
 *
 * A second rule watches what a GPU form costs on the GPU itself, which the CPU never waits for
 * unless it reads the outputs back: a form whose blends pile onto few texels, as atomics on one
 * word do, can take a hundred milliseconds a dispatch where its CPU module takes ten. The device
 * samples a form's GPU time, with `EXT_disjoint_timer_query_webgl2` where the context has it and
 * otherwise with fences around the form's draws, polled at later dispatches. Past `COSTLY_MS`, when
 * its invocations could run in less on the CPU, it tries the CPU module once and keeps whichever
 * costs less. A GPU form
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
/**
 * GPU milliseconds a dispatch of a GPU form may take, in two samples running, before its CPU
 * module is tried: half a 60 Hz frame, which other work on a busy GPU does not lift a small form
 * to. Below it a move saves little, and may bring a readback of the form's inputs every dispatch.
 */
const COSTLY_MS = 8;
/** Dispatches between GPU time samples, once the first two are in: one probe in flight a pipeline. */
const SAMPLE_EVERY = 16;
/**
 * A form's first dispatches, which are not sampled: their GPU time includes the driver's first use
 * of the form's programs and targets, several milliseconds that later dispatches do not pay.
 */
const UNSAMPLED = 4;
/**
 * What a CPU run costs an invocation of a simple kernel, in nanoseconds, as LIMITATIONS gives it: a
 * form is tried only when its GPU time beats such a run by `KEEP_SHARE`, so a busy form of many
 * cheap invocations is not.
 */
const CPU_NS = 40;
/**
 * The most invocations a costly form may have to be tried on the CPU: a million simple ones run
 * in about 11 ms once, where the forms this rule moves take ten times that on the GPU every time.
 */
const MOST_TRIED = 1 << 20;
/** A CPU run stays only below this share of the GPU form's time, and returns above its inverse. */
const KEEP_SHARE = 0.75;

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
  /** Whether GL alone held an input at the last dispatch, which a CPU run would read back. */
  heldNow: boolean;
  /** Why it runs on the CPU: its outputs were read back, or its GPU form cost too much. */
  why: 'readback' | 'costly';
  /** The GPU time of the form's last sampled dispatches, in milliseconds. */
  samples: number[];
  /** The sample in flight, if any. */
  probe: Probe | undefined;
  /** Dispatches since the last sample began. */
  sinceSample: number;
  /** The invocations of the last sampled dispatch. */
  invocations: number;
  /** The GPU time a CPU run competes with, from the samples that moved it. */
  gpuTime: number;
  /** Whether the CPU was tried for cost and lost, so it is not tried again. */
  triedCost: boolean;
  /** Whether the next CPU run is the trial of a move for cost. */
  trial: boolean;
  /** Why the cost rule last kept the form on the GPU, which the readback rule's notes keep. */
  costNote: string;
  costReturns: number;
}

/** A dispatch's GPU time in flight: a timer query, or fences before and after its draws. */
interface Probe {
  readonly query: WebGLQuery | undefined;
  readonly before: WebGLSync | undefined;
  readonly after: WebGLSync | undefined;
  /** When a poll first saw the fence before the draws signalled. */
  seenBefore: number | undefined;
}

/** Each device's records with a probe in flight, and whether a poll is due or done this task. */
const polls = new WeakMap<Ctx, { probing: Set<Record>; ticking: boolean; polled: boolean }>();

function pollsOf(ctx: Ctx) {
  let state = polls.get(ctx);
  if (!state) {
    state = { probing: new Set(), ticking: false, polled: false };
    polls.set(ctx, state);
  }
  return state;
}

/**
 * Polls fences between tasks while any is in flight, as WebGL updates their status only then: the
 * time they give is no finer than the time between polls, which dispatches alone space a frame
 * apart. It stops when no fence is left, so it runs only while a sample is in flight.
 */
function tick(ctx: Ctx): void {
  const state = pollsOf(ctx);
  if (state.ticking || ctx.lost) return;
  state.ticking = true;
  setTimeout(() => {
    state.ticking = false;
    pollProbes(ctx);
    if ([...state.probing].some((record) => record.probe?.after)) tick(ctx);
  }, 0);
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
      heldNow: false,
      why: 'readback',
      samples: [],
      probe: undefined,
      sinceSample: 0,
      invocations: 0,
      gpuTime: 0,
      triedCost: false,
      trial: false,
      costNote: '',
      costReturns: 0,
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
  pollProbes(ctx);
  const record = recordOf(ctx, pipeline, label);
  record.dispatches++;
  const held = heldOnGl(module, groups, record);
  record.heldNow = held !== undefined;
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
  if (record.onCpu) return restart(ctx, record);
  const moves =
    !record.pinned &&
    reads * 2 >= dispatches &&
    cpuInputs * 2 >= dispatches &&
    invocations <= MOST_INVOCATIONS;
  if (moves) {
    record.onCpu = true;
    record.why = 'readback';
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
    ctx.stats.placement.set(label, `gpu: ${why}${record.costNote ? `; ${record.costNote}` : ''}`);
  }
  restart(ctx, record);
}

/** Notes a CPU run of a pipeline that has a GPU form, and moves it back if the CPU costs more. */
export function ranOnCpu(ctx: Ctx, pipeline: object, ms: number): void {
  const record = records.get(pipeline);
  if (!record?.onCpu) return;
  if (record.why === 'costly') return ranCostly(ctx, record, ms);
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

/**
 * Starts sampling a GPU form's dispatch, if this one is sampled: the first two of a pipeline, then
 * one in `SAMPLE_EVERY`, never two in flight. Call around the form's draws, then `endProbe`.
 */
export function beginProbe(
  ctx: Ctx,
  pipeline: object,
  label: string,
  invocations: number,
): Probe | undefined {
  if (!ctx.placement || ctx.lost) return undefined;
  const record = recordOf(ctx, pipeline, label);
  record.sinceSample++;
  if (record.probe || record.triedCost || record.sinceSample <= UNSAMPLED) return undefined;
  if (record.samples.length >= 2 && record.sinceSample < UNSAMPLED + SAMPLE_EVERY) return undefined;
  record.sinceSample = UNSAMPLED;
  record.invocations = invocations;
  const { gl } = ctx;
  const timer = ctx.ext.timerQuery;
  if (timer) {
    const query = gl.createQuery();
    if (!query) return undefined;
    gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
    return { query, before: undefined, after: undefined, seenBefore: undefined };
  }
  const before = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) ?? undefined;
  return { query: undefined, before, after: undefined, seenBefore: undefined };
}

/** Ends a sample `beginProbe` started; `ran` false when the form drew nothing and ran elsewhere. */
export function endProbe(ctx: Ctx, pipeline: object, probe: Probe | undefined, ran: boolean): void {
  if (!probe) return;
  const { gl } = ctx;
  const timer = ctx.ext.timerQuery;
  if (probe.query && timer) gl.endQuery(timer.TIME_ELAPSED_EXT);
  const record = records.get(pipeline);
  if (!ran || !record) {
    discardProbe(ctx, probe);
    return;
  }
  const after = probe.query
    ? undefined
    : (gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) ?? undefined);
  // A fence signals only once the commands before it reach the GPU
  if (after) gl.flush();
  record.probe = { ...probe, after };
  pollsOf(ctx).probing.add(record);
  if (after) tick(ctx);
}

function discardProbe(ctx: Ctx, probe: Probe): void {
  const { gl } = ctx;
  if (probe.query) gl.deleteQuery(probe.query);
  if (probe.before) gl.deleteSync(probe.before);
  if (probe.after) gl.deleteSync(probe.after);
}

/**
 * Reads the samples whose results arrived. A timer query gives the GPU time; fences give the time
 * between the polls that first saw each signalled, which errs by up to the time between polls,
 * as much either way, since each fence signals at some time before the poll that sees it.
 */
function pollProbes(ctx: Ctx): void {
  const state = pollsOf(ctx);
  // Results change only between tasks, and the disjoint flag may cost a round trip to read: a
  // task polls once, however many dispatches it makes
  if (state.polled) return;
  state.polled = true;
  queueMicrotask(() => {
    state.polled = false;
  });
  const now = performance.now();
  if (state.probing.size === 0) return;
  const { gl } = ctx;
  const timer = ctx.ext.timerQuery;
  const disjoint = timer ? (gl.getParameter(timer.GPU_DISJOINT_EXT) as boolean) : false;
  for (const record of state.probing) {
    const probe = record.probe as Probe;
    let ms: number | undefined;
    let done = false;
    if (probe.query) {
      if (disjoint) done = true;
      else if (gl.getQueryParameter(probe.query, gl.QUERY_RESULT_AVAILABLE)) {
        ms = (gl.getQueryParameter(probe.query, gl.QUERY_RESULT) as number) / 1e6;
        done = true;
      }
    } else if (probe.before && probe.after) {
      const signalled = (sync: WebGLSync) =>
        gl.getSyncParameter(sync, gl.SYNC_STATUS) === gl.SIGNALED;
      if (probe.seenBefore === undefined && signalled(probe.before)) probe.seenBefore = now;
      if (probe.seenBefore !== undefined && signalled(probe.after)) {
        ms = now - probe.seenBefore;
        done = true;
      }
    } else done = true;
    if (!done) continue;
    discardProbe(ctx, probe);
    record.probe = undefined;
    state.probing.delete(record);
    if (ms !== undefined) sampled(ctx, record, ms);
  }
}

/** Takes a GPU time sample, and tries the CPU module once two samples running pass `COSTLY_MS`. */
function sampled(ctx: Ctx, record: Record, ms: number): void {
  record.samples.push(ms);
  if (record.samples.length > 2) record.samples.shift();
  if (record.onCpu || record.triedCost || record.samples.length < 2) return;
  const least = Math.min(...record.samples);
  // A form whose inputs GL alone holds would read them back every CPU run, a wait the GPU time
  // does not count
  const cpuMs = record.invocations * CPU_NS * 1e-6;
  if (least < COSTLY_MS || cpuMs >= least * KEEP_SHARE || record.heldNow) return;
  if (record.invocations > MOST_TRIED) {
    ctx.stats.placement.set(
      record.label,
      `gpu: its GPU form took ${least.toFixed(1)} ms a dispatch on the GPU, but ${record.invocations} invocations are too many to try on the CPU`,
    );
    return;
  }
  record.onCpu = true;
  record.why = 'costly';
  record.gpuTime = least;
  record.settling = true;
  restart(ctx, record);
  ctx.stats.placement.set(
    record.label,
    `cpu: trying the CPU, as its GPU form took ${least.toFixed(1)} ms a dispatch on the GPU`,
  );
}

/**
 * Judges a CPU run of a form moved for its GPU cost: the trial keeps the CPU while it costs less
 * than `KEEP_SHARE` of the GPU time, and later runs send it back above the inverse of that share.
 */
function ranCostly(ctx: Ctx, record: Record, ms: number): void {
  const gpu = record.gpuTime.toFixed(1);
  // The first run waits for the GPU to finish the form's last dispatches and reads back what they
  // wrote: the cost being left, not the CPU's, so the trial is the run after it
  if (record.settling) {
    record.settling = false;
    record.trial = true;
    return;
  }
  if (record.trial) {
    record.trial = false;
    if (ms < record.gpuTime * KEEP_SHARE) {
      ctx.stats.placement.set(
        record.label,
        `cpu: its GPU form took ${gpu} ms a dispatch on the GPU, and a CPU run ${ms.toFixed(1)} ms`,
      );
      restart(ctx, record);
      return;
    }
    record.onCpu = false;
    record.triedCost = true;
    record.samples = [];
    record.costNote = `a CPU run took ${ms.toFixed(1)} ms, not less than its GPU form's ${gpu} ms`;
    ctx.stats.placement.set(record.label, `gpu: ${record.costNote}`);
    restart(ctx, record);
    return;
  }
  record.cpuMs += ms;
  if (record.dispatches < WINDOW) return;
  const each = record.cpuMs / record.dispatches;
  if (each > record.gpuTime / KEEP_SHARE) {
    record.onCpu = false;
    record.samples = [];
    record.costReturns++;
    record.triedCost = record.costReturns >= MOST_RETURNS;
    ctx.stats.placement.set(
      record.label,
      `gpu: CPU runs took ${each.toFixed(1)} ms, more than its GPU form's ${gpu} ms${record.triedCost ? ', and it stays there' : ''}`,
    );
  }
  restart(ctx, record);
}
