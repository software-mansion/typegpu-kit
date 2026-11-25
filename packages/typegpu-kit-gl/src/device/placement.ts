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
 * its inputs stop being on the CPU. A kernel whose inputs other GPU forms rewrite before each of
 * its dispatches would read them back instead, so it moves with those forms, and theirs, as one
 * group of a few small forms, judged and sent back as one.
 */

import type { CpuModule } from '../cpu/types.ts';
import { type Bound, bufferEntry } from './binding.ts';
import type { BufferImpl } from './buffer.ts';
import type { Ctx } from './context.ts';
import type { GpuCompute } from './gpuCompute.ts';
import type { TextureImpl } from './texture.ts';
import { disjointFor, mayTime } from './timestamps.ts';

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
 * The most pipelines that move to the CPU together: a form whose outputs the CPU reads back, and
 * the forms that rewrite its inputs on the GPU, which it could not leave behind.
 */
const MOST_GROUPED = 4;
/**
 * Leader windows a group's CPU runs are judged over: moving several forms shifts the waits of the
 * forms around them for a while, which one window would mistake for the group's own.
 */
const GROUP_WINDOWS = 3;
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
/**
 * Milliseconds a sample may stay in flight before it is dropped unread: a fence a driver never
 * signals, or a query result a context never makes available, would otherwise be polled forever.
 */
const PROBE_MS = 2000;
/**
 * Milliseconds from a sample's start to the first poll of samples in flight that does not wait
 * for a dispatch, which dispatches every frame make first; each such poll doubles the wait, to
 * `IDLE_MS` at most, until no sample is in flight.
 */
const FIRST_IDLE_MS = 50;
const IDLE_MS = 400;

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
  /** GPU dispatches it has made, which number what it wrote. */
  written: number;
  /** The last of those that `readBack` counted. */
  counted: number;
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
  /** The last input GL rewrote before a dispatch, for the stats. */
  held: string;
  /** Whether GL rewrote an input before the last dispatch, which a CPU run would read back. */
  heldNow: boolean;
  /** The GPU form's write each GL-newer buffer held when this pipeline last dispatched over it. */
  readonly seen: WeakMap<BufferImpl, Writer>;
  /** The invocations of the last dispatch. */
  upcoming: number;
  /** The GPU forms that rewrote its inputs before its dispatches, in this window and the last. */
  blockers: Set<Record>;
  lastBlockers: Set<Record>;
  /** Whether GL rewrote an input of it that no known GPU form wrote, this window or the last. */
  loose: boolean;
  lastLoose: boolean;
  /** The pipelines it moved to the CPU with, if it did. */
  group: Group | undefined;
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

/**
 * Pipelines on the CPU together: one whose outputs the CPU read back, the leader, and the GPU forms
 * that rewrote its inputs, transitively, which a CPU run of it alone would read back every time.
 * They are judged as one, at the leader's windows: their CPU runs against their GPU forms' issue
 * and the device's readback waits before and after.
 */
interface Group {
  readonly leader: Record;
  readonly members: ReadonlySet<Record>;
  /** CPU milliseconds of every member since judging began, the leader's dispatches and windows. */
  cpuMs: number;
  dispatches: number;
  windows: number;
  /** The device's readback wait when judging began. */
  waitAt: number;
  /** At the move: the members' GPU issue, and the device's readback wait, per leader dispatch. */
  readonly gpuIssue: number;
  readonly waitBefore: number;
  /** Just moved: the first window reads back what GL held, so it is not judged. */
  settling: boolean;
}

/** A dispatch's GPU time in flight: a timer query, or fences before and after its draws. */
interface Probe {
  readonly query: WebGLQuery | undefined;
  readonly before: WebGLSync | undefined;
  readonly after: WebGLSync | undefined;
  /** When a poll first saw the fence before the draws signalled. */
  seenBefore: number | undefined;
  /** When the draws it times were issued. */
  ended: number;
}

interface Polls {
  readonly probing: Set<Record>;
  /** Whether this task polled, as results change only between tasks. */
  polled: boolean;
  /** The timeout that polls fences in flight at the next task, if one is set. */
  tick: ReturnType<typeof setTimeout> | undefined;
  /** The timeout that polls samples in flight once dispatches stop, if one is set. */
  idle: ReturnType<typeof setTimeout> | undefined;
  /** Milliseconds until that timeout, which doubles at each poll it makes. */
  wait: number;
}

/** Each device's records with a probe in flight, and the timeouts that poll them. */
const polls = new WeakMap<Ctx, Polls>();

function pollsOf(ctx: Ctx): Polls {
  let state = polls.get(ctx);
  if (!state) {
    state = { probing: new Set(), polled: false, tick: undefined, idle: undefined, wait: 0 };
    polls.set(ctx, state);
  }
  return state;
}

/**
 * Sets the timeouts that poll samples in flight, and clears them when none is. Fences are polled
 * at every task while one is in flight, as WebGL updates their status only between tasks and the
 * time they give is no finer than the time between polls. Timer queries give their own time, so
 * dispatches poll them, and one timeout reads what is left when dispatches stop, sooner first and
 * then backing off to `IDLE_MS`, until `PROBE_MS` drops it. No timeout is pending while no sample
 * is in flight, so a page that stops dispatching is left alone.
 */
function schedule(ctx: Ctx, idle = false): void {
  const state = pollsOf(ctx);
  if (state.probing.size === 0 || ctx.lost) {
    if (state.idle !== undefined) clearTimeout(state.idle);
    if (state.tick !== undefined) clearTimeout(state.tick);
    state.idle = state.tick = undefined;
    return;
  }
  // A dispatch's poll leaves the timeout as it is: only a page that stops dispatching meets it
  if (state.idle !== undefined) return pollFences(ctx, state);
  state.wait = idle ? Math.min(IDLE_MS, state.wait * 2) : FIRST_IDLE_MS;
  state.idle = setTimeout(() => {
    state.idle = undefined;
    pollProbes(ctx, true);
  }, state.wait);
  pollFences(ctx, state);
}

/** Sets the timeout that polls fences in flight at the next task, if any is in flight. */
function pollFences(ctx: Ctx, state: Polls): void {
  const fences = [...state.probing].some((record) => record.probe?.after);
  if (fences && state.tick === undefined) {
    state.tick = setTimeout(() => {
      state.tick = undefined;
      pollProbes(ctx);
    }, 0);
  }
}

/** Deletes every sample in flight on the device, as it is destroyed. */
export function discardProbes(ctx: Ctx): void {
  const state = pollsOf(ctx);
  for (const record of state.probing) {
    if (record.probe) discardProbe(ctx, record.probe);
    record.probe = undefined;
  }
  state.probing.clear();
  schedule(ctx);
}

const records = new WeakMap<object, Record>();
/** Milliseconds every readback of a device has waited. */
const waits = new WeakMap<Ctx, { ms: number }>();

function waitedOf(ctx: Ctx): { ms: number } {
  let waited = waits.get(ctx);
  if (!waited) waits.set(ctx, (waited = { ms: 0 }));
  return waited;
}
/** The pipeline whose GPU form last wrote each resource, and which of its dispatches did. */
interface Writer {
  readonly record: Record;
  readonly dispatch: number;
}
const writers = new WeakMap<BufferImpl | TextureImpl, Writer>();

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
      written: 0,
      counted: 0,
      waitMs: 0,
      gpuMs: 0,
      cpuInputs: 0,
      cpuMs: 0,
      waitAt: waitedOf(ctx).ms,
      gpuIssue: 0,
      waitBefore: 0,
      held: '',
      heldNow: false,
      seen: new WeakMap(),
      upcoming: 0,
      blockers: new Set(),
      lastBlockers: new Set(),
      loose: false,
      lastLoose: false,
      group: undefined,
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
  record.lastBlockers = record.blockers;
  record.blockers = new Set();
  record.lastLoose = record.loose;
  record.loose = false;
}

/**
 * The label of a resource the dispatch binds that GL rewrote since this pipeline's last dispatch,
 * which a CPU run would read back each time, or undefined when there is none. A buffer GL holds
 * that no GPU form wrote again costs a CPU run one readback in all, after which it stays current
 * on the CPU, so it counts only the first time this pipeline sees it, or when no GPU form is known
 * to have written it. Only GPU forms leave a buffer with GL; a texture draws and copies write as
 * well, so one GL holds counts at every dispatch. What the pipeline's own GPU form wrote last
 * does not count: on the CPU, the pipeline would have written it there.
 */
function heldOnGl(
  module: CpuModule,
  groups: readonly (Bound | undefined)[],
  record: Record,
): string | undefined {
  let held: string | undefined;
  const rewritten = (buffer: BufferImpl) => {
    const writer = writers.get(buffer);
    if (writer?.record === record) return false;
    // What a pipeline of its group wrote on the GPU before the move is read back once, settling
    const grouped = writer !== undefined && record.group?.members.has(writer.record) === true;
    const again = !grouped && (writer === undefined || record.seen.get(buffer) !== writer);
    if (writer) record.seen.set(buffer, writer);
    if (again && writer) record.blockers.add(writer.record);
    else if (again) record.loose = true;
    return again;
  };
  for (const use of module.bindings) {
    if (use.kind === 'sampler') continue;
    if (use.kind === 'buffer') {
      const buffer = bufferEntry(groups[use.group], use.binding)?.buffer;
      if (buffer?.gpuNewer && rewritten(buffer)) {
        held ??= buffer.label || `@group(${use.group}) @binding(${use.binding})`;
      }
      continue;
    }
    const entry = groups[use.group]?.group.entries.get(use.binding);
    if (entry?.kind === 'texture') {
      const { texture } = entry.view;
      // Draws and copies write textures too and leave no writer, so one GL holds always counts
      if (texture.written && texture.home === 'gl' && writers.get(texture)?.record !== record) {
        held ??= texture.label || 'a texture';
        record.loose = true;
      }
    }
  }
  return held;
}

/**
 * Whether the dispatch should run its GPU form. A move for cost takes this dispatch's size and
 * inputs: samples that arrive now judge it, and a form on the CPU for cost returns to the GPU
 * before a run past `MOST_TRIED` invocations or one that would read back what GL rewrote.
 */
export function runsOnGpu(
  ctx: Ctx,
  pipeline: object,
  label: string,
  module: CpuModule,
  groups: readonly (Bound | undefined)[],
  invocations: number,
): boolean {
  if (!ctx.placement) return true;
  const record = recordOf(ctx, pipeline, label);
  record.dispatches++;
  const held = heldOnGl(module, groups, record);
  record.heldNow = held !== undefined;
  if (held === undefined) record.cpuInputs++;
  else record.held = held;
  record.upcoming = invocations;
  // A GPU form outside the group rewrote what a member reads: the group goes back as one
  if (record.group && held !== undefined) {
    breakGroup(ctx, record.group, `a GPU form outside its group rewrote '${held}'`);
  }
  pollProbes(ctx);
  if (record.onCpu && record.why === 'costly') {
    const why = costlyRefusal(record);
    if (why !== undefined) {
      backFromCost(ctx, record);
      ctx.stats.placement.set(record.label, `gpu: ${why}`);
    }
  }
  return !record.onCpu;
}

/** Why a CPU run of the next dispatch cannot try or keep a costly form, if it cannot. */
function costlyRefusal(record: Record): string | undefined {
  if (record.upcoming > MOST_TRIED) {
    return `${record.upcoming} invocations are too many to try on the CPU`;
  }
  if (record.heldNow) {
    return `GL rewrote its input '${record.held}', which a CPU run would read back`;
  }
  return undefined;
}

/**
 * Sends a form moved for cost back to the GPU before a dispatch the CPU cannot take. One that ran
 * on the CPU counts as a return, so a form whose dispatches alternate settles on the GPU.
 */
function backFromCost(ctx: Ctx, record: Record): void {
  if (!record.settling) {
    record.costReturns++;
    record.triedCost = record.costReturns >= MOST_RETURNS;
  }
  record.onCpu = false;
  record.settling = false;
  record.trial = false;
  record.samples = [];
  restart(ctx, record);
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
  replaced: readonly BufferImpl[] = [],
): void {
  const record = recordOf(ctx, pipeline, label);
  record.gpuMs += ms;
  const writer = { record, dispatch: ++record.written };
  for (const buffer of replaced) writers.set(buffer, writer);
  for (const { target } of gpu.passes) {
    if (target.kind === 'buffer') {
      const buffer = bufferEntry(groups[target.group], target.binding)?.buffer;
      if (buffer) writers.set(buffer, writer);
    } else {
      const entry = groups[target.group]?.group.entries.get(target.binding);
      if (entry?.kind === 'texture') writers.set(entry.view.texture, writer);
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
  } else if (
    !record.pinned &&
    reads * 2 >= dispatches &&
    cpuInputs * 2 < dispatches &&
    moveGroup(ctx, record, reads)
  ) {
    // The forms that rewrote its inputs moved with it
  } else if (!record.pinned) {
    const why =
      reads * 2 < dispatches
        ? `its outputs were read back after ${reads} of ${dispatches} dispatches`
        : cpuInputs * 2 < dispatches
          ? `GL rewrote its input '${record.held}' before ${dispatches - cpuInputs} of ${dispatches} dispatches`
          : `${invocations} invocations are too many to try on the CPU`;
    ctx.stats.placement.set(label, `gpu: ${why}${record.costNote ? `; ${record.costNote}` : ''}`);
  }
  restart(ctx, record);
}

/**
 * The pipelines that must move with `leader` for its CPU runs to stop reading back what GL wrote:
 * the GPU forms that rewrote its inputs, theirs, and so on, or undefined when that cannot be. Each
 * must be on the GPU, free to move, small enough to try, and blocked only by known GPU forms.
 */
function groupOf(leader: Record): Set<Record> | undefined {
  const members = new Set<Record>([leader]);
  const queue = [leader];
  for (let member = queue.pop(); member; member = queue.pop()) {
    if (member.loose || member.lastLoose) return undefined;
    if (member.pinned || member.onCpu || member.group || member.upcoming > MOST_INVOCATIONS) {
      return undefined;
    }
    for (const blocker of [...member.blockers, ...member.lastBlockers]) {
      if (members.has(blocker)) continue;
      if (members.size === MOST_GROUPED) return undefined;
      members.add(blocker);
      queue.push(blocker);
    }
  }
  return members.size > 1 ? members : undefined;
}

/** Moves `leader` and the forms that rewrite its inputs to the CPU together, if they can be. */
function moveGroup(ctx: Ctx, leader: Record, reads: number): boolean {
  const members = groupOf(leader);
  if (!members) return false;
  const perDispatch = (member: Record) => member.gpuMs / Math.max(1, member.dispatches);
  const group: Group = {
    leader,
    members,
    cpuMs: 0,
    dispatches: 0,
    windows: 0,
    waitAt: 0,
    gpuIssue: [...members].reduce((sum, member) => sum + perDispatch(member), 0),
    waitBefore: (waitedOf(ctx).ms - leader.waitAt) / leader.dispatches,
    settling: true,
  };
  const others = [...members].filter((member) => member !== leader);
  const names = others.map((member) => `'${member.label}'`).join(', ');
  ctx.stats.placement.set(
    leader.label,
    `cpu: the CPU read back its outputs after ${reads} of ${leader.dispatches} dispatches, waiting ${(leader.waitMs / leader.dispatches).toFixed(2)} ms each, and ${names} rewrote its inputs, so they moved with it`,
  );
  for (const member of members) {
    member.onCpu = true;
    member.why = 'readback';
    member.settling = false;
    member.group = group;
    if (member !== leader) {
      restart(ctx, member);
      ctx.stats.placement.set(
        member.label,
        `cpu: it rewrote an input of '${leader.label}', whose outputs the CPU read back, so it moved with it`,
      );
    }
  }
  return true;
}

/**
 * Sends every pipeline of a group back to the GPU: for good when its CPU runs cost more than they
 * saved, or, when a GPU form outside it rewrote a member's input, counting a return for each.
 */
function breakGroup(ctx: Ctx, group: Group, why: string, lost = false): void {
  for (const member of group.members) {
    member.onCpu = false;
    member.group = undefined;
    if (lost) member.pinned = true;
    else {
      member.returns++;
      member.pinned = member.returns >= MOST_RETURNS;
    }
    restart(ctx, member);
    const stays = member.pinned && !lost ? ', and it stays there' : '';
    ctx.stats.placement.set(member.label, `gpu: ${why}${stays}`);
  }
}

/**
 * Notes a CPU run of a member of a group, judged at the leader's windows as the single rule judges
 * one pipeline: every member's CPU time against their GPU issue and the readback waits saved.
 */
function ranGrouped(ctx: Ctx, record: Record, group: Group, ms: number): void {
  group.cpuMs += ms;
  if (record !== group.leader || record.dispatches < WINDOW) return;
  if (group.settling) {
    group.settling = false;
    return judgeFrom(ctx, record, group);
  }
  group.dispatches += record.dispatches;
  restart(ctx, record);
  if (++group.windows < GROUP_WINDOWS) return;
  const { dispatches } = group;
  const extra = group.cpuMs / dispatches - group.gpuIssue;
  const saved = group.waitBefore - (waitedOf(ctx).ms - group.waitAt) / dispatches;
  const cost = `CPU runs of its group cost ${extra.toFixed(2)} ms more than their GPU forms, and saved ${saved.toFixed(2)} ms of readback waits`;
  if (extra > CHEAP_MS && extra > saved) return breakGroup(ctx, group, cost, true);
  ctx.stats.placement.set(record.label, `cpu: ${cost}`);
  judgeFrom(ctx, record, group);
}

/** Starts the group's next judging span at the leader's next window. */
function judgeFrom(ctx: Ctx, leader: Record, group: Group): void {
  restart(ctx, leader);
  group.cpuMs = 0;
  group.dispatches = 0;
  group.windows = 0;
  group.waitAt = waitedOf(ctx).ms;
}

/** Notes a CPU run of a pipeline that has a GPU form, and moves it back if the CPU costs more. */
export function ranOnCpu(ctx: Ctx, pipeline: object, ms: number): void {
  const record = records.get(pipeline);
  if (!record?.onCpu) return;
  // The form's first dispatches back upload what CPU runs wrote, which a sample would count
  record.sinceSample = 0;
  if (record.why === 'costly') return ranCostly(ctx, record, ms);
  if (record.group) return ranGrouped(ctx, record, record.group, ms);
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

/**
 * Notes that the CPU read `resource` back from GL, which charges the GPU form that wrote it: the
 * wait for each resource, and the dispatch once, however many of its outputs were read.
 */
export function readBack(ctx: Ctx, resource: BufferImpl | TextureImpl, ms: number): void {
  waitedOf(ctx).ms += ms;
  const writer = writers.get(resource);
  if (!writer) return;
  writers.delete(resource);
  const { record } = writer;
  if (record.counted !== writer.dispatch) record.readBack++;
  record.counted = writer.dispatch;
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
  if (timer && mayTime(ctx)) {
    const query = gl.createQuery();
    if (!query) return undefined;
    gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
    return { query, before: undefined, after: undefined, seenBefore: undefined, ended: 0 };
  }
  const before = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) ?? undefined;
  return { query: undefined, before, after: undefined, seenBefore: undefined, ended: 0 };
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
  // A fence signals, and a query's result arrives, only once the commands before it reach the GPU:
  // a device that idles after this dispatch would otherwise leave the sample in flight
  gl.flush();
  record.probe = { ...probe, after, ended: performance.now() };
  pollsOf(ctx).probing.add(record);
  schedule(ctx);
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
function pollProbes(ctx: Ctx, idle = false): void {
  const state = pollsOf(ctx);
  // Results change only between tasks, and the disjoint flag may cost a round trip to read: a
  // task polls once, however many dispatches it makes
  if (state.polled) return;
  state.polled = true;
  queueMicrotask(() => {
    state.polled = false;
  });
  const now = performance.now();
  if (state.probing.size === 0) return schedule(ctx, idle);
  const { gl } = ctx;
  const timer = ctx.ext.timerQuery;
  const disjoint = timer ? disjointFor(ctx, state) : false;
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
    if (!done && now - probe.ended > PROBE_MS) done = true;
    if (!done) continue;
    discardProbe(ctx, probe);
    record.probe = undefined;
    state.probing.delete(record);
    if (ms !== undefined) sampled(ctx, record, ms);
  }
  schedule(ctx, idle);
}

/** Takes a GPU time sample, and tries the CPU module once two samples running pass `COSTLY_MS`. */
function sampled(ctx: Ctx, record: Record, ms: number): void {
  record.samples.push(ms);
  if (record.samples.length > 2) record.samples.shift();
  if (record.onCpu || record.triedCost || record.samples.length < 2) return;
  const least = Math.min(...record.samples);
  // A form whose inputs GL rewrites would read them back every CPU run, a wait the GPU time
  // does not count
  const cpuMs = record.invocations * CPU_NS * 1e-6;
  if (least < COSTLY_MS || cpuMs >= least * KEEP_SHARE || record.heldNow) return;
  const most = Math.max(record.invocations, record.upcoming);
  if (most > MOST_TRIED) {
    ctx.stats.placement.set(
      record.label,
      `gpu: its GPU form took ${least.toFixed(1)} ms a dispatch on the GPU, but ${most} invocations are too many to try on the CPU`,
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
