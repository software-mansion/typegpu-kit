import { afterEach, expect, test, vi } from 'vitest';
import type { CpuModule } from '../../src/cpu/types.ts';
import type { Bound } from '../../src/device/binding.ts';
import type { BufferImpl } from '../../src/device/buffer.ts';
import { createStats, type Ctx } from '../../src/device/context.ts';
import type { GpuCompute } from '../../src/device/gpuCompute.ts';
import {
  beginProbe,
  endProbe,
  ranOnCpu,
  ranOnGpu,
  readBack,
  runsOnGpu,
} from '../../src/device/placement.ts';

/**
 * A context that times every sampled dispatch at `gpuMs`, with timer queries or with fences, and
 * counts the GL objects samples create and delete.
 */
function fakeCtx(options: { timer: boolean; gpuMs: number; signals?: boolean }) {
  const made = { queries: [] as object[], fences: [] as object[] };
  const deleted = { queries: [] as object[], fences: [] as object[] };
  const polls = { fences: 0 };
  const gl = {
    QUERY_RESULT: 1,
    QUERY_RESULT_AVAILABLE: 2,
    SYNC_GPU_COMMANDS_COMPLETE: 3,
    SYNC_STATUS: 4,
    SIGNALED: 5,
    UNSIGNALED: 6,
    createQuery: () => {
      const query = {};
      made.queries.push(query);
      return query;
    },
    deleteQuery: (query: object) => deleted.queries.push(query),
    beginQuery: () => {},
    endQuery: () => {},
    getParameter: () => false,
    getQueryParameter: (_: object, name: number) => (name === 2 ? true : options.gpuMs * 1e6),
    fenceSync: () => {
      const fence = {};
      made.fences.push(fence);
      return fence;
    },
    deleteSync: (fence: object) => deleted.fences.push(fence),
    getSyncParameter: () => {
      polls.fences++;
      return options.signals === false ? 6 : 5;
    },
    flush: () => {},
  };
  const ctx = {
    gl,
    ext: { timerQuery: options.timer ? { TIME_ELAPSED_EXT: 7, GPU_DISJOINT_EXT: 8 } : null },
    stats: createStats(),
    placement: true,
    lost: false,
  } as unknown as Ctx;
  return { ctx, made, deleted, polls };
}

const buffer = (label: string) => ({ label, gpuNewer: false }) as BufferImpl;

/** A pipeline that reads one buffer and writes three, dispatched as `compute.ts` does. */
function kernel(ctx: Ctx) {
  const pipeline = {};
  const input = buffer('input');
  const outputs = [buffer('a'), buffer('b'), buffer('c')];
  const entries = new Map(
    [input, ...outputs].map((each, binding) => [
      binding,
      { kind: 'buffer' as const, buffer: each, offset: 0, size: 16 },
    ]),
  );
  const groups: Bound[] = [
    { group: { layout: { entries: [], dynamic: [] }, entries }, offsets: undefined },
  ];
  const module = {
    bindings: [input, ...outputs].map((_, binding) => ({
      group: 0,
      binding,
      kind: 'buffer',
      reads: binding === 0,
      writes: binding > 0,
    })),
  } as unknown as CpuModule;
  const gpu = {
    passes: outputs.map((_, i) => ({ target: { kind: 'buffer', group: 0, binding: i + 1 } })),
  } as unknown as GpuCompute;
  const run = (invocations: number, cpuMs = 1): 'gpu' | 'cpu' => {
    if (runsOnGpu(ctx, pipeline, 'kernel', module, groups, invocations)) {
      const probe = beginProbe(ctx, pipeline, 'kernel', invocations);
      endProbe(ctx, pipeline, probe, true);
      ranOnGpu(ctx, pipeline, 'kernel', gpu, groups, 0.1, invocations);
      return 'gpu';
    }
    ranOnCpu(ctx, pipeline, cpuMs);
    return 'cpu';
  };
  return { ctx, input, outputs, run };
}

/** Lets a task pass, as frames do, so samples arrive and the next dispatch polls again. */
const task = () => new Promise((resolve) => setTimeout(resolve, 0));

afterEach(() => {
  vi.restoreAllMocks();
});

const SAMPLED = 100_000;

/** Six dispatches of `SAMPLED` invocations whose fifth and sixth sample 10 ms on the GPU. */
async function costly(timer: boolean) {
  const fake = fakeCtx({ timer, gpuMs: 10 });
  const { input, run } = kernel(fake.ctx);
  for (let i = 0; i < 6; i++) {
    expect(run(SAMPLED)).toBe('gpu');
    await task();
  }
  return { ...fake, input, run };
}

test('two costly samples move the next dispatch to the CPU, and its trial after it', async () => {
  const { run } = await costly(true);
  expect(run(SAMPLED)).toBe('cpu');
  await task();
  expect(run(SAMPLED)).toBe('cpu');
});

test('a costly form whose next dispatch outgrows the CPU trial stays on the GPU', async () => {
  const { run, ctx } = await costly(true);
  expect(run(2 ** 21)).toBe('gpu');
  expect(ctx.stats.placement.get('kernel')).toMatch(/^gpu: .*2097152 invocations are too many/);
});

test('a costly form whose next dispatch binds an input only GL holds stays on the GPU', async () => {
  const { run, input } = await costly(true);
  input.gpuNewer = true;
  expect(run(SAMPLED)).toBe('gpu');
});

test("a costly form's judged trial is capped at the trial's size", async () => {
  const { run } = await costly(true);
  expect(run(SAMPLED)).toBe('cpu');
  await task();
  expect(run(2 ** 21)).toBe('gpu');
});

test('a readback of three outputs of one dispatch counts that dispatch once', async () => {
  const { ctx, run, outputs } = kernel(fakeCtx({ timer: true, gpuMs: 0.1 }).ctx);
  expect(run(1000)).toBe('gpu');
  for (const output of outputs) readBack(ctx, output, 1);
  for (let i = 0; i < 5; i++) {
    expect(run(1000)).toBe('gpu');
    await task();
  }
  expect(ctx.stats.placement.get('kernel')).toMatch(
    /^gpu: its outputs were read back after 1 of 6 dispatches/,
  );
  expect(run(1000)).toBe('gpu');
});

test('readbacks of every output of every dispatch count each dispatch once and every wait', async () => {
  const { ctx, run, outputs } = kernel(fakeCtx({ timer: true, gpuMs: 0.1 }).ctx);
  // The sixth dispatch judges the window before its own outputs are read
  for (let i = 0; i < 6; i++) {
    expect(run(1000)).toBe('gpu');
    for (const output of outputs) readBack(ctx, output, 1);
    await task();
  }
  expect(ctx.stats.placement.get('kernel')).toBe(
    'cpu: the CPU read back its outputs after 5 of 6 dispatches, waiting 2.50 ms each, and its inputs were on the CPU',
  );
});

test('fences that never signal are deleted once, after a bounded wait, and polling stops', async () => {
  let now = 0;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  const { ctx, made, deleted, polls, run } = (() => {
    const fake = fakeCtx({ timer: false, gpuMs: 0, signals: false });
    return { ...fake, run: kernel(fake.ctx).run };
  })();
  for (let i = 0; i < 5; i++) {
    run(SAMPLED);
    now += 16;
    await task();
  }
  expect(made.fences).toHaveLength(2);
  for (let i = 0; i < 5; i++) await task();
  expect(deleted.fences).toEqual([]);
  expect(polls.fences).toBeGreaterThan(0);

  now += 60_000;
  for (let i = 0; i < 5; i++) await task();
  expect(new Set(deleted.fences)).toEqual(new Set(made.fences));
  expect(deleted.fences).toHaveLength(2);
  const polled = polls.fences;
  for (let i = 0; i < 5; i++) await task();
  expect(polls.fences).toBe(polled);
  expect(ctx.lost).toBe(false);
});

test('a form back from the CPU warms up for four dispatches before it samples again', async () => {
  const { ctx, made, run } = await costly(true);
  expect(run(SAMPLED)).toBe('cpu');
  await task();
  expect(run(SAMPLED, 1)).toBe('cpu');
  await task();
  // CPU runs past the GPU time over the share's inverse send it back after a window
  for (let i = 0; i < 6; i++) {
    expect(run(SAMPLED, 20)).toBe('cpu');
    await task();
  }
  const sampled = made.queries.length;
  expect(ctx.stats.placement.get('kernel')).toMatch(/^gpu: CPU runs took 20.0 ms/);
  for (let i = 0; i < 4; i++) {
    expect(run(SAMPLED)).toBe('gpu');
    await task();
  }
  expect(made.queries).toHaveLength(sampled);
  expect(run(SAMPLED)).toBe('gpu');
  expect(made.queries).toHaveLength(sampled + 1);
});
