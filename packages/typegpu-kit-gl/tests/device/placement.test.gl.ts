import { d, std, tgpu } from 'typegpu';
import { expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot } from './compare.ts';
import { install } from './minimum.ts';

const COUNT = 256;

test('a GPU form whose output a CPU kernel reads back moves to the CPU and stays correct', async () => {
  const root = glRoot();
  try {
    const input = root.createReadonly(d.arrayOf(d.u32, COUNT));
    const doubled = root.createMutable(d.arrayOf(d.u32, COUNT));
    const total = root.createMutable(d.atomic(d.u32));
    const double = tgpu.computeFn({
      workgroupSize: [64],
      in: { id: d.builtin.globalInvocationId },
    })(({ id }) => {
      'use gpu';
      doubled.$[id.x] = input.$[id.x]! * 2;
    });
    // A barrier keeps this kernel on the CPU, so it reads back what `double` wrote
    const sum = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(({
      id,
    }) => {
      'use gpu';
      std.workgroupBarrier();
      std.atomicAdd(total.$, doubled.$[id.x]!);
    });
    const doubling = root.createComputePipeline({ compute: double }).$name('doubling');
    const summing = root.createComputePipeline({ compute: sum }).$name('summing');
    const totals: number[] = [];
    for (let frame = 1; frame <= 16; frame++) {
      input.write(Array.from({ length: COUNT }, () => frame));
      total.write(0);
      doubling.dispatchWorkgroups(COUNT / 64);
      summing.dispatchWorkgroups(COUNT / 64);
      totals.push(await total.read());
    }
    expect(totals).toEqual(Array.from({ length: 16 }, (_, i) => (i + 1) * 2 * COUNT));
    const { placement } = (root.device as GLDevice).stats;
    expect(placement.get('doubling')).toMatch(/^cpu:/);
  } finally {
    root.destroy();
  }
});

const CROWD = 100_000;

/**
 * Adds to or ors into `cells` from `CROWD` invocations a dispatch: into one word when `crowded`,
 * which piles every record onto one texel, or each into its own word.
 */
function crowding(root: ReturnType<typeof glRoot>, kind: 'add' | 'or', crowded: boolean) {
  const cells = root.createMutable(d.arrayOf(d.atomic(d.u32), crowded ? 4 : CROWD));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      if (id.x >= CROWD) return;
      const cell = crowded ? d.u32(0) : id.x;
      if (kind === 'add') std.atomicAdd(cells.$[cell], 1);
      else std.atomicOr(cells.$[cell], d.u32(1) << (id.x % 32));
    },
  );
  const label = `${kind} ${crowded ? 'crowded' : 'spread'}`;
  const pipeline = root.createComputePipeline({ compute: kernel }).$name(label);
  return { cells, pipeline, label };
}

/** Dispatches with a task between each, as frames would, so samples arrive; reads every third. */
async function frames(root: ReturnType<typeof glRoot>, kind: 'add' | 'or', crowded: boolean) {
  const { cells, pipeline, label } = crowding(root, kind, crowded);
  const errors: string[] = [];
  root.device.addEventListener('uncapturederror', (e) =>
    errors.push((e as GPUUncapturedErrorEvent).error.message.slice(0, 300)),
  );
  const words: number[] = [];
  for (let frame = 1; frame <= 24; frame++) {
    pipeline.dispatchWorkgroups(Math.ceil(CROWD / 64));
    await new Promise((resolve) => setTimeout(resolve, 20));
    if (frame % 3 === 0) words.push((await cells.read())[crowded ? 0 : 7] as number);
  }
  const { stats } = root.device as GLDevice;
  return { words, label, placement: stats.placement.get(label) ?? '', stats, errors };
}

test.each(['add', 'or'] as const)(
  "a GPU form that piles '%s' records onto one word moves to the CPU, and stays correct",
  async (kind) => {
    const root = glRoot();
    try {
      const { words, label, placement, stats, errors } = await frames(root, kind, true);
      expect(errors).toEqual([]);
      // A counter gains 100 000 a dispatch, wrapping; ors of every bit fill the word at once
      const expected = Array.from({ length: 8 }, (_, i) =>
        kind === 'add' ? ((i + 1) * 3 * CROWD) >>> 0 : 0xffffffff,
      );
      expect(words).toEqual(expected);
      expect(placement).toMatch(
        /^cpu: its GPU form took [\d.]+ ms a dispatch on the GPU, and a CPU run/,
      );
      expect([...stats.kernels.keys()]).toContain(label);
      expect([...stats.kernels.keys()]).toContain(`${label} (gpu)`);
      expect((root.device as GLDevice).gl.getError()).toBe(0);
    } finally {
      root.destroy();
    }
  },
);

test('the same records spread over many words stay on the GPU', async () => {
  const root = glRoot();
  try {
    const { words, label, placement, stats } = await frames(root, 'add', false);
    expect(words).toEqual(Array.from({ length: 8 }, (_, i) => (i + 1) * 3));
    expect(placement.startsWith('cpu')).toBe(false);
    expect([...stats.kernels.keys()]).not.toContain(label);
  } finally {
    root.destroy();
  }
});

// Fences time a form to within the time between polls, so this one is far past the threshold
test('without timer queries, fences around the draws move a crowded form to the CPU', async () => {
  const undo = install({ minimum: false, hide: ['EXT_disjoint_timer_query_webgl2'] });
  const root = glRoot();
  try {
    expect((root.device as GLDevice).gl.getExtension('EXT_disjoint_timer_query_webgl2')).toBeNull();
    const { words, placement } = await frames(root, 'or', true);
    expect(words).toEqual(Array.from({ length: 8 }, () => 0xffffffff));
    expect(placement).toMatch(
      /^cpu: its GPU form took [\d.]+ ms a dispatch on the GPU, and a CPU run/,
    );
  } finally {
    root.destroy();
    undo();
  }
});
