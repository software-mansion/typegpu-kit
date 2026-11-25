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

test('a crowded form whose input only GL holds stays on the GPU, as a CPU run would read it back', async () => {
  const root = glRoot();
  try {
    const fed = root.createMutable(d.arrayOf(d.u32, CROWD));
    const cells = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
    const feeding = root
      .createComputePipeline({
        compute: tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
          ({ id }) => {
            'use gpu';
            if (id.x >= CROWD) return;
            fed.$[id.x] = d.u32(1);
          },
        ),
      })
      .$name('feeding');
    const adding = root
      .createComputePipeline({
        compute: tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
          ({ id }) => {
            'use gpu';
            if (id.x >= CROWD) return;
            std.atomicAdd(cells.$[0], fed.$[id.x]);
          },
        ),
      })
      .$name('fed crowded');
    const words: number[] = [];
    for (let frame = 1; frame <= 24; frame++) {
      feeding.dispatchWorkgroups(Math.ceil(CROWD / 64));
      adding.dispatchWorkgroups(Math.ceil(CROWD / 64));
      await new Promise((resolve) => setTimeout(resolve, 20));
      if (frame % 3 === 0) words.push((await cells.read())[0] as number);
    }
    const { stats } = root.device as GLDevice;
    expect(words).toEqual(Array.from({ length: 8 }, (_, i) => ((i + 1) * 3 * CROWD) >>> 0));
    expect([...stats.kernels.keys()]).toContain('fed crowded (gpu)');
    expect([...stats.kernels.keys()]).not.toContain('fed crowded');
    expect(stats.placement.get('fed crowded')?.startsWith('cpu')).toBe(false);
  } finally {
    root.destroy();
  }
});

type Gl = WebGL2RenderingContext;

/**
 * Counts the timer queries and fences every WebGL 2 context makes and deletes, and queries begun
 * over another, from the prototype, which `install`'s proxy reaches too. Fails a draw while `fail`
 * is set. Returns its undo.
 */
function watchSamples() {
  const live = { queries: new Set<WebGLQuery>(), fences: new Set<WebGLSync>() };
  const seen = { queries: 0, fences: 0, unknown: 0, overlapping: 0, fail: false };
  let active = false;
  const prototype = WebGL2RenderingContext.prototype;
  const { createQuery, deleteQuery, beginQuery, endQuery, fenceSync, deleteSync, drawArrays } =
    prototype;
  Object.assign(prototype, {
    createQuery(this: Gl) {
      const query = createQuery.call(this);
      if (query) live.queries.add(query);
      seen.queries++;
      return query;
    },
    deleteQuery(this: Gl, query: WebGLQuery | null) {
      if (!query || !live.queries.delete(query)) seen.unknown++;
      deleteQuery.call(this, query);
    },
    beginQuery(this: Gl, target: number, query: WebGLQuery) {
      if (active) seen.overlapping++;
      active = true;
      beginQuery.call(this, target, query);
    },
    endQuery(this: Gl, target: number) {
      active = false;
      endQuery.call(this, target);
    },
    fenceSync(this: Gl, condition: number, flags: number) {
      const fence = fenceSync.call(this, condition, flags);
      if (fence) live.fences.add(fence);
      seen.fences++;
      return fence;
    },
    deleteSync(this: Gl, fence: WebGLSync | null) {
      if (!fence || !live.fences.delete(fence)) seen.unknown++;
      deleteSync.call(this, fence);
    },
    drawArrays(this: Gl, mode: number, first: number, count: number) {
      if (seen.fail) {
        seen.fail = false;
        throw new Error('a draw failed');
      }
      drawArrays.call(this, mode, first, count);
    },
  });
  const undo = () =>
    Object.assign(prototype, {
      createQuery,
      deleteQuery,
      beginQuery,
      endQuery,
      fenceSync,
      deleteSync,
      drawArrays,
    });
  return { live, seen, undo };
}

/** A small form that stays on the GPU, as nothing reads its output back. */
function filling(root: ReturnType<typeof glRoot>) {
  const out = root.createMutable(d.arrayOf(d.u32, COUNT));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      out.$[id.x] = id.x + 1;
    },
  );
  const pipeline = root.createComputePipeline({ compute: kernel }).$name('filling');
  return () => pipeline.dispatchWorkgroups(COUNT / 64);
}

const settle = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

test.each([
  ['timer queries', []],
  ['fences', ['EXT_disjoint_timer_query_webgl2']],
] as const)('a sampled GPU dispatch that throws closes its sample, with %s', async (_, hide) => {
  const undo = install({ minimum: false, hide });
  const { live, seen, undo: unwatch } = watchSamples();
  const root = glRoot();
  try {
    const { gl } = root.device as GLDevice;
    const errors: string[] = [];
    root.device.addEventListener('uncapturederror', (e) =>
      errors.push((e as GPUUncapturedErrorEvent).error.message),
    );
    const dispatch = filling(root);
    // The fifth dispatch is the first sampled
    for (let frame = 1; frame <= 10; frame++) {
      seen.fail = frame === 5;
      dispatch();
      await settle(20);
    }
    await settle(200);
    expect(errors).toEqual(['a draw failed']);
    expect(seen.overlapping).toBe(0);
    expect(seen.unknown).toBe(0);
    expect(gl.getError()).toBe(0);
    if (hide.length > 0) expect(live.fences.size).toBe(0);
    else expect(live.queries.size).toBeLessThanOrEqual(1);
  } finally {
    root.destroy();
    unwatch();
    undo();
  }
});
