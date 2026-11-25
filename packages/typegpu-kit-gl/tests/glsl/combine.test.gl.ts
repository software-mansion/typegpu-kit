import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot, roots } from '../device/compare.ts';
import { install } from '../device/minimum.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});

// A draw WebGL refuses leaves its error behind and the targets as they were
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

const gid = { gid: d.builtin.globalInvocationId };
const INVOCATIONS = 300;
const CELLS = 6;

/** Whether `root` ran `label` as a GPU form, throwing with its reason if it ran on the CPU. */
const onGpu = (root: TgpuRoot, label: string) => {
  const { stats } = root.device as unknown as GLDevice;
  const reason = stats.fallbacks.get(label);
  if (reason) throw new Error(`'${label}' ran on the CPU: ${reason}`);
  return [...stats.kernels.keys()].includes(`${label} (gpu)`);
};

/**
 * Values across the whole `u32` range from an invocation index and a dispatch's salt. Their high
 * halves come from four that span it, 0, 1, 0x7fff and 0xffff, so many values share one, and their
 * low halves are scattered, so the least or greatest low half of a cell is seldom that of the
 * values with its least or greatest high half.
 */
const spread = (i: number, salt: number) => {
  'use gpu';
  const mixed = (i * d.u32(0x9e3779b1) + salt * d.u32(0x85ebca6b)) ^ (i >>> 3);
  const highs = [d.u32(0), d.u32(1), d.u32(0x7fff), d.u32(0xffff)];
  return ((highs[(mixed >>> 16) % 4] as number) << 16) | (mixed & 0xffff);
};

type Kind = 'min' | 'max' | 'or' | 'and';

/**
 * Combines into `CELLS` words by `kind` from five dispatches with their own salts, three in one
 * pass and two in later submissions, through a helper at two sites, one of them conditional,
 * beside a write to the invocation's own cell.
 */
async function combined(root: TgpuRoot, kind: Kind, label: string) {
  const layout = tgpu.bindGroupLayout({
    cells: { storage: d.arrayOf(d.atomic(d.u32), CELLS), access: 'mutable' },
    out: { storage: d.arrayOf(d.u32, INVOCATIONS), access: 'mutable' },
    salt: { uniform: d.u32 },
  });
  const apply = (cell: number, value: number) => {
    'use gpu';
    if (kind === 'min') std.atomicMin(layout.$.cells[cell % CELLS], value);
    if (kind === 'max') std.atomicMax(layout.$.cells[cell % CELLS], value);
    if (kind === 'or') std.atomicOr(layout.$.cells[cell % CELLS], d.u32(1) << (value % 32));
    if (kind === 'and') std.atomicAnd(layout.$.cells[cell % CELLS], ~(d.u32(1) << (value % 32)));
  };
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    const i = id.x;
    if (i >= INVOCATIONS) return;
    const salt = layout.$.salt;
    apply(i * 7, spread(i, salt));
    if (i % 3 === 0) apply(i + 1, spread(i + 11, salt));
    layout.$.out[i] = i * 2 + salt;
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name(label);
  // Even cells start where no record reaches them, so their old words must survive
  const start = (c: number, k: number) =>
    kind === 'min'
      ? c % 2 === 0
        ? k
        : 0xffffffff - c
      : kind === 'max'
        ? c % 2 === 0
          ? 0xffffffff - k
          : c
        : kind === 'or'
          ? c * 0x01010101 * k
          : 0xffffffff - c * k;
  const groups = [0, 1, 2, 3, 4].map((k) => {
    const cells = root
      .createBuffer(
        d.arrayOf(d.atomic(d.u32), CELLS),
        Array.from({ length: CELLS }, (_, c) => start(c, k + 1) >>> 0),
      )
      .$usage('storage');
    const out = root.createBuffer(d.arrayOf(d.u32, INVOCATIONS)).$usage('storage');
    return {
      cells,
      out,
      group: root.createBindGroup(layout, {
        cells,
        out,
        salt: root.createUniform(d.u32, 17 * (k + 1)).buffer,
      }),
    };
  });
  const run = (from: number, to: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const { group } of groups.slice(from, to)) {
      pass.setBindGroup(layout, group);
      pass.dispatchWorkgroups(Math.ceil(INVOCATIONS / 64));
    }
    pass.end();
    encoder.submit();
  };
  run(0, 3);
  run(3, 4);
  run(4, 5);
  run(0, 1);
  const read = [];
  for (const { cells, out } of groups)
    read.push({ cells: await cells.read(), out: await out.read() });
  return read;
}

it.each(['min', 'max', 'or', 'and'] as const)(
  "combines by '%s' at shared cells as WebGPU does, over several dispatches",
  async (kind) => {
    const actual = await combined(gl, kind, kind);
    expect(onGpu(gl, kind)).toBe(true);
    expect(actual).toEqual(await combined(gpu, kind, kind));
  },
);

/** Every bit of a word, each from its own invocation, then cleared again but one. */
async function bits(root: TgpuRoot, label: string) {
  const word = root.createMutable(d.arrayOf(d.atomic(d.u32), 2), [0, 0xffffffff]);
  const kernel = tgpu.computeFn({ workgroupSize: [32], in: gid })(({ gid: id }) => {
    'use gpu';
    std.atomicOr(word.$[0], d.u32(1) << id.x);
    if (id.x !== 13) std.atomicAnd(word.$[1], ~(d.u32(1) << id.x));
  });
  root.createComputePipeline({ compute: kernel }).$name(label).dispatchWorkgroups(1);
  return word.read();
}

it('sets and clears every bit of a word', async () => {
  // One buffer combines by one operation: two words of one buffer by two refuses the form
  const actual = await bits(gl, 'bits');
  expect((gl.device as unknown as GLDevice).stats.fallbacks.get('bits')).toContain(
    "which it also combines by 'or'",
  );
  expect(actual).toEqual([0xffffffff, 1 << 13]);
  expect(actual).toEqual(await bits(gpu, 'bits'));
});

/**
 * The least or greatest of values with every pair of high and low halves from 0, 1, 0x7fff and
 * 0xffff, but for the winning high half only low halves that lose to another high half's.
 */
async function extremes(root: TgpuRoot, kind: 'min' | 'max', label: string) {
  const cell = root.createMutable(d.atomic(d.u32), kind === 'min' ? 0xffffffff : 0);
  const kernel = tgpu.computeFn({ workgroupSize: [16], in: gid })(({ gid: id }) => {
    'use gpu';
    const halves = [d.u32(0), d.u32(1), d.u32(0x7fff), d.u32(0xffff)];
    const high = halves[(id.x >>> 2) % 4] as number;
    let low = halves[id.x % 4] as number;
    if (kind === 'min') {
      if (high === 0) low = std.max(low, d.u32(0x7fff));
      std.atomicMin(cell.$, (high << 16) | low);
    } else {
      if (high === 0xffff) low = std.min(low, d.u32(1));
      std.atomicMax(cell.$, (high << 16) | low);
    }
  });
  root.createComputePipeline({ compute: kernel }).$name(label).dispatchWorkgroups(1);
  return cell.read();
}

it('keeps the least and greatest of values across the whole u32 range', async () => {
  const least = await extremes(gl, 'min', 'least');
  const greatest = await extremes(gl, 'max', 'greatest');
  expect(onGpu(gl, 'least') && onGpu(gl, 'greatest')).toBe(true);
  expect([least, greatest]).toEqual([0x7fff, 0xffff0001]);
  expect([least, greatest]).toEqual([
    await extremes(gpu, 'min', 'least'),
    await extremes(gpu, 'max', 'greatest'),
  ]);
});

it('combines on a device of four draw buffers in more rounds, to the same result', async () => {
  const undo = install({ minimum: true });
  const small = glRoot();
  try {
    const { gl: context } = small.device as unknown as GLDevice;
    expect(context.getParameter(context.MAX_DRAW_BUFFERS)).toBe(4);
    for (const kind of ['min', 'max', 'or', 'and'] as const) {
      const actual = await combined(small, kind, `${kind} halves`);
      expect(onGpu(small, `${kind} halves`)).toBe(true);
      expect(actual).toEqual(await combined(gpu, kind, `${kind} halves`));
    }
    expect(await bits(small, 'bits halves')).toEqual([0xffffffff, 1 << 13]);
    expect(context.getError()).toBe(0);
  } finally {
    small.destroy();
    undo();
  }
});
