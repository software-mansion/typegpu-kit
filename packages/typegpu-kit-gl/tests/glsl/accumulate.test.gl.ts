import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { mostAdded } from '../../src/glsl/computeModule.ts';
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

const stats = () => (gl.device as unknown as GLDevice).stats;

/** Whether the fallback ran `label` as a GPU form, throwing with its reason if it ran on the CPU. */
const onGpu = (label: string) => {
  const reason = stats().fallbacks.get(label);
  if (reason) throw new Error(`'${label}' ran on the CPU: ${reason}`);
  return [...stats().kernels.keys()].includes(`${label} (gpu)`);
};

/**
 * Adds to eight counters through a helper at two sites, one of them conditional, from five
 * dispatches with amounts of their own, three in a pass and two in later submissions, beside a
 * write to the invocation's own cell.
 */
async function counted(root: TgpuRoot, invocations: number, label: string) {
  const layout = tgpu.bindGroupLayout({
    counts: { storage: d.arrayOf(d.atomic(d.u32), 8), access: 'mutable' },
    signed: { storage: d.arrayOf(d.atomic(d.i32), 4), access: 'mutable' },
    out: { storage: d.arrayOf(d.u32, invocations), access: 'mutable' },
    amount: { uniform: d.u32 },
  });
  const count = (cell: number, amount: number) => {
    'use gpu';
    std.atomicAdd(layout.$.counts[cell % 8], amount);
  };
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    const i = id.x;
    if (i >= invocations) return;
    count(i * 7, layout.$.amount + i);
    if (i % 3 === 0) count(i + 1, 0x01010101 * (i & 3));
    // Wraps past 2^32 and below 0, and carries between bytes
    std.atomicSub(layout.$.counts[i % 5], 0x7fffffff);
    std.atomicAdd(layout.$.signed[i % 4], d.i32(i) - 300);
    layout.$.out[i] = i * 2 + layout.$.amount;
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name(label);
  const groups = [0, 1, 2, 3, 4].map((k) => {
    const counts = root
      .createBuffer(
        d.arrayOf(d.atomic(d.u32), 8),
        Array.from({ length: 8 }, (_, c) => (c * 0x10203 * (k + 1)) >>> 0),
      )
      .$usage('storage');
    const signed = root
      .createBuffer(d.arrayOf(d.atomic(d.i32), 4), [k, -k, 5, -2_000_000_000])
      .$usage('storage');
    const out = root.createBuffer(d.arrayOf(d.u32, invocations)).$usage('storage');
    return {
      counts,
      signed,
      out,
      group: root.createBindGroup(layout, {
        counts,
        signed,
        out,
        amount: root.createUniform(d.u32, 0xfe * (k + 1)).buffer,
      }),
    };
  });
  const run = (from: number, to: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const { group } of groups.slice(from, to)) {
      pass.setBindGroup(layout, group);
      pass.dispatchWorkgroups(Math.ceil(invocations / 64));
    }
    pass.end();
    encoder.submit();
  };
  run(0, 3);
  run(3, 4);
  run(4, 5);
  run(0, 1);
  const read = [];
  for (const { counts, signed, out } of groups) {
    read.push({ counts: await counts.read(), signed: await signed.read(), out: await out.read() });
  }
  return read;
}

it('adds through helpers at shared cells as WebGPU does, over several dispatches', async () => {
  const actual = await counted(gl, 300, 'counted');
  expect(onGpu('counted')).toBe(true);
  expect(actual).toEqual(await counted(gpu, 300, 'counted'));
});

/** The most records the device of `root` sums exactly into one cell: nibbles with eight targets. */
function mostOf(root: TgpuRoot) {
  const { gl: context } = root.device as unknown as GLDevice;
  const targets = Math.min(
    context.getParameter(context.MAX_DRAW_BUFFERS) as number,
    context.getParameter(context.MAX_COLOR_ATTACHMENTS) as number,
  );
  return mostAdded(targets >= 8 ? 8 : 4);
}

it('runs a dispatch on the CPU where a cell could take more records than a sum keeps exact', async () => {
  // Three additions to the counts an invocation, one invocation past the bound's third; the
  // device counts every invocation of the workgroups dispatched
  const invocations = Math.floor(mostOf(gl) / 3) + 1;
  const actual = await counted(gl, invocations, 'crowded');
  expect(stats().fallbacks.get('crowded')).toBe(
    `the dispatch may add ${Math.ceil(invocations / 64) * 64 * 3} times to one cell, past ${mostOf(gl)}`,
  );
  expect(actual).toEqual(await counted(gpu, invocations, 'crowded'));
});

/**
 * Adds the largest amount once an invocation to one cell from `workgroups` groups of 241: both
 * bounds, 2^24 / 15 = 241 * 4641 and 2^24 / 255 = 241 * 273, are whole dispatches of them.
 */
async function crowded(root: TgpuRoot, workgroups: number, label: string) {
  const counts = root.createMutable(d.arrayOf(d.atomic(d.u32), 2), [7, 0]);
  const kernel = tgpu.computeFn({ workgroupSize: [241] })(() => {
    'use gpu';
    std.atomicAdd(counts.$[0], 0xffffffff);
  });
  root.createComputePipeline({ compute: kernel }).$name(label).dispatchWorkgroups(workgroups);
  return counts.read();
}

/** Expects the most records a cell may take to sum exactly on `root`, and one group more to fall back. */
async function bounded(root: TgpuRoot, label: string) {
  const most = mostOf(root);
  const { stats: counts } = root.device as unknown as GLDevice;
  const actual = await crowded(root, most / 241, label);
  expect(counts.fallbacks.get(label)).toBeUndefined();
  expect([...counts.kernels.keys()]).toContain(`${label} (gpu)`);
  // Each addition of 2^32 - 1 takes 1 off, modulo 2^32
  expect(actual).toEqual([(7 - most) >>> 0, 0]);
  const over = await crowded(root, most / 241 + 1, `${label} past`);
  expect(counts.fallbacks.get(`${label} past`)).toBe(
    `the dispatch may add ${most + 241} times to one cell, past ${most}`,
  );
  expect(over).toEqual([(7 - most - 241) >>> 0, 0]);
  return actual;
}

it('sums the most records a dispatch may put on one cell exactly, every piece at its largest', async () => {
  // 2^24 / 15 records where the device sums nibbles, 2^24 / 255 where it sums bytes
  const actual = await bounded(gl, 'worst');
  expect(actual).toEqual(await crowded(gpu, mostOf(gl) / 241, 'worst'));
});

it('sums bytes on a device of four draw buffers, to their own bound', async () => {
  const undo = install({ minimum: true });
  const small = glRoot();
  try {
    expect(mostOf(small)).toBe(mostAdded(4));
    await bounded(small, 'bytes');
    expect((small.device as unknown as GLDevice).gl.getError()).toBe(0);
  } finally {
    small.destroy();
    undo();
  }
});

const schema = (n: number) => d.arrayOf(d.atomic(d.u32), n);

/**
 * Adds 24 times an invocation, through a helper at three sites, to `words` counters: a fixed few
 * that share a record per word, or a runtime-sized array whose records pass WebGL's stride.
 */
async function many(root: TgpuRoot, words: number | 'runtime', label: string) {
  const layout = tgpu.bindGroupLayout({
    counts: { storage: words === 'runtime' ? schema : schema(words), access: 'mutable' },
  });
  const size = words === 'runtime' ? 3 : words;
  const eight = (i: number, amount: number) => {
    'use gpu';
    std.atomicAdd(layout.$.counts[(i + 0) % size], amount + d.u32(0));
    std.atomicAdd(layout.$.counts[(i + 1) % size], amount + d.u32(1));
    std.atomicAdd(layout.$.counts[(i + 2) % size], amount + d.u32(2));
    std.atomicAdd(layout.$.counts[(i + 3) % size], amount + d.u32(3));
    std.atomicAdd(layout.$.counts[(i + 4) % size], amount + d.u32(4));
    std.atomicAdd(layout.$.counts[(i + 5) % size], amount + d.u32(5));
    std.atomicAdd(layout.$.counts[(i + 6) % size], amount + d.u32(6));
    std.atomicAdd(layout.$.counts[(i + 7) % size], amount + d.u32(7));
  };
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    if (id.x >= 200) return;
    eight(id.x, id.x * d.u32(0x01000193));
    eight(id.x * 3, d.u32(0xfffffff0));
    if (id.x % 2 === 0) eight(id.x + 1, d.u32(7));
  });
  const counts = root
    .createBuffer(schema(size), [1, 2, 0xffffffff].slice(0, size))
    .$usage('storage');
  const pipeline = root.createComputePipeline({ compute: kernel }).$name(label);
  const group = root.createBindGroup(layout, { counts });
  for (let k = 0; k < 2; k++) pipeline.with(group).dispatchWorkgroups(4);
  return counts.read();
}

it('keeps a record per word of a small buffer, however many times an invocation adds to it', async () => {
  const actual = await many(gl, 3, 'merged');
  expect(onGpu('merged')).toBe(true);
  expect(actual).toEqual(await many(gpu, 3, 'merged'));
});

it("runs a kernel on the CPU where its records pass WebGL's stride", async () => {
  const actual = await many(gl, 'runtime', 'strided');
  expect(stats().fallbacks.get('strided')).toBe(
    'its writes need 288 bytes an invocation; WebGL 2 reads 255 a vertex',
  );
  expect(actual).toEqual(await many(gpu, 'runtime', 'strided'));
});
