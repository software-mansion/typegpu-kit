import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

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

it('runs a dispatch on the CPU where a cell could take more records than a sum keeps exact', async () => {
  // 16 448 invocations of four additions each pass 65 536 records
  const actual = await counted(gl, 16_448, 'crowded');
  expect(stats().fallbacks.get('crowded')).toBe('the dispatch may add 65792 times to one cell');
  expect(actual).toEqual(await counted(gpu, 16_448, 'crowded'));
});

/** Adds the largest amount four times an invocation to one cell, `invocations` times over. */
async function crowded(root: TgpuRoot, invocations: number) {
  const counts = root.createMutable(d.arrayOf(d.atomic(d.u32), 2), [7, 0]);
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    if (id.x >= invocations) return;
    std.atomicAdd(counts.$[0], 0xffffffff);
    std.atomicAdd(counts.$[0], 0xffffffff);
    std.atomicAdd(counts.$[0], 0xffffffff);
    std.atomicAdd(counts.$[0], 0xffffffff);
  });
  root
    .createComputePipeline({ compute: kernel })
    .$name('worst')
    .dispatchWorkgroups(Math.ceil(invocations / 64));
  return counts.read();
}

it('sums the most records a dispatch may put on one cell exactly, every byte at its largest', async () => {
  const actual = await crowded(gl, 16_384);
  expect(onGpu('worst')).toBe(true);
  // 65 536 additions of 2^32 - 1 take 65 536 off, modulo 2^32
  expect(actual).toEqual([(7 - 65_536) >>> 0, 0]);
  expect(actual).toEqual(await crowded(gpu, 16_384));
});
