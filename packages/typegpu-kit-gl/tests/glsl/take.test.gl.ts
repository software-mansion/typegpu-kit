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

afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

const gid = { gid: d.builtin.globalInvocationId };
/** Larger than the 83 emissions, so the ring wraps past its end without overwriting any. */
const RING = 128;

const onGpu = (label: string) => {
  const stats = (gl.device as unknown as GLDevice).stats;
  const reason = stats.fallbacks.get(label);
  if (reason) throw new Error(`'${label}' ran on the CPU: ${reason}`);
  return [...stats.kernels.keys()].includes(`${label} (gpu)`);
};

/**
 * Emits into a ring as particles do: each emission takes the next of a counter's values and writes
 * its slot, none, once or twice an invocation, from three dispatches with values of their own.
 */
async function emitted(root: TgpuRoot) {
  const head = root.createMutable(d.atomic(d.u32), 100);
  const pool = root.createMutable(
    d.arrayOf(d.vec4u, RING),
    Array.from({ length: RING }, () => d.vec4u(0)),
  );
  const layout = tgpu.bindGroupLayout({ round: { uniform: d.u32 } });
  const emit = (value: d.v4u) => {
    'use gpu';
    const slot = std.atomicAdd(head.$, 1) % RING;
    pool.$[slot] = d.vec4u(value);
  };
  // A second helper between the kernel and the take, which counts its takes through both
  const spit = (x: number, round: number) => {
    'use gpu';
    emit(d.vec4u(x, round, 2, 7));
  };
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    if (at.x >= 30) return;
    const round = layout.$.round;
    if ((at.x + round) % 3 !== 0) emit(d.vec4u(at.x, round, 1, 0));
    if (at.x % 4 === round) spit(at.x, round);
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name('emitting');
  const groups = [0, 1, 2].map((round) =>
    root.createBindGroup(layout, { round: root.createUniform(d.u32, round).buffer }),
  );
  const run = (from: number, to: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const group of groups.slice(from, to)) {
      pass.setBindGroup(layout, group);
      pass.dispatchWorkgroups(4);
    }
    pass.end();
    encoder.submit();
  };
  run(0, 2);
  run(2, 3);
  return {
    head: await head.read(),
    pool: (await pool.read())
      .map((v) => [v.x, v.y, v.z, v.w])
      .toSorted((a, b) => a[0]! - b[0]! || a[1]! - b[1]! || a[2]! - b[2]! || a[3]! - b[3]!),
  };
}

it('takes ring slots in some order WebGPU may give, over several dispatches', async () => {
  const actual = await emitted(gl);
  expect(onGpu('emitting')).toBe(true);
  // No emission overwrites another, so which slot each took does not change the pool's contents
  expect(actual).toEqual(await emitted(gpu));
  expect(actual.head).toBe(183);
});
