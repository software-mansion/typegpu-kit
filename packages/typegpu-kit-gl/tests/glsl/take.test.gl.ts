import { cameras, Particle, particles } from '@typegpu/kit';
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

/**
 * Emits Kit particles, sixteen words a row, from a kernel through `emit`, as stampede's dust and
 * ember-mirror's embers do, over three dispatches, and reads every field of the pool.
 */
async function emittedParticles(root: TgpuRoot) {
  const view = cameras.flat(root, { size: { width: 8, height: 8 }, height: 10 });
  view.update(0);
  const sparks = particles.create(root, view, { capacity: 256, step: 1 / 60 });
  const layout = tgpu.bindGroupLayout({ round: { uniform: d.u32 } });
  const kernel = tgpu.computeFn({ workgroupSize: [16], in: gid })(({ gid: at }) => {
    'use gpu';
    const round = layout.$.round;
    if ((at.x + round) % 3 === 0) return;
    const i = d.f32(at.x);
    const r = d.f32(round);
    sparks.emit(
      Particle({
        position: d.vec3f(i, r, i * r + 0.5),
        age: r * 0.25,
        velocity: d.vec3f(-i, 2 * r, 3),
        life: 1 + i * 0.125,
        color: d.vec4f(i / 16, r / 3, 0.5, 1 - i / 32),
        size: 0.25 + r,
        seed: i * 0.0625 + r,
      }),
    );
  });
  const pipeline = root
    .pipe(sparks.inject())
    .createComputePipeline({ compute: kernel })
    .$name('emitting particles');
  for (const round of [0, 1, 2]) {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      layout,
      root.createBindGroup(layout, { round: root.createUniform(d.u32, round).buffer }),
    );
    pass.dispatchWorkgroups(2);
    pass.end();
    encoder.submit();
  }
  const rows = (await sparks.pool.read()).map((p) => [
    p.position.x,
    p.position.y,
    p.position.z,
    p.age,
    p.velocity.x,
    p.velocity.y,
    p.velocity.z,
    p.life,
    p.color.x,
    p.color.y,
    p.color.z,
    p.color.w,
    p.size,
    p.seed,
  ]);
  sparks.destroy();
  return rows.toSorted((a, b) => {
    for (const [k, value] of a.entries()) if (value !== b[k]) return value - (b[k] as number);
    return 0;
  });
}

it('emits whole particle rows through a kernel as WebGPU does, every field', async () => {
  const actual = await emittedParticles(gl);
  expect(onGpu('emitting particles')).toBe(true);
  expect(actual.filter((row) => row.some((value) => value !== 0)).length).toBe(64);
  expect(actual).toEqual(await emittedParticles(gpu));
});
