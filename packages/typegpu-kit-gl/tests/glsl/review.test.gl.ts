import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

// Kernels a soundness review found accepted as GPU forms with wrong results: each must now run
// as WebGPU does, refused to the CPU where no form is sound

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
const reasonOf = (label: string) =>
  (gl.device as unknown as GLDevice).stats.fallbacks.get(label) ?? '';

/** F1: a taken value kept in an array element decides whether to take again. */
async function indexed(root: TgpuRoot) {
  const head = root.createMutable(d.atomic(d.u32), 0);
  const out = root.createMutable(d.arrayOf(d.u32, 3));
  const kernel = tgpu.computeFn({ workgroupSize: [3], in: gid })(({ gid: at }) => {
    'use gpu';
    const v = d.arrayOf(d.u32, 2)();
    v[0] = std.atomicAdd(head.$, 1);
    if (v[0] > 0) v[0] = std.atomicAdd(head.$, 1);
    out.$[at.x] = v[0] as number;
  });
  root.createComputePipeline({ compute: kernel }).$name('indexed').dispatchWorkgroups(1);
  return [await out.read(), await head.read()];
}

/** F2: loads the counter it takes from. */
async function loaded(root: TgpuRoot) {
  const head = root.createMutable(d.atomic(d.u32), 0);
  const out = root.createMutable(d.arrayOf(d.vec2u, 3));
  const kernel = tgpu.computeFn({ workgroupSize: [3], in: gid })(({ gid: at }) => {
    'use gpu';
    const slot = std.atomicAdd(head.$, 1);
    out.$[at.x] = d.vec2u(slot, std.atomicLoad(head.$));
  });
  root.createComputePipeline({ compute: kernel }).$name('loaded').dispatchWorkgroups(1);
  return [(await out.read()).map((v) => [v.x, v.y]), await head.read()];
}

/** F3: loads a counter it adds to through a helper resolved before the addition. */
async function reread(root: TgpuRoot) {
  const head = root.createMutable(d.arrayOf(d.atomic(d.u32), 1), [0]);
  const out = root.createMutable(d.arrayOf(d.vec2u, 3));
  const read = () => {
    'use gpu';
    return std.atomicLoad(head.$[0]);
  };
  const kernel = tgpu.computeFn({ workgroupSize: [3], in: gid })(({ gid: at }) => {
    'use gpu';
    const before = read();
    std.atomicAdd(head.$[0], 1);
    const after = read();
    out.$[at.x] = d.vec2u(before, after);
  });
  root.createComputePipeline({ compute: kernel }).$name('reread').dispatchWorkgroups(1);
  return [(await out.read()).map((v) => [v.x, v.y]), await head.read()];
}

/** F5: takes from two counters of one binding. */
async function paired(root: TgpuRoot) {
  const Heads = d.struct({ a: d.atomic(d.u32), b: d.atomic(d.u32) });
  const head = root.createMutable(Heads, { a: 10, b: 100 });
  const out = root.createMutable(d.arrayOf(d.vec2u, 3));
  const kernel = tgpu.computeFn({ workgroupSize: [3], in: gid })(({ gid: at }) => {
    'use gpu';
    const first = std.atomicAdd(head.$.a, 1);
    const second = std.atomicAdd(head.$.b, 1);
    out.$[at.x] = d.vec2u(first, second);
  });
  root.createComputePipeline({ compute: kernel }).$name('paired').dispatchWorkgroups(1);
  const heads = await head.read();
  return [(await out.read()).map((v) => [v.x, v.y]), [heads.a, heads.b]];
}

// WebGPU leaves the order of the three invocations' atomics open, so racing kernels are checked
// against the outcome the CPU gives, which WebGPU allows too, and the deterministic one against it
it.each([
  ['indexed', indexed, 'decides a branch on a value it takes', [[0, 2, 4], 5]],
  [
    'loaded',
    loaded,
    "reads 'head",
    [
      [
        [0, 1],
        [1, 2],
        [2, 3],
      ],
      3,
    ],
  ],
  [
    'reread',
    reread,
    "reads 'head",
    [
      [
        [0, 1],
        [1, 2],
        [2, 3],
      ],
      [3],
    ],
  ],
] as const)(
  'refuses %s as a GPU form and runs it in order',
  async (label, run, reason, expected) => {
    expect(await run(gl)).toEqual(expected);
    expect(reasonOf(label)).toContain(reason);
  },
);

it('refuses taking from two counters of one binding, and runs it as WebGPU does', async () => {
  const actual = await paired(gl);
  expect(actual).toEqual(await paired(gpu));
  expect(reasonOf('paired')).toContain('more than one atomic');
});
