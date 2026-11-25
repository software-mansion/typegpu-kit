import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

// Invocations that own a constant block of a grid's cells, written through a helper that keeps
// its column inside the row, judged against WebGPU in the same page

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

const W = 10;
const H = 7;
const gid = { gid: d.builtin.globalInvocationId };

const ranAs = (label: string) => {
  const stats = (gl.device as unknown as GLDevice).stats;
  const reason = stats.fallbacks.get(label);
  if (reason) return reason;
  return [...stats.kernels.keys()].includes(`${label} (gpu)`) ? 'gpu' : 'cpu';
};

function grid(root: TgpuRoot) {
  const cells = root
    .createBuffer(
      d.arrayOf(d.u32, W * H),
      Array.from({ length: W * H }, (_, i) => (i * 37) % 11),
    )
    .$usage('storage');
  const writable = cells.as('mutable');
  const weights = root.createReadonly(
    d.arrayOf(d.u32, W * H),
    Array.from({ length: W * H }, (_, i) => i % 5),
  );
  const inside = (p: d.v2i) => {
    'use gpu';
    return std.all(std.ge(p, d.vec2i())) && p.x < W && p.y < H;
  };
  const at = (p: d.v2i) => {
    'use gpu';
    if (!inside(p)) return d.u32(0);
    return writable.$[d.u32(p.y) * W + d.u32(p.x)] as number;
  };
  const put = (p: d.v2i, value: number) => {
    'use gpu';
    if (!inside(p)) return;
    writable.$[d.u32(p.y) * W + d.u32(p.x)] = value;
  };
  return { cells, weights, at, put };
}

/** 2 × 2 blocks shifted by a phase each round, each rotating its cells and mixing in weights. */
async function blocks(root: TgpuRoot) {
  const { cells, weights, at, put } = grid(root);
  const layout = tgpu.bindGroupLayout({ shift: { uniform: d.vec2u } });
  const kernel = tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
    'use gpu';
    const origin = d.vec2i(g.xy) * 2 - d.vec2i(layout.$.shift);
    const a = at(origin);
    const b = at(origin + d.vec2i(1, 0));
    const c = at(origin + d.vec2i(0, 1));
    const e = at(origin + d.vec2i(1));
    put(origin, b + 1);
    put(origin + d.vec2i(1, 0), e);
    // Reads back a cell of its own block that it has just written
    put(origin + d.vec2i(1), c * 2 + at(origin));
    put(origin + d.vec2i(0, 1), a + (weights.$[(g.y * W + g.x) % (W * H)] as number));
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name('blocks');
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  for (const shift of [d.vec2u(0, 0), d.vec2u(1, 0), d.vec2u(1, 1), d.vec2u(0, 1)]) {
    pass.setBindGroup(
      layout,
      root.createBindGroup(layout, { shift: root.createUniform(d.vec2u, shift).buffer }),
    );
    pass.dispatchWorkgroups(2, 2);
  }
  pass.end();
  encoder.submit();
  return cells.read();
}

/** Pairs along rows of one parity, at an offset that alternates, as liquids flow. */
async function pairs(root: TgpuRoot) {
  const { cells, at, put } = grid(root);
  const make = (parity: number, offset: number) =>
    root
      .createComputePipeline({
        compute: tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: g }) => {
          'use gpu';
          const left = d.vec2i(d.i32(g.x) * 2 - offset, d.i32(g.y) * 2 + parity);
          const right = left + d.vec2i(1, 0);
          const l = at(left);
          const r = at(right);
          if (l === r) return;
          // Reads the row below, which no invocation of this pass writes
          const below = at(left + d.vec2i(0, 1));
          put(left, r + (below & 1));
          put(right, l);
        }),
      })
      .$name(`pairs ${parity} ${offset}`);
  const passes = [make(0, 0), make(1, 0), make(0, 1), make(1, 1)];
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const pipeline of [...passes, ...passes]) {
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(1, 4);
  }
  pass.end();
  encoder.submit();
  return cells.read();
}

it('owns 2 × 2 blocks at alternating phases, reading back its own writes, as WebGPU does', async () => {
  const actual = await blocks(gl);
  expect(ranAs('blocks')).toBe('gpu');
  expect(actual).toEqual(await blocks(gpu));
});

it('owns pairs along rows at alternating offsets, as WebGPU does', async () => {
  const actual = await pairs(gl);
  for (const label of ['pairs 0 0', 'pairs 1 0', 'pairs 0 1', 'pairs 1 1']) {
    expect(ranAs(label)).toBe('gpu');
  }
  expect(actual).toEqual(await pairs(gpu));
});
