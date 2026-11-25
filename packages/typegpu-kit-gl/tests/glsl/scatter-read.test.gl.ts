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

const stats = () => (gl.device as unknown as GLDevice).stats;
const gid = { gid: d.builtin.globalInvocationId };
const START = Array.from({ length: 128 }, (_, i) => (i * 37 + 11) % 1000);

/**
 * Each invocation reads two cells of one half of a buffer, then writes one computed cell of the
 * other half, in an arm that returns or after it. `(i * 5) % 64` gives every invocation its own
 * cell, and no invocation reads the half being written, so WebGPU's outcome is determined. A uniform
 * swaps the halves, and a second dispatch reads what the first wrote.
 */
async function swapped(root: TgpuRoot, label: string) {
  const cells = root.createMutable(d.arrayOf(d.u32, 128), START);
  const base = root.createUniform(d.u32, 0);
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    const i = id.x;
    const from = base.$;
    const to = d.u32(64) - from;
    const seen = (cells.$[from + i] as number) + (cells.$[from + ((i + 3) % 64)] as number);
    if (i % 3 === 0) {
      cells.$[to + ((i * 5) % 64)] = seen + 7;
      return;
    }
    cells.$[to + ((i * 5) % 64)] = seen + 1;
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name(label);
  const seen: number[][] = [];
  for (const from of [0, 64]) {
    base.write(from);
    pipeline.dispatchWorkgroups(1);
    seen.push(await cells.read());
  }
  return seen;
}

/** The same in JavaScript: every read before any write, the order the form runs them in. */
function swappedInJs() {
  const cells = [...START];
  const seen: number[][] = [];
  for (const from of [0, 64]) {
    const to = 64 - from;
    const before = [...cells];
    for (let i = 0; i < 64; i++) {
      const read = (before[from + i] as number) + (before[from + ((i + 3) % 64)] as number);
      cells[to + ((i * 5) % 64)] = read + (i % 3 === 0 ? 7 : 1);
    }
    seen.push([...cells]);
  }
  return seen;
}

it('reads a buffer before writing it at computed cells, on the GPU, as WebGPU does', async () => {
  const actual = await swapped(gl, 'swapped');
  expect(stats().fallbacks.get('swapped')).toBeUndefined();
  expect([...stats().kernels.keys()]).toContain('swapped (gpu)');
  expect(actual).toEqual(swappedInJs());
  expect(actual).toEqual(await swapped(gpu, 'swapped'));
});

/** As `swapped`, over two rows of a storage texture: reads one row, writes the other. */
async function rows(root: TgpuRoot, label: string) {
  const texture = root
    .createTexture({ size: [64, 2], format: 'r32uint' })
    .$usage('storage', 'sampled');
  const view = texture.createView(d.textureStorage2d('r32uint', 'read-write'));
  const init = root.createReadonly(d.arrayOf(d.u32, 128), START);
  const base = root.createUniform(d.u32, 0);
  const fill = tgpu.computeFn({ workgroupSize: [64, 2], in: gid })(({ gid: id }) => {
    'use gpu';
    std.textureStore(view.$, id.xy, d.vec4u(init.$[id.y * 64 + id.x] as number, 0, 0, 1));
  });
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    const i = id.x;
    const from = base.$;
    const to = d.u32(1) - from;
    const seen =
      std.textureLoad(view.$, d.vec2u(i, from)).x +
      std.textureLoad(view.$, d.vec2u((i + 3) % 64, from)).x;
    std.textureStore(view.$, d.vec2u((i * 5) % 64, to), d.vec4u(seen + 1, 0, 0, 1));
  });
  const out = root.createMutable(d.arrayOf(d.u32, 128));
  const copy = tgpu.computeFn({ workgroupSize: [64, 2], in: gid })(({ gid: id }) => {
    'use gpu';
    out.$[id.y * 64 + id.x] = std.textureLoad(view.$, id.xy).x;
  });
  root.createComputePipeline({ compute: fill }).dispatchWorkgroups(1);
  const pipeline = root.createComputePipeline({ compute: kernel }).$name(label);
  const copying = root.createComputePipeline({ compute: copy });
  const seen: number[][] = [];
  for (const from of [0, 1]) {
    base.write(from);
    pipeline.dispatchWorkgroups(1);
    copying.dispatchWorkgroups(1);
    seen.push(await out.read());
  }
  return seen;
}

it('reads a storage texture before storing to it at computed texels, on the GPU, as WebGPU does', async () => {
  const actual = await rows(gl, 'rows');
  expect(stats().fallbacks.get('rows')).toBeUndefined();
  expect([...stats().kernels.keys()]).toContain('rows (gpu)');
  expect(actual).toEqual(await rows(gpu, 'rows'));
});

/** Writes its computed cell, then reads it back: the read must see the write. */
async function reread(root: TgpuRoot, label: string) {
  const cells = root.createMutable(d.arrayOf(d.u32, 64), START.slice(0, 64));
  const out = root.createMutable(d.arrayOf(d.u32, 64));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    const at = (id.x * 5) % 64;
    cells.$[at] = id.x + 100;
    out.$[id.x] = (cells.$[at] as number) * 2;
  });
  root.createComputePipeline({ compute: kernel }).$name(label).dispatchWorkgroups(1);
  return [await cells.read(), await out.read()];
}

it('runs on the CPU a kernel that reads a cell after writing it, as WebGPU does', async () => {
  const actual = await reread(gl, 'reread');
  expect(stats().fallbacks.get('reread')).toContain('after writing it');
  expect(actual[1]).toEqual(Array.from({ length: 64 }, (_, i) => (i + 100) * 2));
  expect(actual).toEqual(await reread(gpu, 'reread'));
});

/** R1: stores 5 to its permuted texel, then stores what it loads back plus 1. */
async function restored(root: TgpuRoot, label: string) {
  const texture = root
    .createTexture({ size: [64, 1], format: 'r32uint' })
    .$usage('storage', 'sampled');
  const view = texture.createView(d.textureStorage2d('r32uint', 'read-write'));
  const out = root.createMutable(d.arrayOf(d.u32, 64));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    const cell = d.vec2u(id.x ^ 1, 0);
    std.textureStore(view.$, cell, d.vec4u(5, 0, 0, 1));
    std.textureStore(view.$, cell, d.vec4u(std.textureLoad(view.$, cell).x + 1, 0, 0, 1));
  });
  const copy = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    out.$[id.x] = std.textureLoad(view.$, d.vec2u(id.x, 0)).x;
  });
  root.createComputePipeline({ compute: kernel }).$name(label).dispatchWorkgroups(1);
  root.createComputePipeline({ compute: copy }).dispatchWorkgroups(1);
  return out.read();
}

it('stores what it loads back after storing to that texel, as WebGPU does', async () => {
  const actual = await restored(gl, 'restored');
  expect(actual).toEqual(Array<number>(64).fill(6));
  expect(actual).toEqual(await restored(gpu, 'restored'));
});

/** R2: writes 5 to its permuted cell, then adds that cell to a counter. */
async function summedAfter(root: TgpuRoot, label: string) {
  const a = root.createMutable(d.arrayOf(d.u32, 64));
  const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
    'use gpu';
    const i = id.x ^ 1;
    a.$[i] = 5;
    std.atomicAdd(sums.$[i], a.$[i] as number);
  });
  root.createComputePipeline({ compute: kernel }).$name(label).dispatchWorkgroups(1);
  return [await a.read(), await sums.read()];
}

it('adds what it reads back after writing that cell, as WebGPU does', async () => {
  const actual = await summedAfter(gl, 'summedAfter');
  expect(actual[1]).toEqual(Array<number>(64).fill(5));
  expect(actual).toEqual(await summedAfter(gpu, 'summedAfter'));
});
