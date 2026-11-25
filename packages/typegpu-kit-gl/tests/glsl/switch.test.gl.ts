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

const reasonOf = (label: string) => (gl.device as unknown as GLDevice).stats.fallbacks.get(label);

/** G1: one case adds and returns; the other adds twice and loads what it added to. */
async function cased(root: TgpuRoot) {
  const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
  const out = root.createMutable(d.arrayOf(d.u32, 64));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      switch (gid.x) {
        case 0: {
          std.atomicAdd(a.$[0], 1);
          return;
        }
        default: {
          std.atomicAdd(a.$[1], 1);
          std.atomicAdd(a.$[2], 1);
          out.$[gid.x] = std.atomicLoad(a.$[1]);
        }
      }
    },
  );
  root.createComputePipeline({ compute: kernel }).$name('cased').dispatchWorkgroups(1);
  return { a: await a.read(), out: await out.read() };
}

it('adds every case and loads after its own addition, as one order of the atomics gives', async () => {
  const { a, out } = await cased(gl);
  expect(reasonOf('cased')).toContain('where it may have added to it');
  expect(a).toEqual([1, 63, 63, 0]);
  // Run in order, each invocation loads after the additions of those before it and its own
  expect(out).toEqual(Array.from({ length: 64 }, (_, i) => i));
  const webgpu = await cased(gpu);
  expect(webgpu.a).toEqual(a);
  // WebGPU's loads race, but each sees at least its own addition
  expect(webgpu.out.slice(1).every((value) => value >= 1 && value <= 63)).toBe(true);
});

/** G2: the helper's guard sits in one case of a switch, and the write after it every time. */
async function guarded(root: TgpuRoot) {
  const W = 10;
  const cells = root.createMutable(d.arrayOf(d.u32, W * 6));
  const inside = (p: d.v2i) => {
    'use gpu';
    return std.all(std.ge(p, d.vec2i())) && p.x < W && p.y < 6;
  };
  const put = (p: d.v2i, value: number) => {
    'use gpu';
    switch (p.y) {
      case 0: {
        if (!inside(p)) return;
        break;
      }
      default: {
        break;
      }
    }
    cells.$[d.u32(p.y) * W + d.u32(p.x)] = value;
  };
  const kernel = tgpu.computeFn({
    workgroupSize: [4, 2],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    put(d.vec2i(gid.xy) * 2 + d.vec2i(9, 0), d.u32(7));
  });
  root.createComputePipeline({ compute: kernel }).$name('guarded').dispatchWorkgroups(1);
  return cells.read();
}

it('writes past a guard that only one switch case holds, as WebGPU does', async () => {
  const actual = await guarded(gl);
  expect(reasonOf('guarded')).toContain('at a column it does not keep');
  // Invocation (1, 1) writes (11, 2), element 31, which the guard in case 0 does not skip
  expect(actual[31]).toBe(7);
  expect(actual).toEqual(await guarded(gpu));
});
