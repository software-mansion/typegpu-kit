import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

// Kernels a user could write against the scatter, sum and take forms, each judged by comparison
// with WebGPU in the same page: exactly, or as multisets where WebGPU leaves the order open

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
const N = 24;

/** How the fallback ran `label`: 'gpu', or the reason it ran on the CPU. */
const ranAs = (label: string) => {
  const stats = (gl.device as unknown as GLDevice).stats;
  const reason = stats.fallbacks.get(label);
  if (reason) return reason;
  return [...stats.kernels.keys()].includes(`${label} (gpu)`) ? 'gpu' : 'cpu';
};

const sortRows = (rows: number[][]) =>
  rows.toSorted((a, b) => {
    for (const [k, value] of a.entries()) if (value !== b[k]) return value - (b[k] as number);
    return 0;
  });

const words = (values: readonly (number | d.v2u | d.v4u)[]) =>
  values.map((v) => (typeof v === 'number' ? [v] : [...(v as unknown as number[])]));

describe('scatter sites in helpers under data-dependent control', () => {
  const cases: [string, (root: TgpuRoot, label: string) => Promise<number[]>][] = [
    [
      'reaches one site through two call paths',
      (root, label) => {
        const out = root.createMutable(d.arrayOf(d.u32, 2 * N), Array(2 * N).fill(99));
        const put = (cell: number, value: number) => {
          'use gpu';
          out.$[cell] = value;
        };
        const relay = (cell: number, value: number) => {
          'use gpu';
          put(cell + N, value + 1000);
        };
        const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
          'use gpu';
          const cell = (at.x * 5) % N;
          put(cell, at.x);
          relay(cell, at.x);
        });
        root
          .createComputePipeline({ compute: kernel })
          .$name(label)
          .dispatchWorkgroups(N / 8);
        return out.read();
      },
    ],
    [
      'calls the helper only when the invocation chooses to',
      (root, label) => {
        const out = root.createMutable(d.arrayOf(d.u32, 2 * N), Array(2 * N).fill(99));
        const put = (cell: number, value: number) => {
          'use gpu';
          out.$[cell] = value;
        };
        const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
          'use gpu';
          if ((at.x * 7) % 3 === 1) put((at.x * 5) % N, at.x + 1);
        });
        root
          .createComputePipeline({ compute: kernel })
          .$name(label)
          .dispatchWorkgroups(N / 8);
        return out.read();
      },
    ],
    [
      'returns early in the helper before its site',
      (root, label) => {
        const out = root.createMutable(d.arrayOf(d.u32, 2 * N), Array(2 * N).fill(99));
        const put = (cell: number, value: number) => {
          'use gpu';
          if (value % 4 === 0) return;
          out.$[cell] = value;
        };
        const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
          'use gpu';
          put((at.x * 5) % N, at.x);
          put(N + ((at.x * 7) % N), at.x + 2);
        });
        root
          .createComputePipeline({ compute: kernel })
          .$name(label)
          .dispatchWorkgroups(N / 8);
        return out.read();
      },
    ],
    [
      'writes one resource from both arms of an if',
      (root, label) => {
        const out = root.createMutable(d.arrayOf(d.u32, 2 * N), Array(2 * N).fill(99));
        const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
          'use gpu';
          if (at.x % 2 === 0) {
            out.$[(at.x * 5) % N] = at.x;
          } else {
            out.$[N + ((at.x * 5) % N)] = at.x + 500;
          }
        });
        root
          .createComputePipeline({ compute: kernel })
          .$name(label)
          .dispatchWorkgroups(N / 8);
        return out.read();
      },
    ],
    [
      'writes from the cases of a switch',
      (root, label) => {
        const out = root.createMutable(d.arrayOf(d.u32, 2 * N), Array(2 * N).fill(99));
        const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
          'use gpu';
          const cell = (at.x * 5) % N;
          switch (at.x % 3) {
            case 0:
              out.$[cell] = at.x;
              break;
            case 1:
              out.$[cell + N] = at.x + 100;
              break;
            default:
              out.$[cell] = at.x + 200;
              out.$[cell + N] = at.x + 300;
          }
        });
        root
          .createComputePipeline({ compute: kernel })
          .$name(label)
          .dispatchWorkgroups(N / 8);
        return out.read();
      },
    ],
    [
      'calls one helper twice, at two cells and then at the same cell',
      (root, label) => {
        const out = root.createMutable(d.arrayOf(d.u32, 2 * N), Array(2 * N).fill(99));
        const put = (cell: number, value: number) => {
          'use gpu';
          out.$[cell] = value;
        };
        const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
          'use gpu';
          const cell = (at.x * 5) % N;
          put(cell, at.x);
          put(cell + N, at.x + 1);
          // Within one invocation the later write wins
          put(cell + N, at.x + 2);
        });
        root
          .createComputePipeline({ compute: kernel })
          .$name(label)
          .dispatchWorkgroups(N / 8);
        return out.read();
      },
    ],
    [
      'writes after an early return of the entry',
      (root, label) => {
        const out = root.createMutable(d.arrayOf(d.u32, 2 * N), Array(2 * N).fill(99));
        const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
          'use gpu';
          if (at.x % 5 === 2) return;
          out.$[(at.x * 7) % N] = at.x;
        });
        root
          .createComputePipeline({ compute: kernel })
          .$name(label)
          .dispatchWorkgroups(N / 8);
        return out.read();
      },
    ],
  ];
  it.each(cases)('%s, as WebGPU does', async (name, run) => {
    const label = name.replaceAll(' ', '-');
    const actual = await run(gl, label);
    expect(actual).toEqual(await run(gpu, label));
    expect(ranAs(label)).toBe('gpu');
  });
});

/** Adds mixed signs, zero, wide amounts and subtractions to shared cells, beside an own write. */
async function summed(root: TgpuRoot) {
  const unsigned = root.createMutable(d.arrayOf(d.atomic(d.u32), 6), [1, 2, 3, 4, 5, 6]);
  const signed = root.createMutable(d.arrayOf(d.atomic(d.i32), 2), [7, -7]);
  const own = root.createMutable(d.arrayOf(d.u32, N));
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    const i = at.x;
    std.atomicAdd(signed.$[0], d.i32(i) - 12);
    std.atomicAdd(signed.$[1], -1000000 * d.i32(i % 3));
    std.atomicAdd(unsigned.$[0], 0);
    std.atomicAdd(unsigned.$[1], 256 + i);
    std.atomicAdd(unsigned.$[2], 65536 * (i + 1));
    std.atomicAdd(unsigned.$[3], 0xffffffff);
    std.atomicSub(unsigned.$[4], i * 3);
    // One cell from two sites in one invocation
    std.atomicAdd(unsigned.$[5], 1);
    std.atomicAdd(unsigned.$[5], i);
    own.$[i] = i * 10;
  });
  root
    .createComputePipeline({ compute: kernel })
    .$name('sums')
    .dispatchWorkgroups(N / 8);
  return [await unsigned.read(), await signed.read(), await own.read()];
}

describe('sums of atomic additions', () => {
  it('sums mixed signs, zero, wide amounts and subtractions on shared cells, as WebGPU does', async () => {
    const actual = await summed(gl);
    expect(actual).toEqual(await summed(gpu));
    expect(ranAs('sums')).toBe('gpu');
  });

  it('drops an addition past the bound array and touches no neighbour', async () => {
    const counts = gl.createMutable(d.arrayOf(d.atomic(d.u32), 8), [0, 0, 0, 0, 0, 0, 0, 0]);
    const after = gl.createMutable(d.arrayOf(d.u32, 8), [5, 5, 5, 5, 5, 5, 5, 5]);
    const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
      'use gpu';
      // Indices 8 and 9 lie past the array
      std.atomicAdd(counts.$[at.x + 2], 1);
      after.$[at.x] = 5;
    });
    gl.createComputePipeline({ compute: kernel }).$name('past').dispatchWorkgroups(1);
    expect(await counts.read()).toEqual([0, 0, 1, 1, 1, 1, 1, 1]);
    expect(await after.read()).toEqual([5, 5, 5, 5, 5, 5, 5, 5]);
    expect(ranAs('past')).toBe('gpu');
  });
});

/** An emitting kernel: each invocation takes a slot of `head` when `keep` holds for it. */
function emitter(
  root: TgpuRoot,
  head: { $: number },
  ring: { $: d.v2u[] },
  label: string,
  size: [number, number, number],
) {
  const kernel = tgpu.computeFn({ workgroupSize: size, in: gid })(({ gid: at }) => {
    'use gpu';
    if ((at.x + at.y * 3 + at.z * 5) % 4 === 1) return;
    const slot = std.atomicAdd(head.$ as never, 1) % 64;
    ring.$[slot] = d.vec2u(at.x + at.y * 100 + at.z * 1000, 7);
  });
  return root.createComputePipeline({ compute: kernel }).$name(label);
}

/** Serves a take, a take of another counter, a sum and the take again from one bind group. */
async function servedByOneGroup(root: TgpuRoot) {
  const layout = tgpu.bindGroupLayout({
    head: { storage: d.atomic(d.u32), access: 'mutable' },
    other: { storage: d.atomic(d.u32), access: 'mutable' },
    ring: { storage: d.arrayOf(d.vec2u, 64), access: 'mutable' },
  });
  const take = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    const slot = std.atomicAdd(layout.$.head, 1) % 64;
    layout.$.ring[slot] = d.vec2u(at.x, 1);
  });
  const takeOther = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    const slot = std.atomicAdd(layout.$.other, 1) % 64;
    layout.$.ring[(slot + 32) % 64] = d.vec2u(at.x, 2);
  });
  const sum = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    std.atomicAdd(layout.$.head, at.x);
  });
  const group = root.createBindGroup(layout, {
    head: root.createBuffer(d.atomic(d.u32), 1).$usage('storage'),
    other: root.createBuffer(d.atomic(d.u32), 0).$usage('storage'),
    ring: root.createBuffer(d.arrayOf(d.vec2u, 64)).$usage('storage'),
  });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setBindGroup(layout, group);
  for (const [compute, label] of [
    [take, 'take'],
    [takeOther, 'take other'],
    [sum, 'sum'],
    [take, 'take again'],
  ] as const) {
    pass.setPipeline(root.createComputePipeline({ compute }).$name(label));
    pass.dispatchWorkgroups(1);
  }
  pass.end();
  encoder.submit();
  const out = root.createMutable(d.arrayOf(d.vec2u, 66));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        for (let i = d.u32(0); i < 64; i++) out.$[i] = d.vec2u(layout.$.ring[i] as d.v2u);
        out.$[64] = d.vec2u(std.atomicLoad(layout.$.head), 0);
        out.$[65] = d.vec2u(std.atomicLoad(layout.$.other), 0);
      }),
    })
    .with(group)
    .dispatchWorkgroups(1);
  const rows = words(await out.read());
  return [rows.slice(64), sortRows(rows.slice(0, 64))];
}

/** Takes from one dispatch, none from an empty one and more from a third, all in one pass. */
async function twiceInAPass(root: TgpuRoot) {
  const head = root.createMutable(d.atomic(d.u32), 3);
  const ring = root.createMutable(d.arrayOf(d.vec2u, 64));
  const pipeline = emitter(root, head as never, ring as never, 'twice in a pass', [4, 2, 1]);
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(2, 2, 1);
  pass.dispatchWorkgroups(0, 1, 1);
  pass.dispatchWorkgroups(1, 3, 1);
  pass.end();
  encoder.submit();
  return [await head.read(), sortRows(words(await ring.read()))];
}

/** Takes, reads the ring and head in a kernel on the CPU, then takes again. */
async function aroundACpuRead(root: TgpuRoot) {
  const head = root.createMutable(d.atomic(d.u32), 0);
  const ring = root.createMutable(d.arrayOf(d.vec2u, 64));
  const seen = root.createMutable(d.arrayOf(d.u32, 2));
  const first = emitter(root, head as never, ring as never, 'deep first', [2, 1, 2]);
  const second = emitter(root, head as never, ring as never, 'deep second', [1, 2, 2]);
  // A barrier keeps the reader on the CPU, so the device reads the ring and head back
  const reader = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      std.workgroupBarrier();
      let sum = d.u32(0);
      for (let i = d.u32(0); i < 64; i++) sum += (ring.$[i] as d.v2u).x;
      seen.$[0] = sum;
      seen.$[1] = std.atomicLoad(head.$ as never);
    }),
  });
  first.dispatchWorkgroups(2, 2, 2);
  reader.dispatchWorkgroups(1);
  const before = await seen.read();
  second.dispatchWorkgroups(2, 1, 2);
  return [before, await head.read(), sortRows(words(await ring.read()))];
}

describe('sequences of takes', () => {
  it('takes from the head an earlier dispatch in the same pass left, as WebGPU does', async () => {
    const actual = await twiceInAPass(gl);
    expect(actual).toEqual(await twiceInAPass(gpu));
    expect(ranAs('twice in a pass')).toBe('gpu');
  });

  it('takes again after a CPU kernel read the ring and the head, with workgroups deep in z', async () => {
    const actual = await aroundACpuRead(gl);
    expect(actual).toEqual(await aroundACpuRead(gpu));
    expect(ranAs('deep first')).toBe('gpu');
    expect(ranAs('deep second')).toBe('gpu');
  });

  it('keeps each form apart when one bind group serves a take, a sum and a take again', async () => {
    const actual = await servedByOneGroup(gl);
    expect(actual).toEqual(await servedByOneGroup(gpu));
    for (const label of ['take', 'take other', 'sum', 'take again'])
      expect(ranAs(label)).toBe('gpu');
  });
});
