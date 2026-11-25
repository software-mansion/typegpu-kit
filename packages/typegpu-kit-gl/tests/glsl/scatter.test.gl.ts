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

const gid = { gid: d.builtin.globalInvocationId };

// A draw WebGL refuses leaves its error behind and the targets as they were, so a pass that fails
// must not pass for one that wrote what it was meant to
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

/** Whether the fallback ran the last dispatch of `label` as a GPU form. */
const onGpu = (label: string) => {
  const stats = (gl.device as unknown as GLDevice).stats;
  const reason = stats.fallbacks.get(label);
  if (reason) throw new Error(`'${label}' ran on the CPU: ${reason}`);
  return [...stats.kernels.keys()].includes(`${label} (gpu)`);
};

const HEIGHT = 5;
const DIRECTIONS = 3;
const STEP = 4;
const Packed = d.arrayOf(d.u32, 2);

/** Writes pairs of words at a mixed-radix index of the invocation, at two sites, as cascades do. */
async function cascade(root: TgpuRoot) {
  const count = 2 * DIRECTIONS * HEIGHT;
  const size = (2 * (STEP + 1) + 1) * HEIGHT;
  // As the cascades bind it: a runtime-sized array of pairs
  const buffer = root
    .createBuffer(
      d.arrayOf(Packed, size),
      Array.from({ length: size }, (_, i) => [i, 1000 + i]),
    )
    .$usage('storage');
  const layout = tgpu.bindGroupLayout({
    target: { storage: d.arrayOf(Packed), access: 'mutable' },
  });
  const kernel = tgpu.computeFn({ workgroupSize: [8, 2], in: gid })(({ gid: at }) => {
    'use gpu';
    const id = at.x + at.y * 8;
    if (id >= count) return;
    const y = id % HEIGHT;
    const direction = (std.intdiv(id, HEIGHT) % DIRECTIONS) * 2;
    const x = std.intdiv(id, HEIGHT * DIRECTIONS);
    const index = (x * (STEP + 1) + direction) * HEIGHT + y;
    if (direction < STEP) {
      layout.$.target[index + HEIGHT] = Packed([id * 3, x + 7]);
    }
    layout.$.target[index] = Packed([id, direction]);
  });
  root
    .createComputePipeline({ compute: kernel })
    .$name('cascade')
    .with(root.createBindGroup(layout, { target: buffer }))
    .dispatchWorkgroups(1, 2);
  return buffer.read();
}

/** Swaps the cells of each 2×2 block of a grid, at an offset a uniform holds. */
async function blocks(root: TgpuRoot) {
  const W = 8;
  const H = 6;
  const cells = root.createMutable(
    d.arrayOf(d.f32, W * H),
    Array.from({ length: W * H }, (_, i) => i * 0.5),
  );
  const source = root.createReadonly(
    d.arrayOf(d.f32, W * H),
    Array.from({ length: W * H }, (_, i) => 100 + i),
  );
  const shift = root.createUniform(d.u32, 1);
  const kernel = tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: at }) => {
    'use gpu';
    const x = at.x * 2 + shift.$;
    const y = at.y * 2 + shift.$;
    if (x + 1 >= W || y + 1 >= H) return;
    const a = source.$[y * W + x] as number;
    const b = source.$[y * W + x + 1] as number;
    const c = source.$[(y + 1) * W + x] as number;
    const e = source.$[(y + 1) * W + x + 1] as number;
    cells.$[y * W + x] = e;
    cells.$[y * W + x + 1] = c;
    cells.$[(y + 1) * W + x] = b;
    if (a > 110) cells.$[(y + 1) * W + x + 1] = a;
  });
  root.createComputePipeline({ compute: kernel }).$name('blocks').dispatchWorkgroups(1, 1);
  return cells.read();
}

/** Moves each pair of cells into a target, one cell to the other's place where it is heavier. */
async function pairs(root: TgpuRoot) {
  const N = 40;
  const from = root.createReadonly(
    d.arrayOf(d.vec2i, N),
    Array.from({ length: N }, (_, i) => d.vec2i((i * 7) % 11, i)),
  );
  const into = root.createMutable(
    d.arrayOf(d.vec2i, N),
    Array.from({ length: N }, () => d.vec2i(-1, -1)),
  );
  const kernel = tgpu.computeFn({ workgroupSize: [16], in: gid })(({ gid: at }) => {
    'use gpu';
    const first = at.x * 2;
    if (first + 1 >= N) return;
    const a = from.$[first] as d.v2i;
    const b = from.$[first + 1] as d.v2i;
    const heavier = a.x > b.x;
    into.$[std.select(first, first + 1, heavier)] = d.vec2i(a);
    into.$[std.select(first + 1, first, heavier)] = d.vec2i(b);
  });
  root.createComputePipeline({ compute: kernel }).$name('pairs').dispatchWorkgroups(2);
  return into.read();
}

/** Stores each invocation's texel at (id % W, id / W) of a texture, and into a 2d-array layer. */
async function texels(root: TgpuRoot) {
  const W = 7;
  const H = 5;
  const flat = root
    .createTexture({ size: [W, H], format: 'rgba16float' })
    .$usage('storage', 'sampled');
  const layered = root
    .createTexture({ size: [W, H, 3], format: 'rgba16float' })
    .$usage('storage', 'sampled');
  const layout = tgpu.bindGroupLayout({
    flat: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
    layered: { storageTexture: d.textureStorage2dArray('rgba16float', 'write-only') },
  });
  const kernel = tgpu.computeFn({ workgroupSize: [16], in: gid })(({ gid: at }) => {
    'use gpu';
    const id = at.x;
    if (id >= W * H) return;
    const pixel = d.vec2i(d.i32(id % W), d.i32(std.intdiv(id, W)));
    std.textureStore(layout.$.flat, pixel, d.vec4f(d.f32(id), 1, 0, 1));
    const shifted = d.vec2i(std.select(pixel.x - 1, W - 1, pixel.x === 0), pixel.y);
    std.textureStore(layout.$.layered, shifted, 2, d.vec4f(d.f32(id) * 0.5, 0, 1, 1));
  });
  root
    .createComputePipeline({ compute: kernel })
    .$name('texels')
    .with(root.createBindGroup(layout, { flat, layered }))
    .dispatchWorkgroups(3);
  const read = tgpu.bindGroupLayout({
    flat: { texture: d.texture2d(d.f32) },
    layered: { texture: d.texture2dArray(d.f32) },
  });
  const out = root.createMutable(d.arrayOf(d.vec4f, W * H * 4));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        for (let y = d.u32(0); y < H; y++) {
          for (let x = d.u32(0); x < W; x++) {
            out.$[y * W + x] = std.textureLoad(read.$.flat, d.vec2u(x, y), 0);
            for (let z = d.u32(0); z < 3; z++) {
              out.$[(z + 1) * W * H + y * W + x] = std.textureLoad(
                read.$.layered,
                d.vec2u(x, y),
                z,
                0,
              );
            }
          }
        }
      }),
    })
    .with(root.createBindGroup(read, { flat, layered }))
    .dispatchWorkgroups(1);
  return (await out.read()).map((v) => [v.x, v.y, v.z, v.w]);
}

it('writes pairs of words at two computed sites, as radiance cascades do', async () => {
  const actual = await cascade(gl);
  expect(onGpu('cascade')).toBe(true);
  expect(actual).toEqual(await cascade(gpu));
});

it('writes the cells of 2×2 blocks, one of them conditionally', async () => {
  expect(await blocks(gl)).toEqual(await blocks(gpu));
  expect(onGpu('blocks')).toBe(true);
});

it('writes a pair of cells at indices it chooses', async () => {
  expect(await pairs(gl)).toEqual(await pairs(gpu));
  expect(onGpu('pairs')).toBe(true);
});

it('stores texels at computed coordinates of a texture and a 2d-array layer', async () => {
  const actual = await texels(gl);
  expect(onGpu('texels')).toBe(true);
  expect(actual).toEqual(await texels(gpu));
});

/** Writes elements of four words, and of three words that straddle data texels, in reverse. */
async function widths(root: TgpuRoot) {
  const N = 21;
  const Triple = d.arrayOf(d.u32, 3);
  const quads = root.createMutable(
    d.arrayOf(d.vec4u, N),
    Array.from({ length: N }, (_, i) => d.vec4u(i, i, i, i)),
  );
  const tripleBuffer = root
    .createBuffer(
      d.arrayOf(Triple, N),
      Array.from({ length: N }, (_, i) => [i, 2 * i, 3 * i]),
    )
    .$usage('storage');
  const layout = tgpu.bindGroupLayout({
    triples: { storage: d.arrayOf(Triple), access: 'mutable' },
  });
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    if (at.x >= N - 1) return;
    const to = N - 1 - at.x;
    quads.$[to] = d.vec4u(at.x, at.x + 100, at.x + 200, at.x + 300);
    layout.$.triples[to] = Triple([at.x * 7, at.x + 1, 9]);
  });
  root
    .createComputePipeline({ compute: kernel })
    .$name('widths')
    .with(root.createBindGroup(layout, { triples: tripleBuffer }))
    .dispatchWorkgroups(3);
  return [await quads.read(), await tripleBuffer.read()];
}

it('writes elements of four words, and of three that straddle texels', async () => {
  const actual = await widths(gl);
  expect(onGpu('widths')).toBe(true);
  expect(actual).toEqual(await widths(gpu));
});

/**
 * Dispatches one scatter kernel three times in a pass and twice more in later submissions, each
 * with its own target, permutation and value, into a buffer and a texture.
 */
async function repeated(root: TgpuRoot, stored: boolean) {
  const N = 24;
  const W = 6;
  const layout = tgpu.bindGroupLayout({
    order: { storage: d.arrayOf(d.u32, N), access: 'readonly' },
    out: { storage: d.arrayOf(d.vec2u, N), access: 'mutable' },
    image: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
    add: { uniform: d.u32 },
  });
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    if (at.x >= N) return;
    const to = layout.$.order[at.x] as number;
    layout.$.out[to] = d.vec2u(at.x + layout.$.add, to);
    if (!stored) return;
    std.textureStore(
      layout.$.image,
      d.vec2u(to % W, std.intdiv(to, W)),
      d.vec4f(d.f32(at.x), d.f32(layout.$.add), 1, 0.5),
    );
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name('repeated');
  const targets = [0, 1, 2, 3, 4].map((k) => {
    // A permutation of its own per dispatch: each multiplier is coprime to N, so no two
    // invocations write one cell, which would leave which value wins open as on WebGPU
    const order = Array.from({ length: N }, (_, i) => (i * ([5, 7, 11, 13, 17][k] ?? 1) + k) % N);
    const out = root
      .createBuffer(
        d.arrayOf(d.vec2u, N),
        Array.from({ length: N }, () => d.vec2u(7, 7)),
      )
      .$usage('storage');
    const image = root
      .createTexture({ size: [W, N / W], format: 'rgba16float' })
      .$usage('storage', 'sampled');
    const group = root.createBindGroup(layout, {
      order: root.createBuffer(d.arrayOf(d.u32, N), order).$usage('storage'),
      out,
      image,
      add: root.createUniform(d.u32, 1000 * (k + 1)).buffer,
    });
    return { out, image, group };
  });
  const run = (from: number, to: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    for (const { group } of targets.slice(from, to)) {
      pass.setBindGroup(layout, group);
      pass.dispatchWorkgroups(N / 8);
    }
    pass.end();
    encoder.submit();
  };
  run(0, 3);
  run(3, 4);
  run(4, 5);
  const read = tgpu.bindGroupLayout({ from: { texture: d.texture2d(d.f32) } });
  const seen = root.createMutable(d.arrayOf(d.vec4f, N));
  const copy = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      for (let i = d.u32(0); i < N; i++) {
        seen.$[i] = std.textureLoad(read.$.from, d.vec2u(i % W, std.intdiv(i, W)), 0);
      }
    }),
  });
  const results = [];
  for (const { out, image } of targets) {
    copy.with(root.createBindGroup(read, { from: image })).dispatchWorkgroups(1);
    results.push({
      out: (await out.read()).map((v) => [v.x, v.y]),
      image: (await seen.read()).map((v) => [v.x, v.y, v.z, v.w]),
    });
  }
  return results;
}

it.each([false, true])(
  'writes what each of several dispatches of one kernel scatters, in a pass and across submits (texture %s)',
  async (stored) => {
    const actual = await repeated(gl, stored);
    expect(onGpu('repeated')).toBe(true);
    expect(actual).toEqual(await repeated(gpu, stored));
  },
);

/**
 * Writes cells it computes through a helper, twice an invocation and once only on a condition,
 * into a buffer and a texture, over two dispatches.
 */
async function helped(root: TgpuRoot) {
  const N = 40;
  const W = 8;
  const out = root.createMutable(
    d.arrayOf(d.vec2u, 2 * N),
    Array.from({ length: 2 * N }, () => d.vec2u(9, 9)),
  );
  const image = root
    .createTexture({ size: [W, (2 * N) / W], format: 'rgba16float' })
    .$usage('storage', 'sampled');
  const layout = tgpu.bindGroupLayout({
    image: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
    round: { uniform: d.u32 },
  });
  const put = (cell: number, value: number) => {
    'use gpu';
    out.$[cell] = d.vec2u(value, layout.$.round);
    std.textureStore(
      layout.$.image,
      d.vec2u(cell % W, std.intdiv(cell, W)),
      d.vec4f(d.f32(value), d.f32(layout.$.round), 0, 1),
    );
  };
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: gid })(({ gid: at }) => {
    'use gpu';
    if (at.x >= N) return;
    put((at.x * 7 + layout.$.round) % N, at.x);
    if (at.x % 3 !== 0) put(N + ((at.x * 11) % N), at.x + 100);
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name('helped');
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  for (const round of [1, 2]) {
    pass.setBindGroup(
      layout,
      root.createBindGroup(layout, { image, round: root.createUniform(d.u32, round).buffer }),
    );
    pass.dispatchWorkgroups(N / 8);
  }
  pass.end();
  encoder.submit();
  const read = tgpu.bindGroupLayout({ from: { texture: d.texture2d(d.f32) } });
  const seen = root.createMutable(d.arrayOf(d.vec4f, 2 * N));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        for (let i = d.u32(0); i < 2 * N; i++) {
          seen.$[i] = std.textureLoad(read.$.from, d.vec2u(i % W, std.intdiv(i, W)), 0);
        }
      }),
    })
    .with(root.createBindGroup(read, { from: image }))
    .dispatchWorkgroups(1);
  return {
    out: (await out.read()).map((v) => [v.x, v.y]),
    image: (await seen.read()).map((v) => [v.x, v.y, v.z, v.w]),
  };
}

it('writes cells it computes through a helper called twice, as WebGPU does', async () => {
  const actual = await helped(gl);
  expect(onGpu('helped')).toBe(true);
  expect(actual).toEqual(await helped(gpu));
});

/** Scatters elements of eight words, two captured vectors each, over two dispatches. */
async function wide(root: TgpuRoot) {
  const Pair = d.struct({ a: d.vec4u, b: d.vec4u });
  const out = root.createMutable(d.arrayOf(Pair, 6));
  const layout = tgpu.bindGroupLayout({ round: { uniform: d.u32 } });
  const kernel = tgpu.computeFn({ workgroupSize: [3], in: gid })(({ gid: at }) => {
    'use gpu';
    const round = layout.$.round;
    out.$[((at.x + 1) % 3) + round * 3] = Pair({
      a: d.vec4u(1, 2, 3, 4) + at.x,
      b: d.vec4u(5, 6, 7, 8) * (round + 1) + at.x,
    });
  });
  const pipeline = root.createComputePipeline({ compute: kernel }).$name('wide');
  for (const round of [0, 1]) {
    pipeline
      .with(root.createBindGroup(layout, { round: root.createUniform(d.u32, round).buffer }))
      .dispatchWorkgroups(1);
  }
  return (await out.read()).map(({ a, b }) => [a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w]);
}

it('scatters elements wider than four words whole, as WebGPU does', async () => {
  const actual = await wide(gl);
  expect(onGpu('wide')).toBe(true);
  expect(actual).toEqual(await wide(gpu));
});
