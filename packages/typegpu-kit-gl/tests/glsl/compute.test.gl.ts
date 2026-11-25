import { d, std, tgpu, type TgpuBindGroupLayout, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { ComputeModule } from '../../src/glsl/computeModule.ts';
import { dispatch, type Resource, type TextureResource } from './executor.ts';
import { glslRoot } from './utils.ts';

/** A compute entry function, which both roots take. */
type Kernel = object;

type Input =
  | {
      readonly kind: 'buffer';
      readonly schema: d.AnyWgslData;
      readonly value: unknown;
      readonly uniform?: boolean;
    }
  | (Omit<TextureResource, 'data'> & { readonly data: Float32Array; readonly storage: boolean });

interface Scenario {
  readonly layout: TgpuBindGroupLayout;
  /** A compute entry, dispatched as `workgroups`. */
  readonly kernel?: Kernel;
  readonly workgroups?: readonly [number, number, number];
  /** Or a guarded pipeline's callback, dispatched as `threads`. */
  readonly guarded?: (...ids: number[]) => void;
  readonly threads?: readonly [number, number, number];
  /** In the layout's order, so each one's binding is its index. */
  readonly inputs: Record<string, Input>;
}

/** The workgroup size TypeGPU gives a guarded pipeline with `count` dimensions. */
const GUARDED_SIZE = [
  [1, 1, 1],
  [256, 1, 1],
  [16, 16, 1],
  [8, 8, 4],
] as const;

function workgroupsOf(scenario: Scenario): [number, number, number] {
  if (scenario.workgroups) return [...scenario.workgroups];
  const size = GUARDED_SIZE[scenario.guarded?.length ?? 0] as readonly number[];
  const threads = scenario.threads as readonly number[];
  return [0, 1, 2].map((i) => Math.ceil((threads[i] as number) / (size[i] as number))) as [
    number,
    number,
    number,
  ];
}

/** The compute pipeline inside a guarded one, which TypeGPU keeps behind a symbol. */
function innerPipeline(guarded: object): object {
  const state = Object.getOwnPropertySymbols(guarded)
    .map((key) => (guarded as Record<symbol, { pipeline?: object }>)[key])
    .find((value) => value?.pipeline);
  return state?.pipeline as object;
}

let gpu: TgpuRoot;
beforeAll(async () => {
  gpu = await tgpu.init();
});
afterAll(() => gpu.destroy());

const BYTES: Record<TextureResource['format'], number> = {
  rgba32float: 16,
  r32float: 4,
  rgba8unorm: 4,
  rgba32uint: 16,
};

async function readGpu(root: TgpuRoot, source: GPUBuffer, size: number): Promise<Uint8Array> {
  const staging = root.device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, size);
  root.device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(staging.getMappedRange()).slice();
  staging.destroy();
  return bytes;
}

/** Runs the kernel on WebGPU and returns each resource's contents afterwards, by layout key. */
async function onWebGpu(scenario: Scenario): Promise<Record<string, Uint8Array>> {
  const { device } = gpu;
  const entries: Record<string, unknown> = {};
  const reads: Record<string, () => Promise<Uint8Array>> = {};
  for (const [key, input] of Object.entries(scenario.inputs)) {
    if (input.kind === 'buffer') {
      const buffer = gpu
        .createBuffer(input.schema as d.AnyWgslData, input.value as never)
        .$usage(input.uniform ? 'uniform' : 'storage');
      entries[key] = buffer;
      const raw = gpu.unwrap(buffer);
      reads[key] = () => readGpu(gpu, raw, raw.size);
      continue;
    }
    const [width, height, depth] = input.size;
    const texture = device.createTexture({
      size: [width, height, depth],
      format: input.format,
      dimension: input.dimension === '3d' ? '3d' : '2d',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC |
        (input.storage ? GPUTextureUsage.STORAGE_BINDING : 0),
    });
    const row = width * BYTES[input.format];
    device.queue.writeTexture({ texture }, input.data, { bytesPerRow: row, rowsPerImage: height }, [
      width,
      height,
      depth,
    ]);
    entries[key] = texture.createView({ dimension: input.dimension });
    reads[key] = async () => {
      const padded = Math.ceil(row / 256) * 256;
      const staging = device.createBuffer({
        size: padded * height * depth,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: staging, bytesPerRow: padded, rowsPerImage: height },
        [width, height, depth],
      );
      device.queue.submit([encoder.finish()]);
      const bytes = await readGpu(gpu, staging, padded * height * depth);
      const tight = new Uint8Array(row * height * depth);
      for (let r = 0; r < height * depth; r++) {
        tight.set(bytes.subarray(r * padded, r * padded + row), r * row);
      }
      return tight;
    };
  }
  const group = gpu.createBindGroup(scenario.layout, entries as never);
  if (scenario.guarded) {
    gpu
      .createGuardedComputePipeline(scenario.guarded as never)
      .with(group)
      .dispatchThreads(...(scenario.threads as [number, number, number]));
  } else {
    gpu
      .createComputePipeline({ compute: scenario.kernel as never })
      .with(group)
      .dispatchWorkgroups(...workgroupsOf(scenario));
  }
  const result: Record<string, Uint8Array> = {};
  for (const [key, read] of Object.entries(reads)) result[key] = await read();
  return result;
}

/** The kernel's GPU form, resolved with the fallback's generator. */
function gpuForm(scenario: Scenario): ComputeModule {
  const { root, computes } = glslRoot({ compute: true });
  if (scenario.guarded) {
    const guarded = root.createGuardedComputePipeline(scenario.guarded as never);
    root.unwrap(innerPipeline(guarded) as never);
  } else {
    root.unwrap(root.createComputePipeline({ compute: scenario.kernel as never }));
  }
  const [verdict] = computes();
  if (!verdict || !('module' in verdict)) {
    throw new Error(`No GPU form: ${verdict && 'reason' in verdict ? verdict.reason : 'none'}`);
  }
  return verdict.module;
}

/** Runs the GPU form on WebGL 2 and returns each resource's contents afterwards, by layout key. */
function onWebGl(scenario: Scenario): Record<string, Uint8Array> {
  const module = gpuForm(scenario);
  const { root, recording } = glslRoot();
  const keys = Object.keys(scenario.inputs);
  const resources = new Map<string, Resource>();
  for (const [index, key] of keys.entries()) {
    const input = scenario.inputs[key] as Input;
    if (input.kind === 'buffer') {
      const buffer = root.createBuffer(input.schema as d.AnyWgslData, input.value as never);
      resources.set(`0:${index}`, { kind: 'buffer', bytes: recording.bytes(buffer).slice() });
    } else {
      resources.set(`0:${index}`, { ...input, data: input.data.slice() });
    }
  }
  if (scenario.threads) {
    // A guarded pipeline's own uniform holds the thread counts
    const size = module.blocks.find((block) => block.block.startsWith('sizeUniform'));
    if (size) {
      const words = new Uint32Array([...scenario.threads, 0]);
      resources.set(`${size.group}:${size.binding}`, {
        kind: 'buffer',
        bytes: new Uint8Array(words.buffer),
      });
    }
  }
  const gl = document.createElement('canvas').getContext('webgl2') as WebGL2RenderingContext;
  dispatch(gl, module, resources, workgroupsOf(scenario));
  return Object.fromEntries(
    keys.map((key, index) => {
      const each = resources.get(`0:${index}`) as Resource;
      return [key, each.kind === 'buffer' ? each.bytes : new Uint8Array(each.data.buffer)];
    }),
  );
}

/** Bytes as floats, so a mismatch reads as the values the kernel computed. */
const words = (bytes: Uint8Array) =>
  Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));

async function expectSame(scenario: Scenario) {
  const expected = await onWebGpu(scenario);
  const actual = onWebGl(scenario);
  for (const key of Object.keys(scenario.inputs)) {
    expect(words(actual[key] as Uint8Array), key).toEqual(words(expected[key] as Uint8Array));
  }
}

const gid = { gid: d.builtin.globalInvocationId };
const range = <T>(count: number, f: (i: number) => T): T[] =>
  Array.from({ length: count }, (_, i) => f(i));

describe('GPU forms run as WebGPU runs their kernels', () => {
  it('steps particles through an alias, with early returns and a uniform', async () => {
    const Agent = d.struct({ position: d.vec2f, fed: d.f32, awake: d.u32 });
    const layout = tgpu
      .bindGroupLayout({
        agents: { storage: d.arrayOf(Agent, 100), access: 'mutable' },
        speed: { uniform: d.f32 },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x >= 100) return;
        const agent = layout.$.agents[id.x] as d.Infer<typeof Agent>;
        agent.fed -= 0.25;
        if (agent.fed < 0) {
          agent.awake = 0;
          return;
        }
        layout.$.agents[id.x] = Agent({
          position: agent.position + d.vec2f(layout.$.speed, 2),
          fed: agent.fed * 2,
          awake: agent.awake + 1,
        });
      }),
      workgroups: [2, 1, 1],
      inputs: {
        agents: {
          kind: 'buffer',
          schema: d.arrayOf(Agent, 100),
          value: range(100, (i) => i).map((i) => ({
            position: d.vec2f(i, -i),
            fed: (i % 7) * 0.1,
            awake: i % 3,
          })),
        },
        speed: { kind: 'buffer', schema: d.f32, value: 1.5, uniform: true },
      },
    });
  });

  it('writes two buffers, of 4- and 12-byte elements, and reads its own writes back', async () => {
    const Triple = d.struct({ x: d.f32, y: d.f32, n: d.u32 });
    const layout = tgpu
      .bindGroupLayout({
        values: { storage: d.arrayOf(d.f32, 37), access: 'mutable' },
        triples: { storage: d.arrayOf(Triple, 37), access: 'mutable' },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const i = id.x;
        if (i >= 37) return;
        layout.$.values[i] = (layout.$.values[i] as number) * 2 + 1;
        layout.$.triples[i] = Triple({ x: d.f32(i), y: layout.$.values[i] as number, n: i * 3 });
        (layout.$.triples[i] as d.Infer<typeof Triple>).n += 1;
      }),
      workgroups: [1, 1, 1],
      inputs: {
        values: { kind: 'buffer', schema: d.arrayOf(d.f32, 37), value: range(37, (i) => i / 4) },
        triples: {
          kind: 'buffer',
          schema: d.arrayOf(Triple, 37),
          value: range(37, () => ({ x: 9, y: 9, n: 9 })),
        },
      },
    });
  });

  it('reads the neighbours of the buffer it writes, as they were', async () => {
    const layout = tgpu
      .bindGroupLayout({ cells: { storage: d.arrayOf(d.f32, 64), access: 'mutable' } })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x % 2 === 1 || id.x === 0) return;
        const left = layout.$.cells[id.x - 1] as number;
        const right = layout.$.cells[std.min(id.x + 1, 63)] as number;
        layout.$.cells[id.x] = left + right + (layout.$.cells[id.x] as number) * 10;
      }),
      workgroups: [1, 1, 1],
      inputs: {
        cells: { kind: 'buffer', schema: d.arrayOf(d.f32, 64), value: range(64, (i) => i * i) },
      },
    });
  });

  it('steps 48-byte elements, and leaves those past a partial dispatch alone', async () => {
    const Body = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f });
    const layout = tgpu
      .bindGroupLayout({ bodies: { storage: d.arrayOf(Body, 80), access: 'mutable' } })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [32], in: gid })(({ gid: id }) => {
        'use gpu';
        const body = layout.$.bodies[id.x] as d.Infer<typeof Body>;
        body.position += body.velocity;
        body.color.w += 1;
      }),
      workgroups: [2, 1, 1],
      inputs: {
        bodies: {
          kind: 'buffer',
          schema: d.arrayOf(Body, 80),
          value: range(80, (i) => ({
            position: d.vec3f(i, 0, 1),
            velocity: d.vec3f(1, i, -1),
            color: d.vec4f(0.5, 0.25, 1, i),
          })),
        },
      },
    });
  });

  it('writes an array that follows a header in its buffer', async () => {
    const Pool = d.struct({ count: d.u32, scale: d.f32, items: d.arrayOf(d.vec2f, 24) });
    const layout = tgpu.bindGroupLayout({ pool: { storage: Pool, access: 'mutable' } }).$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x >= layout.$.pool.count) return;
        layout.$.pool.items[id.x] = (layout.$.pool.items[id.x] as d.v2f) * layout.$.pool.scale;
      }),
      workgroups: [1, 1, 1],
      inputs: {
        pool: {
          kind: 'buffer',
          schema: Pool,
          value: { count: 20, scale: 3, items: range(24, (i) => d.vec2f(i, -i)) },
        },
      },
    });
  });

  it('diffuses one 2d texture into another', async () => {
    const layout = tgpu
      .bindGroupLayout({
        source: { texture: d.texture2d(), sampleType: 'unfilterable-float' },
        target: { storageTexture: d.textureStorage2d('rgba32float', 'write-only') },
      })
      .$idx(0);
    const [width, height] = [16, 8];
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x >= 16 || id.y >= 8) return;
        const cell = d.vec2i(id.xy);
        const here = std.textureLoad(layout.$.source, cell, 0);
        const right = std.textureLoad(
          layout.$.source,
          std.min(cell + d.vec2i(1, 0), d.vec2i(15, 7)),
          0,
        );
        const down = std.textureLoad(
          layout.$.source,
          std.min(cell + d.vec2i(0, 1), d.vec2i(15, 7)),
          0,
        );
        std.textureStore(layout.$.target, id.xy, (here + right + down) * 0.25);
      }),
      workgroups: [2, 1, 1],
      inputs: {
        source: {
          kind: 'texture',
          format: 'rgba32float',
          dimension: '2d',
          size: [width, height, 1],
          data: new Float32Array(range(width * height * 4, (i) => (i * 37) % 101)),
          storage: false,
        },
        target: {
          kind: 'texture',
          format: 'rgba32float',
          dimension: '2d',
          size: [width, height, 1],
          data: new Float32Array(width * height * 4).fill(-1),
          storage: true,
        },
      },
    });
  });

  it('steps a 3d texture in place', async () => {
    const layout = tgpu
      .bindGroupLayout({
        volume: { storageTexture: d.textureStorage3d('r32float', 'read-write') },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [4, 4, 4], in: gid })(({ gid: id }) => {
        'use gpu';
        const value = std.textureLoad(layout.$.volume, id);
        std.textureStore(layout.$.volume, id, value * 0.5 + d.vec4f(d.f32(id.z)));
      }),
      workgroups: [1, 1, 1],
      inputs: {
        volume: {
          kind: 'texture',
          format: 'r32float',
          dimension: '3d',
          size: [4, 4, 4],
          data: new Float32Array(range(64, (i) => i)),
          storage: true,
        },
      },
    });
  });
  it('sums a 3d texture along z, a column per invocation', async () => {
    const layout = tgpu
      .bindGroupLayout({
        source: { texture: d.texture3d(), sampleType: 'unfilterable-float' },
        summed: { storageTexture: d.textureStorage3d('rgba32float', 'write-only') },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [2, 2, 1], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x >= 1 && id.y >= 3) return;
        let total = d.vec4f();
        for (let k = d.u32(0); k < 5; k++) {
          total += std.textureLoad(layout.$.source, d.vec3u(id.xy, k), 0);
          if (k % 2 === 0 || id.y === 1) {
            std.textureStore(layout.$.summed, d.vec3u(id.xy, k), total);
          }
        }
      }),
      workgroups: [1, 2, 1],
      inputs: {
        source: {
          kind: 'texture',
          format: 'rgba32float',
          dimension: '3d',
          size: [4, 4, 5],
          data: new Float32Array(range(320, (i) => (i % 7) - 2)),
          storage: false,
        },
        summed: {
          kind: 'texture',
          format: 'rgba32float',
          dimension: '3d',
          size: [4, 4, 5],
          data: new Float32Array(range(320, (i) => -i)),
          storage: true,
        },
      },
    });
  });
});

/** Writes one half of a buffer from the other, as the uniform's `x` picks. */
const halves = (source: number) => {
  const layout = tgpu
    .bindGroupLayout({
      cells: { storage: d.arrayOf(d.u32, 128), access: 'mutable' },
      pass: { uniform: d.vec2u },
    })
    .$idx(0);
  return expectSame({
    layout,
    kernel: tgpu.computeFn({ workgroupSize: [8, 4], in: gid })(({ gid: id }) => {
      'use gpu';
      if (id.x >= 8 || id.y >= 6) return;
      const from = layout.$.pass.x;
      const here = id.y * 8 + id.x;
      const left = layout.$.cells[here + from * 64] as number;
      const next = layout.$.cells[((here + 1) % 48) + from * 64] as number;
      layout.$.cells[here + (1 - from) * 64] = left * 3 + next + layout.$.pass.y;
    }),
    workgroups: [1, 2, 1],
    inputs: {
      cells: { kind: 'buffer', schema: d.arrayOf(d.u32, 128), value: range(128, (i) => i * 7) },
      pass: { kind: 'buffer', schema: d.vec2u, value: d.vec2u(source, 5), uniform: true },
    },
  });
};

describe('GPU forms with offsets and strides run as WebGPU runs them', () => {
  it('writes one half of a buffer from the other', () => halves(0));
  it('writes the other half when the uniform flips', () => halves(1));

  it('copies one half into the other at offsets read through an alias of a uniform', async () => {
    const Halves = d.struct({ now: d.u32, before: d.u32 });
    const layout = tgpu
      .bindGroupLayout({
        cells: { storage: d.arrayOf(d.f32, 96), access: 'mutable' },
        pass: { uniform: Halves },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [16], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x >= 48 || id.x % 5 === 0) return;
        const time = layout.$.pass;
        layout.$.cells[time.before + id.x] = (layout.$.cells[time.now + id.x] as number) * 2;
      }),
      workgroups: [3, 1, 1],
      inputs: {
        cells: { kind: 'buffer', schema: d.arrayOf(d.f32, 96), value: range(96, (i) => i - 40) },
        pass: { kind: 'buffer', schema: Halves, value: { now: 48, before: 0 }, uniform: true },
      },
    });
  });

  it('writes every third element from a constant phase', async () => {
    const layout = tgpu
      .bindGroupLayout({ out: { storage: d.arrayOf(d.f32, 200), access: 'mutable' } })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const at = id.x * 3 + 2;
        layout.$.out[at] = (layout.$.out[at] as number) * 2 + (layout.$.out[at - 1] as number);
      }),
      workgroups: [1, 1, 1],
      inputs: {
        out: { kind: 'buffer', schema: d.arrayOf(d.f32, 200), value: range(200, (i) => i - 50) },
      },
    });
  });

  it('writes every other element from a phase a uniform masks', async () => {
    const layout = tgpu
      .bindGroupLayout({
        out: { storage: d.arrayOf(d.u32, 130), access: 'mutable' },
        parity: { uniform: d.u32 },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        layout.$.out[id.x * 2 + (layout.$.parity & 1)] = id.x + 1000;
      }),
      workgroups: [1, 1, 1],
      inputs: {
        out: { kind: 'buffer', schema: d.arrayOf(d.u32, 130), value: range(130, (i) => i) },
        parity: { kind: 'buffer', schema: d.u32, value: 3, uniform: true },
      },
    });
  });

  for (const parity of [0, 1]) {
    it(`sweeps the ${parity ? 'black' : 'red'} cells of a 3d grid`, async () => {
      const layout = tgpu
        .bindGroupLayout({
          value: { storageTexture: d.textureStorage3d('r32float', 'read-write') },
        })
        .$idx(0);
      await expectSame({
        layout,
        kernel: tgpu.computeFn({ workgroupSize: [2, 4, 2], in: gid })(({ gid: id }) => {
          'use gpu';
          const size = d.vec3i(std.textureDimensions(layout.$.value));
          const cell = d.vec3i(
            d.i32(id.x * 2 + ((id.y + id.z + parity) & 1)),
            d.i32(id.y),
            d.i32(id.z),
          );
          if (cell.x >= size.x || cell.y >= size.y || cell.z >= size.z) return;
          let around = d.f32(0);
          if (cell.x > 0) around += std.textureLoad(layout.$.value, cell - d.vec3i(1, 0, 0)).x;
          if (cell.x < size.x - 1) {
            around += std.textureLoad(layout.$.value, cell + d.vec3i(1, 0, 0)).x;
          }
          const was = std.textureLoad(layout.$.value, cell).x;
          std.textureStore(layout.$.value, cell, d.vec4f(was * 0.5 + around * 0.25, 0, 0, 0));
        }),
        workgroups: [2, 1, 2],
        inputs: {
          value: {
            kind: 'texture',
            format: 'r32float',
            dimension: '3d',
            size: [7, 4, 4],
            data: new Float32Array(range(112, (i) => (i * 13) % 17)),
            storage: true,
          },
        },
      });
    });
  }
});

it('stores a checkerboard a uniform turns', async () => {
  const layout = tgpu
    .bindGroupLayout({
      target: { storageTexture: d.textureStorage2d('rgba32float', 'write-only') },
      turn: { uniform: d.u32 },
    })
    .$idx(0);
  await expectSame({
    layout,
    kernel: tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: id }) => {
      'use gpu';
      const at = d.vec2u(id.x * 2 + ((id.y + layout.$.turn) % 2), id.y);
      if (at.x >= 7) return;
      std.textureStore(layout.$.target, at, d.vec4f(d.f32(id.x), d.f32(id.y), 1, 0.5));
    }),
    workgroups: [1, 2, 1],
    inputs: {
      target: {
        kind: 'texture',
        format: 'rgba32float',
        dimension: '2d',
        size: [7, 6, 1],
        data: new Float32Array(range(168, (i) => -i)),
        storage: true,
      },
      turn: { kind: 'buffer', schema: d.u32, value: 3, uniform: true },
    },
  });
});

it('writes a grid from a dispatch wider than it, past a guard that returns', async () => {
  const layout = tgpu
    .bindGroupLayout({ out: { storage: d.arrayOf(d.f32, 40), access: 'mutable' } })
    .$idx(0);
  await expectSame({
    layout,
    kernel: tgpu.computeFn({ workgroupSize: [4, 2], in: gid })(({ gid: id }) => {
      'use gpu';
      if (id.x >= 6 || id.y >= 5) return;
      layout.$.out[id.y * 6 + id.x] = d.f32(id.x * 10 + id.y);
    }),
    workgroups: [2, 3, 1],
    inputs: {
      out: { kind: 'buffer', schema: d.arrayOf(d.f32, 40), value: range(40, (i) => -i) },
    },
  });
});

describe('GPU forms with one chosen writer run as WebGPU runs them', () => {
  const Pick = d.struct({ hit: d.u32, at: d.vec2f, count: d.atomic(d.u32) });
  const View = d.struct({ cursor: d.vec2f, size: d.vec2u });

  const picking = (cursor: d.v2f) => {
    const layout = tgpu
      .bindGroupLayout({
        image: { storage: d.arrayOf(d.f32, 64), access: 'mutable' },
        pick: { storage: Pick, access: 'mutable' },
        view: { uniform: View },
      })
      .$idx(0);
    return expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: id }) => {
        'use gpu';
        const view = View(layout.$.view);
        if (id.x >= view.size.x || id.y >= view.size.y) return;
        const value = d.f32(id.x * 3 + id.y) * 0.5;
        if (d.u32(view.cursor.x) === id.x && id.y === d.u32(view.cursor.y)) {
          layout.$.pick.hit = d.u32(value > 2);
          layout.$.pick.at = d.vec2f(id.xy) + value;
          std.atomicAdd(layout.$.pick.count, 1);
          if (value > 100) return;
        }
        layout.$.image[id.y * 8 + id.x] = value;
      }),
      workgroups: [2, 2, 1],
      inputs: {
        image: { kind: 'buffer', schema: d.arrayOf(d.f32, 64), value: range(64, (i) => -i) },
        pick: { kind: 'buffer', schema: Pick, value: { hit: 7, at: d.vec2f(-1, -2), count: 5 } },
        view: {
          kind: 'buffer',
          schema: View,
          value: { cursor, size: d.vec2u(7, 8) },
          uniform: true,
        },
      },
    });
  };

  it('writes the pick of the invocation under the cursor', () => picking(d.vec2f(5.5, 3.2)));
  it('keeps the pick when no invocation is under the cursor', () => picking(d.vec2f(7.5, 3.2)));
});

describe('GPU forms of wider kernels run as WebGPU runs them', () => {
  it('runs a guarded pipeline that writes a buffer at its linear index', async () => {
    const layout = tgpu
      .bindGroupLayout({ out: { storage: d.arrayOf(d.f32, 256), access: 'mutable' } })
      .$idx(0);
    await expectSame({
      layout,
      guarded: (x: number, y: number) => {
        'use gpu';
        const index = x + y * 16;
        layout.$.out[index] = d.f32(x * 100 + y) + (layout.$.out[index] as number);
      },
      threads: [16, 12, 1],
      inputs: {
        out: { kind: 'buffer', schema: d.arrayOf(d.f32, 256), value: range(256, (i) => -i) },
      },
    });
  });

  it('runs a guarded pipeline that stores a 2d texture', async () => {
    const layout = tgpu
      .bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('rgba32float', 'write-only') },
      })
      .$idx(0);
    await expectSame({
      layout,
      guarded: (x: number, y: number) => {
        'use gpu';
        std.textureStore(layout.$.target, d.vec2u(x, y), d.vec4f(d.f32(x), d.f32(y), 1, 0.5));
      },
      threads: [20, 9, 1],
      inputs: {
        target: {
          kind: 'texture',
          format: 'rgba32float',
          dimension: '2d',
          size: [24, 12, 1],
          data: new Float32Array(24 * 12 * 4).fill(-1),
          storage: true,
        },
      },
    });
  });

  it('writes a 3d grid at its Horner index, through a helper', async () => {
    const Cell = d.struct({ value: d.f32, hits: d.u32, tag: d.vec2u });
    const layout = tgpu
      .bindGroupLayout({ cells: { storage: d.arrayOf(Cell, 8 * 4 * 2), access: 'mutable' } })
      .$idx(0);
    const visit = (x: number, y: number, z: number) => {
      'use gpu';
      const cell = layout.$.cells[(z * 4 + y) * 8 + x] as d.Infer<typeof Cell>;
      cell.hits += 1;
      cell.tag = d.vec2u(x, y * 10 + z);
    };
    await expectSame({
      layout,
      kernel: tgpu.computeFn({
        workgroupSize: [4, 4, 2],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid: id }) => {
        'use gpu';
        if (id.x >= 7) return;
        visit(id.x, id.y, id.z);
      }),
      workgroups: [2, 1, 1],
      inputs: {
        cells: {
          kind: 'buffer',
          schema: d.arrayOf(Cell, 64),
          value: range(64, (i) => ({ value: i, hits: i % 5, tag: d.vec2u(9, 9) })),
        },
      },
    });
  });

  it('writes state and a history entry from a single invocation', async () => {
    const State = d.struct({ count: d.u32, total: d.f32, last: d.vec2f });
    const layout = tgpu
      .bindGroupLayout({
        state: { storage: State, access: 'mutable' },
        history: { storage: d.arrayOf(d.f32, 8), access: 'mutable' },
        input: { storage: d.arrayOf(d.f32, 4), access: 'readonly' },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        const state = layout.$.state;
        state.count += 1;
        let sum = d.f32(0);
        for (let i = 0; i < 4; i++) sum += layout.$.input[i] as number;
        state.total += sum;
        state.last = d.vec2f(sum, state.total);
        layout.$.history[state.count % 8] = state.total;
      }),
      workgroups: [1, 1, 1],
      inputs: {
        state: {
          kind: 'buffer',
          schema: State,
          value: { count: 6, total: 1.5, last: d.vec2f(0, 0) },
        },
        history: { kind: 'buffer', schema: d.arrayOf(d.f32, 8), value: range(8, (i) => i) },
        input: { kind: 'buffer', schema: d.arrayOf(d.f32, 4), value: [1, 2, 3, 4.5] },
      },
    });
  });
});

describe('GPU forms of kernels with atomics on their own cells run as WebGPU runs them', () => {
  it('takes and clears its own counters, as a drop-and-diffuse step does', async () => {
    const layout = tgpu
      .bindGroupLayout({
        dropped: { storage: d.arrayOf(d.atomic(d.u32), 16 * 8), access: 'mutable' },
        totals: { storage: d.arrayOf(d.u32, 16 * 8), access: 'mutable' },
      })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })(
        ({ gid: id }) => {
          'use gpu';
          if (id.x >= 16 || id.y >= 8) return;
          const index = id.y * 16 + id.x;
          const fresh = std.atomicLoad(layout.$.dropped[index] as d.atomicU32);
          std.atomicStore(layout.$.dropped[index] as d.atomicU32, 0);
          const before = std.atomicAdd(layout.$.dropped[index] as d.atomicU32, id.x);
          std.atomicMax(layout.$.dropped[index] as d.atomicU32, 5);
          layout.$.totals[index] = (layout.$.totals[index] as number) + fresh * 10 + before;
        },
      ),
      workgroups: [2, 1, 1],
      inputs: {
        dropped: { kind: 'buffer', schema: d.arrayOf(d.u32, 128), value: range(128, (i) => i * 3) },
        totals: { kind: 'buffer', schema: d.arrayOf(d.u32, 128), value: range(128, (i) => i) },
      },
    });
  });

  it('counts with atomics from a single invocation', async () => {
    const Counters = d.struct({ alive: d.atomic(d.u32), spawned: d.atomic(d.i32), last: d.u32 });
    const layout = tgpu
      .bindGroupLayout({ counters: { storage: Counters, access: 'mutable' } })
      .$idx(0);
    await expectSame({
      layout,
      kernel: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        const previous = std.atomicExchange(layout.$.counters.alive, 7);
        std.atomicSub(layout.$.counters.spawned, 3);
        std.atomicOr(layout.$.counters.alive, 16);
        layout.$.counters.last = previous + std.atomicLoad(layout.$.counters.alive);
      }),
      workgroups: [1, 1, 1],
      inputs: {
        counters: {
          kind: 'buffer',
          schema: Counters,
          value: { alive: 40, spawned: -2, last: 0 },
        },
      },
    });
  });
});
