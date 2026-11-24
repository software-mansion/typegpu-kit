import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { PASS_VERTEX } from '../../src/glsl/computeModule.ts';
import { glslRoot, type ComputeVerdict } from './utils.ts';
import { problems, validator, type Validator } from './validator.ts';

let gl: Validator;
beforeAll(async () => {
  gl = await validator();
});
afterAll(() => gl.close());

const Agent = d.struct({ position: d.vec2f, fed: d.f32, awake: d.u32 });
const Holder = d.struct({ count: d.u32, total: d.f32 });

type Make = (root: TgpuRoot) => object;

const gid = { gid: d.builtin.globalInvocationId };
/** A kernel over a line of 64-invocation workgroups. */
const line = tgpu.computeFn({ workgroupSize: [64], in: gid });

/** Each kernel with what the GPU form must say about it: a grid, or the start of a reason. */
const kernels: [name: string, make: Make, verdict: number | string][] = [
  [
    'a particle step through an alias, with early returns',
    (root) => {
      const agents = root.createMutable(d.arrayOf(Agent, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x >= 64) return;
        const agent = agents.$[id.x] as d.Infer<typeof Agent>;
        agent.fed -= 0.1;
        if (agent.fed < 0) return;
        agents.$[id.x] = Agent({
          position: agent.position + d.vec2f(1, 0),
          fed: agent.fed,
          awake: 1,
        });
      });
    },
    1,
  ],
  [
    'writes to two buffers, through a const index and casts',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.f32, 64));
      const b = root.createMutable(d.arrayOf(d.vec4u, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const i = id.x;
        a.$[i] = (a.$[i] as number) + 1;
        b.$[d.u32(d.i32(id.x))] = d.vec4u(i);
      });
    },
    1,
  ],
  [
    'reads the neighbours of the buffer it writes',
    (root) => {
      const cells = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const left = cells.$[std.max(id.x, 1) - 1] as number;
        cells.$[id.x] = (cells.$[id.x] as number) * 0.5 + left * 0.5;
      });
    },
    1,
  ],
  [
    'diffuses one texture into another',
    (root) => {
      const layout = tgpu.bindGroupLayout({
        source: { texture: d.texture2d() },
        target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
      });
      void root;
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        const cell = d.vec2i(id.xy);
        const here = std.textureLoad(layout.$.source, cell, 0);
        const right = std.textureLoad(layout.$.source, cell + d.vec2i(1, 0), 0);
        std.textureStore(layout.$.target, id.xy, (here + right) * 0.5);
      });
    },
    2,
  ],
  [
    'steps a 3d texture in place',
    () => {
      const layout = tgpu.bindGroupLayout({
        volume: { storageTexture: d.textureStorage3d('r32float', 'read-write') },
      });
      return tgpu.computeFn({ workgroupSize: [4, 4, 4], in: gid })(({ gid: id }) => {
        'use gpu';
        const value = std.textureLoad(layout.$.volume, id);
        std.textureStore(layout.$.volume, id, value * 0.5);
      });
    },
    3,
  ],
  [
    'stores to a 2d-array layer from the z axis',
    () => {
      const layout = tgpu.bindGroupLayout({
        layers: { storageTexture: d.textureStorage2dArray('rgba8unorm', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        std.textureStore(layout.$.layers, id.xy, id.z, d.vec4f(1));
      });
    },
    3,
  ],
  [
    'reads its position from other builtins',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 256));
      return tgpu.computeFn({
        workgroupSize: [64],
        in: {
          id: d.builtin.globalInvocationId,
          index: d.builtin.localInvocationIndex,
          group: d.builtin.workgroupId,
          groups: d.builtin.numWorkgroups,
        },
      })(({ id, index, group, groups }) => {
        'use gpu';
        out.$[id.x] = index + group.x * 1000 + groups.x * 100000;
      });
    },
    1,
  ],
  [
    'writes a flattened index',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 4096));
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[id.y * 64 + id.x] = 1;
      });
    },
    2,
  ],
  [
    'writes the next element',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      return line(({ gid: id }) => {
        'use gpu';
        out.$[id.x + 1] = 1;
      });
    },
    "writes an element of 'out",
  ],
  [
    'writes in a loop',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      return line(({ gid: id }) => {
        'use gpu';
        for (let i = d.u32(0); i < 4; i++) {
          out.$[i] = d.f32(id.x);
        }
      });
    },
    "writes an element of 'out",
  ],
  [
    'adds atomically',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 1));
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicAdd(counter.$[0] as d.atomicU32, 1);
        out.$[id.x] = 1;
      });
    },
    "'atomicAdd'",
  ],
  [
    'waits at a barrier',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.workgroupBarrier();
        out.$[id.x] = 1;
      });
    },
    "'workgroupBarrier'",
  ],
  [
    'shares a workgroup variable',
    (root) => {
      const shared = tgpu.workgroupVar(d.arrayOf(d.f32, 64));
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        shared.$[0] = 1;
        out.$[id.x] = shared.$[0] as number;
      });
    },
    "A 'workgroup' variable",
  ],
  [
    'writes in a helper',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      const write = (i: number) => {
        'use gpu';
        out.$[i] = 1;
      };
      return line(({ gid: id }) => {
        'use gpu';
        write(id.x);
      });
    },
    1,
  ],
  [
    'writes nothing',
    (root) => {
      const input = root.createReadonly(d.arrayOf(d.f32, 64));
      return line(({ gid: id }) => {
        'use gpu';
        let value = input.$[id.x] as number;
        value += 1;
      });
    },
    'writes nothing',
  ],
  [
    'writes a struct outside an array',
    (root) => {
      const holder = root.createMutable(Holder);
      return line(({ gid: id }) => {
        'use gpu';
        if (id.x === 0) holder.$.count = 1;
      });
    },
    "writes 'holder' outside an array",
  ],
  [
    'stores to another texel',
    () => {
      const layout = tgpu.bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        std.textureStore(layout.$.target, id.yx, d.vec4f(1));
      });
    },
    "stores to a texel of 'target",
  ],
  [
    'stores to a format WebGL 2 cannot render to',
    () => {
      const layout = tgpu.bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('rgba8snorm', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        std.textureStore(layout.$.target, id.xy, d.vec4f(1));
      });
    },
    "stores to a 'rgba8snorm' texture",
  ],
  [
    'moves the index it writes by',
    () => {
      const layout = tgpu.bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        const cell = d.vec2u(id.xy);
        cell.x += 1;
        std.textureStore(layout.$.target, cell, d.vec4f(1));
      });
    },
    "modifies 'cell'",
  ],
  [
    'writes a buffer from a square workgroup',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[id.x] = 1;
      });
    },
    'writes a buffer from a workgroup larger than its indices cover',
  ],
  [
    'reads the whole array it writes',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 4));
      return line(({ gid: id }) => {
        'use gpu';
        let sum = d.f32(0);
        for (const value of out.$) sum += value;
        out.$[id.x] = sum;
      });
    },
    "reads all of 'out",
  ],
  [
    'writes a 3d index in Horner form',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 256));
      return tgpu.computeFn({ workgroupSize: [4, 4, 4], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[(id.z * 4 + id.y) * 8 + id.x] = 1;
      });
    },
    3,
  ],
  [
    'writes through a helper at the indices it is given',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 256));
      const put = (x: number, y: number) => {
        'use gpu';
        const index = x + y * 16;
        out.$[index] = d.f32(x);
      };
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        put(id.x, id.y);
      });
    },
    2,
  ],
  [
    'stores through a helper at the texel it is given',
    () => {
      const layout = tgpu.bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
      });
      const paint = (cell: d.v2u) => {
        'use gpu';
        std.textureStore(layout.$.target, cell, d.vec4f(1));
      };
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        paint(id.xy);
      });
    },
    2,
  ],
  [
    'calls a writing helper with another index too',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      const put = (i: number) => {
        'use gpu';
        out.$[i] = 1;
      };
      return line(({ gid: id }) => {
        'use gpu';
        put(id.x);
        put(id.x + 1);
      });
    },
    "calls 'put' with other invocation indices",
  ],
  [
    'writes a strided index',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 192));
      return line(({ gid: id }) => {
        'use gpu';
        out.$[id.x * 3 + 1] = 1;
      });
    },
    "writes an element of 'out",
  ],
  [
    'writes a flattened index of a width it reads from a uniform',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 4096));
      const width = root.createUniform(d.u32, 64);
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[id.y * width.$ + id.x] = 1;
      });
    },
    "writes an element of 'out",
  ],
  [
    'writes a large buffer from one invocation',
    (root) => {
      const big = root.createMutable(d.arrayOf(d.f32, 1024));
      return tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        big.$[3] = 1;
      });
    },
    "writes 'big' whole, which is larger than 1024 bytes",
  ],
  [
    'reads through a helper at its own and other indices',
    (root) => {
      const input = root.createReadonly(d.arrayOf(d.f32, 64));
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      const at = (i: number) => {
        'use gpu';
        return input.$[i] as number;
      };
      return line(({ gid: id }) => {
        'use gpu';
        out.$[id.x] = at(id.x) + at(std.min(id.x + 1, 63));
      });
    },
    1,
  ],
];

function verdictOf(make: Make): ComputeVerdict {
  const { root, computes } = glslRoot();
  const pipeline = root.createComputePipeline({ compute: make(root) as never });
  root.unwrap(pipeline);
  const [verdict] = computes();
  if (!verdict) throw new Error('No compute module');
  return verdict;
}

describe('GPU forms of compute kernels', () => {
  it('owns whole bindings in a dispatch of one invocation', () => {
    const verdict = verdictOf((root) => {
      const holder = root.createMutable(Holder);
      const history = root.createMutable(d.arrayOf(d.f32, 8));
      return tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        holder.$.count += 1;
        history.$[holder.$.count % 8] = holder.$.total;
      });
    });
    if (!('module' in verdict)) throw new Error(verdict.reason);
    expect(verdict.module.single).toBe(true);
    expect(verdict.module.passes.map((pass) => pass.target.kind)).toEqual(['buffer', 'buffer']);
  });

  it.each(kernels)('%s', async (_, make, expected) => {
    const verdict = verdictOf(make);
    if (typeof expected === 'string') {
      expect('reason' in verdict ? verdict.reason : 'eligible').toContain(expected);
      return;
    }
    if ('reason' in verdict) throw new Error(`Ineligible: ${verdict.reason}`);
    expect(verdict.module.grid).toBe(expected);
    const reports = await gl.check(
      verdict.module.passes.map((pass) => ({
        kind: 'render' as const,
        vertex: PASS_VERTEX,
        fragment: pass.fragment,
        buffers: [],
        textures: [],
        levels: [],
      })),
    );
    for (const [i, report] of reports.entries()) {
      expect(problems(report), verdict.module.passes[i]?.fragment).toEqual([]);
    }
  });
});
