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
/** A grid, a reason it has no form, or `scatter` for a form with a pass of points. */
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
    'writes a column of a 3d texture per invocation',
    () => {
      const layout = tgpu.bindGroupLayout({
        volume: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        for (let k = d.u32(0); k < 4; k++) {
          std.textureStore(layout.$.volume, d.vec3u(id.xy, k), d.vec4f(d.f32(k)));
        }
      });
    },
    2,
  ],
  [
    'writes a column of a 2d-array texture per invocation',
    () => {
      const layout = tgpu.bindGroupLayout({
        layers: { storageTexture: d.textureStorage2dArray('rgba8unorm', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        std.textureStore(layout.$.layers, id.xy, id.x % 3, d.vec4f(1));
      });
    },
    2,
  ],
  [
    'reads back the column it writes',
    () => {
      const layout = tgpu.bindGroupLayout({
        volume: { storageTexture: d.textureStorage3d('r32float', 'read-write') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        for (let k = d.u32(1); k < 4; k++) {
          const below = std.textureLoad(layout.$.volume, d.vec3u(id.xy, k - 1));
          std.textureStore(layout.$.volume, d.vec3u(id.xy, k), below + 1);
        }
      });
    },
    "loads from 'volume', of which an invocation writes a whole column",
  ],
  [
    "writes another invocation's column",
    () => {
      const layout = tgpu.bindGroupLayout({
        volume: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        std.textureStore(layout.$.volume, d.vec3u(id.yx, 0), d.vec4f(1));
      });
    },
    'scatter',
  ],
  [
    'writes a pick from the invocation a uniform chooses',
    (root) => {
      const cursor = root.createUniform(d.vec2u);
      const pick = root.createMutable(Holder);
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        const at = d.vec2u(cursor.$);
        if (id.x === at.x && id.y === at.y && id.x > 2) {
          pick.$.count = id.x;
        }
      });
    },
    2,
  ],
  [
    'chooses by a variable it reassigns',
    (root) => {
      const cursor = root.createUniform(d.vec2u);
      const pick = root.createMutable(Holder);
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        let at = d.vec2u(cursor.$);
        for (let i = 0; i < 2; i++) {
          if (id.x === at.x && id.y === at.y) {
            pick.$.count = id.x;
          }
          at = id.xy;
        }
      });
    },
    "finds its cells through 'at', which it reassigns",
  ],
  [
    'chooses on fewer axes than its workgroup',
    (root) => {
      const cursor = root.createUniform(d.vec2u);
      const pick = root.createMutable(Holder);
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x === cursor.$.x) pick.$.count = id.y;
      });
    },
    'chooses an invocation on fewer axes than its workgroup spans',
  ],
  [
    'chooses by what another invocation writes',
    (root) => {
      const pick = root.createMutable(Holder);
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x === pick.$.count) pick.$.total = d.f32(id.x);
      });
    },
    "writes 'pick' outside an array",
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
    1,
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
    'at computed cells inside a loop',
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
    'scatter',
  ],
  [
    'adds through a helper at cells it computes',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.f32, 64));
      const off = false;
      const deposit = (cell: number, amount: number) => {
        'use gpu';
        if (!off) {
          std.atomicAdd(counter.$[cell % 4], d.u32(amount));
        }
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        deposit(id.x, 2);
        if (id.x > 3) deposit(id.x + 1, 1);
        out.$[id.x] = 1;
      });
    },
    'scatter',
  ],
  [
    'writes cells it computes through a helper, twice',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 128));
      const put = (cell: number, value: number) => {
        'use gpu';
        out.$[cell] = value;
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        put((id.x * 5) % 64, d.f32(1));
        if (id.x > 7) put(64 + ((id.x * 3) % 64), d.f32(2));
      });
    },
    'scatter',
  ],
  [
    'writes cells it computes through a helper inside a loop',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 128));
      const put = (cell: number) => {
        'use gpu';
        out.$[cell] = 1;
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        for (let k = d.u32(0); k < 2; k++) put((id.x * 5 + k) % 128);
      });
    },
    'at computed cells inside a loop',
  ],
  [
    'takes slots of a counter it then writes',
    (root) => {
      const head = root.createMutable(d.atomic(d.u32));
      const ring = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const slot = std.atomicAdd(head.$, 1) % 64;
        ring.$[slot] = d.f32(id.x);
      });
    },
    'scatter',
  ],
  [
    'decides by a slot it takes',
    (root) => {
      const head = root.createMutable(d.atomic(d.u32));
      const ring = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const slot = std.atomicAdd(head.$, 1);
        if (slot < 64) ring.$[slot] = d.f32(id.x);
      });
    },
    'decides a branch on a value it takes from a counter',
  ],
  [
    'takes slots by more than one',
    (root) => {
      const head = root.createMutable(d.atomic(d.u32));
      const ring = root.createMutable(d.arrayOf(d.f32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const slot = std.atomicAdd(head.$, 2) % 64;
        ring.$[slot] = d.f32(id.x);
      });
    },
    "adds other than 1 to 'head",
  ],
  [
    'adds to shared cells inside a loop',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        for (let k = d.u32(0); k < id.x; k++) std.atomicAdd(counter.$[k % 4], 1);
      });
    },
    'adds to shared cells inside a loop',
  ],
  [
    'reads a buffer it adds to',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicAdd(counter.$[id.x % 4], 1);
        out.$[id.x] = std.atomicLoad(counter.$[0]);
      });
    },
    'which it adds to at shared cells',
  ],
  [
    'uses the old value an addition returns',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[id.x] = std.atomicAdd(counter.$[id.x % 4], 1);
      });
    },
    "'atomicAdd' on an element of 'counter' that is not the invocation's own",
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
    'writes a struct from the one invocation a constant chooses',
    (root) => {
      const holder = root.createMutable(Holder);
      return line(({ gid: id }) => {
        'use gpu';
        if (id.x === 0) holder.$.count = 1;
      });
    },
    1,
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
    'scatter',
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
    1,
  ],
  [
    'writes at a stride past its phase',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.f32, 192));
      return line(({ gid: id }) => {
        'use gpu';
        out.$[id.x * 2 + 3] = 1;
      });
    },
    'scatter',
  ],
  [
    'writes its own element past an offset a uniform holds',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 128));
      const half = root.createUniform(d.u32);
      return line(({ gid: id }) => {
        'use gpu';
        const from = half.$;
        out.$[id.x + (1 - from) * 64] = out.$[id.x + from * 64] as number;
      });
    },
    1,
  ],
  [
    'writes at an offset it reads from what it writes',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 128));
      return line(({ gid: id }) => {
        'use gpu';
        out.$[id.x + (out.$[0] as number)] = 1;
      });
    },
    "writes an element of 'out",
  ],
  [
    'writes at an offset it reassigns',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 128));
      const half = root.createUniform(d.u32);
      return line(({ gid: id }) => {
        'use gpu';
        let from = half.$;
        if (id.x > 3) from = d.u32(2);
        out.$[id.x + from] = 1;
      });
    },
    "finds its cells through 'from",
  ],
  [
    'stores red cells of a red-black grid',
    () => {
      const layout = tgpu.bindGroupLayout({
        volume: { storageTexture: d.textureStorage3d('r32float', 'read-write') },
      });
      return tgpu.computeFn({ workgroupSize: [4, 4, 4], in: gid })(({ gid: id }) => {
        'use gpu';
        const cell = d.vec3i(d.i32(id.x * 2 + ((id.y + id.z + 1) & 1)), d.i32(id.y), d.i32(id.z));
        const was = std.textureLoad(layout.$.volume, cell).x;
        const left = std.textureLoad(layout.$.volume, cell - d.vec3i(1, 0, 0)).x;
        std.textureStore(layout.$.volume, cell, d.vec4f(was * 0.5 + left, 0, 0, 0));
      });
    },
    3,
  ],
  [
    'stores a checkerboard a uniform turns',
    (root) => {
      const turn = root.createUniform(d.u32);
      const layout = tgpu.bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        const at = d.vec2u(id.x * 2 + ((id.y + turn.$) % 2), id.y);
        std.textureStore(layout.$.target, at, d.vec4f(1));
      });
    },
    2,
  ],
  [
    'stores at a remainder that is not a mask',
    () => {
      const layout = tgpu.bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('rgba16float', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        std.textureStore(layout.$.target, d.vec2u(id.x * 3 + (id.y % 3), id.y), d.vec4f(1));
      });
    },
    'scatter',
  ],
  [
    'stores at a residue of its own axis',
    () => {
      const layout = tgpu.bindGroupLayout({
        target: { storageTexture: d.textureStorage2d('r32float', 'write-only') },
      });
      return tgpu.computeFn({ workgroupSize: [8, 8], in: gid })(({ gid: id }) => {
        'use gpu';
        std.textureStore(layout.$.target, d.vec2u(id.x * 2 + (id.x & 1), id.y), d.vec4f(1));
      });
    },
    'scatter',
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
    'scatter',
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
    if (expected === 'scatter') {
      if ('reason' in verdict) throw new Error(`Ineligible: ${verdict.reason}`);
      expect(verdict.module.passes.some((pass) => pass.scatter)).toBe(true);
    } else if (typeof expected === 'string') {
      expect('reason' in verdict ? verdict.reason : 'eligible').toContain(expected);
      return;
    }
    if ('reason' in verdict) throw new Error(`Ineligible: ${verdict.reason}`);
    if (expected !== 'scatter') expect(verdict.module.grid).toBe(expected);
    const reports = await gl.check(
      verdict.module.passes.map((pass) => ({
        kind: 'render' as const,
        vertex: pass.scatter?.vertex ?? PASS_VERTEX,
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
