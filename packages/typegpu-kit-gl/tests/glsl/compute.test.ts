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
    'reads cells through a helper resolved first, then writes at computed cells',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const peek = (i: number) => {
        'use gpu';
        return out.$[i % 64] as number;
      };
      const poke = (i: number, value: number) => {
        'use gpu';
        out.$[(i * 5) % 64] = value;
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const seen = peek(id.x + 1);
        poke(id.x, seen + 1);
      });
    },
    'scatter',
  ],
  [
    'reads a cell, then writes at a computed cell',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const seen = out.$[id.x] as number;
        out.$[(id.x * 5) % 64] = seen + 1;
      });
    },
    'scatter',
  ],
  [
    'reads cells in a constant loop, then writes at a computed cell',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        let sum = d.u32(0);
        for (let k = d.u32(0); k < 4; k++) sum += out.$[(id.x + k) % 64] as number;
        out.$[(id.x * 5) % 64] = sum;
      });
    },
    'scatter',
  ],
  [
    'writes at a computed cell in an arm that returns, then reads',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x % 2 === 0) {
          out.$[(id.x * 5) % 64] = 1;
          return;
        }
        out.$[(id.x * 3) % 64] = (out.$[id.x] as number) + 2;
      });
    },
    'scatter',
  ],
  [
    'writes at a computed cell, then reads it',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[(id.x * 5) % 64] = 1;
        out.$[(id.x * 3) % 64] = (out.$[id.x] as number) + 1;
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    'reads and writes at computed cells in one loop',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        for (let k = d.u32(0); k < 2; k++) {
          out.$[(id.x * 5 + k) % 64] = (out.$[(id.x + k) % 64] as number) + 1;
        }
      });
    },
    'inside a loop',
  ],
  [
    'writes through a helper before reading through another',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const peek = (i: number) => {
        'use gpu';
        return out.$[i % 64] as number;
      };
      const poke = (i: number, value: number) => {
        'use gpu';
        out.$[(i * 5) % 64] = value;
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        poke(id.x, d.u32(1));
        out.$[(id.x * 3) % 64] = peek(id.x + 1) + 1;
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    'takes from a counter through a pointer',
    (root) => {
      const head = root.createMutable(d.atomic(d.u32));
      const ring = root.createMutable(d.arrayOf(d.u32, 64));
      const take = tgpu.fn(
        [d.ptrStorage(d.atomic(d.u32), 'read-write')],
        d.u32,
      )((counter) => {
        'use gpu';
        return std.atomicAdd(counter as unknown as d.atomicU32, 1);
      });
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        ring.$[take(head.$ as never) % 64] = id.x;
      });
    },
    'a helper function takes the pointer',
  ],
  ...blockKernels(),
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
    'loads a buffer it adds to after adding to it',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicAdd(counter.$[id.x % 4], 1);
        out.$[id.x] = std.atomicLoad(counter.$[0]);
      });
    },
    'where it may have added to it',
  ],
  [
    'loads a buffer before adding to it',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const seen = std.atomicLoad(counter.$[0]);
        std.atomicAdd(counter.$[id.x % 4], 1);
        out.$[id.x] = seen;
      });
    },
    'scatter',
  ],
  [
    'loads a buffer in a constant loop before adding to it',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        let seen = d.u32(0);
        for (let k = d.u32(0); k < 4; k++) seen += std.atomicLoad(counter.$[k]);
        std.atomicAdd(counter.$[id.x % 4], 1);
        out.$[id.x] = seen;
      });
    },
    'scatter',
  ],
  [
    'loads a buffer after an arm that adds and returns',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x % 2 === 0) {
          std.atomicAdd(counter.$[id.x % 4], 1);
          return;
        }
        out.$[id.x] = std.atomicLoad(counter.$[1]);
      });
    },
    'scatter',
  ],
  [
    'loads a buffer after an arm that adds and falls through',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x % 2 === 0) {
          std.atomicAdd(counter.$[id.x % 4], 1);
        }
        out.$[id.x] = std.atomicLoad(counter.$[1]);
      });
    },
    'where it may have added to it',
  ],
  [
    'loads through a helper before adding through another',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const read = (i: number) => {
        'use gpu';
        return std.atomicLoad(counter.$[i % 4]);
      };
      const add = (i: number) => {
        'use gpu';
        std.atomicAdd(counter.$[i % 4], 1);
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const seen = read(id.x);
        add(id.x);
        out.$[id.x] = seen;
      });
    },
    'scatter',
  ],
  [
    'adds through a helper before loading through another',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const read = (i: number) => {
        'use gpu';
        return std.atomicLoad(counter.$[i % 4]);
      };
      const add = (i: number) => {
        'use gpu';
        std.atomicAdd(counter.$[i % 4], 1);
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        add(id.x);
        out.$[id.x] = read(id.x);
      });
    },
    'where it may have added to it',
  ],
  [
    'loads and adds in one loop',
    (root) => {
      const counter = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        let seen = d.u32(0);
        for (let k = d.u32(0); k < 2; k++) {
          seen += std.atomicLoad(counter.$[k]);
          std.atomicAdd(counter.$[id.x % 4], 1);
        }
        out.$[id.x] = seen;
      });
    },
    'inside a loop',
  ],
  [
    'takes again only when a taken value decides a short-circuit',
    (root) => {
      const head = root.createMutable(d.atomic(d.u32));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const first = std.atomicAdd(head.$, 1);
        const again = first > 2 && std.atomicAdd(head.$, 1) > 0;
        out.$[id.x] = d.u32(again);
      });
    },
    'by a value it takes',
  ],
  [
    'keeps the least key in one atomic, as the picker does',
    (root) => {
      const key = root.createMutable(d.atomic(d.u32));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x % 3 === 0) return;
        std.atomicMin(key.$, (id.x << 8) | 7);
      });
    },
    'scatter',
  ],
  [
    "keeps the greatest of each invocation's bits, as a sphere fit does",
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicMax(a.$[6], std.bitcast(d.f32, d.u32)(d.f32(id.x) * 0.5));
      });
    },
    'scatter',
  ],
  [
    'keeps the least bound of three cells from three lanes, as a mesh bake does',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const lane = id.x % 4;
        if (lane < 3) std.atomicMin(a.$[lane], id.x * 3);
      });
    },
    'scatter',
  ],
  [
    'sets message bits through a helper, as machines send them',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      const send = (i: number, mask: number) => {
        'use gpu';
        std.atomicOr(a.$[i % 8], d.u32(mask));
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        send(id.x, 1 << (id.x % 5));
        if (id.x % 2 === 0) send(id.x + 1, 32);
      });
    },
    'scatter',
  ],
  [
    'keeps the least value of a struct field beside a count it reads',
    (root) => {
      const Bounds = d.struct({ low: d.atomic(d.u32), count: d.u32 });
      const holder = root.createMutable(Bounds);
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicMin(holder.$.low, id.x + holder.$.count);
      });
    },
    'scatter',
  ],
  [
    'clears bits of signed cells, unrolled over three cells',
    (root) => {
      const b = root.createMutable(d.arrayOf(d.atomic(d.i32), 4));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        for (const k of tgpu.unroll([0, 1, 2])) std.atomicAnd(b.$[k], d.i32(id.x) - 1);
      });
    },
    'scatter',
  ],
  [
    'uses the old value a minimum returns',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[id.x] = std.atomicMin(a.$[0], id.x);
      });
    },
    "that is not the invocation's own",
  ],
  [
    'combines one buffer by both a minimum and a maximum',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicMin(a.$[0], id.x);
        std.atomicMax(a.$[1], id.x);
      });
    },
    'which it also combines by',
  ],
  [
    'combines one buffer by both a sum and a bitwise or',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicAdd(a.$[0], id.x);
        std.atomicOr(a.$[1], id.x);
      });
    },
    'which it also combines by',
  ],
  [
    'keeps the least of signed cells',
    (root) => {
      const b = root.createMutable(d.arrayOf(d.atomic(d.i32), 4));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicMin(b.$[0], d.i32(id.x) - 32);
      });
    },
    "whose cells are not 'u32'",
  ],
  [
    'loads a buffer after combining into it',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicMax(a.$[0], id.x);
        out.$[id.x] = std.atomicLoad(a.$[0]);
      });
    },
    'where it may have added to it',
  ],
  [
    'keeps a maximum inside a runtime loop',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 8));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        for (let k = d.u32(0); k < id.x % 4; k++) std.atomicMax(a.$[k], id.x);
      });
    },
    'inside a loop',
  ],
  [
    'takes from a counter it also keeps the maximum of',
    (root) => {
      const head = root.createMutable(d.atomic(d.u32));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        out.$[id.x] = std.atomicAdd(head.$, 1);
        std.atomicMax(head.$, id.x);
      });
    },
    'takes values of',
  ],
  [
    'stores to its own cell of a buffer it keeps the least of',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        std.atomicStore(a.$[id.x], 0);
        std.atomicMin(a.$[0], id.x);
      });
    },
    "'atomicMin' on an element of 'a' that is not the invocation's own",
  ],
  [
    "loads a summed cell inside a later addition's operand",
    (root) => {
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      const other = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const i = id.x ^ 1;
        std.atomicAdd(sums.$[i], 1);
        std.atomicAdd(other.$[i], std.atomicLoad(sums.$[i]));
      });
    },
    'where it may have added to it',
  ],
  [
    "loads a summed cell inside a later addition's address",
    (root) => {
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      const other = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const i = id.x ^ 1;
        std.atomicAdd(sums.$[i], 1);
        std.atomicAdd(other.$[std.atomicLoad(sums.$[i]) % 64], 1);
      });
    },
    'where it may have added to it',
  ],
  [
    "loads a summed cell inside a later scattered store's value",
    (root) => {
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      const store = root.createTexture({ size: [64, 1], format: 'r32uint' }).$usage('storage');
      const view = store.createView(d.textureStorage2d('r32uint', 'write-only'));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const i = id.x ^ 1;
        std.atomicAdd(sums.$[i], 1);
        std.textureStore(view.$, d.vec2u(i, 0), d.vec4u(std.atomicLoad(sums.$[i]), 0, 0, 1));
      });
    },
    'where it may have added to it',
  ],
  [
    "loads a summed cell inside an addition's operand before adding to it",
    (root) => {
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      const other = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const i = id.x ^ 1;
        std.atomicAdd(other.$[i], std.atomicLoad(sums.$[i]));
        std.atomicAdd(sums.$[i], 1);
      });
    },
    'scatter',
  ],
  [
    "(a) reads a stored texture inside a later store's value, after a write",
    (root) => {
      const texture = root.createTexture({ size: [64, 1], format: 'r32uint' }).$usage('storage');
      const view = texture.createView(d.textureStorage2d('r32uint', 'read-write'));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        std.textureStore(view.$, d.vec2u((id.x * 5) % 64, 0), d.vec4u(1, 0, 0, 1));
        std.textureStore(
          view.$,
          d.vec2u((id.x * 3) % 64, 0),
          d.vec4u(std.textureLoad(view.$, d.vec2u(k, 0)).x, 0, 0, 1),
        );
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    "(a) reads a stored texture inside a later store's value, before any write",
    (root) => {
      const texture = root.createTexture({ size: [64, 1], format: 'r32uint' }).$usage('storage');
      const view = texture.createView(d.textureStorage2d('r32uint', 'read-write'));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        std.textureStore(
          view.$,
          d.vec2u((id.x * 3) % 64, 0),
          d.vec4u(std.textureLoad(view.$, d.vec2u(k, 0)).x, 0, 0, 1),
        );
        std.textureStore(view.$, d.vec2u((id.x * 5) % 64, 0), d.vec4u(1, 0, 0, 1));
      });
    },
    'scatter',
  ],
  [
    "(b) reads a stored texture inside a later store's coordinate, after a write",
    (root) => {
      const texture = root.createTexture({ size: [64, 1], format: 'r32uint' }).$usage('storage');
      const view = texture.createView(d.textureStorage2d('r32uint', 'read-write'));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        std.textureStore(view.$, d.vec2u((id.x * 5) % 64, 0), d.vec4u(1, 0, 0, 1));
        std.textureStore(
          view.$,
          d.vec2u(std.textureLoad(view.$, d.vec2u(k, 0)).x % 64, 0),
          d.vec4u(2, 0, 0, 1),
        );
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    "(b) reads a stored texture inside a later store's coordinate, before any write",
    (root) => {
      const texture = root.createTexture({ size: [64, 1], format: 'r32uint' }).$usage('storage');
      const view = texture.createView(d.textureStorage2d('r32uint', 'read-write'));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        std.textureStore(
          view.$,
          d.vec2u(std.textureLoad(view.$, d.vec2u(k, 0)).x % 64, 0),
          d.vec4u(2, 0, 0, 1),
        );
        std.textureStore(view.$, d.vec2u((id.x * 5) % 64, 0), d.vec4u(1, 0, 0, 1));
      });
    },
    'scatter',
  ],
  [
    "(c) reads a written buffer inside an addition's operand, after a write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        out.$[(id.x * 5) % 64] = 1;
        std.atomicAdd(sums.$[k], out.$[k] as number);
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    "(c) reads a written buffer inside an addition's operand, before any write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        std.atomicAdd(sums.$[k], out.$[k] as number);
        out.$[(id.x * 5) % 64] = 1;
      });
    },
    'scatter',
  ],
  [
    "(d) reads a written buffer inside an addition's address, after a write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        out.$[(id.x * 5) % 64] = 1;
        std.atomicAdd(sums.$[(out.$[k] as number) % 64], 1);
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    "(d) reads a written buffer inside an addition's address, before any write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const sums = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        std.atomicAdd(sums.$[(out.$[k] as number) % 64], 1);
        out.$[(id.x * 5) % 64] = 1;
      });
    },
    'scatter',
  ],
  [
    "(e) reads a written buffer inside a helper call's argument, after a write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const other = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        out.$[(id.x * 5) % 64] = 1;
        other.$[id.x] = twice(out.$[k] as number);
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    "(e) reads a written buffer inside a helper call's argument, before any write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const other = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        other.$[id.x] = twice(out.$[k] as number);
        out.$[(id.x * 5) % 64] = 1;
      });
    },
    'scatter',
  ],
  [
    "(f) reads a written buffer inside a later write's index, after a write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        out.$[(id.x * 5) % 64] = 1;
        out.$[((out.$[k] as number) * 3) % 64] = 2;
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    "(f) reads a written buffer inside a later write's index, as the first write",
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        out.$[((out.$[k] as number) * 3) % 64] = 2;
      });
    },
    'scatter',
  ],
  [
    '(g) reads a written buffer into a local used later, after a write',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const other = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        out.$[(id.x * 5) % 64] = 1;
        const kept = out.$[k] as number;
        other.$[id.x] = kept + 1;
      });
    },
    'which it writes at computed cells, after writing it',
  ],
  [
    '(g) reads a written buffer into a local used later, before any write',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const other = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        const k = id.x ^ 1;
        const kept = out.$[k] as number;
        other.$[id.x] = kept + 1;
        out.$[(id.x * 5) % 64] = 1;
      });
    },
    'scatter',
  ],
  [
    'loads after adding in a switch case beside one that adds and returns',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        switch (id.x) {
          case 0: {
            std.atomicAdd(a.$[0], 1);
            return;
          }
          default: {
            std.atomicAdd(a.$[1], 1);
            std.atomicAdd(a.$[2], 1);
            out.$[id.x] = std.atomicLoad(a.$[1]);
          }
        }
      });
    },
    'where it may have added to it',
  ],
  [
    'loads after a loop that adds and breaks out before returning',
    (root) => {
      const a = root.createMutable(d.arrayOf(d.atomic(d.u32), 4));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const add = () => {
        'use gpu';
        std.atomicAdd(a.$[1], 1);
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: id }) => {
        'use gpu';
        if (id.x % 2 === 0) {
          add();
          for (let k = d.u32(0); k < 4; k++) {
            if (k === id.x % 3) break;
          }
        }
        out.$[id.x] = std.atomicLoad(a.$[1]);
      });
    },
    'where it may have added to it',
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
    'writes its own element past an offset a uniform-derived texelSize holds',
    (root) => {
      const out = root.createMutable(d.arrayOf(d.u32, 128));
      const half = root.createUniform(d.u32);
      return line(({ gid: id }) => {
        'use gpu';
        // Named like a texel read, but read from a uniform
        const texelSize = half.$;
        out.$[id.x + texelSize * 64] = 1;
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

/** Kernels that write a 10-wide grid through a helper, `put`, at cells of a block. */
/** Doubles a value, as the helper a read is passed to in the audit's kernels. */
const twice = (v: number) => {
  'use gpu';
  return v * 2;
};

function blockKernels(): [string, Make, number | string][] {
  const W = 10;
  const parts = (root: TgpuRoot, guarded: boolean) => {
    const cells = root.createMutable(d.arrayOf(d.u32, W * 6));
    const inside = (p: d.v2i) => {
      'use gpu';
      return std.all(std.ge(p, d.vec2i())) && p.x < W && p.y < 6;
    };
    const put = (p: d.v2i, value: number) => {
      'use gpu';
      if (guarded && !inside(p)) return;
      cells.$[d.u32(p.y) * W + d.u32(p.x)] = value;
    };
    return { cells, put };
  };
  return [
    [
      'owns a 2 × 2 block through a guarded helper',
      (root) => {
        const { put } = parts(root, true);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          const o = d.vec2i(g.xy) * 2;
          put(o, d.u32(1));
          put(o + d.vec2i(1, 0), d.u32(2));
          put(o + d.vec2i(0, 1), d.u32(3));
          put(o + d.vec2i(1), d.u32(4));
        });
      },
      2,
    ],
    [
      'owns a block through a guarded helper it first calls under a condition',
      (root) => {
        const { put } = parts(root, true);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          const o = d.vec2i(g.xy) * 2;
          if (g.x % 2 === 0) put(o, d.u32(1));
          put(o + d.vec2i(1, 0), d.u32(2));
        });
      },
      2,
    ],
    [
      'owns a 1 × 2 pair through a guarded helper',
      (root) => {
        const { put } = parts(root, true);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          const o = d.vec2i(d.i32(g.x) * 2 - 1, d.i32(g.y));
          put(o, d.u32(1));
          put(o + d.vec2i(1, 0), d.u32(2));
        });
      },
      2,
    ],
    [
      'writes a block whose column it does not keep inside the row',
      (root) => {
        const { put } = parts(root, false);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          put(d.vec2i(g.xy) * 2, d.u32(1));
        });
      },
      'at a column it does not keep below 10',
    ],
    [
      'writes a block at an offset that is not constant',
      (root) => {
        const { put } = parts(root, true);
        const shift = root.createUniform(d.vec2i);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          const o = d.vec2i(g.xy) * 2;
          put(o, d.u32(1));
          put(o + shift.$, d.u32(2));
        });
      },
      'blocks of two shapes',
    ],
    [
      'writes blocks wider than their step, overlapping a neighbour',
      (root) => {
        const { put } = parts(root, true);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          const o = d.vec2i(g.xy) * 2;
          put(o, d.u32(1));
          put(o + d.vec2i(2, 0), d.u32(2));
        });
      },
      'wider than their step',
    ],
    [
      'writes blocks whose axes are swapped between calls',
      (root) => {
        const { put } = parts(root, true);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          put(d.vec2i(g.xy) * 2, d.u32(1));
          put(d.vec2i(g.yx) * 2 + 1, d.u32(2));
        });
      },
      'at no block of its own',
    ],
    [
      'writes blocks with steps that differ between calls',
      (root) => {
        const { put } = parts(root, true);
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          put(d.vec2i(g.xy) * 2, d.u32(1));
          put(d.vec2i(g.xy) * 3, d.u32(2));
        });
      },
      'blocks of two shapes',
    ],
    [
      'writes through a block-shaped helper from another helper, as points',
      (root) => {
        const { cells, put } = parts(root, true);
        const land = (p: d.v2i) => {
          'use gpu';
          if ((cells.$[d.u32(p.y) * W + d.u32(p.x)] as number) === 0) put(p, d.u32(9));
        };
        return tgpu.computeFn({ workgroupSize: [4, 4], in: gid })(({ gid: g }) => {
          'use gpu';
          land(d.vec2i(d.i32((g.x * 3) % 10), d.i32(g.y)));
        });
      },
      'scatter',
    ],
    [
      'guards its block write only inside a switch case',
      (root) => {
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
        return tgpu.computeFn({ workgroupSize: [4, 2], in: gid })(({ gid: g }) => {
          'use gpu';
          put(d.vec2i(g.xy) * 2 + d.vec2i(9, 0), d.u32(7));
        });
      },
      'at a column it does not keep',
    ],
  ];
}

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

  it("keeps each pass of sums its own buffer's records, one a word where the buffer is fixed", () => {
    const verdict = verdictOf((root) => {
      const counts = root.createMutable(d.arrayOf(d.atomic(d.u32), 2));
      const many = root.createMutable(d.arrayOf(d.atomic(d.u32), 64));
      const head = root.createMutable(d.atomic(d.u32));
      const out = root.createMutable(d.arrayOf(d.u32, 64));
      const add = (i: number) => {
        'use gpu';
        std.atomicAdd(counts.$[i % 2], d.u32(1));
        std.atomicAdd(counts.$[(i + 1) % 2], d.u32(2));
        std.atomicAdd(many.$[i % 64], d.u32(3));
      };
      return tgpu.computeFn({ workgroupSize: [64], in: gid })(({ gid: g }) => {
        'use gpu';
        add(g.x);
        add(g.x + 1);
        add(g.x + 2);
        out.$[g.x] = std.atomicAdd(head.$, 1) + std.atomicAdd(head.$, 1);
      });
    });
    if ('reason' in verdict) throw new Error(`Ineligible: ${verdict.reason}`);
    // Six additions to two words keep two records, three to 64 words three, and two takes two
    const sites = verdict.module.passes
      .filter((pass) => pass.scatter?.accumulate)
      .map((pass) => [pass.scatter?.sites, pass.scatter?.takes !== undefined]);
    expect(sites.toSorted()).toEqual([
      [2, false],
      [2, true],
      [3, false],
    ]);
  });
});
