import { recordingRoot } from '@typegpu/kit/testing';
import { d, std, tgpu, type TgpuMutable } from 'typegpu';
import { expect, it } from 'vitest';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile } from '../../src/cpu/index.ts';

type Out = TgpuMutable<d.WgslArray<d.U32>>;

/**
 * Whether a CPU module says it writes `data`, a mutable binding: `step` makes the kernel's body
 * over `data` and `out`, which the body always writes.
 */
function writesData<T extends d.AnyWgslData>(
  schema: T,
  step: (data: TgpuMutable<T>, out: Out) => (i: number) => void,
) {
  const { root } = recordingRoot();
  const data = root.createMutable<T>(schema as never).$name('data');
  const out = root.createMutable(d.arrayOf(d.u32, 4)).$name('out');
  const body = step(data, out);
  const kernel = tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    body(id.x);
  });
  const code = tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() });
  const { bindings } = compile(code);
  root.destroy();
  const uses = bindings.filter((use) => use.kind === 'buffer');
  expect(uses).toHaveLength(2);
  const marked = uses.filter((use) => use.writes).length;
  expect(marked).toBeGreaterThanOrEqual(1);
  return marked === 2;
}

const words = d.arrayOf(d.u32, 8);
const Pair = d.struct({ a: d.u32, b: d.vec2f });

it('leaves a mutable buffer that is only read unwritten', () => {
  expect(
    writesData(words, (data, out) => (i: number) => {
      'use gpu';
      out.$[i % 4] = data.$[i] + data.$[(i + 1) % 8];
      if (data.$[i] === 0) out.$[1] = data.$[(i * 3) % 8];
    }),
  ).toBe(false);
});

it.each([
  [
    'an element write',
    (data: TgpuMutable<typeof words>, out: Out) => (i: number) => {
      'use gpu';
      out.$[0] = 1;
      data.$[i] = 2;
    },
  ],
  [
    'a compound assignment',
    (data: TgpuMutable<typeof words>, out: Out) => (i: number) => {
      'use gpu';
      out.$[0] = 1;
      data.$[i] += 3;
    },
  ],
  [
    'a write after a read of the same element',
    (data: TgpuMutable<typeof words>, out: Out) => (i: number) => {
      'use gpu';
      out.$[0] = data.$[i];
      if (data.$[i] === 0) data.$[(i * 3) % 8] = out.$[0];
    },
  ],
] as const)('counts %s to a mutable buffer as a write', (_, step) => {
  expect(writesData(words, step)).toBe(true);
});

it('counts an atomic on a mutable buffer as a write', () => {
  expect(
    writesData(d.arrayOf(d.atomic(d.u32), 8), (data, out) => (i: number) => {
      'use gpu';
      out.$[0] = 1;
      std.atomicAdd(data.$[i], 1);
    }),
  ).toBe(true);
});

it('tells reads from writes of a struct binding', () => {
  expect(
    writesData(Pair, (data, out) => () => {
      'use gpu';
      out.$[0] = data.$.a + d.u32(data.$.b.y);
    }),
  ).toBe(false);
  expect(
    writesData(Pair, (data, out) => () => {
      'use gpu';
      out.$[0] = 1;
      data.$.b.x = 2;
    }),
  ).toBe(true);
});

it('counts a write through a pointer a helper takes', () => {
  const bump = tgpu.fn([d.ptrStorage(d.u32, 'read-write')])((cell) => {
    'use gpu';
    cell.$ = cell.$ + 1;
  });
  expect(
    writesData(words, (data, out) => (i: number) => {
      'use gpu';
      out.$[0] = 1;
      // A storage element passes as a pointer the way WGSL passes `&data[i]`
      bump(data.$[i] as unknown as d.ref<number>);
    }),
  ).toBe(true);
});

/** The words a dispatch of `body` over a 64-word `data` reports it stored to. */
function storedWords(
  body: (data: TgpuMutable<d.WgslArray<d.U32>>) => (i: number) => void,
  schema: d.AnyWgslData = d.arrayOf(d.u32, 64),
) {
  const { root } = recordingRoot();
  const data = root.createMutable(schema as d.WgslArray<d.U32>).$name('data');
  const step = body(data);
  const kernel = tgpu.computeFn({ workgroupSize: [4], in: { id: d.builtin.globalInvocationId } })(({
    id,
  }) => {
    'use gpu';
    step(id.x);
  });
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const bytes = new ArrayBuffer(64 * 4);
  module.dispatch(1, 1, 1, {
    buffer: () => ({ bytes, offset: 0, size: bytes.byteLength }),
    texture: () => {
      throw new Error('unexpected texture');
    },
    sampler: () => {
      throw new Error('unexpected sampler');
    },
  });
  root.destroy();
  return module.writtenWords?.(0, 0);
}

it('reports the words a dispatch stored to, from the lowest to past the highest', () => {
  expect(
    storedWords((data) => (i: number) => {
      'use gpu';
      data.$[10 + i * 3] = data.$[i] + 1;
    }),
  ).toEqual([10, 20]);
  expect(
    storedWords((data) => (i: number) => {
      'use gpu';
      if (i === 9) data.$[i] = 1;
      data.$[0] += data.$[1];
    }),
  ).toEqual([0, 1]);
});

it('cannot tell the words an atomic stored to', () => {
  expect(
    storedWords(
      (data) => (i: number) => {
        'use gpu';
        std.atomicAdd(data.$[i] as never, 1);
      },
      d.arrayOf(d.atomic(d.u32), 64),
    ),
  ).toBeUndefined();
});

/** Dispatches `module` once over one buffer per binding, of `words` words each: their words. */
function runOver(module: ReturnType<typeof compile>, words: number[], workgroups = 1) {
  const buffers = words.map((count) => new ArrayBuffer(count * 4));
  module.dispatch(workgroups, 1, 1, {
    buffer: (_group, binding) => {
      const bytes = buffers[binding]!;
      return { bytes, offset: 0, size: bytes.byteLength };
    },
    texture: () => {
      throw new Error('unexpected texture');
    },
    sampler: () => {
      throw new Error('unexpected sampler');
    },
  });
  return buffers.map((bytes) => [...new Uint32Array(bytes)]);
}

it('tracks a store inside the index of another store, each in its own range', () => {
  // `idx[…] = …` stores inside the index of the store to `out`, as an inline adapter's setter does
  const module = compile(
    [
      '//#kit-gl compute',
      '//#binding {"group":0,"binding":0,"kind":"buffer","reads":true,"writes":true}',
      'const out = $buffer(0, 0, Uint32Array);',
      '//#binding {"group":0,"binding":1,"kind":"buffer","reads":true,"writes":true}',
      'const idx = $buffer(0, 1, Uint32Array);',
      '//#entry {"name":"main","size":[4,1,1],"inputs":["global_invocation_id"],"generator":false}',
      'function main(gid) {',
      '  (out[((idx[(gid[0] + 2)] = (gid[0] * 3)), idx[(gid[0] + 2)])]) = (gid[0] + 40);',
      '}',
    ].join('\n'),
  );
  const [out, idx] = runOver(module, [16, 8]);
  expect(out!.slice(0, 10)).toEqual([40, 0, 0, 41, 0, 0, 42, 0, 0, 43]);
  expect(idx).toEqual([0, 0, 0, 3, 6, 9, 0, 0]);
  expect(module.writtenWords?.(0, 0)).toEqual([0, 10]);
  expect(module.writtenWords?.(0, 1)).toEqual([2, 6]);
});
