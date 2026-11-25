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
