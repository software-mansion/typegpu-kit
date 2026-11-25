import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';

const operations = [
  {
    name: 'reverseBits',
    fn: std.reverseBits,
    reference: (v: number) =>
      Number(
        BigInt(`0b${(v >>> 0).toString(2).padStart(32, '0').split('').toReversed().join('')}`),
      ),
  },
  {
    name: 'firstTrailingBit',
    fn: std.firstTrailingBit,
    reference: (v: number) => (v === 0 ? -1 : Math.log2((v & -v) >>> 0)),
  },
  {
    name: 'countOneBits',
    fn: std.countOneBits,
    reference: (v: number) => (v >>> 0).toString(2).replaceAll('0', '').length,
  },
  { name: 'countLeadingZeros', fn: std.countLeadingZeros, reference: (v: number) => Math.clz32(v) },
  {
    name: 'countTrailingZeros',
    fn: std.countTrailingZeros,
    reference: (v: number) => (v === 0 ? 32 : Math.log2((v & -v) >>> 0)),
  },
  { name: 'abs', fn: std.abs, reference: (v: number) => Math.abs(v) },
  { name: 'sign', fn: std.sign, reference: (v: number) => Math.sign(v) },
  {
    name: 'min',
    fn: (v: number) => {
      'use gpu';
      return std.min(v, 1);
    },
    reference: (v: number) => Math.min(v, 1),
  },
  {
    name: 'max',
    fn: (v: number) => {
      'use gpu';
      return std.max(v, 1);
    },
    reference: (v: number) => Math.max(v, 1),
  },
  {
    name: 'clamp',
    fn: (v: number) => {
      'use gpu';
      return std.clamp(v, 0, 1);
    },
    reference: (v: number) => Math.min(Math.max(v, 0), 1),
  },
  {
    name: 'select',
    fn: (v: number) => {
      'use gpu';
      return std.select(v, 0, v < 0);
    },
    reference: (v: number) => (v < 0 ? 0 : v),
  },
] as const;

for (const signed of [true, false]) {
  for (const { name, fn, reference } of operations) {
    if (!signed && (name === 'abs' || name === 'sign')) continue;
    it(`${name} returns canonical ${signed ? 'i32' : 'u32'} values inside expressions`, async () => {
      const root = cpuRoot();
      const schema = signed ? d.i32 : d.u32;
      const values = signed
        ? [0, 1, -1, -2147483648, 2147483647]
        : [0, 1, 4294967295, 2147483648, 2147483647];
      const input = root.createReadonly(d.arrayOf(schema, values.length), values);
      const output = root.createMutable(d.arrayOf(d.vec2f, values.length));
      root
        .createComputePipeline({
          compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
            ({ id }) => {
              'use gpu';
              output.$[id.x] = d.vec2f(d.f32(fn(input.$[id.x])), d.f32(fn(input.$[id.x]) < 0));
            },
          ),
        })
        .dispatchWorkgroups(values.length);
      expect((await output.read()).map((v) => [...v])).toEqual(
        values.map((value) => {
          const raw = reference(value);
          const canonical = signed ? raw | 0 : raw >>> 0;
          return [Math.fround(canonical), canonical < 0 ? 1 : 0];
        }),
      );
      root.destroy();
    });
  }
}

it('keeps signed vector builtin components canonical before float conversion', async () => {
  const root = cpuRoot();
  const input = root.createReadonly(d.vec2i, d.vec2i(1, -2147483648));
  const output = root.createMutable(d.arrayOf(d.vec2f, 3));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        output.$[0] = d.vec2f(std.reverseBits(input.$));
        output.$[1] = d.vec2f(std.abs(input.$));
        output.$[2] = d.vec2f(std.firstTrailingBit(input.$ - input.$));
      }),
    })
    .dispatchWorkgroups(1);
  expect((await output.read()).map((v) => [...v])).toEqual([
    [-2147483648, 1],
    [1, -2147483648],
    [-1, -1],
  ]);
  root.destroy();
});

it.each([
  ['firstLeadingBit', std.firstLeadingBit],
  [
    'extractBits',
    (v: number) => {
      'use gpu';
      return std.extractBits(v, 0, 1);
    },
  ],
  [
    'insertBits',
    (v: number) => {
      'use gpu';
      return std.insertBits(v, d.u32(1), 0, 1);
    },
  ],
] as const)('rejects unsupported %s at generation time with its name', (name, operation) => {
  const root = cpuRoot();
  const input = root.createReadonly(d.u32, 1);
  const output = root.createMutable(d.u32);
  const pipeline = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      output.$ = operation(input.$);
    }),
  });
  expect(() => pipeline.dispatchWorkgroups(1)).toThrow(`unsupported builtin ${name}`);
  root.destroy();
});

it('wraps signed vector negation before converting to float', async () => {
  const root = cpuRoot();
  const input = root.createReadonly(d.vec2i, d.vec2i(-2147483648, 1));
  const output = root.createMutable(d.vec2f);
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
        'use gpu';
        output.$ = d.vec2f(-input.$);
      }),
    })
    .dispatchWorkgroups(1);
  expect([...(await output.read())]).toEqual([-2147483648, -1]);
  root.destroy();
});

it('rejects unsigned vector negation through TypeGPU validation', () => {
  const root = cpuRoot();
  const input = root.createReadonly(d.vec2u, d.vec2u(1));
  const output = root.createMutable(d.vec2f);
  const compute = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    output.$ = d.vec2f(-input.$);
  });
  expect(() => root.createComputePipeline({ compute }).dispatchWorkgroups(1)).toThrow(
    /Unary operator - cannot be applied.*vec2u/,
  );
  root.destroy();
});
