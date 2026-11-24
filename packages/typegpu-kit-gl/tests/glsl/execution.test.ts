import { common, d, std, tgpu } from 'typegpu';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { glslRoot } from './utils.ts';
import { validator, type Validator } from './validator.ts';

let gl: Validator;
beforeAll(async () => {
  gl = await validator();
});
afterAll(() => gl.close());

const Inner = d.struct({ a: d.vec2f, b: d.i32 });
const Item = d.struct({
  p: d.vec3f,
  s: d.f32,
  m: d.mat3x3f,
  n: d.mat2x2f,
  k: d.arrayOf(d.f32, 3),
  inner: Inner,
  flag: d.u32,
  q: d.vec4i,
  counter: d.atomic(d.u32),
});

const items = [
  {
    p: d.vec3f(1, 2, 3),
    s: 4,
    m: d.mat3x3f(5, 6, 7, 8, 9, 10, 11, 12, 13),
    n: d.mat2x2f(14, 15, 16, 17),
    k: [18, 19, 20],
    inner: { a: d.vec2f(21, 22), b: -23 },
    flag: 24,
    q: d.vec4i(-25, 26, -27, 28),
    counter: 29,
  },
  {
    p: d.vec3f(31, 32, 33),
    s: 34,
    m: d.mat3x3f(35, 36, 37, 38, 39, 40, 41, 42, 43),
    n: d.mat2x2f(44, 45, 46, 47),
    k: [48, 49, 50],
    inner: { a: d.vec2f(51, 52), b: -53 },
    flag: 54,
    q: d.vec4i(-55, 56, -57, 58),
    counter: 59,
  },
];

const layout = tgpu.bindGroupLayout({
  items: { storage: d.arrayOf(Item), access: 'readonly' },
});

const storagePixel = (i: number) => {
  'use gpu';
  const item = layout.$.items[1] as unknown as d.Infer<typeof Item>;
  if (i === 0) return d.vec4f(item.p, item.s);
  if (i === 1) return d.vec4f(item.m.columns[2] as d.v3f, d.f32(item.inner.b));
  if (i === 2) return d.vec4f(layout.$.items[0].n.columns[1] as d.v2f, layout.$.items[0].inner.a);
  if (i === 3)
    return d.vec4f(item.k[0] as number, item.k[1] as number, item.k[2] as number, d.f32(item.flag));
  if (i === 4) return d.vec4f(item.q);
  if (i === 5) {
    return d.vec4f(
      d.f32(layout.$.items.length),
      d.f32(std.atomicLoad(layout.$.items[0].counter)),
      0,
      0,
    );
  }
  return d.vec4f(item.m.columns[0] as d.v3f, item.n.columns[0].y);
};

/** Reads what a buffer holds into a `width` × 1 target, pixel by pixel. */
async function draw(
  width: number,
  pixel: (index: number) => d.v4f,
  buffers: (recording: ReturnType<typeof glslRoot>['recording']) => Record<string, Uint8Array>,
  setup: (root: ReturnType<typeof glslRoot>['root']) => void = () => {},
) {
  const { root, recording, modules } = glslRoot();
  setup(root);
  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(
      ({ position }) => {
        'use gpu';
        return pixel(d.u32(position.x));
      },
    ),
    targets: { format: 'rgba32float' },
  });
  root.unwrap(pipeline);
  const [module] = modules();
  if (!module) throw new Error('No render module');
  if (process.env.SHOW) console.log(module.fragment, module.buffers, module.blocks);
  const pixels = await gl.run(module, width, buffers(recording));
  return Array.from({ length: width }, (_, i) => pixels.slice(i * 4, i * 4 + 4));
}

describe('buffer reads', () => {
  it('read each member of storage structs where WGSL lays it out', async () => {
    let sized: object | undefined;
    const pixels = await draw(
      7,
      storagePixel,
      (recording) => ({ items: recording.bytes(sized as never) }),
      (root) => {
        sized = root.createBuffer(d.arrayOf(Item, 2), items).$usage('storage');
      },
    );
    expect(pixels).toEqual([
      [31, 32, 33, 34],
      [41, 42, 43, -53],
      [16, 17, 21, 22],
      [48, 49, 50, 54],
      [-55, 56, -57, 58],
      [2, 29, 0, 0],
      [35, 36, 37, 45],
    ]);
  });

  it('read uniform buffers from uniform blocks with the same layout', async () => {
    const Plain = d.struct({ p: d.vec3f, m: d.mat3x3f, inner: Inner });
    const Params = d.struct({ scale: d.f32, item: Plain, tail: d.vec2u });
    const params = {
      scale: 0.5,
      item: { p: d.vec3f(31, 32, 33), m: items[1]?.m as d.m3x3f, inner: items[1]?.inner as never },
      tail: d.vec2u(60, 61),
    };
    let buffer: object | undefined;
    let usage: { $: d.Infer<typeof Params> } | undefined;

    const pixels = await draw(
      3,
      (i: number) => {
        'use gpu';
        const p = (usage as { $: d.Infer<typeof Params> }).$;
        if (i === 0) return d.vec4f(p.item.p, p.scale);
        if (i === 1) return d.vec4f(p.item.m.columns[1] as d.v3f, d.f32(p.item.inner.b));
        return d.vec4f(d.vec2f(p.tail), p.item.inner.a);
      },
      (recording) => ({ params_block: recording.bytes(buffer as never) }),
      (root) => {
        buffer = root.createBuffer(Params, params).$usage('uniform').$name('params');
        usage = (buffer as unknown as { as(u: 'uniform'): { $: d.Infer<typeof Params> } }).as(
          'uniform',
        );
      },
    );
    expect(pixels).toEqual([
      [31, 32, 33, 0.5],
      [38, 39, 40, -53],
      [60, 61, 51, 52],
    ]);
  });
});

const clz = (x: number) => Math.clz32(x);
const ctz = (x: number) => (x === 0 ? 32 : 31 - Math.clz32(x & -x));
const ones = (x: number) => x.toString(2).replaceAll('0', '').length;
const reverse = (x: number) =>
  Number.parseInt(x.toString(2).padStart(32, '0').split('').toReversed().join(''), 2);

const leading = (x: number) => (x === 0 ? -1 : 31 - clz(x));
const leadingSigned = (x: number) => {
  const signed = x | 0;
  const bits = signed < 0 ? ~signed >>> 0 : signed;
  return bits === 0 ? -1 : 31 - clz(bits);
};

describe('builtins without a GLSL ES 3.00 counterpart', () => {
  const values = [0, 1, 0x80000000, 0x00f0f000, 12345678, 0xffffffff, 6, 0x10];
  it('count, find and reverse bits as WGSL does', async () => {
    let buffer: object | undefined;
    let usage: { $: number[] } | undefined;
    const pixels = await draw(
      16,
      (i: number) => {
        'use gpu';
        const x = (usage as { $: number[] }).$[i % 8] as number;
        if (i < 8) {
          return d.vec4f(
            d.f32(std.countOneBits(x)),
            d.f32(std.countTrailingZeros(x)),
            d.f32(std.countLeadingZeros(x)),
            d.f32(d.i32(std.firstLeadingBit(x))),
          );
        }
        const reversed = std.reverseBits(x);
        return d.vec4f(
          d.f32(d.i32(std.firstTrailingBit(x))),
          d.f32(reversed & 0xffff),
          d.f32(reversed >>> 16),
          d.f32(std.firstLeadingBit(d.i32(x))),
        );
      },
      (recording) => ({ bits: recording.bytes(buffer as never) }),
      (root) => {
        buffer = root.createBuffer(d.arrayOf(d.u32, 8), values).$usage('storage').$name('bits');
        usage = (buffer as unknown as { as(u: 'readonly'): { $: number[] } }).as('readonly');
      },
    );
    expect(pixels).toEqual([
      ...values.map((x) => [ones(x), ctz(x), clz(x), leading(x)]),
      ...values.map((x) => [
        x === 0 ? -1 : ctz(x),
        reverse(x) & 0xffff,
        reverse(x) >>> 16,
        leadingSigned(x),
      ]),
    ]);
  });

  it('selects integer vectors per component and writes ternaries as selects', async () => {
    let buffer: object | undefined;
    let usage: { $: number[] } | undefined;
    const pixels = await draw(
      2,
      (i: number) => {
        'use gpu';
        const x = (usage as { $: number[] }).$[i] as number;
        const picked = std.select(d.vec2i(1, 2), d.vec2i(3, 4), d.vec2b(x > 0, x < 0));
        const unsigned = std.select(d.vec3u(5), d.vec3u(6, 7, 8), d.vec3b(true, x > 0, false));
        const ternary = x > 0 ? 9 : 10;
        return d.vec4f(
          d.f32(picked.x + picked.y),
          d.f32(unsigned.y),
          d.f32(ternary),
          d.f32(unsigned.x),
        );
      },
      (recording) => ({ signs: recording.bytes(buffer as never) }),
      (root) => {
        buffer = root.createBuffer(d.arrayOf(d.i32, 2), [1, -1]).$usage('storage').$name('signs');
        usage = (buffer as unknown as { as(u: 'readonly'): { $: number[] } }).as('readonly');
      },
    );
    expect(pixels).toEqual([
      [3 + 2, 7, 9, 6],
      [1 + 4, 5, 10, 6],
    ]);
  });
});
