import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterEach, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot } from './compare.ts';

let root: TgpuRoot | undefined;

afterEach(() => {
  root?.destroy();
  root = undefined;
});

const mix = (i: number, salt: number) => (Math.imul(i, 0x9e3779b1) ^ salt) >>> 0;

/**
 * A CPU kernel that copies `from` into `into` word by word, which makes the device read `from`
 * back where a GPU form wrote it last: a barrier keeps it on the CPU at every size.
 */
function copier(owner: TgpuRoot, from: d.WgslArray<d.U32>, label: string) {
  const count = from.elementCount;
  const source = owner.createMutable(from).$name(label);
  const into = owner.createMutable(from).$name(`${label} copy`);
  const copy = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    std.workgroupBarrier();
    for (let i = d.u32(0); i < count; i++) into.$[i] = source.$[i];
  });
  const pipeline = owner.createComputePipeline({ compute: copy }).$name(`${label} copying`);
  return {
    source,
    copied: async () => {
      pipeline.dispatchWorkgroups(1);
      return [...(await into.read())];
    },
  };
}

/**
 * A buffer of `count` words that a GPU form fills with `mix(i, salt)`, read back through the
 * device's readback for a CPU kernel to copy: what the CPU then holds.
 */
function filler(owner: TgpuRoot, count: number, label: string) {
  const { source: words, copied } = copier(owner, d.arrayOf(d.u32, count), label);
  const salt = owner.createUniform(d.u32);
  const fill = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.x >= count) return;
    words.$[gid.x] = (gid.x * d.u32(0x9e3779b1)) ^ salt.$;
  });
  const pipeline = owner.createComputePipeline({ compute: fill }).$name(`${label} fill`);
  return async (value: number) => {
    salt.write(value);
    pipeline.dispatchWorkgroups(Math.ceil(count / 64));
    return copied();
  };
}

const expected = (count: number, salt: number) =>
  Array.from({ length: count }, (_, i) => mix(i, salt));

test('reads back buffers of every row shape, larger then smaller, with no bytes of another', async () => {
  // Placement would move the small fills to the CPU, where nothing is read back
  root = glRoot({ placement: false });
  const { stats } = root.device as unknown as GLDevice;
  // Whole rows of 1024 texels, rows and a part, one part of a row, and a last texel part-filled
  const shapes = [
    ['rows', 2 * 1024 * 4],
    ['rows and a part', 5000],
    ['a part', 37],
    ['one word', 1],
  ] as const;
  const fills = shapes.map(
    ([label, count]) => [label, count, filler(root!, count, label)] as const,
  );
  for (const salt of [0x11111111, 0x22222222]) {
    for (const [label, count, fill] of fills.toReversed()) {
      expect(await fill(salt), `${label} after smaller reads`).toEqual(expected(count, salt));
    }
    for (const [label, count, fill] of fills) {
      expect(await fill(salt ^ 0xff), `${label} after larger reads`).toEqual(
        expected(count, salt ^ 0xff),
      );
    }
  }
  for (const [label, count] of shapes) {
    const reads = stats.buffers.get(`${label} ${count * 4}`)?.reads;
    expect(reads, `${label} read back for the CPU kernel`).toBe(4);
    expect([...stats.kernels.keys()]).toContain(`${label} copying`);
  }
});

test('reads back a buffer the CPU wrote one part of and a GPU form the other', async () => {
  // Placement would move the small fills to the CPU, where nothing is read back
  root = glRoot({ placement: false });
  const Halves = d.struct({ head: d.arrayOf(d.u32, 13), tail: d.arrayOf(d.u32, 1500) });
  const both = root.createMutable(Halves).$name('halves');
  const salt = root.createUniform(d.u32);
  const fill = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.x >= 1500) return;
    both.$.tail[gid.x] = (gid.x * d.u32(0x9e3779b1)) ^ salt.$;
  });
  const pipeline = root.createComputePipeline({ compute: fill }).$name('tail fill');
  const into = root.createMutable(Halves).$name('halves copy');
  const copy = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    for (let i = d.u32(0); i < 13; i++) into.$.head[i] = both.$.head[i];
    for (let i = d.u32(0); i < 1500; i++) into.$.tail[i] = both.$.tail[i];
  });
  const copying = root.createComputePipeline({ compute: copy }).$name('halves copying');
  for (const value of [5, 6]) {
    const head = Array.from({ length: 13 }, (_, i) => value * 1000 + i);
    both.patch({ head });
    salt.write(value);
    pipeline.dispatchWorkgroups(Math.ceil(1500 / 64));
    copying.dispatchWorkgroups(1);
    const read = await into.read();
    expect(read.head).toEqual(head);
    expect(read.tail).toEqual(expected(1500, value));
  }
  const { stats } = root.device as unknown as GLDevice;
  expect([...stats.kernels.keys()]).toContain('tail fill (gpu)');
});
