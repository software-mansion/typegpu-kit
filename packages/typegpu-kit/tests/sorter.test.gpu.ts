import { gpuRoot, type GpuRoot } from '@typegpu/kit/testing/dawn';
import { d } from 'typegpu';
import { expect, it } from 'vitest';
import * as sorter from '../src/sorter.ts';

const COUNT = 5000;

/** Sorts `keys` with their indices as the payload and compares with a stable sort on the CPU. */
async function expectSorted(
  root: GpuRoot,
  schema: sorter.Key,
  keys: number[],
  options: sorter.Options,
  order: (a: number, b: number) => number,
) {
  const keyBuffer = root.createBuffer(d.arrayOf(schema, COUNT), keys).$usage('storage');
  const indices = Array.from({ length: COUNT }, (_, i) => i);
  const values = root.createBuffer(d.arrayOf(d.u32, COUNT), indices).$usage('storage');
  const out = options.out ?? { keys: keyBuffer, values };
  const sort = sorter.create(root, keyBuffer, { ...options, values });

  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  sort.sort(pass);
  pass.end();
  encoder.submit();

  const expected = indices.toSorted((a, b) => order(keys[a], keys[b]));
  expect(await out.values!.read()).toEqual(expected);
  expect(await out.keys.read()).toEqual(expected.map((i) => keys[i]));
  sort.destroy();
}

it('sorts u32 keys in place with their payload, keeping ties in order', async () =>
  // Three passes of eight bits end in the scratch copy, which copies back
  expectSorted(
    await gpuRoot(),
    d.u32,
    Array.from({ length: COUNT }, (_, i) => (i * 7919) % 613),
    { keyBits: 24 },
    (a, b) => a - b,
  ));

it('sorts f32 keys descending into other buffers', async () => {
  const root = await gpuRoot();
  const keys = Array.from({ length: COUNT }, (_, i) => Math.round(Math.sin(i) * 64) / 4);
  await expectSorted(
    root,
    d.f32,
    keys,
    {
      direction: 'descending',
      out: {
        keys: root.createBuffer(d.arrayOf(d.f32, COUNT)).$usage('storage'),
        values: root.createBuffer(d.arrayOf(d.u32, COUNT)).$usage('storage'),
      },
    },
    (a, b) => b - a,
  );
});

it('sorts f32 keys within a range in fewer bits, keeping ties in order', async () =>
  expectSorted(
    await gpuRoot(),
    d.f32,
    Array.from({ length: COUNT }, (_, i) => Math.round(Math.cos(i) * 64) / 4),
    { range: [-16, 16], keyBits: 16 },
    (a, b) => a - b,
  ));
