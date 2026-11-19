import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import * as workgroup from '../src/workgroup.ts';

const SIZE = 64;

/**
 * Reduces and scans `sample(lid)` over one workgroup and checks both against a fold of the same
 * values with the monoid's functions on the CPU. Each invocation reduces twice, once with the
 * values in reverse, so the second call would overwrite lanes the first still reads.
 */
async function expectFolds<T extends d.AnyWgslData>(
  monoid: workgroup.Monoid<T>,
  sample: (lid: number) => d.Infer<T>,
) {
  const root = await gpuRoot();
  const totals = root.createMutable(d.arrayOf(monoid.schema, SIZE));
  const prefixes = root.createMutable(d.arrayOf(monoid.schema, SIZE));
  const total = workgroup.reduce(monoid, SIZE);
  const prefix = workgroup.scan(monoid, SIZE);
  const kernel = tgpu.computeFn({
    workgroupSize: [SIZE],
    in: { lid: d.builtin.localInvocationIndex },
  })(({ lid }) => {
    'use gpu';
    const value = sample(lid);
    const forward = total(value, lid);
    (totals.$ as d.Infer<T>[])[lid] = monoid.combine(forward, total(sample(SIZE - 1 - lid), lid));
    (prefixes.$ as d.Infer<T>[])[lid] = prefix(value, lid);
  });
  const encoder = root.createCommandEncoder();
  root.createComputePipeline({ compute: kernel }).with(encoder).dispatchWorkgroups(1);
  encoder.submit();

  const expected = [monoid.identity()];
  for (let lid = 0; lid < SIZE; lid++) {
    expected.push(monoid.combine(expected[lid], sample(lid)));
  }
  const whole = expected.pop()!;
  expect(await totals.read()).toEqual(Array(SIZE).fill(monoid.combine(whole, whole)));
  expect(await prefixes.read()).toEqual(expected);
}

it('sums u32 values', () =>
  expectFolds(workgroup.sum(d.u32), (lid) => {
    'use gpu';
    return d.u32((lid * 37) % 11);
  }));

const Tally = d.struct({ count: d.u32, weight: d.vec3f, parts: d.arrayOf(d.f32, 2) });

it('sums a struct field by field', () =>
  expectFolds(workgroup.sum(Tally), (lid) => {
    'use gpu';
    const k = d.f32(lid % 5);
    return Tally({ count: d.u32(1), weight: d.vec3f(k, 1, -k), parts: [k * 2, 0.5] });
  }));

const Box = workgroup.bounds(d.vec3i);

it('keeps the bounds of i32 points, from the extremes of i32', () =>
  expectFolds(Box, (lid) => {
    'use gpu';
    const p = d.vec3i(d.i32((lid * 3) % 17) - 8, d.i32((lid * 5) % 13), -d.i32(lid));
    return Box.schema({ low: p, high: p });
  }));

it('keeps the largest i32, from the lowest i32', () =>
  expectFolds(workgroup.max(d.i32), (lid) => {
    'use gpu';
    return d.i32((lid * 7) % 23) - 11;
  }));
