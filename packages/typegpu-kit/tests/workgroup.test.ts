import { d } from 'typegpu';
import { expect, it } from 'vitest';
import { workgroup } from '../src/index.ts';

it('sums vectors per component from the schema’s zero, and only numbers', () => {
  const total = workgroup.sum(d.struct({ count: d.u32, weight: d.vec3f }));
  const one = { count: 2, weight: d.vec3f(1, 2, 3) };
  expect(total.combine(total.combine(total.identity(), one), one)).toEqual({
    count: 4,
    weight: d.vec3f(2, 4, 6),
  });

  const Hit = d.struct({ count: d.u32, hit: d.bool });
  // @ts-expect-error a bool is not a number
  expect(() => workgroup.sum(Hit)).toThrow('workgroup.sum adds numbers and vectors, not bool.');
});

it('sums a decorated schema from its inner zero, keeping the decoration', () => {
  const Aligned = d.align(16, d.u32);
  const total = workgroup.sum(Aligned);
  const zero: number = total.identity();
  expect(total.combine(total.combine(zero, 2), 3)).toBe(5);
  expect(total.schema).toBe(Aligned);
});
