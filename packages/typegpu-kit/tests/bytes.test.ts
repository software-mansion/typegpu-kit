import { d } from 'typegpu';
import { expect, it } from 'vitest';
import { byteViews, writerOf } from '../src/bytes.ts';

it('lays out a struct with padding, skipping fields it is not given', () => {
  const Part = d.struct({ a: d.f32, b: d.vec3f, c: d.arrayOf(d.u32, 2) });
  const views = byteViews(new ArrayBuffer(d.sizeOf(Part)));
  views.f32.fill(-1);
  writerOf(Part, 'part')(views, 0, { b: [1, 2, 3], c: [4, 5] });

  expect(views.f32[0]).toBe(-1); // a, untouched
  expect([...views.f32.subarray(4, 7)]).toEqual([1, 2, 3]);
  expect([...views.u32.subarray(7, 9)]).toEqual([4, 5]); // c follows b at 28
});

it('steps array elements by their alignment', () => {
  const views = byteViews(new ArrayBuffer(d.sizeOf(d.arrayOf(d.vec3f, 2))));
  writerOf(d.arrayOf(d.vec3f, 2), 'points')(views, 0, [
    [1, 2, 3],
    [4, 5, 6],
  ]);

  expect([...views.f32]).toEqual([1, 2, 3, 0, 4, 5, 6, 0]);
});

it('lays out a matrix with padding, from an instance or a plain array', () => {
  const padded = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0];
  const views = byteViews(new ArrayBuffer(2 * d.sizeOf(d.mat3x3f)));
  const write = writerOf(d.mat3x3f, 'm');

  write(views, 0, d.mat3x3f(1, 2, 3, 4, 5, 6, 7, 8, 9));
  write(views, 48, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  expect([...views.f32.subarray(0, 12)]).toEqual(padded);
  expect([...views.f32.subarray(12)]).toEqual(padded);
});

it('refuses types the CPU cannot write, naming the field', () => {
  const Pose = d.struct({ feet: d.arrayOf(d.vec2h, 2) });

  expect(() => writerOf(d.bool, 'flag')).toThrow("'flag' has type bool");
  expect(() => writerOf(d.f16, 'heat')).toThrow("'heat' has type f16");
  expect(() => writerOf(Pose, 'pose')).toThrow("'pose.feet[]' has type vec2h");
});
