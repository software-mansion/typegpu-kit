import { expect, it } from 'vitest';
import { prunePureLocals } from '../../src/cpu/dead-locals.ts';

it('removes unused pure dependencies while preserving calls with effects', () => {
  const body = `{
let box = bounds();
let sphere = { center: place(box), radius: 2 };
let discarded = change();
return state();
}`;
  const optimized = prunePureLocals(
    body,
    new Map([
      ['bounds', false],
      ['place', false],
      ['change', true],
    ]),
  );
  for (const source of [body, optimized]) {
    let state = 0;
    const run = new Function('bounds', 'place', 'change', 'state', source) as (
      ...args: unknown[]
    ) => number;
    expect(
      run(
        () => 7,
        (value: number) => value * 2,
        () => ++state,
        () => state,
      ),
    ).toBe(1);
  }
  expect(optimized).not.toContain('bounds()');
  expect(optimized).not.toContain('place(box)');
  expect(optimized).toContain('change()');
});
