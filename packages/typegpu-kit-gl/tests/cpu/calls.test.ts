import { expect, it } from 'vitest';
import {
  callSnapshot,
  callSwizzles,
  callValues,
  callLocalSnapshot,
  vectorPrecedence,
  callEffects,
  vectorResults,
} from './calls.ts';
import { cpuRoot } from './device.ts';

it('preserves argument evaluation order and vector arguments across a helper barrier', async () => {
  const root = cpuRoot();
  try {
    expect(await callValues(root)).toEqual(Array.from({ length: 8 }, () => [1, 2, 3, 80]).flat());
    expect(await callSnapshot(root)).toEqual([4, 5]);
    expect(await callLocalSnapshot(root)).toEqual([4, 5]);
    expect(await callEffects(root)).toEqual([4, 5, 1]);
    expect(await vectorResults(root)).toEqual(
      Array.from({ length: 8 }, (_, i) => [6 * i + 51, 18 * i + 153, i + 7, i + 8]).flat(),
    );
    expect(await vectorPrecedence(root)).toEqual([-32, 18, 2, -48, 27, 3, 8, -12, 20]);
    expect(await callSwizzles(root)).toEqual([1, 2, 9, 5, 5, 6, 9, 5]);
  } finally {
    root.destroy();
  }
});
