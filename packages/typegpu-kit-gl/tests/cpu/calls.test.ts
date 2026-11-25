import { expect, it } from 'vitest';
import {
  conditionalValues,
  callSnapshot,
  callSwizzles,
  callValues,
  callLocalSnapshot,
  vectorPrecedence,
  callEffects,
  vectorResults,
  recordValues,
  uniformArguments,
  matrixLocals,
  recordFields,
  literalFields,
  addressReturns,
} from './calls.ts';
import { cpuRoot } from './device.ts';

it('preserves argument evaluation order and vector arguments across a helper barrier', async () => {
  const root = cpuRoot();
  try {
    expect(await conditionalValues(root)).toEqual([
      2,
      3,
      Math.fround(0.3),
      22,
      4,
      5,
      6,
      12,
      2,
      3,
      Math.fround(0.3),
      22,
      4,
      5,
      8,
      12,
    ]);
    expect(await callValues(root)).toEqual(Array.from({ length: 8 }, () => [1, 2, 3, 80]).flat());
    expect(await callSnapshot(root)).toEqual([4, 5]);
    expect(await callLocalSnapshot(root)).toEqual([4, 5]);
    expect(await uniformArguments(root)).toEqual([
      2, 3, 7, 8, 7, 8, 2, 3, 2, 3, 7, 8, 2, 3, 7, 8, 12, 13, 17, 18, 17, 18, 12, 13, 12, 13, 17,
      18, 12, 13, 17, 18,
    ]);
    expect(await addressReturns(root)).toEqual([4, 5, 6, 7, 99, 99, 99, 99]);
    expect(await literalFields(root)).toEqual([3, 4, 1, 100, 99]);
    expect(await recordValues(root)).toEqual([103, 4, 8, 9, 0, 1, 2]);
    expect(await matrixLocals(root)).toEqual([4, 4, 2, 3, 3, 2, 1, 91]);
    expect(await recordFields(root)).toEqual([
      1, 7, 2, 3, 4, 5, 6, 2, 2, 2, 2, 3, 90, 7, 2, 3, 90, 2, 3, 7,
    ]);
    expect(await callEffects(root)).toEqual([4, 5, 1]);
    expect(await vectorResults(root)).toEqual(
      Array.from({ length: 8 }, (_, i) => [9 * i + 75, 27 * i + 225, i + 10, i + 11]).flat(),
    );
    expect(await vectorPrecedence(root)).toEqual([-32, 18, 2, -48, 27, 3, 8, -12, 20]);
    expect(await callSwizzles(root)).toEqual([1, 2, 9, 5, 5, 6, 9, 5]);
  } finally {
    root.destroy();
  }
});
