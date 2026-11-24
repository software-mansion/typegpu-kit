import { expect, it } from 'vitest';
import { callSnapshot, callSwizzles, callValues, callLocalSnapshot } from './calls.ts';
import { cpuRoot } from './device.ts';

it('preserves argument evaluation order and vector arguments across a helper barrier', async () => {
  const root = cpuRoot();
  try {
    expect(await callValues(root)).toEqual(Array.from({ length: 8 }, () => [1, 2, 3, 80]).flat());
    expect(await callSnapshot(root)).toEqual([4, 5]);
    expect(await callLocalSnapshot(root)).toEqual([4, 5]);
    expect(await callSwizzles(root)).toEqual([1, 2, 9, 5, 5, 6, 9, 5]);
  } finally {
    root.destroy();
  }
});
