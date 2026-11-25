import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { randomValues, scaledSeeds, storedVectors } from './noise.ts';

it('reads and updates the private vector state of randf', async () => {
  const root = cpuRoot();
  try {
    const values = await randomValues(root);
    expect(values[0]).toBeCloseTo(0.0992, 3);
    expect(values[1]).toBeCloseTo(0.4676, 3);
    expect(new Set(values).size).toBe(8);
    expect((await scaledSeeds(root))[5]).toBeCloseTo(0.7195, 4);
    expect(await storedVectors(root)).toEqual([2, 1, 4, 3, 6, 5, 8, 7, 2, 1, 4, 3, 6, 5, 8, 7]);
  } finally {
    root.destroy();
  }
});
