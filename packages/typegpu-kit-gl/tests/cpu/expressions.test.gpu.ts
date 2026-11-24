import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { expressionCases, expressionValues } from './expressions.ts';

it('matches dawn for composed vector, scalar and matrix expressions', async () => {
  const gpu = await gpuRoot(),
    cpu = cpuRoot();
  try {
    const expected = await expressionValues(gpu),
      actual = await expressionValues(cpu);
    const differences = expected.flatMap((value, word) => {
      if (
        Number.isFinite(actual[word]) &&
        Math.abs(actual[word]! - value) <= 1e-4 * Math.max(1, Math.abs(value))
      )
        return [];
      const index = Math.floor(word / 3) % expressionCases.length;
      return [
        `input ${Math.floor(word / (3 * expressionCases.length))}, component ${word % 3}: ${expressionCases[index]!.source}: CPU=${actual[word]}, dawn=${value}`,
      ];
    });
    expect(actual.length).toBe(expected.length);
    expect(differences, `${differences.length} differing expression components`).toEqual([]);
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
}, 120000);
