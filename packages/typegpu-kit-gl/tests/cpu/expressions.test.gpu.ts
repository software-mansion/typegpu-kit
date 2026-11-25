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
    const cpuBits = new Uint32Array(Float32Array.from(actual).buffer);
    const gpuBits = new Uint32Array(Float32Array.from(expected).buffer);
    const differences = expected.flatMap((value, word) => {
      const index = Math.floor(word / 3) % expressionCases.length;
      const source = expressionCases[index]!.source;
      const approximate =
        /std\.(?:normalize|refract|length|distance|sin|cos|exp|pow|inverseSqrt)\(/.test(source);
      const ulps = Math.abs(cpuBits[word]! - gpuBits[word]!);
      if (
        Number.isFinite(actual[word]) &&
        (cpuBits[word] === gpuBits[word] ||
          (approximate && Math.sign(actual[word]!) === Math.sign(value) && ulps <= 8))
      )
        return [];
      return [
        `input ${Math.floor(word / (3 * expressionCases.length))}, component ${word % 3}: ${source}: CPU=${actual[word]}, dawn=${value}, ${ulps} ulp (${approximate ? 'approximate' : 'exact'})`,
      ];
    });
    expect(actual.length).toBe(expected.length);
    expect(differences, `${differences.length} differing expression components`).toEqual([]);
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
}, 120000);
