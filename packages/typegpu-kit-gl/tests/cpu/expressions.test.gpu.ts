import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { expressionCases, expressionValues, inputPairs } from './expressions.ts';

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
      const quotient = /[/%]|std\.mod\(/.test(source);
      const input = Math.floor(word / (3 * expressionCases.length));
      const magnitude = Math.max(
        1,
        Math.abs(value),
        ...[...inputPairs[input * 2]!, ...inputPairs[input * 2 + 1]!].map(Math.abs),
      );
      const acceptable = approximate
        ? Math.abs(actual[word]! - value) <= 1e-5 * magnitude
        : quotient && Math.sign(actual[word]!) === Math.sign(value) && ulps <= 3;
      if (Number.isFinite(actual[word]) && (actual[word] === value || acceptable)) return [];
      return [
        `input ${Math.floor(word / (3 * expressionCases.length))}, component ${word % 3}: ${source}: CPU=${actual[word]} (0x${cpuBits[word]!.toString(16)}), dawn=${value} (0x${gpuBits[word]!.toString(16)}), ${ulps} ulp (${approximate ? 'approximate' : quotient ? 'division/remainder' : 'exact'})`,
      ];
    });
    expect(actual.length).toBe(expected.length);
    expect(differences, `${differences.length} differing expression components`).toEqual([]);
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
}, 120000);
