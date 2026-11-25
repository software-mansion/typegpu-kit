import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { perlinCache, randomValues, scaledSeeds, storedVectors } from './noise.ts';

it('matches dawn for randf private vector state', async () => {
  const gpu = await gpuRoot();
  const cpu = cpuRoot();
  try {
    for (const run of [randomValues, storedVectors, scaledSeeds, perlinCache]) {
      const expected = await run(gpu);
      const actual = await run(cpu);
      const cpuBits = new Uint32Array(Float32Array.from(actual).buffer);
      const gpuBits = new Uint32Array(Float32Array.from(expected).buffer);
      const differences = expected.flatMap((value, i) =>
        cpuBits[i] === gpuBits[i]
          ? []
          : [
              `${run.name} word ${i}: CPU=${actual[i]} (0x${cpuBits[i]!.toString(16)}), dawn=${value} (0x${gpuBits[i]!.toString(16)}), bit distance ${Math.abs(cpuBits[i]! - gpuBits[i]!)}`,
            ],
      );
      expect(actual.length).toBe(expected.length);
      expect(
        differences.length,
        `${run.name}: ${differences.length}/${expected.length} bit differences\n${differences.slice(0, 30).join('\n')}`,
      ).toBe(0);
    }
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
});
