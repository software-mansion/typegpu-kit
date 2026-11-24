import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { randomValues, storedVectors } from './noise.ts';

it('matches dawn for randf private vector state', async () => {
  const gpu = await gpuRoot();
  const cpu = cpuRoot();
  try {
    for (const run of [randomValues, storedVectors]) {
      const expected = await run(gpu);
      const actual = await run(cpu);
      expect(
        expected.flatMap((value, i) =>
          Math.abs(actual[i]! - value) < 1e-7
            ? []
            : [`${run.name} word ${i}: CPU=${actual[i]}, dawn=${value}`],
        ),
      ).toEqual([]);
    }
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
});
