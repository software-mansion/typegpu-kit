import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { callSnapshot, callSwizzles, callValues } from './calls.ts';
import { cpuRoot } from './device.ts';

it('matches dawn for helper argument order, composite snapshots and vector barriers', async () => {
  const gpu = await gpuRoot();
  const cpu = cpuRoot();
  try {
    for (const run of [callValues, callSnapshot, callSwizzles]) {
      const expected = await run(gpu);
      const actual = await run(cpu);
      const failures = expected.flatMap((value, i) =>
        actual[i] === value ? [] : [`${run.name} word ${i}: CPU=${actual[i]}, dawn=${value}`],
      );
      expect(actual.length).toBe(expected.length);
      expect(failures).toEqual([]);
    }
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
});
