import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { boxBounds, cylinder, posePhase } from './render.ts';

for (const run of [boxBounds, cylinder, posePhase]) {
  it(`matches dawn for render geometry ${run.name}`, async () => {
    const gpu = await gpuRoot();
    const cpu = cpuRoot();
    try {
      const expected = await run(gpu);
      const actual = await run(cpu);
      const differences = expected.flatMap((value, i) =>
        !Number.isFinite(actual[i]) || Math.abs(actual[i]! - value) > 1e-5
          ? [`${run.name} word ${i}: CPU=${actual[i]}, dawn=${value}`]
          : [],
      );
      expect(actual.length).toBe(expected.length);
      expect(differences.slice(0, 20)).toEqual([]);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  });
}
