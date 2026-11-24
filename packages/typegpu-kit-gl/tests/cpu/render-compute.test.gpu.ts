import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import {
  cullingValues,
  pickerValues,
  exposureValues,
  environmentValues,
  focusValues,
  impostorValues,
} from './render-compute.ts';

for (const run of [
  cullingValues,
  pickerValues,
  exposureValues,
  environmentValues,
  focusValues,
  impostorValues,
]) {
  it(`matches dawn for render compute ${run.name}`, async () => {
    const gpu = await gpuRoot();
    const cpu = cpuRoot({ generator: SweepGenerator });
    try {
      const expected = await run(gpu),
        actual = await run(cpu);
      const tolerance = run === environmentValues ? 0.005 : 0.0001;
      const differences = expected.flatMap((value, i) =>
        Number.isFinite(actual[i]) && Math.abs(actual[i]! - value) <= tolerance
          ? []
          : [`${run.name} word ${i}: CPU=${actual[i]}, dawn=${value}`],
      );
      expect(actual.length).toBe(expected.length);
      expect(differences.slice(0, 20), `${differences.length} differing words`).toEqual([]);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  }, 120000);
}
