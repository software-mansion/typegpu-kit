import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { moduleCases } from './modules.ts';

for (const [name, run] of moduleCases) {
  it(`matches dawn for Kit ${name}`, async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const expected = await run(gpu),
        actual = await run(cpu);
      expect(actual.length).toBe(expected.length);
      const failures: string[] = [];
      for (let i = 0; i < expected.length; i++) {
        const tolerance = name.startsWith('particles') ? 1e-4 : 0;
        if (Math.abs(actual[i]! - expected[i]!) > tolerance || !Number.isFinite(actual[i]))
          failures.push(`word ${i}: CPU=${actual[i]}, dawn=${expected[i]}`);
      }
      expect(failures, name).toEqual([]);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  });
}
