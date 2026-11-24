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
  it(`runs render compute ${run.name}`, async () => {
    const root = cpuRoot({ generator: SweepGenerator });
    try {
      const values = await run(root);
      expect(values.length).toBeGreaterThan(0);
      expect(values.every(Number.isFinite)).toBe(true);
      if (run === pickerValues) expect(values).toEqual([3, 150, 0xffffffff]);
      if (run === exposureValues) expect(values[0]).toBeCloseTo(0.09, 4);
      if (run === environmentValues)
        expect(values.every((value) => value > 0.45 && value < 1.55)).toBe(true);
      if (run === focusValues) {
        expect(values[1]).toBeGreaterThan(0);
        expect(values[1]).toBeLessThan(values[0]!);
        expect(values[0]).toBeLessThan(10);
      }
      if (run === impostorValues) {
        expect(values.slice(0, 3)).toEqual([2, 3, 4]);
        expect(values.slice(4, 7)).toEqual([1, 0.5, 0.25]);
      }
      if (run === cullingValues)
        expect(values.slice(2008).filter((_v, i) => i % 5 === 1)).toEqual([
          400, 0, 200, 0, 300, 0, 150, 0,
        ]);
    } finally {
      root.destroy();
    }
  });
}
