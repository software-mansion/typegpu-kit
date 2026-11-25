import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { sampleCorners, cornerValues } from './cube-corners.ts';
import { cpuRoot } from './device.ts';
import { recordResults } from './results.ts';

it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'records implementation-defined cube-corner taps',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    const results = [];
    try {
      for (const size of [1, 2, 4]) {
        const expected = await sampleCorners(gpu, size),
          actual = await sampleCorners(cpu, size);
        for (let row = 0; row < 16; row++) {
          const corner = row % 8,
            layer = Math.floor(row / 8),
            faces = [
              cornerValues[corner & 1 ? 1 : 0]!,
              cornerValues[corner & 2 ? 3 : 2]!,
              cornerValues[corner & 4 ? 5 : 4]!,
            ].map((v) => v + layer * 192);
          for (const output of [expected, actual]) {
            expect(output[row]![0]).toBeGreaterThanOrEqual(Math.min(...faces));
            expect(output[row]![0]).toBeLessThanOrEqual(Math.max(...faces));
          }
        }
        results.push({ size, cpu: actual, dawn: expected });
      }
      recordResults('cubeCorners', results);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
);
