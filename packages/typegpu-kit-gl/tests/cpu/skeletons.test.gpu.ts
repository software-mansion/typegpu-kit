import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { poseField, skeletons } from './skeletons.ts';

it('matches dawn for seeded skeleton machines, pose tracks and joint rows after four seconds', async () => {
  const gpu = await gpuRoot();
  const cpu = cpuRoot({ generator: SweepGenerator });
  try {
    const expected = await skeletons(gpu);
    const actual = await skeletons(cpu);
    expect(expected.length).toBeGreaterThanOrEqual(3);
    expect(actual.map((buffer) => buffer.label)).toEqual(expected.map((buffer) => buffer.label));
    const differences: string[] = [];
    for (let buffer = 0; buffer < expected.length; buffer++) {
      const left = expected[buffer]!,
        right = actual[buffer]!;
      expect(right.bytes.byteLength, left.label).toBe(left.bytes.byteLength);
      const a = new Uint32Array(left.bytes),
        b = new Uint32Array(right.bytes);
      const af = new Float32Array(left.bytes),
        bf = new Float32Array(right.bytes);
      for (let i = 0; i < a.length; i++) {
        if (
          a[i] === b[i] ||
          (Number.isFinite(af[i]) &&
            Number.isFinite(bf[i]) &&
            Math.abs(af[i]! - bf[i]!) <= 1e-4 * Math.max(1, Math.abs(af[i]!)))
        )
          continue;
        differences.push(
          `${left.label} word ${i} ${poseField(left.label, i)}: CPU=${bf[i]} (u32 ${b[i]}), dawn=${af[i]} (u32 ${a[i]})`,
        );
      }
    }
    expect(differences.slice(0, 20), `${differences.length} differing words`).toEqual([]);
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
}, 120000);
