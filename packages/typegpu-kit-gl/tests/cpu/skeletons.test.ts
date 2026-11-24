import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { SweepGenerator } from './scene-generator.ts';
import { skeletons } from './skeletons.ts';

it('executes seeded skeleton machines and produces finite, nonzero pose joint rows', async () => {
  const root = cpuRoot({ generator: SweepGenerator });
  try {
    const buffers = await skeletons(root);
    expect(buffers.length).toBeGreaterThanOrEqual(3);
    const joints = buffers.find((buffer) => buffer.label.startsWith('poses_joints'))!;
    const values = new Float32Array(joints.bytes);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values.some((value) => Math.abs(value) > 0.5)).toBe(true);
  } finally {
    root.destroy();
  }
}, 120000);
