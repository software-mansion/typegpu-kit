import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { expressionCases, expressionValues } from './expressions.ts';

it('compiles and evaluates the composed expression matrix on the CPU', async () => {
  const root = cpuRoot();
  try {
    const values = await expressionValues(root);
    expect(values.length).toBe(expressionCases.length * 4 * 3);
    expect(values.every(Number.isFinite)).toBe(true);
  } finally {
    root.destroy();
  }
}, 120000);
