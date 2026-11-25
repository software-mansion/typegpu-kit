import { expect, it } from 'vitest';
import { compileJavascript } from '../../src/cpu/javascript.ts';

it('rejects undeclared assignments instead of leaking globals', () => {
  expect(Object.hasOwn(globalThis, 'cpuUndeclaredRegression')).toBe(false);
  const run = compileJavascript([], 'cpuUndeclaredRegression = 1;');
  expect(run).toThrow(ReferenceError);
  expect(Object.hasOwn(globalThis, 'cpuUndeclaredRegression')).toBe(false);
});
