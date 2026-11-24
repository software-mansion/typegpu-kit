import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { moduleCases } from './modules.ts';

for (const [name, run, expected] of moduleCases) {
  it(`executes Kit ${name}`, async () => {
    const root = cpuRoot();
    try {
      const actual = await run(root);
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++)
        expect(actual[i], `${name} word ${i}`).toBeCloseTo(expected[i]!, 4);
    } finally {
      root.destroy();
    }
  });
}
