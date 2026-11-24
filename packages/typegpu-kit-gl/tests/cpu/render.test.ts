import { writeFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { boxBounds, cylinder } from './render.ts';

it('bakes mesh bounds and keeps the cylinder cap ring', async () => {
  let modules = 0;
  const root = cpuRoot({
    onModule: (code) => {
      if (process.env.CPU_DIAGNOSTICS === '1')
        writeFileSync(new URL(`./mesh-${modules++}.generated.txt`, import.meta.url), code);
    },
  });
  try {
    expect(await boxBounds(root)).toEqual([0, 0, 0, 0, 1, 0.5, 0.25, 0]);
    const vertices = await cylinder(root);
    for (let i = 67; i <= 98; i++) {
      expect(Math.hypot(vertices[i * 4]!, vertices[i * 4 + 2]!)).toBeCloseTo(0.55, 5);
      expect(vertices[i * 4 + 1]).toBeCloseTo(0.2, 5);
    }
  } finally {
    root.destroy();
  }
});
