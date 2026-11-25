import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { scopedLocals } from './scopes.ts';
it('keeps CPU aliases within their declaring blocks', async () => {
  let code = '';
  const root = cpuRoot({ onModule: (value) => (code = value) });
  try {
    expect(await scopedLocals(root), code).toEqual([8, 80, 7, 70, 58, 8, 5, 50, 8, 20]);
  } finally {
    root.destroy();
  }
});
