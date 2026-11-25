import { expect, it } from 'vitest';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { cpuRoot } from './device.ts';
import { scopedLocals } from './scopes.ts';
it('matches dawn for sibling, loop and nested local scopes', async () => {
  const gpu = await gpuRoot();
  const cpu = cpuRoot();
  try {
    expect(await scopedLocals(cpu)).toEqual(await scopedLocals(gpu));
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
});
