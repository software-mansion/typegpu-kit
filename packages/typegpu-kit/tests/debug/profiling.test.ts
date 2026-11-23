import { expect, it } from 'vitest';
import { captureProfile, startProfiling, stopProfiling } from '../../src/debug/profiling.ts';

it('rejects a capture still waiting for its frame when profiling stops', async () => {
  startProfiling();
  const capture = captureProfile();
  stopProfiling();

  await expect(capture).rejects.toThrow('Recording stopped before the capture finished.');
});
