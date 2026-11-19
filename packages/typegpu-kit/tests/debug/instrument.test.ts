import { describe, expect, it } from 'vitest';
import { install } from '../../src/debug/gpuDebug.ts';
import { instrument } from '../../src/debug/instrument.ts';
import { createRecorder } from '../../src/debug/recorder.ts';

// Node has no WebGPU unless a test loads dawn.node, as a browser without it has none.
describe('without WebGPU', () => {
  it('patches nothing and reports nothing', () => {
    expect(typeof GPUCommandEncoder).toBe('undefined');

    const instrumentation = instrument();
    const recorder = createRecorder();
    instrumentation.start(recorder);
    instrumentation.stop();
    instrumentation[Symbol.dispose]();

    expect(recorder.bars()).toEqual([]);
  });

  it('installs and uninstalls', () => {
    const debug = install();
    expect(debug.running).toBe(false);
    debug.destroy();
  });
});
