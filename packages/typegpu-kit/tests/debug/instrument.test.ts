import { describe, expect, it } from 'vitest';
import { install } from '../../src/debug/gpuDebug.ts';
import { instrument, type Interfaces } from '../../src/debug/instrument.ts';
import { createRecorder } from '../../src/debug/recorder.ts';

const createCommandEncoder = () => {};

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

  // WebKit without WebGPU still defines `GPUCommandEncoder`, `GPUQueue` and the pass encoders
  it("patches nothing where the browser defines only some of WebGPU's interfaces", () => {
    const some = ['GPUCommandEncoder', 'GPUQueue', 'GPUComputePassEncoder', 'GPURenderPassEncoder'];
    const given = Object.fromEntries(
      ['GPUDevice', 'GPUTexture', ...some].map((name) => [
        name,
        { prototype: { createCommandEncoder, createView() {} } },
      ]),
    ) as unknown as NonNullable<Parameters<typeof instrument>[0]>['interfaces'];
    for (const name of some) Object.assign(globalThis, { [name]: { prototype: {} } });
    try {
      const alone = instrument();
      alone.start(createRecorder());
      alone[Symbol.dispose]();

      // The interfaces a device brings are patched all the same
      const withGiven = instrument({ interfaces: given! });
      expect(given!.GPUDevice.prototype.createCommandEncoder).not.toBe(createCommandEncoder);
      withGiven[Symbol.dispose]();
      expect(given!.GPUDevice.prototype.createCommandEncoder).toBe(createCommandEncoder);
    } finally {
      for (const name of some) delete (globalThis as Record<string, unknown>)[name];
    }
  });

  it("patches the browser's own interfaces once when they are given too", () => {
    const names = [
      'GPUDevice',
      'GPUTexture',
      'GPUQueue',
      'GPUCommandEncoder',
      'GPUComputePassEncoder',
      'GPURenderPassEncoder',
    ];
    let patched = 0;
    const counting = new Proxy<Record<string, unknown>>(
      { createCommandEncoder, createView() {} },
      {
        set(target, key, value) {
          if (key === 'createCommandEncoder') patched++;
          target[key as string] = value;
          return true;
        },
      },
    );
    const own = Object.fromEntries(
      names.map((name) => [name, { prototype: name === 'GPUDevice' ? counting : {} }]),
    );
    Object.assign(globalThis, own);
    try {
      const both = instrument({ interfaces: own as unknown as Interfaces });
      expect(patched).toBe(1);
      both[Symbol.dispose]();
      expect(counting.createCommandEncoder).toBe(createCommandEncoder);
    } finally {
      for (const name of names) delete (globalThis as Record<string, unknown>)[name];
    }
  });
});
