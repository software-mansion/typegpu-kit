import { afterEach, expect, it, vi } from 'vitest';
import * as backends from '../src/backends.ts';
import { recordingRoot } from '../src/testing.ts';

afterEach(() => {
  vi.unstubAllGlobals();
});

it('reads WebGPU from a device that says nothing, by root or by device', () => {
  const { root } = recordingRoot();
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });

  const backend = backends.of(root);

  expect(backend.kind).toBe('webgpu');
  expect(backend.compute).toBe('gpu');
  expect(backend.preferredCanvasFormat).toBe('bgra8unorm');
  expect(backends.of(root.device).kind).toBe('webgpu');
});

it('configures a WebGPU canvas for the device', () => {
  const { root } = recordingRoot();
  const context = { configure: vi.fn() };
  const canvas = { getContext: vi.fn(() => context) } as unknown as HTMLCanvasElement;

  const configured = backends.of(root).configure(canvas, { format: 'rgba8unorm' });

  expect(configured).toBe(context);
  expect(context.configure).toHaveBeenCalledWith({ format: 'rgba8unorm', device: root.device });
  expect(() =>
    backends
      .of(root)
      .configure({ getContext: () => null } as unknown as HTMLCanvasElement, {
        format: 'rgba8unorm',
      }),
  ).toThrow(/no WebGPU context/);
});

it("returns a device's own description from its brand", () => {
  const { root } = recordingRoot();
  const own: backends.Backend = {
    kind: 'webgl',
    compute: 'cpu',
    preferredCanvasFormat: 'rgba8unorm',
    configure: () => ({}) as GPUCanvasContext,
  };
  Object.assign(root.device, { [backends.brand]: own });

  expect(backends.of(root)).toBe(own);
  expect(backends.of(root.device)).toBe(own);
});
