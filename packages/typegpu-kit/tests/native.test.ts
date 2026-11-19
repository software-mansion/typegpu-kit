import { expect, it, vi } from 'vitest';
import { frames } from '../src/index.ts';
import * as native from '../src/native.ts';
import { recordingRoot } from '../src/testing.ts';

function fakeCanvas() {
  let texture = { createView: vi.fn(() => ({})) };

  // react-native-webgpu's CanvasRef has no size; its context's canvas does.
  const context = {
    canvas: { width: 0, height: 0 },
    configure: vi.fn(),
    unconfigure: vi.fn(),
    present: vi.fn(),
    getCurrentTexture: () => texture,

    swap: () => {
      texture = { createView: vi.fn(() => ({})) };
    },
  };
  const canvas = { getContext: () => context } as unknown as native.Canvas;

  return { canvas, context };
}

it('sizes the drawing buffer from layout points and a capped pixel ratio', () => {
  const { root } = recordingRoot();
  const { canvas } = fakeCanvas();
  const surface = native.surface(root, canvas, { pixelRatio: 3, format: 'bgra8unorm' });

  surface.resize(390.5, 844);

  expect([surface.width, surface.height]).toEqual([781, 1688]);
  expect(surface.pixelRatio).toBe(2);
  expect(canvas.getContext('webgpu')?.canvas).toMatchObject({ width: 781, height: 1688 });
});

it('presents after each submitted frame that took the view, once', () => {
  const { root, log } = recordingRoot();
  const { canvas, context } = fakeCanvas();
  const surface = native.surface(root, canvas, { pixelRatio: 2, format: 'bgra8unorm' });

  let draw = true;
  const loop = frames.create(root, (frame) => {
    if (!draw) return;
    surface.view(frame);
    surface.view(frame);
    expect(log.submissions).toBe(frame.index);
  });

  loop.run(0);
  draw = false;
  loop.run(1);

  expect(context.present).toHaveBeenCalledTimes(1);
});

it('creates one view per canvas texture', () => {
  const { root } = recordingRoot();
  const { canvas, context } = fakeCanvas();
  const surface = native.surface(root, canvas, { pixelRatio: 2, format: 'bgra8unorm' });

  const loop = frames.create(root, (frame) => {
    const a = surface.view(frame);
    expect(surface.view(frame)).toBe(a);
  });

  loop.run(0);
  context.swap();
  loop.run(1);

  expect(context.present).toHaveBeenCalledTimes(2);
});

it('reconfigures the context on resume', () => {
  const { root } = recordingRoot();
  const { canvas, context } = fakeCanvas();
  const surface = native.surface(root, canvas, { pixelRatio: 2, format: 'bgra8unorm' });

  surface.pause();
  expect(surface.paused).toBe(true);

  surface.resume();
  expect(surface.paused).toBe(false);
  expect(context.configure).toHaveBeenCalledTimes(2);
});

it('throws from view while paused, naming the surface, and submits nothing', () => {
  const { root, log } = recordingRoot();
  const { canvas, context } = fakeCanvas();
  const surface = native.surface(root, canvas, { pixelRatio: 2, format: 'bgra8unorm' });
  const loop = frames.create(root, (frame) => {
    surface.view(frame);
  });

  surface.pause();

  expect(() => loop.run(0)).toThrow(/^native\.surface cannot call view\(\) while paused/);
  expect(log.submissions).toBe(0);
  expect(context.present).not.toHaveBeenCalled();
});

const device = (features: string[]) => ({ features: new Set(features) }) as unknown as GPUDevice;

it('throws for a device without implicit-device-synchronization', () => {
  expect(() => native.checkDevice(device(['implicit-device-synchronization']))).not.toThrow();
  expect(() => native.checkDevice(device([]))).toThrow(/lacks implicit-device-synchronization/);
});
