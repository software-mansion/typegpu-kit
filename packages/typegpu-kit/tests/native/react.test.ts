import type { LayoutChangeEvent } from 'react-native';
import type { CanvasRef } from 'react-native-webgpu';
import type { TgpuRoot } from 'typegpu';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { Frame } from '../../src/frames.ts';
import type { CreateScene, Scene, UseSceneOptions } from '../../src/native/react.ts';
import type { Surface } from '../../src/targets.ts';
import { mount } from '../fake-react.ts';
import * as fake from '../fake-worklets.ts';

const appState = vi.hoisted(() => {
  const listeners = new Set<(state: string) => void>();

  return {
    currentState: 'active',

    addEventListener(_type: 'change', listener: (state: string) => void) {
      listeners.add(listener);
      return { remove: () => listeners.delete(listener) };
    },

    change(state: string) {
      for (const listener of listeners) listener(state);
    },
  };
});

vi.mock('react', () => import('../fake-react.ts'));
vi.mock('react-native', () => ({ AppState: appState, PixelRatio: { get: () => 2 } }));
vi.mock('react-native-worklets', () => import('../fake-worklets.ts'));
vi.mock('react-native-webgpu', () => ({
  installWebGPU: vi.fn(),
  useMainDevice() {
    throw new Error('No GPUDeviceProvider.');
  },
}));

const { useScene } = await import('../../src/native/react.ts');
const { recordingRoot } = await import('../../src/testing.ts');

const RN = fake.RuntimeKind.ReactNative;

beforeEach(() => {
  fake.reset();
  fake.worklet(sky);
  started.length = 0;
  vi.stubGlobal('requestAnimationFrame', fake.requestAnimationFrame);
  vi.stubGlobal('cancelAnimationFrame', fake.cancelAnimationFrame);
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

interface SkyOptions {
  readonly label: string;
  readonly onFrame?: (label: string) => void;
}

/** The label and runtime of each scene started. */
const started: { label: string; runtime: number }[] = [];

/** A worklet once each test marks it. */
function sky(_root: TgpuRoot, surface: Surface, options: SkyOptions) {
  started.push({ label: options.label, runtime: fake.currentRuntime() });

  return {
    frame(frame: Frame) {
      surface.view(frame);
      options.onFrame?.(options.label);
    },

    destroy() {},
  };
}

/** A canvas context that knows whether it is configured. */
function fakeContext() {
  const texture = { createView: () => ({}) };

  const context = {
    canvas: { width: 0, height: 0 },
    configured: false,
    configure: vi.fn(() => {
      context.configured = true;
    }),
    unconfigure: vi.fn(() => {
      context.configured = false;
    }),
    present: vi.fn(),

    getCurrentTexture() {
      if (!context.configured) throw new Error('The context is not configured.');
      return texture;
    },
  };

  return context;
}

interface Props {
  readonly options: SkyOptions;
  readonly settings?: UseSceneOptions;
}

/** `useScene(sky)` on a canvas that is laid out once `layout` is called. */
function mountSky(
  props: Props,
  device = recordingDevice(),
  createScene: CreateScene<SkyOptions, Scene> = sky,
) {
  const context = fakeContext();
  const view = mount(
    ({ options, settings }: Props) => useScene(createScene, options, { device, ...settings }),
    props,
  );
  view.current.canvas.ref.current = { getContext: () => context } as unknown as CanvasRef;

  const layout = () => {
    const layoutEvent = { nativeEvent: { layout: { width: 100, height: 50 } } };
    view.current.canvas.onLayout(layoutEvent as LayoutChangeEvent);
  };

  return { view, context, layout };
}

function recordingDevice() {
  return recordingRoot({ features: ['implicit-device-synchronization'] }).root.device;
}

/** Runs each display frame and what it schedules. */
async function display(...times: number[]) {
  for (const time of times) {
    fake.display(time);
    await fake.settle();
  }
}

it('starts each host with the options of the render it starts in, and the latest functions', async () => {
  const frames: string[] = [];
  const { view, layout } = mountSky({
    options: { label: 'first', onFrame: (label) => frames.push(`first render: ${label}`) },
  });

  view.render({
    options: { label: 'second', onFrame: (label) => frames.push(`second render: ${label}`) },
  });
  layout();
  await fake.settle();
  await display(0);

  view.render({ options: { label: 'third' }, settings: { maxPixelRatio: 3 } });
  await fake.settle();

  expect(started.map(({ label }) => label)).toEqual(['second', 'third']);
  expect(frames).toEqual(['second render: second']);
});

it('hosts the scene again on the same worker, named after it, and leaves the canvas configured', async () => {
  const { view, context, layout } = mountSky({ options: { label: 'sky' } });
  layout();
  await fake.settle();

  view.render({ options: { label: 'sky' }, settings: { maxPixelRatio: 3 } });
  await fake.settle();

  const [first, second] = started.map(({ runtime }) => runtime);
  expect(started).toHaveLength(2);
  expect(first).not.toBe(RN);
  expect(second).toBe(first);
  expect(fake.runtimeName(first!)).toBe('sky');
  expect(context.unconfigure).toHaveBeenCalledOnce();
  expect(context.configured).toBe(true);

  view.unmount();
  await fake.settle();

  expect(context.configured).toBe(false);
});

it('hosts the scene again on a new canvas, such as after a remount', async () => {
  const { view, context, layout } = mountSky({ options: { label: 'sky' } });
  layout();
  await fake.settle();

  const remounted = fakeContext();
  view.current.canvas.ref.current = null;
  view.current.canvas.ref.current = { getContext: () => remounted } as unknown as CanvasRef;
  await fake.settle();

  expect(started).toHaveLength(2);
  expect(context.configured).toBe(false);
  expect(remounted.configured).toBe(true);
});

it('stops frames while the app is in the background', async () => {
  const frames: string[] = [];
  const { layout } = mountSky({
    options: { label: 'sky', onFrame: (label) => frames.push(label) },
  });
  layout();
  await fake.settle();
  await display(0, 16);

  appState.change('background');
  await fake.settle();
  await display(32, 48);

  appState.change('active');
  await fake.settle();
  await display(64);

  expect(frames).toEqual(['sky', 'sky', 'sky']);
});

it.each([
  ['loss', 'The GPU device was lost: removed'],
  ['uncaptured error', 'bad'],
] as const)('reports the first %s of the device and releases the scene', async (kind, message) => {
  const { promise: lost, resolve: lose } = Promise.withResolvers<GPUDeviceLostInfo>();
  const events = new EventTarget();
  const device = {
    ...recordingDevice(),
    lost,
    addEventListener: events.addEventListener.bind(events),
  } as GPUDevice;

  const { view, context, layout } = mountSky({ options: { label: 'sky' } }, device);
  layout();
  await fake.settle();
  if (kind === 'loss') lose({ message: 'removed', reason: 'unknown' } as GPUDeviceLostInfo);
  else {
    for (const error of ['bad', 'worse']) {
      events.dispatchEvent(
        Object.assign(new Event('uncapturederror'), { error: { message: error } }),
      );
    }
  }
  await fake.settle();

  expect(view.current.error?.message).toBe(message);
  expect(context.configured).toBe(false);
});

/** `sky`, but never marked as a worklet. */
const plainSky: CreateScene<SkyOptions, Scene> = (root, surface, options) =>
  sky(root, surface, options);

it('reports a createScene that has to be a worklet as its error', async () => {
  const { view, layout } = mountSky({ options: { label: 'sky' } }, recordingDevice(), plainSky);
  layout();
  await fake.settle();

  expect(view.current.error?.message).toMatch(/^createScene must be a worklet/);
  expect(started).toHaveLength(0);
});

it('requests a device of its own with the optional features the adapter has', async () => {
  const requested: GPUDeviceDescriptor[] = [];
  const adapter = {
    features: new Set(['implicit-device-synchronization', 'indirect-first-instance']),

    requestDevice: async (descriptor: GPUDeviceDescriptor) => {
      requested.push(descriptor);
      return recordingDevice();
    },
  };
  vi.stubGlobal('navigator', {
    gpu: { getPreferredCanvasFormat: () => 'bgra8unorm', requestAdapter: async () => adapter },
  });

  const settings = { optionalFeatures: ['indirect-first-instance', 'timestamp-query'] } as const;
  const view = mount(() => useScene(sky, { label: 'sky' }, settings), {});
  view.current.canvas.ref.current = { getContext: () => fakeContext() } as unknown as CanvasRef;

  const layoutEvent = { nativeEvent: { layout: { width: 100, height: 50 } } };
  view.current.canvas.onLayout(layoutEvent as LayoutChangeEvent);
  await fake.settle();

  expect(requested).toEqual([{ requiredFeatures: ['indirect-first-instance'] }]);
  expect(started.map(({ label }) => label)).toEqual(['sky']);
});

it('hosts the scene again only once the last host has let go of the canvas', async () => {
  const { promise: built, resolve: build } = Promise.withResolvers<void>();
  const slow = fake.worklet(async (root: TgpuRoot, surface: Surface, options: SkyOptions) => {
    await built;
    return sky(root, surface, options);
  });

  const frames: string[] = [];
  const options = { label: 'sky', onFrame: (label: string) => frames.push(label) };
  const { view, context, layout } = mountSky({ options }, recordingDevice(), slow);
  layout();
  await fake.settle();

  view.render({ options, settings: { maxPixelRatio: 3 } });
  await fake.settle();
  build();
  await fake.settle();
  await display(0);

  expect(context.unconfigure).toHaveBeenCalledOnce();
  expect(context.configured).toBe(true);
  expect(view.current.error).toBeUndefined();
  expect(frames).toEqual(['sky']);
});

it('destroys a device it requested once the scene has let it go', async () => {
  const device = recordingDevice();
  const context = fakeContext();
  const destroyed: boolean[] = [];
  device.destroy = () => void destroyed.push(context.configured);

  const adapter = { features: new Set(), requestDevice: async () => device };
  vi.stubGlobal('navigator', {
    gpu: { getPreferredCanvasFormat: () => 'bgra8unorm', requestAdapter: async () => adapter },
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});

  const view = mount(() => useScene(sky, { label: 'sky' }), {});
  view.current.canvas.ref.current = { getContext: () => context } as unknown as CanvasRef;

  const layoutEvent = { nativeEvent: { layout: { width: 100, height: 50 } } };
  view.current.canvas.onLayout(layoutEvent as LayoutChangeEvent);
  await fake.settle();
  expect(context.configured).toBe(true);

  view.unmount();
  await fake.settle();

  expect(destroyed).toEqual([false]);
});
