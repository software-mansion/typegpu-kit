import type { TgpuRoot } from 'typegpu';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { Frame } from '../../src/frames.ts';
import type { SceneRuntime } from '../../src/native/host.ts';
import type { Surface } from '../../src/targets.ts';
import * as fake from '../fake-worklets.ts';

vi.mock('react-native-worklets', () => import('../fake-worklets.ts'));
vi.mock('react-native-webgpu', () => ({ installWebGPU: vi.fn() }));

const { host } = await import('../../src/native/host.ts');
const { recordingRoot } = await import('../../src/testing.ts');

const RN = fake.RuntimeKind.ReactNative;

beforeEach(() => {
  fake.reset();
  vi.stubGlobal('requestAnimationFrame', fake.requestAnimationFrame);
  vi.stubGlobal('cancelAnimationFrame', fake.cancelAnimationFrame);
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function fakeContext() {
  const texture = { createView: () => ({}) };

  return {
    canvas: { width: 0, height: 0 },
    configure: vi.fn(),
    unconfigure: vi.fn(),
    present: vi.fn(),
    getCurrentTexture: () => texture,
  };
}

function hosting(on: SceneRuntime, onError?: (error: unknown) => void) {
  const { root } = recordingRoot({ features: ['implicit-device-synchronization'] });
  const context = fakeContext();
  const options = {
    on,
    device: root.device,
    context: context as unknown as GPUCanvasContext & { present(): void },
    width: 100,
    height: 50,
    pixelRatio: 2,
    ...(onError ? { onError } : {}),
  };

  return { context, options };
}

/** A scene as the examples write one: it draws to the surface and reports its frames. */
function counting(
  _root: TgpuRoot,
  surface: Surface,
  options: { onFrame?: (count: number) => void },
) {
  const runtimes: number[] = [];

  return {
    runtimes,

    frame(frame: Frame) {
      surface.view(frame);
      runtimes.push(fake.currentRuntime());
      options.onFrame?.(runtimes.length);
    },

    destroy: vi.fn(),
  };
}

it('hosts a scene on the JS thread, on animation frames', async () => {
  const { context, options } = hosting('js');
  const onFrame = vi.fn();
  const hosted = host(counting, { onFrame }, options);

  await fake.settle();
  fake.display(0);
  fake.display(16);

  expect(onFrame.mock.calls).toEqual([[1], [2]]);
  expect(context.canvas).toEqual({ width: 200, height: 100 });
  expect(context.present).toHaveBeenCalledTimes(2);

  hosted.resize(10, 10);
  hosted.pause();
  await fake.settle();
  fake.display(32);

  expect(onFrame).toHaveBeenCalledTimes(2);
  expect(context.canvas).toEqual({ width: 20, height: 20 });

  hosted.resume();
  await fake.settle();
  fake.display(48);
  const scene = await hosted.run((stage) => stage.scene);
  hosted.destroy();
  await fake.settle();
  fake.display(64);

  expect(onFrame).toHaveBeenCalledTimes(3);
  expect(scene.destroy).toHaveBeenCalledOnce();
  expect(context.unconfigure).toHaveBeenCalledOnce();
});

it('hosts a worklet scene on a worker, paced by the display, calling back on React Native', async () => {
  const { options } = hosting('worker');
  const called: number[] = [];
  const onFrame = () => called.push(fake.currentRuntime());
  const hosted = host(fake.worklet(counting), { onFrame }, options);

  await fake.settle();
  fake.display(0);
  await fake.settle();
  fake.display(16);
  await fake.settle();

  const ran = await Promise.all([hosted.run(({ scene }) => [...scene.runtimes]), fake.settle()]);
  expect(ran[0]).toHaveLength(2);
  expect(ran[0]).not.toContain(RN);
  expect(called).toEqual([RN, RN]);
});

/** One display frame, and what it scheduled on every runtime. */
async function display(milliseconds: number) {
  fake.display(milliseconds);
  await fake.settle();
}

it('asks the display for no frames while paused on a worker, and resumes at a dt of 0', async () => {
  const { options } = hosting('worker');
  const dts: number[] = [];
  const scene = fake.worklet((_root: TgpuRoot, surface: Surface) => ({
    frame(frame: Frame) {
      surface.view(frame);
      dts.push(frame.dt);
    },
  }));
  const hosted = host(scene, {}, options);

  await fake.settle();
  await display(0);
  await display(16);
  hosted.pause();
  await fake.settle();
  await display(32);
  await display(48);
  expect(fake.requestedFrames()).toBe(0);

  hosted.resume();
  await fake.settle();
  expect(fake.requestedFrames()).toBe(1);
  await display(1000);
  await display(1016);
  expect(dts).toEqual([0, 0.016, 0, 0.016]);
});

it('shows the worklet to write when createScene is not one', () => {
  const { options } = hosting('ui');

  expect(() => host(counting, {}, options)).toThrow(
    "createScene must be a worklet to run on the ui runtime. In a file where Worklets forwards imports, pass a wrapper: (root, surface, options) => { 'worklet'; return createScene(root, surface, options); }",
  );
});

it('refuses a device that two threads cannot share before it starts the scene off the JS thread', () => {
  const { options } = hosting('worker');
  const device = recordingRoot().root.device;

  expect(() => host(fake.worklet(counting), {}, { ...options, device })).toThrow(
    "The device lacks implicit-device-synchronization, so two threads can't share it. Request the feature in requiredFeatures",
  );
  expect(() => host(counting, {}, { ...options, device, on: 'js' })).not.toThrow();
});

it('stops its frames when the scene never gets ready, and reports it once', async () => {
  const onError = vi.fn();
  const { options } = hosting('js', onError);
  const frame = vi.fn();
  host(() => ({ frame, destroy() {}, ready: Promise.reject(new Error('never')) }), {}, options);
  await fake.settle();
  fake.display(0);
  fake.display(16);

  expect(onError.mock.calls).toEqual([[expect.objectContaining({ message: 'never' })]]);
  expect(frame).not.toHaveBeenCalled();
});

it.each([
  ['js', 'create'],
  ['worker', 'create (on TypeGPU Kit scene)'],
] as const)(
  'stops the relay and releases the canvas when a scene on %s fails to start',
  async (on, message) => {
    const onError = vi.fn();
    const { context, options } = hosting(on, onError);

    host(
      fake.worklet(() => {
        throw new Error('create');
      }),
      {},
      options,
    );
    await fake.settle();
    fake.display(0);
    await fake.settle();

    expect(onError.mock.calls).toEqual([[expect.objectContaining({ message })]]);
    expect(context.unconfigure).toHaveBeenCalledOnce();
    expect(fake.requestedFrames()).toBe(0);
  },
);
