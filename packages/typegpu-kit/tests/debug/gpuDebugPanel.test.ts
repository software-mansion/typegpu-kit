import { afterEach, beforeEach, expect, expectTypeOf, it, vi } from 'vitest';
import type { GpuDebugPanelProps } from '../../src/debug/gpuDebugPanel.ts';
import { local, type Owned } from '../../src/hosting.ts';
import type { Hosted, Scene } from '../../src/native/host.ts';
import { mount, type Element } from '../fake-react.ts';

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
vi.mock('react-native', () => ({
  AppState: appState,
  PixelRatio: { get: () => 2 },
  Platform: { select: (options: { default: unknown }) => options.default },
  Pressable: 'Pressable',
  ScrollView: 'ScrollView',
  StyleSheet: { create: <T>(styles: T) => styles, absoluteFill: {} },
  Text: 'Text',
  View: 'View',
}));
vi.mock('react-native-webgpu', () => ({
  Canvas: 'Canvas',
  useMainDevice() {
    throw new Error('No GPUDeviceProvider.');
  },
}));

const { GpuDebugPanel } = await import('../../src/debug/gpuDebugPanel.ts');
const { recordingRoot } = await import('../../src/testing.ts');

beforeEach(() => {
  vi.useFakeTimers();
  // No device arrives, so these tests cover the panel without its graph
  vi.stubGlobal('navigator', { gpu: { requestAdapter: () => new Promise(() => undefined) } });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

/** A scene on the calling runtime, as `useScene` with `on: 'js'` hosts one. */
function hostedScene() {
  const owned = local(() => ({}));
  const scene: Pick<Owned<unknown>, 'run' | 'send'> = { run: owned.run, send: owned.send };
  return { scene, run: vi.spyOn(scene, 'run'), send: vi.spyOn(scene, 'send') };
}

function texts(node: unknown): string[] {
  if (typeof node === 'string') return [node];
  if (Array.isArray(node)) return node.flatMap(texts);
  if (node && typeof node === 'object' && 'props' in node) {
    return texts((node as Element).props.children);
  }
  return [];
}

/** The first element `predicate` accepts. */
function find(node: unknown, predicate: (element: Element) => boolean): Element | undefined {
  if (Array.isArray(node)) return node.map((child) => find(child, predicate)).find(Boolean);
  if (!node || typeof node !== 'object' || !('props' in node)) return undefined;
  const element = node as Element;
  return predicate(element) ? element : find(element.props.children, predicate);
}

/** The pressable whose text is `label`. */
function button(tree: unknown, label: string): (() => void) | undefined {
  const visit = (node: unknown): (() => void) | undefined => {
    if (Array.isArray(node)) return node.map(visit).find(Boolean);
    if (!node || typeof node !== 'object' || !('props' in node)) return undefined;
    const element = node as Element;
    if (element.type === 'Pressable' && texts(element).includes(label)) {
      return element.props.onPress as () => void;
    }
    return visit(element.props.children);
  };
  return visit(tree);
}

const mountPanel = (props: GpuDebugPanelProps) => mount(GpuDebugPanel, props);

const names = (calls: unknown[][]) => calls.map(([fn]) => (fn as { name: string }).name);

it('takes any hosted scene', () => {
  interface Game extends Scene {
    readonly score: number;
  }
  expectTypeOf<Hosted<Game>>().toExtend<NonNullable<GpuDebugPanelProps['scene']>>();
});

it('waits for the scene, records while mounted, reads four times a second and stops', async () => {
  const { scene, run, send } = hostedScene();
  const view = mountPanel({ scene: undefined });
  await vi.advanceTimersByTimeAsync(1000);
  expect(texts(view.current)).toContain('Capture');
  expect(send).not.toHaveBeenCalled();

  view.render({ scene });
  await vi.advanceTimersByTimeAsync(0);
  expect(names(send.mock.calls)).toEqual(['startOnScene']);

  await vi.advanceTimersByTimeAsync(1000);
  expect(names(run.mock.calls)).toEqual(Array(4).fill('readOnScene'));

  view.unmount();
  await vi.advanceTimersByTimeAsync(1000);
  expect(names(send.mock.calls)).toEqual(['startOnScene', 'stopOnScene']);
  expect(run).toHaveBeenCalledTimes(4);
});

it('hands Log’s summary to onLog, or shows it under the graph until closed', async () => {
  const { scene } = hostedScene();
  const onLog = vi.fn();
  const view = mountPanel({ scene, onLog });
  await vi.advanceTimersByTimeAsync(250);

  button(view.current, 'Log')!();
  await vi.advanceTimersByTimeAsync(0);
  expect(onLog).toHaveBeenCalledWith(expect.stringMatching(/^0 fps · gpu 0\.00ms over 0 frames/));

  view.render({ scene });
  button(view.current, 'Log')!();
  await vi.advanceTimersByTimeAsync(0);
  expect(texts(view.current).some((text) => text.startsWith('0 fps · gpu'))).toBe(true);

  button(view.current, '×')!();
  await vi.advanceTimersByTimeAsync(0);
  expect(texts(view.current).some((text) => text.startsWith('0 fps · gpu'))).toBe(false);
  view.unmount();
});

it('drops replies sent before a scene change, even once the scene is back, and frees its buttons', async () => {
  let reply!: (summary: string) => void;
  const slow = hostedScene();
  slow.run.mockImplementation(((fn: { name: string }) => {
    if (fn.name === 'summaryOnScene') return new Promise((resolve) => (reply = resolve));
    if (fn.name === 'measureOnScene') return new Promise(() => {});
    return Promise.resolve();
  }) as never);
  const other = hostedScene();
  const view = mountPanel({ scene: slow.scene });
  await vi.advanceTimersByTimeAsync(0);

  button(view.current, 'Log')!();
  button(view.current, 'Measure')!();
  view.render({ scene: other.scene });
  await vi.advanceTimersByTimeAsync(0);
  button(view.current, 'Measure')!();
  expect(names(other.run.mock.calls)).toContain('measureOnScene');

  view.render({ scene: slow.scene });
  reply('stale summary');
  await vi.advanceTimersByTimeAsync(0);
  expect(texts(view.current)).not.toContain('stale summary');
  view.unmount();
});

it('shows Trace only with onTrace, which receives a Chrome trace', async () => {
  const { scene } = hostedScene();
  const view = mountPanel({ scene });
  await vi.advanceTimersByTimeAsync(0);
  expect(button(view.current, 'Trace')).toBeUndefined();

  const onTrace = vi.fn();
  view.render({ scene, onTrace });
  button(view.current, 'Trace')!();
  await vi.advanceTimersByTimeAsync(0);
  expect(JSON.parse(onTrace.mock.calls[0][0] as string)).toHaveProperty('traceEvents');
  view.unmount();
});

it('steps the detail on the scene, from dispatches back to passes, and keeps it in later reads', async () => {
  const { scene, run, send } = hostedScene();
  const view = mountPanel({ scene });
  await vi.advanceTimersByTimeAsync(250);

  button(view.current, 'Per dispatch')!();
  await vi.advanceTimersByTimeAsync(250);
  const reads = run.mock.results.map(({ value }) => value as Promise<{ detail: string }>);
  expect((await reads.at(-1)!).detail).toBe('pass');
  expect(button(view.current, 'Per pass')).toBeDefined();
  expect(names(send.mock.calls)).toEqual(['startOnScene', 'detailOnScene']);
  view.unmount();
});

it('collapses to its title bar', async () => {
  const { scene } = hostedScene();
  const view = mountPanel({ scene });
  await vi.advanceTimersByTimeAsync(0);

  button(view.current, '▾')!();
  await vi.advanceTimersByTimeAsync(0);
  expect(button(view.current, '▸')).toBeDefined();
  const hasCanvas = JSON.stringify(view.current).includes('"Canvas"');
  expect(hasCanvas).toBe(false);
  view.unmount();
});

it('reads and draws only while the app is in the foreground', async () => {
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
  const { root } = recordingRoot({ features: ['implicit-device-synchronization'] });
  const context = {
    canvas: { width: 1, height: 1 },
    configure: vi.fn(),
    unconfigure: vi.fn(),
    present: vi.fn(),
    getCurrentTexture: () => ({ createView: () => ({}) }),
  };
  const { scene, run } = hostedScene();
  const view = mountPanel({ scene, device: root.device });
  const graph = find(view.current, (element) => element.type === 'Canvas')!;
  (graph.props.ref as { current: unknown }).current = { getContext: () => context };
  const box = find(view.current, (element) => typeof element.props.onLayout === 'function')!;
  (box.props.onLayout as (event: unknown) => void)({
    nativeEvent: { layout: { width: 300, height: 150 } },
  });
  await vi.advanceTimersByTimeAsync(500);
  expect(context.present).toHaveBeenCalled();

  appState.change('background');
  await vi.advanceTimersByTimeAsync(0);
  run.mockClear();
  context.present.mockClear();
  await vi.advanceTimersByTimeAsync(1000);

  expect(run).not.toHaveBeenCalled();
  expect(context.present).not.toHaveBeenCalled();
  expect(context.unconfigure).toHaveBeenCalledOnce();

  appState.change('active');
  await vi.advanceTimersByTimeAsync(500);
  expect(names(run.mock.calls)).toEqual(['readOnScene', 'readOnScene']);
  expect(context.present).toHaveBeenCalled();
  view.unmount();
});
