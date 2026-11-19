/**
 * Shows the GPU debug panel over a React Native scene hosted with `useScene` or `host`. Times
 * each pass of the scene's frames on its runtime and graphs GPU and CPU milliseconds per frame,
 * as the web panel does. Records only while `<GpuDebugPanel>` is mounted, and reads and draws only
 * while the app is in the foreground. Request `timestamp-query`
 * with the scene's device, through `useScene`'s `optionalFeatures` or
 * `tgpu.init({ device: { optionalFeatures } })`; without it, the panel shows CPU times only.
 *
 * @module
 * @unstable
 * @kind tool
 * @platform native
 * @flat
 * @requires timestamp-query optional the scene's device needs it for GPU times; without it, CPU
 * times only
 * @owns a device unless given or provided one, a root, 2 buffers, a bind group, a pipeline, a
 * kit/frames and the graph canvas's configuration; on the scene's runtime, up to 16 query sets and
 * 32 buffers per device while it records, and a texture per canvas while it measures
 */

import {
  createElement as h,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactElement,
} from 'react';
import {
  AppState,
  PixelRatio,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
  type StyleProp,
  type ViewStyle,
} from 'react-native';
import { Canvas, useMainDevice, type CanvasRef } from 'react-native-webgpu';
import { tgpu } from 'typegpu';
import { create as createFrames } from '../frames.ts';
import type { Owned } from '../hosting.ts';
import * as native from '../native.ts';
import { createGraphDrawing } from './graphGpu.ts';
import { createGraphLayout, type Label, theme } from './graphLayout.ts';
import { withoutTracking } from './instrument.ts';
import { detailLabels, details, formatMs, type Detail } from './recorder.ts';
// Use named imports because worklets forwards them to the scene runtime
import {
  captureProfile,
  detailProfile,
  measureProfile,
  profileSummary,
  profileTrace,
  readProfile,
  startProfiling,
  stopProfiling,
  type Profile,
} from './profiling.ts';

export type { Profile } from './profiling.ts';

export interface GpuDebugPanelProps {
  /** The hosted scene, from `useScene` or `host`. The panel waits while it is undefined. */
  readonly scene: Pick<Owned<unknown>, 'run' | 'send'> | undefined;
  /**
   * Draws the graph with this device, such as the scene's. Default the nearest
   * `<GPUDeviceProvider>` device, or one the panel requests.
   */
  readonly device?: GPUDevice;
  /**
   * Receives the summary that Log produces, such as to copy or share it. Default the panel shows
   * it under the graph.
   */
  readonly onLog?: (text: string) => void;
  /**
   * Receives the last two seconds of timed commands in Chrome trace format, as JSON for Perfetto,
   * such as to share the file. Shows a Trace button when given. Default none.
   */
  readonly onTrace?: (json: string) => void;
  /** Places the panel, such as `{ position: 'absolute', left: 8, bottom: 8 }`. Default none. */
  readonly style?: StyleProp<ViewStyle>;
  /** Sets the graph height in points. Default 150. */
  readonly height?: number;
}

/** How often the panel reads the recording, in milliseconds. */
const readInterval = 250;
/** Shapes the graph can draw: columns, peaks, budgets, outlines, swatches and their links. */
const capacity = 4096;
const compactWidth = 380;
const timestampNote = 'The device does not support timestamp-query.';

function startOnScene() {
  'worklet';
  startProfiling();
}

function stopOnScene() {
  'worklet';
  stopProfiling();
}

function readOnScene(_: unknown, columns: number) {
  'worklet';
  return readProfile(columns);
}

function detailOnScene(_: unknown, detail: Detail) {
  'worklet';
  detailProfile(detail);
}

function measureOnScene() {
  'worklet';
  return measureProfile();
}

function summaryOnScene() {
  'worklet';
  return profileSummary();
}

function captureOnScene() {
  'worklet';
  return captureProfile();
}

function traceOnScene() {
  'worklet';
  return profileTrace();
}

const ignore = () => undefined;

/**
 * Shows a collapsible GPU debug panel that records the scene's frames while mounted. The detail
 * button splits GPU time per pass, debug group or dispatch. Measure runs frames back to back for
 * a second at the GPU's full clock. Capture lists the next frame's commands with their times. Log
 * gives mean frame times over the last second. Trace gives a Chrome trace for Perfetto.
 *
 * @param props the hosted scene, the graph's device and the panel's callbacks
 * @returns the panel, to place over the scene's canvas
 */
export function GpuDebugPanel(props: GpuDebugPanelProps): ReactElement {
  const { scene, onLog, onTrace, style, height = 150 } = props;
  const device = useGraphDevice(props.device);
  const active = useActive();
  const canvas = useRef<CanvasRef>(null);
  const graph = useRef<PanelGraph>(undefined);
  const [width, setWidth] = useState(0);
  const [profile, setProfile] = useState<Profile>();
  const [labels, setLabels] = useState<Label[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const [text, setText] = useState<string>();
  const [capturing, setCapturing] = useState(false);
  const [measuring, setMeasuring] = useState(false);
  const columns = useRef(0);

  useEffect(() => {
    if (!scene) return;

    scene.send(startOnScene);
    return () => {
      scene.send(stopOnScene);
      setProfile(undefined);
    };
  }, [scene]);

  // Read and draw only in the foreground, where the hosted scene runs
  useEffect(() => {
    if (!scene || !active) return;

    const read = () => {
      scene.run(readOnScene, columns.current).then(setProfile, ignore);
    };
    const timer = setInterval(read, readInterval);
    return () => clearInterval(timer);
  }, [scene, active]);

  const sized = width > 0;
  useEffect(() => {
    const view = canvas.current;
    if (!device || !view || !sized || collapsed || !active) return;

    const created = createPanelGraph(device, view, PixelRatio.get());
    graph.current = created;
    return () => {
      graph.current = undefined;
      created.destroy();
    };
  }, [device, sized, collapsed, active]);

  useEffect(() => {
    const current = graph.current;
    if (!current || !profile) return;
    columns.current = current.columns(width);
    const note = profile.timestampQuery === false ? timestampNote : undefined;
    setLabels(current.draw(profile.bars, width, height, note));
  }, [profile, width, height]);

  const onLayout = useCallback((event: LayoutChangeEvent) => {
    setWidth(event.nativeEvent.layout.width);
  }, []);

  const cycleDetail = () => {
    if (!scene || !profile) return;
    const detail = details[(details.indexOf(profile.detail) + 1) % details.length];
    scene.send(detailOnScene, detail);
    setProfile({ ...profile, detail });
  };

  const measure = () => {
    if (!scene || measuring) return;
    setMeasuring(true);
    scene
      .run(measureOnScene)
      .then(setText, (error: unknown) => setText(`measure failed: ${String(error)}`))
      .finally(() => setMeasuring(false));
  };

  const capture = () => {
    if (!scene || capturing) return;
    setCapturing(true);
    scene
      .run(captureOnScene)
      .then(setText, ignore)
      .finally(() => setCapturing(false));
  };

  const log = () => {
    scene
      ?.run(summaryOnScene)
      .then((summary) => (onLog ? onLog(summary) : setText(summary)), ignore);
  };

  const trace = () => {
    if (onTrace) scene?.run(traceOnScene).then(onTrace, ignore);
  };

  const fps = profile?.fps ?? 0;
  const stats = fps > 0 ? `${fps.toFixed(0)} fps · ${formatMs(1000 / fps)}` : '';
  const buttons: [string, () => void, boolean][] = [
    [detailLabels[profile?.detail ?? 'dispatch'], cycleDetail, false],
    ['Measure', measure, measuring],
    ['Capture', capture, capturing],
    ['Log', log, false],
    ...(onTrace ? [['Trace', trace, false] as [string, () => void, boolean]] : []),
  ];

  return h(
    View,
    { style: [styles.window, style] },
    h(
      View,
      { style: styles.titleBar },
      h(
        Pressable,
        { onPress: () => setCollapsed(!collapsed), hitSlop: 8 },
        h(Text, { style: styles.button }, collapsed ? '▸' : '▾'),
      ),
      width >= compactWidth ? h(Text, { style: styles.title }, 'GPU debug') : null,
      h(Text, { style: styles.stats, numberOfLines: 1 }, stats),
      ...buttons.map(([label, onPress, lit]) =>
        h(
          Pressable,
          { key: label, onPress, hitSlop: 6 },
          h(Text, { style: [styles.button, lit && styles.lit] }, label),
        ),
      ),
    ),
    collapsed
      ? null
      : h(
          View,
          { style: { height }, onLayout },
          h(Canvas, { ref: canvas, style: StyleSheet.absoluteFill }),
          ...labels.map((label, index) =>
            h(
              Text,
              {
                key: index,
                numberOfLines: 1,
                style: [
                  styles.label,
                  {
                    left: label.x,
                    top: label.y,
                    width: label.width,
                    color: label.color,
                    textAlign: label.align,
                  },
                ],
              },
              label.text,
            ),
          ),
        ),
    collapsed || text === undefined
      ? null
      : h(
          View,
          { style: styles.text },
          h(
            Pressable,
            { onPress: () => setText(undefined), hitSlop: 8, style: styles.close },
            h(Text, { style: styles.button }, '×'),
          ),
          h(ScrollView, null, h(Text, { selectable: true, style: styles.mono }, text)),
        ),
  );
}

interface PanelGraph {
  columns(width: number): number;
  /** Draws the newest bars at this size in points and returns the labels to place over them. */
  draw(bars: Profile['bars'], width: number, height: number, note?: string): Label[];
  destroy(): void;
}

function createPanelGraph(device: GPUDevice, canvas: CanvasRef, pixelRatio: number): PanelGraph {
  const root = tgpu.initFromDevice({ device });
  const surface = native.surface(root, canvas, { pixelRatio, alphaMode: 'premultiplied' });
  const layout = createGraphLayout(capacity);
  const drawing = createGraphDrawing(root, surface.format, capacity);
  let drawn = layout.layout([], 1, 1, 0);
  let size = { width: 1, height: 1 };
  let previous: number | undefined;

  const loop = createFrames(root, (frame) => {
    drawing.draw(frame, surface.view(frame), drawn.rects, drawn.count, size.width, size.height);
  });

  return {
    columns: layout.columns,

    draw(bars, width, height, note) {
      const now = performance.now() / 1000;
      const seconds = previous === undefined ? 1 : now - previous;
      previous = now;

      size = { width, height };
      surface.resize(width, height);
      drawn = layout.layout(bars, width, height, seconds, { note });
      withoutTracking(() => loop.run(now));
      return drawn.labels;
    },

    destroy() {
      loop.destroy();
      drawing.destroy();
      surface.destroy();
      root.destroy();
    },
  };
}

function useActive(): boolean {
  const [active, setActive] = useState(AppState.currentState === 'active');

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      setActive(state === 'active');
    });
    return () => subscription.remove();
  }, []);

  return active;
}

function useGraphDevice(given: GPUDevice | undefined): GPUDevice | undefined {
  const provided = useProvidedDevice();
  const chosen = given ?? provided;
  const [requested, setRequested] = useState<GPUDevice>();

  useEffect(() => {
    if (chosen) return;

    let live = true;
    let made: GPUDevice | undefined;
    navigator.gpu
      .requestAdapter()
      .then((adapter) => {
        if (!adapter) throw new Error('No WebGPU adapter for the profiler graph.');
        return adapter.requestDevice();
      })
      .then((next) => {
        if (!live) return next.destroy();
        made = next;
        setRequested(next);
      }, ignore);

    return () => {
      live = false;
      made?.destroy();
      setRequested(undefined);
    };
  }, [chosen]);

  return chosen ?? requested;
}

function useProvidedDevice(): GPUDevice | undefined {
  try {
    return useMainDevice().device ?? undefined;
  } catch {
    return undefined; // react-native-webgpu throws outside a provider
  }
}

const styles = StyleSheet.create({
  window: {
    borderRadius: 6,
    overflow: 'hidden',
    backgroundColor: 'rgba(22, 24, 29, 0.94)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  titleBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    height: 28,
    paddingHorizontal: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
  title: { color: theme.text, fontSize: 12, fontWeight: '600' },
  stats: {
    flex: 1,
    color: theme.muted,
    fontSize: 12,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
  },
  button: { color: '#aab', fontSize: 12, fontWeight: '600' },
  lit: { color: theme.accent },
  label: { position: 'absolute', fontSize: 11, fontWeight: '600', fontVariant: ['tabular-nums'] },
  text: {
    maxHeight: 200,
    paddingHorizontal: 10,
    paddingTop: 6,
    paddingBottom: 10,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.08)',
  },
  close: { position: 'absolute', top: 4, right: 8, zIndex: 1 },
  mono: {
    color: theme.text,
    fontSize: 11,
    lineHeight: 16,
    fontFamily: Platform.select({ ios: 'Menlo', default: 'monospace' }),
  },
});
