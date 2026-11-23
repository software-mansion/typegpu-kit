import { GpuDebugPanel } from '@typegpu/kit/debug/gpuDebugPanel';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Linking, Share, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useCameraPermission } from 'react-native-vision-camera';
import type { Example } from 'typegpu-kit-examples';
import { useGpu } from '../gpu.tsx';
import { CameraFeed } from './CameraFeed.tsx';
import { CornerTexts } from './CornerTexts.tsx';
import { Frame, TopBar } from './Frame.tsx';
import { elementsOf, UNDER_BAR } from './overlay.ts';
import { Settings } from './Settings.tsx';
import { TouchArea } from './TouchArea.tsx';
import { useExample, useFiles, type ExampleHandle } from './useExample.ts';

/** One screen for every example: its canvas, the veil until ready, UI, settings and a restart. */
export function ExampleScreen({ example }: { readonly example: Example }) {
  const [run, setRun] = useState(0);
  return <Session key={run} example={example} onRestart={() => setRun((count) => count + 1)} />;
}

interface SessionProps {
  readonly example: Example;
  readonly onRestart: () => void;
}

/**
 * One run of the example, from loading its files to its end. It renders one `Frame` throughout, so
 * the veil stays up from loading to the first frame.
 */
function Session({ example, onRestart }: SessionProps) {
  const loaded = useFiles(example);
  const hosted = useExample(example, loaded.files);
  const { shape, readings } = hosted;
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [debugging, setDebugging] = useState(false);
  const cameras = elementsOf(example.ui, 'camera');
  const hasSettings =
    !!shape?.settings || elementsOf(example.ui, 'image').length + cameras.length > 0;
  const [cameraError, setCameraError] = useState<Error>();
  const filming = useFilming(cameras[0]?.[0], setCameraError);
  const camera = cameras.find(([name]) => name === filming.element)?.[1];

  return (
    <Frame
      example={example}
      status={loaded.files ? hosted.status : loaded.status}
      error={loaded.error ?? hosted.error ?? cameraError}
      onRestart={onRestart}
      bar={
        <TopBar
          title={example.title}
          debugging={debugging}
          onDebug={() => setDebugging((shown) => !shown)}
          onSettings={hasSettings ? () => setSettingsOpen((shown) => !shown) : undefined}
        />
      }
    >
      {/* The scene starts once its canvas is laid out, which must wait for its files */}
      {loaded.files ? <TouchArea example={example} hosted={hosted} /> : null}
      <CornerTexts ui={example.ui} texts={readings?.texts} />
      {debugging ? <DebugPanel scene={hosted.scene} /> : null}
      {hosted.scene && filming.element && camera ? (
        <CameraFeed
          scene={hosted.scene}
          element={filming.element}
          selfie={camera.selfie ?? true}
          onError={filming.fail}
        />
      ) : null}
      {hasSettings && shape ? (
        <Settings
          title={example.title}
          open={settingsOpen}
          onOpenChange={setSettingsOpen}
          settings={shape.settings}
          ui={example.ui}
          readings={readings}
          write={hosted.write}
          act={hosted.act}
          pick={hosted.pick}
          filming={filming.element}
          film={filming.toggle}
        />
      ) : null}
    </Frame>
  );
}

/**
 * Which camera element streams. Asks for camera access as the example opens and streams to
 * `first` if the reader allows it; a reader who declines keeps the scene's stand-in.
 */
function useFilming(first: string | undefined, onError: (error: Error) => void) {
  const [element, setElement] = useState<string>();
  const { hasPermission, canRequestPermission, requestPermission } = useCameraPermission();
  const asked = useRef(false);

  useEffect(() => {
    if (!first || asked.current) return;
    asked.current = true;
    if (hasPermission) {
      setElement(first);
    } else if (canRequestPermission) {
      requestPermission().then(
        (granted) => granted && setElement(first),
        () => {},
      );
    }
  }, [first, hasPermission, canRequestPermission, requestPermission]);

  const toggle = useCallback(
    (name: string) => {
      if (element === name) {
        setElement(undefined);
        return;
      }
      if (hasPermission) {
        setElement(name);
        return;
      }
      const failed = (reason: unknown) =>
        onError(reason instanceof Error ? reason : Error(String(reason)));
      // iOS asks once; after a denial only Settings can allow the camera
      if (!canRequestPermission) {
        Linking.openSettings().catch(failed);
        return;
      }
      requestPermission().then((granted) => granted && setElement(name), failed);
    },
    [element, hasPermission, canRequestPermission, requestPermission, onError],
  );

  return { element, toggle, fail: onError };
}

/** The debug panel's height in points. */
const PANEL_HEIGHT = 120;

const share = (message: string) => void Share.share({ message });

/** The GPU debug panel under the top bar, on the app's device. Log and Trace open the share sheet. */
function DebugPanel({ scene }: { readonly scene: ExampleHandle['scene'] }) {
  const { device } = useGpu();
  const insets = useSafeAreaInsets();
  return (
    <GpuDebugPanel
      scene={scene}
      device={device}
      onLog={share}
      onTrace={share}
      style={[styles.panel, { top: insets.top + UNDER_BAR }]}
      height={PANEL_HEIGHT}
    />
  );
}

const styles = StyleSheet.create({
  panel: { position: 'absolute', left: 8, right: 8 },
});
