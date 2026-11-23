import * as runtimes from '@typegpu/kit/native/runtimes';
import { useIsFocused } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import { useCamera, useFrameOutput, type Frame } from 'react-native-vision-camera';
import { useGpu } from '../gpu.tsx';
import { useActive } from '../useActive.ts';
import * as worklets from '../worklets.ts';
import type { ExampleHandle } from './useExample.ts';

/** Camera frames at this size, which covers the examples' feeds. */
const RESOLUTION = { width: 1280, height: 720 };

interface CameraFeedProps {
  readonly scene: NonNullable<ExampleHandle['scene']>;
  /** The camera element the frames go to. */
  readonly element: string;
  /** Uses the front camera, mirrored. */
  readonly selfie: boolean;
  readonly onError: (error: Error) => void;
}

/**
 * Streams the camera into an example's camera element while mounted, its screen focused and the app
 * active.
 */
export function CameraFeed({ scene, element, selfie, onError }: CameraFeedProps) {
  const { worker } = useGpu();
  const focused = useIsFocused();
  const active = useActive();
  const [source, setSource] = useState<runtimes.CameraSource>();

  // A source reads frames on one device, so each scene the element binds gets its own
  useEffect(() => {
    const made = runtimes.camera(worker);
    scene.send(worklets.watch, element, made);
    setSource(made);
    return () => {
      scene.send(worklets.watch, element, undefined);
      made.destroy();
    };
  }, [worker, scene, element]);

  const output = useFrameOutput({
    targetResolution: RESOLUTION,
    onFrame: (frame: Frame) => {
      'worklet';
      try {
        source?.write(frame);
      } finally {
        frame.dispose();
      }
    },
  });
  const outputs = useMemo(() => [output], [output]);

  useCamera({
    isActive: focused && active,
    device: selfie ? 'front' : 'back',
    mirrorMode: selfie ? 'on' : 'off',
    outputs,
    orientationSource: 'interface',
    onError,
  });

  return null;
}
