import * as runtimes from '@typegpu/kit/native/runtimes';
import { useEffect, useMemo, useState } from 'react';
import { useCamera, useFrameOutput, type Frame } from 'react-native-vision-camera';
import { useGpu } from '../gpu.tsx';
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

/** Streams the camera into an example's camera element while mounted. */
export function CameraFeed({ scene, element, selfie, onError }: CameraFeedProps) {
  const { worker } = useGpu();
  const [source] = useState(() => runtimes.camera(worker));

  useEffect(() => {
    scene.send(worklets.watch, element, source);
    return () => {
      scene.send(worklets.watch, element, undefined);
      source.destroy();
    };
  }, [scene, element, source]);

  const output = useFrameOutput({
    targetResolution: RESOLUTION,
    onFrame: (frame: Frame) => {
      'worklet';
      try {
        source.write(frame);
      } finally {
        frame.dispose();
      }
    },
  });
  const outputs = useMemo(() => [output], [output]);

  useCamera({
    isActive: true,
    device: selfie ? 'front' : 'back',
    mirrorMode: selfie ? 'on' : 'off',
    outputs,
    orientationSource: 'interface',
    onError,
  });

  return null;
}
