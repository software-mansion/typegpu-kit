import { useScene } from '@typegpu/kit/native/react';
import { useMemo, type ReactNode } from 'react';
import { PixelRatio, StyleSheet, View } from 'react-native';
import {
  GestureDetector,
  GestureStateManager,
  useManualGesture,
  type GestureTouchEvent,
} from 'react-native-gesture-handler';
import { Canvas } from 'react-native-webgpu';
import { useGpu } from '../gpu.tsx';
import { colors } from '../theme.ts';
import { createBanner, releaseBanner, touchBanner } from './worklets.ts';

interface BannerProps {
  /** Stops drawing, such as while the banner is scrolled away or another screen is open. */
  readonly paused: boolean;
  /** What sits on the banner, such as the gallery's heading. Only its pressables take touches. */
  readonly children: ReactNode;
}

/** The banner takes no options. */
const NO_OPTIONS = {};

/** Drifting dithered smoke behind the gallery's heading, lit where a finger rests on it. */
export function Banner({ paused, children }: BannerProps) {
  const { device } = useGpu();
  const { canvas, scene } = useScene(createBanner, NO_OPTIONS, {
    device,
    on: 'ui',
    paused,
    maxPixelRatio: PixelRatio.get(),
  });

  // A gesture that never activates, so the scroll view and the heading's pressables keep their
  // touches, and fails once the last finger lifts. It runs on the UI runtime, beside the banner
  const send = scene?.send;
  const gesture = useManualGesture(
    useMemo(() => {
      const touch = (event: GestureTouchEvent) => {
        'worklet';
        const finger = event.changedTouches[0];
        if (finger) send?.(touchBanner, finger.x, finger.y);
      };
      return {
        onTouchesDown: touch,
        onTouchesMove: touch,
        onTouchesUp: (event: GestureTouchEvent) => {
          'worklet';
          if (event.numberOfTouches === 0) GestureStateManager.fail(event.handlerTag);
        },
        onTouchesCancel: (event: GestureTouchEvent) => {
          'worklet';
          if (event.numberOfTouches === 0) GestureStateManager.fail(event.handlerTag);
        },
        onFinalize: () => {
          'worklet';
          send?.(releaseBanner);
        },
      };
    }, [send]),
  );

  return (
    <GestureDetector gesture={gesture}>
      <View style={styles.banner}>
        <Canvas {...canvas} style={StyleSheet.absoluteFill} pointerEvents="none" />
        <View pointerEvents="box-none">{children}</View>
      </View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  banner: { backgroundColor: colors.ground, overflow: 'hidden' },
});
