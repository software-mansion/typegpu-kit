import { useScene } from '@typegpu/kit/native/react';
import type { ReactNode } from 'react';
import { PixelRatio, StyleSheet, View, type GestureResponderEvent } from 'react-native';
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

  // Touch events reach the banner without taking the scroll view's gesture
  const touch = (event: GestureResponderEvent) => {
    const { locationX, locationY } = event.nativeEvent;
    scene?.send(touchBanner, locationX, locationY);
  };
  const release = () => scene?.send(releaseBanner);

  return (
    <View
      style={styles.banner}
      onTouchStart={touch}
      onTouchMove={touch}
      onTouchEnd={release}
      onTouchCancel={release}
    >
      <Canvas {...canvas} style={StyleSheet.absoluteFill} pointerEvents="none" />
      <View pointerEvents="box-none">{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: { backgroundColor: colors.ground, overflow: 'hidden' },
});
