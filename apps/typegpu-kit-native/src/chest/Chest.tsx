import { useScene } from '@typegpu/kit/native/react';
import { useRef, useState } from 'react';
import { PixelRatio, Pressable, StyleSheet, type HostInstance } from 'react-native';
import { useSharedValue } from 'react-native-reanimated';
import { Canvas } from 'react-native-webgpu';
import { restOf } from 'typegpu-kit-pull/box.ts';
import type { Placed } from 'typegpu-kit-pull/timeline.ts';
import { useGpu } from '../gpu.tsx';
import { createResting } from './worklets.ts';

/** The chest's spot, in points: the site's pedestal box, a little larger for a finger. */
const SPOT = { width: 124, height: 93 } as const;

interface ChestProps {
  /** Stops drawing, such as while the banner is scrolled away or another screen is open. */
  readonly paused: boolean;
  /** Hides it while the pull it opened holds it. */
  readonly away: boolean;
  /** Opens a pull from where the chest stands, in points of the window. */
  readonly onOpen: (from: Placed) => void;
}

/** The docs site's box resting on the banner: a finger lights and presses it, a tap opens it. */
export function Chest({ paused, away, onOpen }: ChestProps) {
  const { device } = useGpu();
  const pressed = useSharedValue(false);
  const over = useSharedValue(0);
  const [options] = useState(() => ({ pressed, over }));
  const { canvas } = useScene(createResting, options, {
    device,
    on: 'ui',
    paused: paused || away,
    alphaMode: 'premultiplied',
    maxPixelRatio: PixelRatio.get(),
  });
  const spot = useRef<HostInstance>(null);

  const press = (down: boolean) => {
    pressed.value = down;
  };
  const open = () =>
    spot.current?.measureInWindow((x, y, width, height) => {
      const { anchor, unit } = restOf(width, height);
      onOpen({ anchor: [x + anchor[0], y + anchor[1]], unit, yaw: 0.1 * over.value, roll: 0 });
    });

  return (
    <Pressable
      ref={spot}
      onPressIn={() => press(true)}
      onPressOut={() => press(false)}
      onPress={open}
      disabled={away}
      accessibilityRole="button"
      accessibilityLabel="Open the box for a random example"
      style={styles.spot}
    >
      <Canvas
        {...canvas}
        style={[StyleSheet.absoluteFill, away && styles.away]}
        pointerEvents="none"
      />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  spot: SPOT,
  away: { opacity: 0 },
});
