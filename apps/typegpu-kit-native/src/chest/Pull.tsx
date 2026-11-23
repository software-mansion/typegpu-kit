import { useScene } from '@typegpu/kit/native/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { BackHandler, PixelRatio, Pressable, StyleSheet, Text, View } from 'react-native';
import { GestureDetector, usePanGesture } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useDerivedValue,
  useFrameCallback,
  useSharedValue,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Canvas } from 'react-native-webgpu';
import { scheduleOnRN } from 'react-native-worklets';
import { boxStage, pulledBox } from 'typegpu-kit-pull/box.ts';
import { cardAt, cardSize, type CardPose, type CardStage } from 'typegpu-kit-pull/card.ts';
import { COLORS } from 'typegpu-kit-pull/dither.ts';
import { FLIGHT, handoffAt, LAUNCH, type Placed } from 'typegpu-kit-pull/timeline.ts';
import { useGpu } from '../gpu.tsx';
import { colors, fonts } from '../theme.ts';
import { CardCanvas, CardTarget } from './Card.tsx';
import type { Entry } from './cards.ts';
import { createPulled } from './worklets.ts';

/** Points under the card for its row, as the site keeps. */
const ROW = 92;
/** The widest the card gets, in points. */
const WIDEST = 360;
/** Milliseconds the screen takes to dim. */
const DIM = 320;
/** Milliseconds from the pull's start until the card is still in its slot. */
const SETTLED = LAUNCH.box + FLIGHT;
/** Milliseconds a pull waits for its card's poster before it plays without it. */
const PATIENCE = 1000;
/** The least a swipe to the right travels to close the pull, in points. */
const SWIPE = 60;

/** A pull as the gallery opens it. */
export interface Opened {
  readonly entry: Entry;
  /** Where the box stood on the banner, in points of the window. */
  readonly from: Placed;
  readonly direction: number;
  readonly phase: number;
}

/** A pull as the UI runtime plays it, with where the box opens and the card's stage. */
interface Playing extends Opened {
  readonly ground: readonly [number, number];
  readonly stage: CardStage;
}

interface PullProps {
  /** The open pull, or null. */
  readonly pull: Opened | null;
  /** Called once the box is drawn over its spot, so the spot can hide its own. */
  readonly onShown: () => void;
  /** Called when the card is tapped. */
  readonly onPick: (key: string) => void;
  readonly onClose: () => void;
}

/**
 * The site's pull over the gallery: the box rises out of its spot on the banner, knocks and opens,
 * and a card leaves it face down, is tossed up and turns over into its slot. Once it has landed,
 * tapping the card opens its example and tapping beside it closes the pull, as a swipe to the right
 * or the back button do at any time. The pull keeps the layout it opened with.
 */
export function Pull({ pull, onShown, onPick, onClose }: PullProps) {
  const { device } = useGpu();
  const insets = useSafeAreaInsets();
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [settled, setSettled] = useState(false);
  // The open pull's card stage, which its drawing, its target and its row keep until it ends
  const [opened, setOpened] = useState<{ readonly pull: Opened; readonly stage: CardStage }>();
  const [ready, setReady] = useState(false);
  const [idle, setIdle] = useState(true);
  const playing = useSharedValue<Playing | null>(null);
  const clock = useSharedValue(-1);
  const started = useSharedValue(-1);
  const dim = useSharedValue(0);

  // The box and the card at this moment: the box until it has sunk, the card once the clock starts
  const moment = useDerivedValue(() => {
    const now = playing.value;
    if (!now) return null;
    const t = clock.value;
    const at = handoffAt(Math.max(t, 0), now.from, now.ground, now.stage.size);
    const { tint, finish } = now.entry;
    const exit = pulledBox(t, now.phase, at.anchor, at.unit, at.yaw, COLORS.accent[0], tint);
    return {
      box: t <= LAUNCH.box + 800 ? exit.input : null,
      card: t < 0 ? null : cardAt('box', t, exit, now.stage, now.direction, finish.flare),
    };
  });
  const pose = useDerivedValue(() => moment.value?.card ?? null);
  const shownBox = useDerivedValue(() => moment.value?.box ?? null);
  const [boxOptions] = useState(() => ({ box: shownBox }));
  const box = useScene(createPulled, boxOptions, {
    device,
    on: 'ui',
    paused: idle || settled,
    alphaMode: 'premultiplied',
    maxPixelRatio: PixelRatio.get(),
  });

  // The card's size and slot, and the box's ground, in points of the screen inside its safe area
  const height = size.height - insets.top - insets.bottom;
  const [cardW, cardH] = cardSize(size.width, height, ROW, WIDEST);
  const placed = boxStage(size.width, height, cardH, ROW);
  const slot = [placed.slot[0], placed.slot[1] + insets.top] as const;
  const ground = [placed.ground[0], placed.ground[1] + insets.top] as const;

  // The frame callback stays one function, which Reanimated would otherwise register again
  const latest = useRef({ onShown, onClose });
  latest.current = { onShown, onClose };
  const shown = useCallback(() => latest.current.onShown(), []);
  const settle = useCallback(() => setSettled(true), []);
  const onReady = useCallback(() => setReady(true), []);
  const tick = useCallback(
    ({ timestamp }: { timestamp: number }) => {
      'worklet';
      if (started.value < 0) {
        started.value = timestamp;
        scheduleOnRN(shown);
      }
      const t = timestamp - started.value;
      const before = clock.value;
      clock.value = Math.min(t, SETTLED);
      if (before < SETTLED && t >= SETTLED) scheduleOnRN(settle);
    },
    [started, clock, shown, settle],
  );
  const frames = useFrameCallback(tick, false);

  // A pull starts its clock once its card holds its poster, and its end stops it
  const open = pull !== null;
  const stage = opened?.pull === pull ? opened.stage : undefined;
  useEffect(() => {
    if (!pull) return undefined;
    const opening = { width: cardW, height: cardH, slot, size: placed.size };
    playing.value = { ...pull, ground, stage: opening };
    setOpened({ pull, stage: opening });
    started.value = -1;
    clock.value = -1;
    dim.value = withTiming(1, { duration: DIM });
    setSettled(false);
    return () => {
      frames.setActive(false);
      setReady(false);
      playing.value = null;
      clock.value = -1;
      dim.value = 0;
    };
    // The pull's layout is read when it opens
  }, [pull]);

  useEffect(() => {
    if (!open) return undefined;
    if (ready) {
      frames.setActive(true);
      return undefined;
    }
    const waiting = setTimeout(onReady, PATIENCE);
    return () => clearTimeout(waiting);
  }, [open, ready, frames, onReady]);

  // The box's canvas runs while a pull is open, and a few frames after, which clear it. It rests
  // once the card has landed, long after the box has sunk
  useEffect(() => {
    if (open) {
      setIdle(false);
      return undefined;
    }
    const pausing = setTimeout(() => setIdle(true), 100);
    return () => clearTimeout(pausing);
  }, [open]);

  useEffect(() => {
    if (settled) frames.setActive(false);
  }, [settled, frames]);

  // The back button and a swipe to the right close it
  useEffect(() => {
    if (!open) return undefined;
    const subscription = BackHandler.addEventListener('hardwareBackPress', () => {
      onClose();
      return true;
    });
    return () => subscription.remove();
  }, [open, onClose]);
  const swipe = usePanGesture({
    enabled: open,
    activeOffsetX: 16,
    failOffsetY: [-8, 8],
    runOnJS: true,
    onDeactivate: ({ canceled, translationX }) => {
      if (!canceled && translationX > SWIPE) latest.current.onClose();
    },
  });

  const backdrop = useAnimatedStyle(() => ({ opacity: dim.value }));

  return (
    <GestureDetector gesture={swipe}>
      <View
        style={StyleSheet.absoluteFill}
        pointerEvents={open ? 'auto' : 'none'}
        onLayout={(event) => setSize(event.nativeEvent.layout)}
      >
        <Animated.View style={[StyleSheet.absoluteFill, styles.backdrop, backdrop]}>
          <Pressable
            style={StyleSheet.absoluteFill}
            onPress={() => settled && onClose()}
            accessibilityLabel="Close"
          />
        </Animated.View>
        <Canvas
          {...box.canvas}
          style={[StyleSheet.absoluteFill, !open && styles.gone]}
          pointerEvents="none"
        />
        {pull && stage ? (
          <>
            <CardCanvas
              key={`${pull.entry.key}-${pull.phase}`}
              entry={pull.entry}
              pose={pose}
              size={[stage.width, stage.height]}
              still={settled}
              onReady={onReady}
            />
            {/* The card opens its example once it has landed face up */}
            {settled ? (
              <CardTarget
                entry={pull.entry}
                slot={stage.slot}
                size={[stage.width, stage.height]}
                onPress={() => onPick(pull.entry.key)}
              />
            ) : null}
            <Row pose={pose} top={stage.slot[1] + stage.height / 2 + 18} />
          </>
        ) : null}
      </View>
    </GestureDetector>
  );
}

/** The quiet line under the landed card. */
function Row({ pose, top }: { readonly pose: SharedValue<CardPose | null>; readonly top: number }) {
  const shown = useAnimatedStyle(() => ({ opacity: pose.value?.row ?? 0 }));
  return (
    <Animated.View style={[styles.row, { top }, shown]} pointerEvents="none">
      <Text style={styles.hint}>TAP THE CARD TO OPEN IT</Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  backdrop: { backgroundColor: 'rgba(10, 13, 18, 0.88)' },
  gone: { opacity: 0 },
  row: { position: 'absolute', left: 0, right: 0, alignItems: 'center' },
  hint: { ...fonts.mono, color: colors.muted, fontSize: 11, letterSpacing: 1.6 },
});
