import { useEffect, useRef, useState, type RefObject } from 'react';
import { StyleSheet, Text, View, type HostInstance } from 'react-native';
import Animated, { useAnimatedStyle, type SharedValue } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { padKey, type PadRect } from '../touches.ts';
import type { AnyButtonsKind, Button } from 'typegpu-kit-examples/ui.ts';
import { colors } from '../theme.ts';
import { EDGE } from './overlay.ts';

/** A pad's height and its least width, in points. */
const PAD = 56;
/** Points around a pad that still press it. */
const PAD_SLOP = 4;

/** Where pads are placed and which are lit. */
interface Placement {
  /** The touch area pads measure themselves in. */
  readonly area: RefObject<HostInstance | null>;
  /** The pads held down, by `padKey`. */
  readonly lit: SharedValue<Readonly<Record<string, boolean>>>;
  readonly onPlace: (rect: PadRect) => void;
}

interface PadsProps extends Placement {
  /** The buttons element's name in the example's UI. */
  readonly element: string;
  readonly buttons: AnyButtonsKind;
}

/**
 * A buttons element as pads at the bottom corners, laid out by its rows. The touch lane presses
 * them; each reports where it sits through `onPlace`.
 */
export function Pads({ element, buttons, ...placement }: PadsProps) {
  const insets = useSafeAreaInsets();
  // Pads measure again whenever the block moves, which leaves their own layout unchanged
  const [moves, setMoves] = useState(0);
  const sides = [
    ['left', buttons.pads.left],
    ['right', buttons.pads.right],
  ] as const;

  return (
    <View
      style={[styles.pads, { bottom: insets.bottom + EDGE }]}
      pointerEvents="none"
      onLayout={() => setMoves((count) => count + 1)}
    >
      {sides.map(([side, rows]) => (
        <View key={side} style={styles.side}>
          {(rows ?? []).map((row, i) => (
            <View key={i} style={styles.row}>
              {row.map((name, j) => {
                const button = name === null ? undefined : buttons.buttons[name];
                if (name === null || !button) return <View key={j} style={styles.gap} />;
                return (
                  <Pad
                    key={name}
                    element={element}
                    name={name}
                    button={button}
                    moves={moves}
                    {...placement}
                  />
                );
              })}
            </View>
          ))}
        </View>
      ))}
    </View>
  );
}

interface PadProps extends Placement {
  readonly element: string;
  readonly name: string;
  readonly button: Button;
  /** Changes whenever the pads' block moves, so the pad measures again. */
  readonly moves: number;
}

/** One pad, lit while a finger holds it, reporting where it sits in the touch area. */
function Pad({ element, name, button, area, lit, moves, onPlace }: PadProps) {
  const pad = useRef<HostInstance>(null);
  const key = padKey(element, name);
  const look = useAnimatedStyle(() => (lit.value[key] ? padDown : padUp));
  const held = button.held === true;
  const latch = held && button.latch === true;

  const measure = () => {
    const into = area.current;
    if (!into) return;
    pad.current?.measureLayout(into, (x, y, width, height) =>
      onPlace({
        element,
        name,
        held,
        latch,
        x: x - PAD_SLOP,
        y: y - PAD_SLOP,
        width: width + 2 * PAD_SLOP,
        height: height + 2 * PAD_SLOP,
      }),
    );
  };

  useEffect(measure, [moves]);

  return (
    <Animated.View
      ref={pad}
      onLayout={measure}
      style={[styles.pad, look]}
      accessible
      accessibilityRole="button"
      accessibilityLabel={button.name ?? button.label}
    >
      <Text style={styles.text}>{button.label}</Text>
    </Animated.View>
  );
}

const padUp = {
  backgroundColor: 'rgba(20, 24, 31, 0.6)',
  borderColor: 'rgba(233, 236, 242, 0.18)',
};
const padDown = { backgroundColor: colors.accentLow, borderColor: colors.accent };

const styles = StyleSheet.create({
  pads: {
    position: 'absolute',
    left: EDGE,
    right: EDGE,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
  },
  side: { gap: 8 },
  row: { flexDirection: 'row', gap: 8, justifyContent: 'center' },
  gap: { width: PAD, height: PAD },
  pad: {
    minWidth: PAD,
    height: PAD,
    paddingHorizontal: 12,
    borderRadius: PAD / 2,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
  text: { color: colors.text, fontSize: 17, fontWeight: '600' },
});
