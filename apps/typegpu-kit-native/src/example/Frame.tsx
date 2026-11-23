import { Host, Image as SymbolImage } from '@expo/ui/swift-ui';
import { router } from 'expo-router';
import type { ComponentProps, ReactNode } from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import type { Example } from 'typegpu-kit-examples';
import { Veil } from '../chrome/Veil.tsx';
import { posterOf } from '../files.ts';
import { colors, fonts } from '../theme.ts';

interface FrameProps {
  readonly example: Example;
  /** What the example is doing while it loads, then undefined once it is ready. */
  readonly status: string | undefined;
  readonly error: Error | undefined;
  readonly onRestart: () => void;
  /** The bar over the top of the screen. */
  readonly bar: ReactNode;
  /** The canvas and what sits on it. */
  readonly children: ReactNode;
}

/** The canvas's surroundings: the veil until the example is ready, its error, and the top bar. */
export function Frame({ example, status, error, onRestart, bar, children }: FrameProps) {
  const insets = useSafeAreaInsets();
  return (
    <View style={styles.screen}>
      {children}
      {error ? (
        <Failure example={example} error={error} onRestart={onRestart} />
      ) : (
        <Veil example={example} ready={status === undefined}>
          <View style={[styles.cover, { paddingBottom: insets.bottom + 40 }]} pointerEvents="none">
            <Text style={styles.status}>{status}</Text>
          </View>
        </Veil>
      )}
      {bar}
    </View>
  );
}

interface FailureProps {
  readonly example: Example;
  readonly error: Error;
  readonly onRestart: () => void;
}

/** The error over the example's blurred poster, with a button that restarts it. */
function Failure({ example, error, onRestart }: FailureProps) {
  const poster = posterOf(example.key);
  return (
    <View style={styles.cover}>
      {poster !== undefined ? (
        <Image source={poster} style={styles.poster} blurRadius={6} resizeMode="cover" />
      ) : null}
      <View style={styles.dim} />
      <View style={styles.failure}>
        <Text style={styles.message}>{error.message}</Text>
        <Pressable onPress={onRestart} style={styles.restart}>
          <Text style={styles.restartText}>Restart</Text>
        </Pressable>
      </View>
    </View>
  );
}

interface TopBarProps {
  readonly title: string;
  readonly debugging: boolean;
  readonly onDebug: () => void;
  /** Toggles the settings sheet; without it the bar has no settings button. */
  readonly onSettings: (() => void) | undefined;
}

/** Back, the example's title, and the debug panel and settings toggles. */
export function TopBar({ title, debugging, onDebug, onSettings }: TopBarProps) {
  const insets = useSafeAreaInsets();
  return (
    <View style={[styles.bar, { top: insets.top + 4 }]} pointerEvents="box-none">
      <Chip symbol="chevron.left" onPress={() => router.back()} label="Back" />
      <Text style={styles.title} numberOfLines={1}>
        {title}
      </Text>
      <Chip
        symbol="chart.bar.xaxis"
        on={debugging}
        onPress={onDebug}
        label={debugging ? 'Hide the GPU debug panel' : 'Show the GPU debug panel'}
      />
      {onSettings ? <Chip symbol="ellipsis" onPress={onSettings} label="Settings" /> : null}
    </View>
  );
}

interface ChipProps {
  /** An SF Symbol's name. */
  readonly symbol: SymbolName;
  readonly onPress: () => void;
  /** Outlines the chip in the accent. Default false. */
  readonly on?: boolean;
  /** What a screen reader says. */
  readonly label: string;
}

type SymbolName = NonNullable<ComponentProps<typeof SymbolImage>['systemName']>;

/** A round button in the top bar, showing an SF Symbol centred by SwiftUI. */
function Chip({ symbol, onPress, on = false, label }: ChipProps) {
  return (
    <Pressable
      onPress={onPress}
      hitSlop={12}
      style={[styles.chip, on && styles.chipOn]}
      accessibilityRole="button"
      accessibilityLabel={label}
    >
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        <Host style={StyleSheet.absoluteFill} alignment="center">
          <SymbolImage systemName={symbol} size={17} color={colors.text} />
        </Host>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#000' },
  cover: { ...StyleSheet.absoluteFill, alignItems: 'center', justifyContent: 'center' },
  poster: { ...StyleSheet.absoluteFill, width: undefined, height: undefined },
  dim: { ...StyleSheet.absoluteFill, backgroundColor: 'rgba(10, 13, 18, 0.55)' },
  status: { ...fonts.mono, color: colors.text2, fontSize: 13, letterSpacing: 0.5 },
  failure: { alignItems: 'center', gap: 16, padding: 32 },
  message: { color: colors.exception, fontSize: 15, textAlign: 'center' },
  restart: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.ruleStrong,
    backgroundColor: colors.ground3,
  },
  restartText: { color: colors.link, fontSize: 15, fontWeight: '600' },
  bar: {
    position: 'absolute',
    left: 12,
    right: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  title: { ...fonts.display, flex: 1, color: colors.text, fontSize: 17 },
  chip: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(20, 24, 31, 0.72)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.ruleStrong,
  },
  chipOn: { borderColor: colors.accent },
});
