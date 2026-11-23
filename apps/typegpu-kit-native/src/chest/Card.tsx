import { useScene } from '@typegpu/kit/native/react';
import { useEffect, useState } from 'react';
import { PixelRatio, Pressable, StyleSheet } from 'react-native';
import { Canvas } from 'react-native-webgpu';
import type { CardPose } from 'typegpu-kit-pull/card.ts';
import { cardFontsOf, posterBitmapOf } from '../files.ts';
import { useGpu } from '../gpu.tsx';
import { useDrawn } from '../useDrawn.ts';
import type { Entry } from './cards.ts';
import type { FaceOptions } from './face.ts';
import { createFace } from './worklets.ts';

interface CardProps {
  readonly entry: Entry;
  readonly pose: { readonly value: CardPose | null };
  /** The card's size in its slot, in points. */
  readonly size: readonly [number, number];
  /** Stops the canvas once the card has landed. */
  readonly still: boolean;
  /** Called once the canvas has compiled the card and drawn a frame, or has failed to. */
  readonly onReady: () => void;
}

type Loaded = Pick<FaceOptions, 'poster' | 'fonts'>;

/**
 * The pulled card, drawn whole on a canvas over the overlay at the screen's resolution, so its type
 * stays sharp through the flight, once its poster has decoded and its fonts are read.
 */
export function CardCanvas({ entry, pose, size, still, onReady }: CardProps) {
  const [loaded, setLoaded] = useState<Loaded | null>();
  useEffect(() => {
    let live = true;
    const poster = posterBitmapOf(entry.key).catch(() => undefined);
    Promise.all([poster, cardFontsOf()]).then(
      ([bitmap, fonts]) => live && setLoaded({ poster: bitmap, fonts }),
      () => live && setLoaded(null),
    );
    return () => {
      live = false;
    };
  }, [entry.key]);
  useEffect(() => {
    if (loaded === null) onReady();
  }, [loaded, onReady]);
  if (!loaded) return null;
  return (
    <Face entry={entry} pose={pose} size={size} still={still} onReady={onReady} loaded={loaded} />
  );
}

function Face({
  entry,
  pose,
  size,
  still,
  onReady,
  loaded,
}: CardProps & { readonly loaded: Loaded }) {
  const { device } = useGpu();
  const [options] = useState((): FaceOptions => ({ ...loaded, size, entry, pose }));
  const drawn = useDrawn(1);
  const { canvas, error } = useScene(createFace, options, {
    device,
    on: 'ui',
    paused: still,
    alphaMode: 'premultiplied',
    maxPixelRatio: PixelRatio.get(),
    onFrame: drawn.onFrame,
  });
  useEffect(() => {
    if (drawn.drawn || error) onReady();
  }, [drawn.drawn, error, onReady]);
  return <Canvas {...canvas} style={StyleSheet.absoluteFill} pointerEvents="none" />;
}

interface CardTargetProps {
  readonly entry: Entry;
  /** The card's slot: its centre and size, in points of the overlay. */
  readonly slot: readonly [number, number];
  readonly size: readonly [number, number];
  readonly onPress: () => void;
}

/** Where a finger opens the card's example: its slot, which it lands in. */
export function CardTarget({ entry, slot, size, onPress }: CardTargetProps) {
  const [width, height] = size;
  return (
    <Pressable
      style={[
        styles.target,
        { left: slot[0] - width / 2, top: slot[1] - height / 2, width, height },
      ]}
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={`${entry.title}, ${entry.tier}. ${entry.line}`}
    />
  );
}

const styles = StyleSheet.create({
  target: { position: 'absolute' },
});
