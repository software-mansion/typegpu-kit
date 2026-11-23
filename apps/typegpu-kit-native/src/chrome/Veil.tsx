import { useScene } from '@typegpu/kit/native/react';
import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { Image, PixelRatio, StyleSheet, View } from 'react-native';
import { Canvas } from 'react-native-webgpu';
import { posterBitmapOf, posterOf } from '../files.ts';
import { useGpu } from '../gpu.tsx';
import { useDrawn } from '../useDrawn.ts';
import { colors } from '../theme.ts';
import { LIFT } from './veil.ts';
import { createVeil, liftVeil } from './worklets.ts';

/** Milliseconds a ready example waits for its veil to come up before showing without one. */
const PATIENCE_MS = 400;
/** Milliseconds the veil stays after its front has crossed, so the last cells clear. */
const LINGER_MS = 100;

interface VeilProps {
  readonly example: { readonly key: string };
  /** Whether the example below can be seen; the veil lifts off it then. */
  readonly ready: boolean;
  /** What shows over the veil while it covers the example, such as its status. */
  readonly children?: ReactNode;
}

/**
 * Covers a loading example with its poster, printed in dithered cells on the UI thread so it
 * moves while the example compiles, and lifts off once it is ready.
 */
export function Veil({ example, ready, children }: VeilProps) {
  // Undefined while the poster decodes, null when there is none
  const [poster, setPoster] = useState<ImageBitmap | null>();
  const [printing, setPrinting] = useState(false);
  const [gone, setGone] = useState(false);
  const source = posterOf(example.key);

  useEffect(() => {
    let live = true;
    posterBitmapOf(example.key).then(
      (bitmap) => live && setPoster(bitmap ?? null),
      () => live && setPoster(null),
    );
    return () => {
      live = false;
    };
  }, [example.key]);

  // A ready example doesn't wait long for a veil that hasn't come up
  useEffect(() => {
    if (!ready || printing) return undefined;
    const timer = setTimeout(() => setGone(true), PATIENCE_MS);
    return () => clearTimeout(timer);
  }, [ready, printing]);

  const onPrinting = useCallback(() => setPrinting(true), []);
  const onGone = useCallback(() => setGone(true), []);

  if (gone) return null;
  const lifting = ready && printing;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
      {!lifting && source !== undefined ? (
        <View style={styles.plain} pointerEvents="none">
          <Image source={source} style={styles.poster} resizeMode="cover" />
        </View>
      ) : null}
      {poster !== undefined ? (
        <Print
          poster={poster ?? undefined}
          lifting={lifting}
          onPrinting={onPrinting}
          onGone={onGone}
        />
      ) : null}
      {lifting ? null : children}
    </View>
  );
}

interface PrintProps {
  readonly poster: ImageBitmap | undefined;
  /** Starts the front that lifts the print off the example. */
  readonly lifting: boolean;
  /** Called once the print's scene has drawn. */
  readonly onPrinting: () => void;
  /** Called once the print has lifted off or failed. */
  readonly onGone: () => void;
}

/** The poster printed in dithered cells, a scene on the UI runtime. */
function Print({ poster, lifting, onPrinting, onGone }: PrintProps) {
  const { device } = useGpu();
  const printed = useDrawn(1);
  const { canvas, scene, error } = useScene(
    createVeil,
    { poster },
    {
      device,
      on: 'ui',
      alphaMode: 'premultiplied',
      maxPixelRatio: PixelRatio.get(),
      onFrame: printed.onFrame,
    },
  );

  useEffect(() => {
    if (printed.drawn) onPrinting();
  }, [printed.drawn, onPrinting]);

  useEffect(() => {
    if (error) onGone();
  }, [error, onGone]);

  useEffect(() => {
    if (!scene || !lifting) return undefined;
    scene.send(liftVeil);
    const timer = setTimeout(onGone, LIFT * 1000 + LINGER_MS);
    return () => clearTimeout(timer);
  }, [scene, lifting, onGone]);

  return <Canvas {...canvas} style={StyleSheet.absoluteFill} pointerEvents="none" />;
}

const styles = StyleSheet.create({
  plain: { ...StyleSheet.absoluteFill, backgroundColor: colors.ground },
  poster: { ...StyleSheet.absoluteFill, width: undefined, height: undefined, opacity: 0.45 },
});
