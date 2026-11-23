import * as runtimes from '@typegpu/kit/native/runtimes';
import { useState } from 'react';
import { createSynchronizable } from 'react-native-worklets';

interface Drawn {
  /** An `onFrame` for `useScene` that counts the frames the scene draws once it is ready. */
  readonly onFrame: (stage: { readonly ready: boolean }) => void;
  /** Whether the scene has drawn `count` frames since it was ready. */
  readonly drawn: boolean;
}

/**
 * Tells when a scene has drawn `count` frames since it was ready, as the docs page waits before it
 * lifts an example's cover. Counts once per component.
 */
export function useDrawn(count: number): Drawn {
  const [drawn, setDrawn] = useState(false);
  const [onFrame] = useState(() => {
    const frames = createSynchronizable(0, { fixedType: true });
    const done = runtimes.post(() => setDrawn(true));
    return (stage: { readonly ready: boolean }) => {
      'worklet';
      const seen = frames.getDirty();
      if (!stage.ready || seen >= count) return;
      frames.setDirty(seen + 1);
      if (seen + 1 === count) done();
    };
  });
  return { onFrame, drawn };
}
