// Plays an example's sounds through Web Audio, once the reader's first key press or touch lets
// the page make a sound

import type { Read } from 'typegpu-kit-examples/contract';
import type { SoundsKind, Speaker } from 'typegpu-kit-examples/ui.ts';

/**
 * Decodes the sounds `part` names, the example `key`'s, and attaches a player for them to
 * `speaker` until `signal` aborts. Sounds the scene asks for before the page may play any are
 * dropped.
 */
export async function speak(
  key: string,
  part: SoundsKind,
  speaker: Speaker,
  read: Read,
  signal: AbortSignal,
) {
  const context = new AudioContext();
  const unlock = () => void context.resume();
  for (const type of ['keydown', 'pointerdown', 'touchend'] as const) {
    window.addEventListener(type, unlock, { signal, capture: true });
  }
  signal.addEventListener('abort', () => {
    speaker.attach(undefined);
    void context.close();
  });

  const entries = await Promise.all(
    Object.entries(part.files).map(async ([name, path]) => {
      const bytes = await read(`${key}/${path}`);
      return [name, await context.decodeAudioData(bytes)] as const;
    }),
  );
  if (signal.aborted) return;
  const buffers = new Map(entries);

  speaker.attach((name, { pan = 0, gain = 1 } = {}) => {
    const buffer = buffers.get(name);
    // A silent sound still plays, as one heard from afar may be placed nearer as it goes
    if (!buffer || context.state !== 'running') return;
    const source = new AudioBufferSourceNode(context, { buffer });
    const level = new GainNode(context, { gain });
    const panner = new StereoPannerNode(context, { pan });
    source.connect(level).connect(panner).connect(context.destination);
    source.start();
    let ended = false;
    source.addEventListener('ended', () => (ended = true), { once: true });
    return {
      get playing() {
        return !ended;
      },
      place(placement) {
        if (ended) return;
        // Glide over a frame or so, so the move doesn't click
        const now = context.currentTime;
        if (placement.gain !== undefined) level.gain.setTargetAtTime(placement.gain, now, 0.01);
        if (placement.pan !== undefined) panner.pan.setTargetAtTime(placement.pan, now, 0.01);
      },
      stop(fade) {
        if (ended) return;
        const now = context.currentTime;
        level.gain.setValueAtTime(level.gain.value, now);
        level.gain.linearRampToValueAtTime(0, now + fade);
        source.stop(now + fade);
        ended = true;
      },
    };
  });
}
