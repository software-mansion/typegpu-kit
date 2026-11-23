// The camera an example's ui.ts declares, for its page and its figures

import * as web from '@typegpu/kit/web';
import type { Camera, CameraKind } from 'typegpu-kit-examples/ui.ts';

/** What the camera button says when `getUserMedia` refuses, by the error's name. */
const REFUSALS: Readonly<Record<string, string>> = {
  NotAllowedError: 'camera blocked: allow it in the address bar',
  NotFoundError: 'no camera found',
  NotReadableError: 'camera busy in another app',
};

/** Off, also once refused; asking for the camera; or on, filming. */
type Status =
  | { readonly kind: 'off' }
  | { readonly kind: 'asking' }
  | { readonly kind: 'on'; readonly stream: MediaStream; readonly source: web.Video };

export interface Filming {
  /** Starts the camera, or stops it while it runs; waits while it is asked for. */
  toggle(): void;
}

/**
 * Asks for the camera at once and attaches it to `camera`. `toggle` turns it off and on. `label`
 * names what the button does next, or why the camera didn't start, while the scene keeps its
 * stand-in. A selfie camera shows mirrored. Stops the camera when `signal` aborts.
 */
export function film(
  part: CameraKind,
  camera: Camera,
  label: (text: string) => void,
  onError: (error: unknown) => void,
  signal: AbortSignal,
): Filming {
  const selfie = part.selfie ?? true;
  let status: Status = { kind: 'off' };

  const stop = () => {
    if (status.kind !== 'on') return;
    camera.attach(undefined);
    status.source.destroy();
    for (const track of status.stream.getTracks()) track.stop();
    status = { kind: 'off' };
    label(part.label);
  };

  const start = async () => {
    status = { kind: 'asking' };
    let stream: MediaStream | undefined;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: selfie ? 'user' : 'environment' },
      });
      const player = document.createElement('video');
      player.muted = true;
      player.playsInline = true;
      player.srcObject = stream;
      await player.play();
      if (signal.aborted) return;
      const source = web.video(player, { mirrored: selfie });
      status = { kind: 'on', stream, source };
      camera.attach(source);
      label('stop camera');
    } finally {
      // Refused, failed, or let go while it was asked for
      if (status.kind === 'asking') {
        status = { kind: 'off' };
        for (const track of stream?.getTracks() ?? []) track.stop();
      }
    }
  };

  const refused = (reason: unknown) => {
    const why = reason instanceof DOMException ? REFUSALS[reason.name] : undefined;
    if (why) label(why);
    else onError(reason);
  };

  signal.addEventListener('abort', stop);
  const toggle = () => {
    if (signal.aborted || !navigator.mediaDevices) return;
    if (status.kind === 'on') stop();
    else if (status.kind === 'off') void start().catch(refused);
  };
  if (!navigator.mediaDevices) {
    label('camera needs HTTPS');
    return { toggle };
  }
  // Ask at once; a reader who declines, or has no camera, keeps the scene's stand-in
  toggle();
  return { toggle };
}
