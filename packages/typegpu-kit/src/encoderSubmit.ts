import type { Frame } from './frames.ts';

/**
 * Keys the hook on a frame from `frames.create` that runs a callback once its current encoder
 * submits. Private to the kit: the package entry does not export this file.
 */
export const encoderSubmit = /* @__PURE__ */ Symbol('encoderSubmit');

export interface Submitting extends Frame {
  [encoderSubmit](callback: () => void): void;
}

/**
 * Runs `callback` once the commands recorded so far into `frame.encoder` submit: at the frame's
 * next `submit()`, so a frame that throws after an early submit still settles what it sent. A
 * frame you build yourself runs it after its last submit instead. Settle recorded copies here, and
 * keep work that needs the whole frame, such as presenting, on `frame.afterSubmit`.
 *
 * @param frame the frame whose encoder holds the recorded work
 * @param callback created once, outside the frame, so recording allocates nothing
 */
export function afterEncoderSubmit(frame: Frame, callback: () => void): void {
  if (encoderSubmit in frame) (frame as Submitting)[encoderSubmit](callback);
  else frame.afterSubmit(callback);
}
