import type { TgpuCommandEncoder, TgpuRoot } from 'typegpu';

export { animate, type AnimateOptions, type Animation, type Schedule } from './animate.ts';

/**
 * Represents one frame being recorded. Each `run` creates a separate object, so modules can
 * compare frames by identity.
 */
export interface Frame {
  /**
   * Records commands for submission after the recorder returns. `submit()` replaces this encoder.
   * Read it at each call instead of keeping it, and let the frame submit it.
   */
  readonly encoder: TgpuCommandEncoder;
  /** Reports seconds since the first submitted frame. */
  readonly time: number;
  /** Reports seconds since the previous submitted frame. The first frame has a `dt` of zero. */
  readonly dt: number;
  /** Counts frames submitted before this one. */
  readonly index: number;
  /**
   * Calls `callback` after the frame's last submit, in registration order. Drops it if the frame
   * does not finish. Throws unless this frame is recording, so a retained frame cannot add
   * callbacks to a later one. Create the callback outside the frame to avoid allocating.
   */
  afterSubmit(callback: () => void): void;
  /**
   * Submits the recorded commands and replaces `encoder` for further recording. Each call adds one
   * `queue.submit`. End every pass first, because an open pass invalidates the command buffer.
   * Only the program's recorder should call this. Throws unless this frame is recording. Does
   * nothing after the loop is destroyed.
   */
  submit(): void;
}

export interface Frames extends Disposable {
  /**
   * Records and submits one frame with a fresh `Frame` and encoder. Requires finite, nondecreasing
   * `seconds`. Throws for an invalid time, a nested frame or an asynchronous recorder. Returns
   * false if the loop is destroyed before or during recording. Runs all after-submit callbacks. If
   * any throw, the frame still counts and `run` rethrows their errors.
   */
  run(seconds: number): boolean;
  /**
   * Reports the `seconds` of the last recorded frame, or undefined before the first. Another
   * animation on this loop continues from this time.
   */
  readonly seconds: number | undefined;
  /** Stops future submissions and keeps borrowed resources alive. Also `[Symbol.dispose]`. */
  destroy(): void;
}

/**
 * Reads the encoder of a frame at the call, or takes an encoder as it is, for a recording method
 * that accepts either.
 *
 * @param target a frame or an encoder
 * @returns the encoder to record into now
 */
export const encoderOf = (target: Frame | TgpuCommandEncoder): TgpuCommandEncoder =>
  'encoder' in target ? target.encoder : target;

/** Stands for no `dt`: long enough for any easing to arrive. */
const AT_ONCE = 1e9;

/**
 * Picks the seconds a recording method eases by: its `dt` option, else the frame's, else, with an
 * encoder alone, a step long enough for any easing to arrive.
 *
 * @param target a frame or an encoder
 * @param dt the method's `dt` option
 * @returns the seconds to ease by
 */
export const dtOf = (target: Frame | TgpuCommandEncoder, dt?: number): number =>
  dt ?? ('encoder' in target ? target.dt : AT_ONCE);

type Recorder = (frame: Frame) => void;

/**
 * Records a frame into one encoder and submits once, with a clock. Borrows the root. The
 * synchronous recorder borrows each frame's encoder.
 *
 * @submits always once a frame after your recorder returns, and again at each `frame.submit()`
 * @param record your frame, recorded into each frame's encoder
 * @returns a loop you `run` with each frame's time and destroy
 */
export function create(root: TgpuRoot, record: Recorder): Frames {
  let origin: number | undefined;
  let previous: number | undefined;
  let submitted = 0;
  let running = false;
  let recording: Frame | undefined;
  let destroyed = false;
  const pending: (() => void)[] = [];

  const recordingOf = (time: number, dt: number, index: number) => {
    // Create another encoder only when read, so a final `submit()` sends no empty command buffer
    let encoder: TgpuCommandEncoder | undefined = root.createCommandEncoder();
    const frame: Frame = {
      time,
      dt,
      index,

      get encoder() {
        return (encoder ??= root.createCommandEncoder());
      },

      afterSubmit(callback) {
        if (frame !== recording) {
          throw new Error('afterSubmit can only be called while its frame is recording.');
        }
        pending.push(callback);
      },

      submit() {
        if (frame !== recording) {
          throw new Error('submit can only be called while its frame is recording.');
        }
        if (destroyed || !encoder) return;

        const finished = encoder;
        encoder = undefined;
        finished.submit();
      },
    };
    return frame;
  };

  const destroy = () => {
    destroyed = true;
  };

  return {
    run(seconds) {
      if (destroyed) return false;
      if (running) throw new Error('A frame cannot be recorded inside another frame.');
      if (!Number.isFinite(seconds) || (previous !== undefined && seconds < previous)) {
        throw new RangeError(`Frame time must be finite and nondecreasing, got ${seconds}.`);
      }

      running = true;
      try {
        const time = seconds - (origin ?? seconds);
        const dt = seconds - (previous ?? seconds);
        const frame = recordingOf(time, dt, submitted);
        recording = frame;
        const result: unknown = record(frame);

        if (typeof (result as PromiseLike<unknown> | undefined)?.then === 'function') {
          throw new TypeError('Frame recorders must be synchronous; prepare async work first.');
        }

        if (destroyed) return false;
        frame.submit();
        recording = undefined;
        origin ??= seconds;
        previous = seconds;
        submitted += 1;

        let errors: unknown[] | undefined;
        for (let i = 0; i < pending.length; i++) {
          try {
            pending[i]();
          } catch (error) {
            (errors ??= []).push(error);
          }
        }

        if (errors) {
          throw errors.length === 1
            ? errors[0]
            : new AggregateError(errors, 'Several after-submit callbacks failed.');
        }

        return true;
      } finally {
        running = false;
        recording = undefined;
        pending.length = 0;
      }
    },

    get seconds() {
      return previous;
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
