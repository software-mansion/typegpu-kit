import type { Frames } from './frames.ts';

export interface Animation extends Disposable {
  /**
   * Stops requesting frames, even when called from a frame. Keeps the loop alive. Also
   * `[Symbol.dispose]`.
   */
  destroy(): void;
}

/**
 * Requests and cancels animation frames, as `requestAnimationFrame` and `cancelAnimationFrame` do.
 */
export interface Schedule {
  /**
   * Calls `callback` with the time in milliseconds at the next frame and returns the request's id.
   */
  request(callback: (milliseconds: number) => void): number;
  /** Cancels the request with this id. */
  cancel(request: number): void;
}

export interface AnimateOptions {
  /**
   * Stops frames and elapsed time while `paused` is true. The first frame after `resume()` has a
   * `dt` of 0. On React Native, pass the canvas surface and pause it from `AppState`. On the web,
   * pause it from an `IntersectionObserver` or a pause menu. Without a surface, frames run until
   * the animation is destroyed. Default none.
   */
  readonly surface?: { readonly paused: boolean };
  /**
   * Limits the time added per frame, in seconds, after a stall or when a tab becomes visible.
   * Default 0.1.
   */
  readonly maxDt?: number;
  /**
   * Supplies animation frames. Use `runtimes.vsync(worker)` on a React Native worker runtime, or a
   * schedule you step by hand in a test. Default `requestAnimationFrame`.
   */
  readonly schedule?: Schedule;
  /**
   * Receives a frame error after the animation stops. Without a handler, the animation frame
   * callback throws the error. Default none.
   * @cpu
   */
  readonly onError?: (error: unknown) => void;
}

/**
 * Runs one frame of `animate` for a profiler, which marks its start and times `run`. `run` takes
 * the frame's timestamp in milliseconds, so a profiler can run several frames in one, each at a
 * later time.
 */
export type FrameWatch = (milliseconds: number, run: (milliseconds: number) => void) => void;

let watch: FrameWatch | undefined;

/**
 * Wraps every frame that `animate` runs on this runtime, so a profiler can time frames where
 * there is no `requestAnimationFrame` to patch, such as on a React Native worker. Holds one watch
 * per runtime; a later call replaces it.
 *
 * @param next the watch, or undefined to stop watching
 * @returns nothing; the watch wraps frames from the next one `animate` runs
 */
export function watchFrames(next: FrameWatch | undefined) {
  watch = next;
}

/**
 * Runs the loop on each scheduled frame. Time advances by at most `maxDt` per frame, never steps
 * back, and stops while the surface is paused. Destroying the animation keeps the loop alive.
 * Another animation on the same loop continues from its recorded time. A frame error, or a `run`
 * that returns false because the loop was destroyed, stops the animation.
 *
 * @param loop the loop to run on each unpaused frame. Continues from its `seconds` if it ran before
 * @returns an animation you destroy to stop the frames
 */
export function animate(
  loop: Pick<Frames, 'run'> & Partial<Pick<Frames, 'seconds'>>,
  options: AnimateOptions = {},
): Animation {
  const { surface, maxDt = 0.1, onError, schedule = animationFrames() } = options;

  let request = 0;
  let destroyed = false;
  let seconds: number | undefined;
  let previous: number | undefined;

  const destroy = () => {
    destroyed = true;
    schedule.cancel(request);
  };

  const step = (milliseconds: number) => {
    // A timestamp behind the last one, as after a profiler ran frames ahead of the display, adds
    // no time
    const elapsed = previous === undefined ? 0 : (milliseconds - previous) / 1000;
    const dt = Math.min(Math.max(elapsed, 0), maxDt);
    // Continue from the loop time even if the schedule uses a different clock
    seconds = seconds === undefined ? (loop.seconds ?? milliseconds / 1000) : seconds + dt;
    previous = milliseconds;
    // A destroyed loop returns false; a loop of your own that returns nothing keeps running
    if (loop.run(seconds) === false) destroy();
  };

  const tick = (milliseconds: number) => {
    if (surface?.paused) {
      previous = undefined;
    } else {
      try {
        if (watch) watch(milliseconds, step);
        else step(milliseconds);
      } catch (error) {
        destroy();
        if (!onError) throw error;
        onError(error);
        return;
      }
    }

    if (!destroyed) request = schedule.request(tick);
  };
  request = schedule.request(tick);

  return { destroy, [Symbol.dispose]: destroy };
}

function animationFrames(): Schedule {
  return {
    request: (callback) => requestAnimationFrame(callback),
    cancel: (request) => cancelAnimationFrame(request),
  };
}
