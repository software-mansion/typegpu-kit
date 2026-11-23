import {
  createShareable,
  createSynchronizable,
  scheduleOnRuntimeWithId,
  scheduleOnUI,
  UIRuntimeId,
  type FixedSynchronizable,
  type Shareable,
} from 'react-native-worklets';
import type { Runtime, Vsync } from './runtimes.ts';

/** A `Vsync` that a host holds while its scene is paused. */
export interface Relay extends Vsync {
  /**
   * Keeps requests from reaching the display while true. Releasing it sends the request that
   * waits. Call it on the relay's runtime.
   */
  hold(held: boolean): void;
}

interface Waiting {
  callback: ((milliseconds: number) => void) | undefined;
  id: number;
  held: boolean;
}

/**
 * The UI runtime's side: its frame callback, whether it has a frame requested, and the last
 * request it relayed a display frame to.
 */
interface Display {
  tick: ((milliseconds: number) => void) | undefined;
  ticking: boolean;
  offered: number;
}

export function relay(runtime: Runtime): Relay {
  const { runtimeId, name } = runtime;
  // The latest request the display should relay a frame to; only the runtime writes it
  const requested = createSynchronizable(0, { fixedType: true });
  const running = createSynchronizable(false, { fixedType: true });
  const stopped = createSynchronizable(false, { fixedType: true });
  const parked = createSynchronizable(false, { fixedType: true });
  // The worklets Babel plugin rejects a method capturing its own name, so avoid naming this
  // `missed`
  const misses = createSynchronizable(0, { fixedType: true });
  const waiting = createShareable<Waiting>(runtimeId, {
    callback: undefined,
    id: 0,
    held: false,
  });
  const display = createShareable<Display>(UIRuntimeId, {
    tick: undefined,
    ticking: false,
    offered: 0,
  });

  scheduleOnUI(start, runtimeId, requested, running, stopped, parked, misses, waiting, display);

  // Wakes the display once after it parked; a second wake finds it ticking
  const want = (request: number) => {
    'worklet';
    requested.setDirty(request);
    if (!parked.getDirty()) return;
    parked.setDirty(false);
    scheduleOnUI(wake, display);
  };

  const destroy = () => {
    'worklet';
    stopped.setDirty(true);
  };

  return {
    request(callback) {
      'worklet';
      if (!waiting.isHost) throw new Error(`Request frames from vsync(${name}) on ${name} only.`);
      waiting.value.callback = callback;
      waiting.value.id += 1;
      if (!waiting.value.held) want(waiting.value.id);
      return waiting.value.id;
    },

    cancel(request) {
      'worklet';
      if (!waiting.isHost || waiting.value.id !== request) return;
      waiting.value.callback = undefined;
    },

    hold(held) {
      'worklet';
      if (!waiting.isHost) return;
      waiting.value.held = held;
      if (!held && waiting.value.callback) want(waiting.value.id);
    },

    missed() {
      'worklet';
      return misses.getDirty();
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

// Calls back only the request the display relayed this frame to, which a cancel or a newer request
// may have replaced since
function fire(
  waiting: Shareable<Waiting>,
  running: FixedSynchronizable<boolean>,
  stopped: FixedSynchronizable<boolean>,
  request: number,
  milliseconds: number,
) {
  'worklet';
  if (!waiting.isHost) return;
  const { callback, id } = waiting.value;

  try {
    if (stopped.getDirty() || id !== request) return;
    waiting.value.callback = undefined;
    callback?.(milliseconds);
  } catch (error) {
    console.error(
      'A frame failed on a worker runtime. Pass onError to animate to handle it.',
      error,
    );
  } finally {
    running.setDirty(false);
  }
}

function start(
  runtimeId: number,
  requested: FixedSynchronizable<number>,
  running: FixedSynchronizable<boolean>,
  stopped: FixedSynchronizable<boolean>,
  parked: FixedSynchronizable<boolean>,
  misses: FixedSynchronizable<number>,
  waiting: Shareable<Waiting>,
  display: Shareable<Display>,
) {
  'worklet';
  if (!display.isHost) return;
  const state = display.value;

  const tick = (milliseconds: number) => {
    state.ticking = false;
    if (stopped.getDirty()) return;

    const request = requested.getDirty();
    if (request !== state.offered) {
      if (running.getDirty()) {
        misses.setDirty(misses.getDirty() + 1);
      } else {
        state.offered = request;
        running.setDirty(true);
        scheduleOnRuntimeWithId(runtimeId, fire, waiting, running, stopped, request, milliseconds);
      }
    } else if (!running.getDirty()) {
      // Park until a request wakes the display, unless one came in meanwhile
      parked.setDirty(true);
      if (requested.getDirty() === state.offered) return;
      parked.setDirty(false);
    }

    state.ticking = true;
    requestAnimationFrame(tick);
  };

  state.tick = tick;
  state.ticking = true;
  requestAnimationFrame(tick);
}

function wake(display: Shareable<Display>) {
  'worklet';
  if (!display.isHost) return;
  const state = display.value;
  if (state.ticking || !state.tick) return;
  state.ticking = true;
  requestAnimationFrame(state.tick);
}
