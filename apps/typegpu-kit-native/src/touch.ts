// The touch lane from finger to scene. One Gesture Handler Manual gesture covers the canvas and the
// pads, never activates (an active handler blocks every other, and touch events arrive in every
// state) and fails once the last finger lifts, which recycles it. Its callbacks run on the UI
// runtime, where the touch router keeps each finger's route: routing needs only the
// scene's shape and the pads' layout, so pads light at once and the scene's worker receives
// semantic input through `send`, in order.

import type { Hosted } from '@typegpu/kit/native/react';
import type { SharedValue } from 'react-native-reanimated';
import type { ExampleScene } from 'typegpu-kit-examples/contract';
import { createTouchRouter, padKey, type TouchLayout } from './touches.ts';
import { lanes, type Lane } from './lanes.ts';
import * as worklets from './worklets.ts';

/** Where a lane's actions go: the scene, once it is up, and the pads lit. */
export interface Target {
  readonly send: Hosted<ExampleScene>['send'] | undefined;
  readonly lit: SharedValue<Readonly<Record<string, boolean>>>;
}

export type TouchKind = 'down' | 'move' | 'up' | 'cancel';

/** `lit` without `key`. */
const unlit = (lit: Readonly<Record<string, boolean>>, key: string) => {
  'worklet';
  const rest: Record<string, boolean> = {};
  for (const each in lit) if (each !== key) rest[each] = true;
  return rest;
};

/** A lane whose router lights pads and sends to whatever its `target` is at the time. */
const createLane = (target: Target): Lane => {
  'worklet';
  const lane: Lane = {
    target,
    router: createTouchRouter({
      press(pad) {
        const { lit, send } = lane.target;
        const key = padKey(pad.element, pad.name);
        // A latched pad lets go on the tap after the one that pressed it
        if (pad.latch && lit.value[key]) {
          lit.value = unlit(lit.value, key);
          send?.(worklets.release, pad.element, pad.name);
          return;
        }
        lit.value = { ...lit.value, [key]: true };
        send?.(worklets.press, pad.element, pad.name);
      },
      release(pad) {
        if (pad.latch) return;
        const { lit, send } = lane.target;
        lit.value = unlit(lit.value, padKey(pad.element, pad.name));
        if (pad.held) send?.(worklets.release, pad.element, pad.name);
      },
      orbit: (dx, dy) => lane.target.send?.(worklets.orbit, dx, dy),
      zoom: (by) => lane.target.send?.(worklets.zoom, by),
      tap: (x, y) => lane.target.send?.(worklets.tap, x, y),
      drag: (x, y, start) => lane.target.send?.(worklets.drag, x, y, start),
      end: () => lane.target.send?.(worklets.end),
    }),
  };
  return lane;
};

/** Lane `id`, created on its first touch, aimed at `target` and configured with `layout`. */
const laneOf = (id: number, target: Target, layout: TouchLayout) => {
  'worklet';
  let lane = lanes.get(id);
  if (!lane) {
    lane = createLane(target);
    lanes.set(id, lane);
  }
  lane.target = target;
  lane.router.configure(layout);
  return lane;
};

/**
 * Routes the touches that changed, as id, x, y triples in the area's points, stamped `at`
 * milliseconds.
 */
export function route(
  id: number,
  kind: TouchKind,
  touches: readonly number[],
  at: number,
  target: Target,
  layout: TouchLayout,
) {
  'worklet';
  const { router } = laneOf(id, target, layout);
  for (let i = 0; i + 2 < touches.length; i += 3) {
    const finger = touches[i];
    const x = touches[i + 1];
    const y = touches[i + 2];
    if (kind === 'down') router.down(finger, x, y, at);
    else if (kind === 'move') router.move(finger, x, y, at);
    else if (kind === 'up') router.up(finger, x, y, at);
    else router.cancel(finger);
  }
}

/** Cancels every finger of lane `id`, as the gesture ends without their lifts. */
export function settle(id: number) {
  'worklet';
  lanes.get(id)?.router.reset();
}

/** Forgets lane `id` once its screen is gone. */
export function forget(id: number) {
  'worklet';
  lanes.get(id)?.router.reset();
  lanes.delete(id);
}
