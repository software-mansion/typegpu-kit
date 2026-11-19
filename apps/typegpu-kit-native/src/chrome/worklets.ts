// Worklets that host the app's chrome on the UI runtime. Imports from the scene files load again
// there, so each worklet calls the scene's code instead of a function on the JS thread. Worklets
// forwards named imports only, not namespaces.

import type { CreateScene, Stage } from '@typegpu/kit/native/host';
import { create as banner, type Banner } from './banner.ts';
import { create as veil, type Veil, type VeilOptions } from './veil.ts';

/** Creates the gallery's banner. */
export const createBanner: CreateScene<object, Banner> = (root, surface) => {
  'worklet';
  return banner(root, surface);
};

/** Moves the banner's light to a finger at `x`, `y` points. */
export function touchBanner({ scene }: Stage<Banner>, x: number, y: number) {
  'worklet';
  scene.touch(x, y);
}

/** Lets the banner's light wander again. */
export function releaseBanner({ scene }: Stage<Banner>) {
  'worklet';
  scene.release();
}

/** Creates the veil over a loading example. */
export const createVeil: CreateScene<VeilOptions, Veil> = (root, surface, options) => {
  'worklet';
  return veil(root, surface, options);
};

/** Lifts the veil off the example. */
export function liftVeil({ scene }: Stage<Veil>) {
  'worklet';
  scene.lift();
}
