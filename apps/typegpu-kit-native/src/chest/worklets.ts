// Worklets that host the chest's scenes on the UI runtime, as `chrome/worklets.ts` hosts the
// banner's. Each calls the scene file's code through a named import, which worklets forwards.

import type { CreateScene, Scene } from '@typegpu/kit/native/host';
import { pulled, resting, type PulledOptions, type RestingOptions } from './box.ts';
import { create as face, type FaceOptions } from './face.ts';

/** Creates the box resting on the banner. */
export const createResting: CreateScene<RestingOptions, Scene> = (root, surface, options) => {
  'worklet';
  return resting(root, surface, options);
};

/** Creates the box of an open pull. */
export const createPulled: CreateScene<PulledOptions, Scene> = (root, surface, options) => {
  'worklet';
  return pulled(root, surface, options);
};

/** Creates the pulled card's canvas. */
export const createFace: CreateScene<FaceOptions, Scene> = (root, surface, options) => {
  'worklet';
  return face(root, surface, options);
};
