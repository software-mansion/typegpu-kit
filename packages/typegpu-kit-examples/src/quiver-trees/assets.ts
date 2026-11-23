import { each, glb, image } from '../assets.ts';
import { MODELS } from './plain.ts';

/**
 * The trees' and boulders' scans, and Poly Haven's CC0 `gravelly_sand` by Dario Barresi, whose RGB
 * holds albedo and alpha holds height.
 */
export const assets = {
  models: each(MODELS, (name) => glb(`models/${name}.glb`)),
  ground: image('textures/gravelly-sand.webp'),
};
