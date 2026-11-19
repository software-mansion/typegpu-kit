import { each, glb } from '../assets.ts';
import { MODELS } from './horde.ts';

/** The skeletons, what they hold and their clips. */
export const assets = { models: each(MODELS, (name) => glb(`models/${name}.glb`)) };
