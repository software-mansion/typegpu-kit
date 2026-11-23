import { each, glb } from '../assets.ts';
import { MODELS } from './island.ts';

/** Each model of the island. */
export const assets = { models: each(MODELS, (name) => glb(`models/${name}.glb`)) };
