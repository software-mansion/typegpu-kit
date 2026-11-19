import { each, glb } from '../assets.ts';
import { MODELS } from './castle.ts';

/** The castle's pieces and the pines. */
export const assets = { models: each(MODELS, (name) => glb(`models/${name}.glb`)) };
