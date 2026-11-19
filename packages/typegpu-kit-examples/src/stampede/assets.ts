import { font, glb } from '../assets.ts';
import { SCENERY } from './scenery.ts';

/** Quaternius horse and clips, KayKit scenery and the shipped Inter font. */
export const assets = {
  horse: glb('models/horse.glb'),
  scenery: glb('models/scenery.glb', { parts: SCENERY }),
  font: font('inter'),
};
