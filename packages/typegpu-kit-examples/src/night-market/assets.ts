import { glb } from '../assets.ts';
import { MODELS } from './stalls.ts';

/**
 * The market's buildings, goods and lamps, each a part of one Draco-compressed file that shares
 * its images, and the seller, the customer and their clips.
 */
export const assets = {
  models: glb('models/market.glb', { parts: MODELS, draco: true }),
  people: {
    mage: glb('models/mage.glb'),
    rogue: glb('models/rogue.glb'),
    clips: glb('models/clips.glb'),
  },
};
