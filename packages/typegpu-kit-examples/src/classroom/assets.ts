import { glb, image } from '../assets.ts';

/** The classroom, Draco compressed, and its daylight and its lamps' light baked in logarithms. */
export const assets = {
  model: glb('models/classroom.glb', { draco: true }),
  daylight: image('textures/lightmap-daylight.webp'),
  lamps: image('textures/lightmap-lamps.webp'),
};
