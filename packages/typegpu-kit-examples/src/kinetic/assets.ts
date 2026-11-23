import { hdr } from '../assets.ts';

/**
 * Three of Poly Haven's CC0 HDR panoramas at 1k: Greg Zaal's `kloofendal_48d_partly_cloudy_puresky`
 * with Jarod Guest's sky edits, Greg Zaal's `venice_sunset` and Sergej Majboroda's
 * `studio_small_09`.
 */
export const assets = {
  skies: {
    meadow: hdr('textures/kloofendal.hdr'),
    venice: hdr('textures/venice-sunset.hdr'),
    studio: hdr('textures/studio.hdr'),
  },
};
