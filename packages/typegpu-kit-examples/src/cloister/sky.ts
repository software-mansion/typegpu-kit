import { d, std } from 'typegpu';

/** A deep blue overhead, paling to a mauve glow low over the roofs, as the sky in the blue hour. */
const ZENITH = d.vec3f(0.012, 0.028, 0.1);
const HORIZON = d.vec3f(0.11, 0.1, 0.2);
const AFTERGLOW = d.vec3f(0.09, 0.04, 0.03);
const GROUND = d.vec3f(0.01, 0.012, 0.025);

/**
 * The court's light without a sun: the sky's blue from above and its dark bounce from the paving,
 * which the cloister's baked occlusion darkens, and a thin blue haze.
 */
export const DUSK = {
  sunIntensity: 0,
  skyTop: [0.17, 0.23, 0.48],
  skyBottom: [0.045, 0.038, 0.05],
  skyIntensity: 1,
  skyReflection: 1,
  fogColor: [0.06, 0.07, 0.14],
  fogDensity: 0.01,
} as const;

/** The blue-hour sky in a direction, for `environment.background`. */
export const sky = (look: d.v3f) => {
  'use gpu';
  const up = look.y;
  // The afterglow lingers in the west, -x, where the sun set
  const west = std.saturate(-look.x * 0.5 + 0.5);
  let above = std.mix(HORIZON, ZENITH, std.pow(std.saturate(up), 0.5));
  above += AFTERGLOW * west * std.exp(-std.max(up, 0) * 8);
  const below = std.mix(HORIZON, GROUND, std.saturate(-up * 6));
  return std.select(below, above, up > 0);
};
