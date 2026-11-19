import { environment, shading } from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { d, std } from 'typegpu';

/** Unit direction towards the low evening sun. */
export const SUN = d.vec3f(0.6, 0.31, -0.737);

/** The air towards the sun, away from it and straight up. */
const WARM = srgbToLinear(d.vec3f(0.98, 0.74, 0.5));
const COOL = srgbToLinear(d.vec3f(0.5, 0.62, 0.68));
const ZENITH = srgbToLinear(d.vec3f(0.24, 0.44, 0.78));
/** The sun's disc, bright enough to bloom. */
const DISC = d.vec3f(40, 28, 16);

/** The colour of the air along a unit direction near the horizon. */
const horizon = (direction: d.v3f) => {
  'use gpu';
  const facing = std.saturate(std.dot(direction, SUN) * 0.5 + 0.5);
  return std.mix(COOL, WARM, std.pow(facing, 5));
};

/** Fogs towards the horizon colour along the view, by the lighting's height fog. */
export const aerial = (point: shading.Point) => {
  'use gpu';
  const hidden = shading.haze(point);
  return shading.Air({ added: horizon(point.view * -1) * hidden, through: d.vec3f(1 - hidden) });
};

/** The sky's light along a unit direction: the air, deepening overhead, and the sun's glow. */
const skyAt = (direction: d.v3f) => {
  'use gpu';
  const sun = std.saturate(std.dot(direction, SUN));
  const glow = WARM * (std.pow(sun, 12) * 0.6) + DISC * std.smoothstep(0.9994, 0.9997, sun);
  return std.mix(horizon(direction), ZENITH, std.smoothstep(0, 0.45, direction.y)) + glow;
};

/** Draws the sky where the trees leave it. */
export const sky = environment.background(skyAt);
