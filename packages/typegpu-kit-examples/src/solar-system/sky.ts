import { environment } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { d, std } from 'typegpu';

/** Linear colours of space overhead, at the ecliptic and below it. */
const ABOVE = d.vec3f(0.006, 0.005, 0.02);
const LEVEL = d.vec3f(0.014, 0.009, 0.03);
const BELOW = d.vec3f(0.003, 0.005, 0.016);
/** A faint band of haze, tilted across the ecliptic. */
const BAND = d.vec3f(0.018, 0.012, 0.034);
const BAND_AXIS = d.vec3f(0.35, 0.9, -0.25);
/** A star in a fifth of the cells, 0.9 pixels across so none falls between pixels. */
const starAt = environment.stars({ cells: 72, chance: 0.2, size: 0.9 });
const STAR_BRIGHTNESS = 0.55;
/** A power distribution gives many faint stars and a few bright ones. */
const STAR_SPREAD = 6;
const STAR_WARM = d.vec3f(1, 0.85, 0.7);
const STAR_COOL = d.vec3f(0.7, 0.8, 1);

const starlight = (direction: d.v3f) => {
  'use gpu';
  const star = starAt(direction);
  const brightness = std.pow(star.seed.x, STAR_SPREAD) * STAR_BRIGHTNESS + 0.02;
  return std.mix(STAR_WARM, STAR_COOL, star.seed.y) * (brightness * star.disc);
};

const spaceAt = (direction: d.v3f) => {
  'use gpu';
  const up = std.mix(LEVEL, ABOVE, std.sqrt(std.saturate(direction.y)));
  const down = std.mix(LEVEL, BELOW, std.sqrt(std.saturate(-direction.y)));
  const base = std.select(down, up, direction.y > 0);

  const across = std.abs(std.dot(direction, std.normalize(BAND_AXIS)));
  const wisps = perlin3d.sample(direction * 3) * 0.5 + 0.5;
  const haze = std.exp(-across * across * 18) * wisps;
  return base + BAND * haze + starlight(direction);
};

/** Deep space: a blue-violet gradient, a faint band of haze and sparse stars. */
export const space = environment.background(spaceAt);
