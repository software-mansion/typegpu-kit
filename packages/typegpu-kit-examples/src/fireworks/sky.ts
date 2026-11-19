import { environment } from '@typegpu/kit';
import { d, std } from 'typegpu';

const ZENITH = d.vec3f(0.0008, 0.0013, 0.0055);
/** The horizon's colour, which the fog matches so far hills fade into it. */
export const HORIZON = d.vec3f(0.007, 0.009, 0.024);
const HORIZON_FALLOFF = 0.45;
const BELOW = d.vec3f(0.004, 0.005, 0.012);
/** Stars in a seventh of the cells, 0.6 pixels across at any resolution. */
const starAt = environment.stars({ cells: 110, chance: 0.14, size: 0.6 });
const STAR_BRIGHTNESS = 0.6;
/** A power distribution gives many faint stars and a few bright ones. */
const STAR_SPREAD = 9;
const STAR_WARM = d.vec3f(1, 0.82, 0.62);
const STAR_COOL = d.vec3f(0.7, 0.82, 1);
/** Star-density falloff toward the hazy horizon. */
const STAR_HAZE_HEIGHT = 0.18;
const TWINKLE_HZ = 1.7;
const TWINKLE_DEPTH = 0.35;

/** Stars that twinkle and thin towards the horizon. */
const starlight = (direction: d.v3f, time: number) => {
  'use gpu';
  const star = starAt(direction);
  const brightness = std.pow(star.seed.x, STAR_SPREAD) * STAR_BRIGHTNESS;
  const phase = star.seed.z * Math.PI * 2;
  const twinkle = 1 - TWINKLE_DEPTH * (0.5 + 0.5 * std.sin(time * TWINKLE_HZ + phase));
  const haze = std.smoothstep(0, STAR_HAZE_HEIGHT, direction.y);
  const tint = std.mix(STAR_WARM, STAR_COOL, star.seed.y);
  return tint * (brightness * twinkle * haze * star.disc);
};

/** A dark sky with twinkling stars that thin towards the horizon, behind everything. */
export function nightSky(time: { readonly $: number }) {
  const nightAt = (direction: d.v3f) => {
    'use gpu';
    const above = std.mix(HORIZON, ZENITH, std.pow(std.saturate(direction.y), HORIZON_FALLOFF));
    const below = std.mix(HORIZON, BELOW, std.saturate(-direction.y * 8));
    return std.select(below, above, direction.y > 0) + starlight(direction, time.$);
  };
  return environment.background(nightAt, { label: 'night sky' });
}
