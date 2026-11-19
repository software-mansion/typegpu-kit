import { environment } from '@typegpu/kit';
import { d, std } from 'typegpu';

/** The sky at the horizon, which the fog matches so the far yard fades into it. */
export const HORIZON = [0.03, 0.045, 0.085] as const;
/** Towards a low moon, so the skeletons cast long shadows across the flagstones. */
export const MOON = [-0.76, 0.4, 0.51] as const;

const LOW = d.vec3f(...HORIZON);
const ZENITH = d.vec3f(0.004, 0.007, 0.022);
const TOWARD_MOON = std.normalize(d.vec3f(...MOON));
const MOONLIGHT = d.vec3f(0.75, 0.85, 1);
const starAt = environment.stars({ cells: 110, chance: 0.16, size: 0.7 });
const STAR_WARM = d.vec3f(1, 0.85, 0.7);
const STAR_COOL = d.vec3f(0.7, 0.8, 1);

/** A deep blue night, stars thinning towards the horizon, and the moon in its halo. */
const nightAt = (direction: d.v3f) => {
  'use gpu';
  const up = direction.y;
  const above = std.mix(LOW, ZENITH, std.pow(std.saturate(up), 0.5));
  const star = starAt(direction);
  const tint = std.mix(STAR_WARM, STAR_COOL, star.seed.y);
  const stars = tint * (std.pow(star.seed.x, 8) * 0.8 * star.disc * std.smoothstep(0, 0.2, up));
  const toward = std.dot(direction, TOWARD_MOON);
  const disc = std.smoothstep(0.99975, 0.99985, toward) * 6;
  const halo = std.exp((toward - 1) * 60) * 0.12 + std.exp((toward - 1) * 8) * 0.03;
  return std.select(LOW, above + stars + MOONLIGHT * (disc + halo), up > 0);
};

export const night = environment.background(nightAt, { label: 'night sky' });
