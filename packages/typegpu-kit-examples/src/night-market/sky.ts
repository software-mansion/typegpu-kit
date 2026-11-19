import { perlin2d } from '@typegpu/noise';
import { d, std } from 'typegpu';

/** A deep blue overhead, lighter and greener near the rooftops, as the sky an hour after dusk. */
const ZENITH = d.vec3f(0.004, 0.009, 0.045);
const HORIZON = d.vec3f(0.03, 0.06, 0.13);
const GROUND = d.vec3f(0.01, 0.015, 0.03);
/** The market's lamps warm the haze over the rooftops, and lit clouds drift above them. */
const HAZE = d.vec3f(0.035, 0.016, 0.01);
const CLOUD = d.vec3f(0.018, 0.022, 0.036);

/** The night sky over the market in a direction, for `environment.background`. */
export const sky = (look: d.v3f) => {
  'use gpu';
  const up = look.y;
  let above = std.mix(HORIZON, ZENITH, std.pow(std.saturate(up), 0.6));
  above += HAZE * std.exp(-std.max(up, 0) * 20);
  // Clouds on a plane overhead, thinning towards the zenith and fading into the haze
  const onCloud = look.xz / (std.max(up, 0) + 0.08);
  const cover = perlin2d.sample(onCloud * 0.35) * 0.65 + perlin2d.sample(onCloud * 0.9 + 7) * 0.35;
  above += CLOUD * std.smoothstep(-0.15, 0.45, cover) * std.smoothstep(0, 0.12, up);
  const below = std.mix(HORIZON, GROUND, std.saturate(-up * 6));
  return std.select(below, above, up > 0);
};
