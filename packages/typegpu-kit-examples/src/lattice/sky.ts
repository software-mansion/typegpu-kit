import { environment } from '@typegpu/kit';
import { d, std } from 'typegpu';

const ZENITH = d.vec3f(0.004, 0.006, 0.03);
const HORIZON = d.vec3f(0.22, 0.07, 0.04);
const GROUND = d.vec3f(0.012, 0.006, 0.006);

const duskAt = (direction: d.v3f) => {
  'use gpu';
  const above = std.mix(HORIZON, ZENITH, std.pow(std.saturate(direction.y), 0.45));
  const below = std.mix(HORIZON, GROUND, std.saturate(-direction.y * 6));
  return std.select(below, above, direction.y > 0);
};

/** A dusk gradient behind everything, drawn after the beams. */
export const sky = environment.background(duskAt, { label: 'sky' });
