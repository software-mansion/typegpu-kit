import { cameras, environment } from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { d, std, tgpu } from 'typegpu';
import {
  caustic,
  timeOf,
  SHALLOW,
  SUNLIGHT,
  SURFACE,
  throughSurface,
  waterAlong,
} from './water.ts';

/** Samples along each view ray for the light shafts, and how far they reach. */
const STEPS = 8;
const REACH = 11;
/** The shafts' colour and brightness. */
const SHAFTS = d.vec3f(0.011, 0.024, 0.036);
/** Cosine of the half-angle of the window through which the sky shows from under the surface. */
const WINDOW = 0.66;

/**
 * Shafts of light coming down through the surface, summed along a ray from the eye. The samples
 * spread over the stretch of the ray below the surface, so the sum changes smoothly as the ray
 * tips up into it.
 */
const shaftsAlong = (eye: d.v3f, direction: d.v3f, time: number) => {
  'use gpu';
  const toSurface = (SURFACE - eye.y) / std.max(direction.y, 0.0001);
  const reach = std.min(d.f32(REACH), std.select(d.f32(REACH), toSurface, direction.y > 0));
  let sum = d.f32(0);
  for (const i of tgpu.unroll(std.range(STEPS))) {
    const t = ((i + 0.5) / STEPS) * reach;
    const at = eye + direction * t;
    const below = std.max(SURFACE - at.y, 0);
    const drifting = throughSurface(at) * 0.42 + d.vec2f(time * 0.012, time * 0.005);
    const beam = perlin3d.sample(d.vec3f(drifting, time * 0.05));
    const lit = std.smoothstep(0.1, 0.65, beam) * std.exp(-below * 0.55);
    sum += lit * (1 - t / (REACH * 1.2));
  }
  return sum * (reach / STEPS);
};

/** The surface seen from below: the sky through Snell's window, the tank reflected around it. */
const surfaceAlong = (eye: d.v3f, direction: d.v3f, time: number) => {
  'use gpu';
  const far = (SURFACE - eye.y) / std.max(direction.y, 0.01);
  const at = eye.xz + direction.xz * far;
  const ripples = caustic(at * 0.7, time * 1.3);
  const reflected = SHALLOW * (0.5 + ripples * 1.6);
  const sky = SUNLIGHT * (0.5 + ripples * 0.4);
  const shown = std.mix(reflected, sky, std.smoothstep(WINDOW - 0.05, WINDOW + 0.06, direction.y));
  return d.vec4f(shown, std.exp(-far * 0.3));
};

/** The water seen from the eye along `direction`: the surface above, and the shafts through it. */
const waterFrom = (direction: d.v3f) => {
  'use gpu';
  const eye = cameras.current.$.position;
  const time = timeOf.$;
  let color = waterAlong(direction);
  if (direction.y > 0) {
    const surface = surfaceAlong(eye, direction, time);
    color = std.mix(color, surface.xyz, surface.w * std.smoothstep(0, 0.08, direction.y));
  }
  return color + SHAFTS * shaftsAlong(eye, direction, time);
};

/** The water behind the jellies, which blend over it. */
export const water = environment.background(waterFrom, { label: 'water' });
