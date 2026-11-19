import { perlin2d } from '@typegpu/noise';
import { d, std } from 'typegpu';
import { MIDDLE } from './tray.ts';

/**
 * The sky's colour overhead and at the horizon, past white so the tone curve keeps it bright, and
 * how much of it the far tray's haze and glossy surfaces take.
 */
const SKY = {
  zenith: d.vec3f(0.55, 0.78, 1.4),
  horizon: d.vec3f(1.3, 1.16, 1.05),
  glow: d.vec3f(1, 0.72, 0.5),
  haze: 0.85,
  mirrored: 0.72,
};
/** Distant gumdrop hills: their heights over the horizon, as sines of elevation, and colours. */
const HILLS = {
  far: d.vec2f(0.035, 0.09),
  near: d.vec2f(0.01, 0.07),
  farColor: d.vec3f(0.78, 0.7, 1.25),
  nearColor: d.vec3f(0.52, 0.95, 0.66),
};
/**
 * Where the fog starts, in world units along a ray past the eye's distance to the tray's middle,
 * and how fast it thickens.
 */
const FOG = { start: 1.68, density: 0.32 };

/** The sky's gradient by how far up a direction points. */
const skyTone = (up: number) => {
  'use gpu';
  return std.mix(SKY.horizon, SKY.zenith, std.saturate(up) ** 0.5);
};

/** The sky as glossy surfaces mirror it, kept warm near the horizon so grazing gloss stays clean. */
export const mirrored = (up: number) => {
  'use gpu';
  return std.mix(SKY.horizon, SKY.zenith, std.saturate(up)) * SKY.mirrored;
};

/** The pastel sky seen along `direction`: clouds, a warm glow round the sun and gumdrop hills. */
export const sky = (direction: d.v3f, toLight: d.v3f, time: number) => {
  'use gpu';
  const up = direction.y;
  let color = skyTone(up);
  const sunward = std.saturate(std.dot(direction, toLight));
  color += SKY.glow * (sunward ** 6 * 0.35 + sunward ** 64 * 0.6);

  const flat = direction.xz / (up + 0.15);
  const at = flat * 0.9 + d.vec2f(time * 0.02, 0);
  const puff = perlin2d.sample(at) + perlin2d.sample(at * 2.3 + 3.7) * 0.45;
  const cloud = std.smoothstep(0.1, 0.4, puff) * std.smoothstep(0.01, 0.12, up);
  const lit = std.mix(d.vec3f(1.05, 1, 1.3), d.vec3f(1.6, 1.55, 1.5), std.saturate(puff * 1.6));
  color = std.mix(color, lit, cloud * 0.85);

  // Heights from the direction round the horizon, so no seam shows behind the view; folding the
  // noise at zero rounds the crests and creases the dips between them
  const round = std.normalize(direction.xz);
  const far = HILLS.far.x + HILLS.far.y * std.abs(perlin2d.sample(round * 4 + 9.1));
  const near = HILLS.near.x + HILLS.near.y * std.abs(perlin2d.sample(round * 6.5 + 1.3));
  // Each range is clearest at its crest and sinks into the haze at the horizon
  if (up < far) color = std.mix(SKY.horizon, HILLS.farColor, std.saturate(up / far) * 0.9);
  if (up < near) color = std.mix(SKY.horizon, HILLS.nearColor, std.saturate(up / near) * 0.95);
  return color;
};

/** Fades a colour seen `distance` away from `eye` along `direction` into the horizon's haze. */
export const fog = (color: d.v3f, eye: d.v3f, direction: d.v3f, distance: number) => {
  'use gpu';
  const past = distance - std.length(eye - d.vec3f(MIDDLE[0], MIDDLE[1], MIDDLE[2]));
  const amount = 1 - std.exp(-std.max(past - FOG.start, 0) * FOG.density);
  return std.mix(color, skyTone(direction.y * 0.5) * SKY.haze, amount);
};
