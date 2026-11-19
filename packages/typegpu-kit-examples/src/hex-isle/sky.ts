import { environment } from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { vec3 } from 'math';
import { d, std } from 'typegpu';

type Rgb = [number, number, number];

const linear = (r: number, g: number, b: number): Rgb => {
  const { x, y, z } = srgbToLinear(d.vec3f(r, g, b));
  return [x, y, z];
};

/**
 * Sky presets at the sun heights where each holds, from night through the blue hour and sunset to
 * day, in sRGB converted to linear light. The sky between two presets blends them.
 */
const MOMENTS = [
  {
    height: -0.3,
    top: linear(0.08, 0.1, 0.2),
    bottom: linear(0.02, 0.02, 0.04),
    horizon: linear(0.07, 0.09, 0.19),
  },
  {
    height: -0.1,
    top: linear(0.2, 0.22, 0.4),
    bottom: linear(0.05, 0.05, 0.08),
    horizon: linear(0.42, 0.4, 0.62),
  },
  {
    height: 0.02,
    top: linear(0.36, 0.34, 0.52),
    bottom: linear(0.2, 0.13, 0.1),
    horizon: linear(0.8, 0.58, 0.56),
  },
  {
    height: 0.35,
    top: linear(0.55, 0.68, 0.85),
    bottom: linear(0.32, 0.28, 0.22),
    horizon: linear(0.7, 0.8, 0.92),
  },
];

/** Sun intensity above the air, which reddens and dims it towards the horizon. */
const SUN_INTENSITY = 1.6;
/** Linear moon colour converted from sRGB, and midnight intensity. */
const MOON = { color: linear(0.55, 0.65, 0.9), intensity: 0.25 };
/** The isle's latitude in radians, which leans the noon sun south, +z. */
const LATITUDE = 0.42;

/** Lighting at one hour: sun and sky settings, the background colour and the lamps' brightness. */
export const createDaylight = () => ({
  sunDirection: [0, 1, 0] as Rgb,
  sunColor: [1, 1, 1] as Rgb,
  sunIntensity: 1,
  skyTop: [0, 0, 0] as Rgb,
  skyBottom: [0, 0, 0] as Rgb,
  /** Background clear colour with alpha. */
  horizon: [0, 0, 0, 1] as [number, number, number, number],
  /** The horizon's colour, which clouds take on where the sun leaves them. There is no fog. */
  fogColor: [0, 0, 0] as Rgb,
  /** Night-light factor, 0 by day and 1 at night. */
  lamps: 0,
});
export type Daylight = ReturnType<typeof createDaylight>;

/** Writes `a` mixed `t` of the way to `b` into the first three components of `out`. */
const mix = <T extends number[]>(out: T, a: Rgb, b: Rgb, t: number) => {
  for (let i = 0; i < 3; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
};

/**
 * Writes lighting for `hour`, from 0 to 24, into `out`. The sun rises at 6 in the east, +x, and
 * sets at 18, on an equinox day. It leans south, +z, at noon. At night, the moon uses the same
 * light from the opposite side.
 */
export function daylight(hour: number, out: Daylight = createDaylight()): Daylight {
  const sun = environment.sunAt(hour, LATITUDE, 0);
  const height = sun.y;
  const up = height >= 0 ? 1 : -1;

  const direction = out.sunDirection;
  direction[0] = sun.x * up;
  direction[1] = sun.y * up;
  direction[2] = sun.z * up;

  const above = MOMENTS.findIndex((moment) => moment.height > height);
  const next = above === -1 ? MOMENTS.length - 1 : Math.max(above, 1);
  const [from, to] = [MOMENTS[next - 1], MOMENTS[next]];
  const t = std.smoothstep(from.height, to.height, height);
  mix(out.skyTop, from.top, to.top, t);
  mix(out.skyBottom, from.bottom, to.bottom, t);
  mix(out.horizon, from.horizon, to.horizon, t);
  mix(out.fogColor, from.horizon, to.horizon, t);

  // Fade each light at the horizon to hide the switch between sun and moon
  if (height >= 0) {
    const color = environment.sunColor(sun);
    out.sunColor[0] = color.x;
    out.sunColor[1] = color.y;
    out.sunColor[2] = color.z;
    out.sunIntensity = SUN_INTENSITY * std.smoothstep(0, 0.12, height);
  } else {
    vec3.copy(out.sunColor, MOON.color);
    out.sunIntensity = MOON.intensity * std.smoothstep(0, 0.2, -height);
  }

  out.lamps = std.smoothstep(0.15, -0.05, height);
  return out;
}
