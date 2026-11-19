import { cameras, environment, instances, materials, meshPool, shading } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { perlin2d } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';

/** Half the length and half the width of the oval the herd runs around, in metres. */
export const LOOP_METRES = [520, 200] as const;
/** The infield's knoll, at the oval's middle. */
const KNOLL_METRES = 16;
const HILL_METRES = 40;
const HILL_WAVE_METRES = 22;
/** Ridges across the hills, metres high. */
const RIDGE_METRES = [22, 16, 9, 5] as const;
const RANGE_METRES = 2100;
const RANGE_RISE_METRES = 900;
const PEAK_METRES = 340;
const PEAK_WAVES_METRES = [120, 60] as const;
/** World units from the middle that the land reaches. */
const LAND_METRES = 4600;
const RINGS = 200;
const SECTORS = 320;
/** Radial-grid exponent. Concentrates rings near the centre. */
const RING_POWER = 2.2;
const NORMAL_STEP_METRES = 0.8;

const SAGE = srgbToLinear(d.vec3f(0.34, 0.45, 0.2));
const OCHRE = srgbToLinear(d.vec3f(0.62, 0.52, 0.24));
const SCRUB = srgbToLinear(d.vec3f(0.2, 0.27, 0.12));
const TRODDEN = srgbToLinear(d.vec3f(0.52, 0.42, 0.3));
const ROCK = srgbToLinear(d.vec3f(0.5, 0.46, 0.42));
const ROUGHNESS = 0.95;
const CLIFF_SLOPE = 0.6;
const PATCH_METRES = 80;
const TUFT_METRES = 7;
const BLADE_METRES = 1.3;
/** Loop radius where hills and darker scrub begin. */
const SCRUB_FROM = 1.35;
/** Worn-track half-width in loop-width units. */
const TRACK_WIDTH = 0.16;

const ZENITH = srgbToLinear(d.vec3f(0.4, 0.56, 0.8));
/** The haze at the horizon, which the fog matches so the far hills fade into it. */
export const HAZE = srgbToLinear(d.vec3f(0.98, 0.8, 0.58));
const HORIZON_POWER = 0.38;
const SUN_GLOW = srgbToLinear(d.vec3f(1, 0.72, 0.4));
/** The sun's disc, in the lighting's sun colour times its intensity. */
const SUN_DISC = 9;

/**
 * Normalised oval radius at a position. One lies on the track, lower values inside and higher
 * values outside.
 */
export const loopAt = (x: number, z: number) => {
  'use gpu';
  const along = x / LOOP_METRES[0];
  const across = z / LOOP_METRES[1];
  return std.sqrt(along * along + across * across);
};

/** Warped ridge pattern from 0 to 1. */
const ridge = (x: number, z: number, alongX: number, alongZ: number) => {
  'use gpu';
  return std.abs(std.sin(x * alongX + z * alongZ + std.sin(z * alongX) * 1.5));
};

/** Height above the valley floor in metres, shared by CPU and GPU calculations. */
export const heightAt = (x: number, z: number) => {
  'use gpu';
  const r = std.sqrt(x * x + z * z);
  const angle = std.atan2(z, x);
  const loop = loopAt(x, z);

  const roll =
    std.sin(x * 0.021 + 1.3) * 0.8 +
    std.sin(z * 0.033 + x * 0.011) * 0.6 +
    std.sin((x + z) * 0.009 + 0.4) * 1.1;
  const knoll = KNOLL_METRES * (1 - std.smoothstep(0.12, 0.66, loop));
  const hills =
    (HILL_METRES +
      std.sin(angle * 5 + 2) * HILL_WAVE_METRES +
      ridge(x, z, 0.0071, 0.0043) * RIDGE_METRES[0] +
      ridge(x, z, -0.0052, 0.0113) * RIDGE_METRES[1] +
      ridge(x, z, 0.0163, -0.0121) * RIDGE_METRES[2] +
      ridge(x, z, 0.031, 0.027) * RIDGE_METRES[3]) *
    std.smoothstep(1.4, 2.6, loop);
  const range =
    (PEAK_METRES +
      std.sin(angle * 4 + 0.7) * PEAK_WAVES_METRES[0] +
      std.sin(angle * 9 + 1.9) * PEAK_WAVES_METRES[1]) *
    std.smoothstep(RANGE_METRES, RANGE_METRES + RANGE_RISE_METRES, r);
  return roll + knoll + hills + range;
};

/** Ground normal at a position. */
export const normalAt = (x: number, z: number) => {
  'use gpu';
  const dx = heightAt(x + NORMAL_STEP_METRES, z) - heightAt(x - NORMAL_STEP_METRES, z);
  const dz = heightAt(x, z + NORMAL_STEP_METRES) - heightAt(x, z - NORMAL_STEP_METRES);
  return std.normalize(d.vec3f(-dx, NORMAL_STEP_METRES * 2, -dz));
};

/**
 * Position on the polar grid at sector fraction `u` and ring fraction `v`, with rings concentrated
 * near the centre.
 */
const placeAt = (u: number, v: number) => {
  'use gpu';
  // The last sector meets the first exactly
  const angle = std.select(u, d.f32(0), u >= 1) * Math.PI * 2;
  return d.vec2f(std.cos(angle), std.sin(angle)) * (LAND_METRES * std.pow(v, RING_POWER));
};

const grassAt = (place: d.v2f, height: number, far: number) => {
  'use gpu';
  const patch = perlin2d.sample(place / PATCH_METRES);
  // Fade fine detail with distance before it shimmers
  const tuft = perlin2d.sample(place / TUFT_METRES) * (1 - std.smoothstep(150, 700, far));
  const blade = perlin2d.sample(place / BLADE_METRES) * (1 - std.smoothstep(25, 110, far));
  const loop = loopAt(place.x, place.y);
  let color = std.mix(SAGE, OCHRE, std.smoothstep(-0.3, 0.3, patch + tuft * 0.4));
  const scrub =
    std.smoothstep(SCRUB_FROM, SCRUB_FROM + 0.3, loop) *
    std.smoothstep(-0.1, 0.25, patch * 0.8 - tuft * 0.5);
  color = std.mix(color, SCRUB, scrub * 0.8);
  const worn = 1 - std.smoothstep(TRACK_WIDTH * 0.5, TRACK_WIDTH, std.abs(loop - 1));
  color = std.mix(color, TRODDEN, worn * (0.4 + tuft * 0.25));
  return color * (0.88 + blade * 0.24 + std.smoothstep(4, 16, height) * 0.06);
};

/** Grass worn along the track, with scrub and rock on the hills and far ranges. */
const Land = instances.define('Land', {
  material: (point) => {
    'use gpu';
    const cliff = 1 - std.smoothstep(CLIFF_SLOPE - 0.15, CLIFF_SLOPE, point.normal.y);
    const far = std.distance(point.position, cameras.current.$.position);
    const color = std.mix(grassAt(point.position.xz, point.position.y, far), ROCK, cliff);
    return materials.glossy(color, ROUGHNESS);
  },
});

/** Bakes the valley into a mesh of its own once and draws it with the scene's lighting. */
export function createLand(root: TgpuRoot) {
  const shape = meshes.parametric(
    {
      at: (u, v) => {
        'use gpu';
        const place = placeAt(u, v);
        return d.vec3f(place.x, heightAt(place.x, place.y), place.y);
      },
      normalAt: (u, v) => {
        'use gpu';
        const place = placeAt(u, v);
        return normalAt(place.x, place.y);
      },
    },
    { cols: SECTORS, rows: RINGS },
  );
  const pool = meshPool.create(root, meshPool.roomFor([shape]));
  const field = instances.create(root, pool, { kind: Land, capacity: 1 });
  field.add(pool.add(shape));
  field.flush();

  return {
    renderer: field.renderer(),

    destroy() {
      field.destroy();
      pool.destroy();
    },
  };
}

/**
 * The sky in a direction: haze at the horizon, blue overhead, and the glow and disc of the
 * lighting's sun.
 */
export const sky = (direction: d.v3f) => {
  'use gpu';
  const toSun = std.saturate(std.dot(direction, shading.towardsSun()));
  const up = std.saturate(direction.y);
  const color = std.mix(HAZE, ZENITH, std.pow(up, HORIZON_POWER));
  const glow = std.pow(toSun, 24) * 0.7 + std.pow(toSun, 400) * 3;
  return color + SUN_GLOW * glow + environment.sunDisc(direction, SUN_DISC);
};
