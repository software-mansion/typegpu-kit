import { instances, materials, meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, std, type TgpuRoot } from 'typegpu';
import { mirrored } from './lake.ts';

/** Height of the rock the castle stands on, above the water. */
export const PLATEAU_METRES = 7;
const PLATEAU_RADIUS_METRES = 38;
const ISLAND_RADIUS_METRES = 62;
const LAKE_FLOOR_METRES = -9;
/** Far-shore shape around the lake. */
const SHORE_METRES = 430;
const SHORE_WAVES_METRES = [70, 35] as const;
const HILL_RISE_METRES = 260;
const HILL_METRES = 46;
const HILL_WAVES_METRES = [22, 11] as const;
const RANGE_METRES = 1500;
const RANGE_RISE_METRES = 700;
const PEAK_METRES = 190;
const PEAK_WAVES_METRES = [80, 45] as const;
/** Metres from the centre that the land reaches. */
const LAND_METRES = 3200;
const RINGS = 168;
const SECTORS = 288;
/** Radial-grid exponent. Concentrates rings near the island. */
const RING_POWER = 2.6;
const NORMAL_STEP_METRES = 0.6;
const GRASS = d.vec3f(0.03, 0.05, 0.028);
const ROCK = d.vec3f(0.07, 0.068, 0.07);
const MUD = d.vec3f(0.045, 0.04, 0.032);
const ROUGHNESS = 0.92;
const CLIFF_SLOPE = 0.55;

/** Height above water in metres, shared by CPU and GPU calculations. */
export const heightAt = (x: number, z: number) => {
  'use gpu';
  const r = std.sqrt(x * x + z * z);
  const angle = std.atan2(z, x);

  const crag = std.sin(angle * 5 + 1.3) * 1.2 + std.sin(angle * 11 + r * 0.05) * 0.7;
  const island =
    LAKE_FLOOR_METRES +
    (PLATEAU_METRES - LAKE_FLOOR_METRES + crag) *
      (1 - std.smoothstep(PLATEAU_RADIUS_METRES, ISLAND_RADIUS_METRES, r + crag * 2));

  const shore =
    SHORE_METRES +
    std.sin(angle * 3 + 0.4) * SHORE_WAVES_METRES[0] +
    std.sin(angle * 7 + 2.1) * SHORE_WAVES_METRES[1];
  const hills =
    (HILL_METRES +
      std.sin(angle * 5 + 2) * HILL_WAVES_METRES[0] +
      std.sin(angle * 13 + r * 0.004) * HILL_WAVES_METRES[1]) *
    std.smoothstep(shore, shore + HILL_RISE_METRES, r);
  const range =
    (PEAK_METRES +
      std.sin(angle * 3 + 2.6) * PEAK_WAVES_METRES[0] +
      std.sin(angle * 8 + 0.4) * PEAK_WAVES_METRES[1]) *
    std.smoothstep(RANGE_METRES, RANGE_METRES + RANGE_RISE_METRES, r);
  const banks = LAKE_FLOOR_METRES * (1 - std.smoothstep(shore - 60, shore, r));

  return std.max(island, banks + hills + range);
};

/**
 * Position on the polar grid at sector fraction `u` and ring fraction `v`, with rings concentrated
 * near the island.
 */
const placeAt = (u: number, v: number) => {
  'use gpu';
  // The last sector meets the first exactly
  const angle = std.select(u, d.f32(0), u >= 1) * Math.PI * 2;
  return d.vec2f(std.cos(angle), std.sin(angle)) * (LAND_METRES * std.pow(v, RING_POWER));
};

/** Grass and rock, wet mud at the waterline. */
const Land = instances.define('Land', {
  material: (pixel) => {
    'use gpu';
    const point = pixel.point;
    // Exclude underwater geometry from the reflection
    if (mirrored.$ && point.position.y < 0) std.discard();
    const cliff = 1 - std.smoothstep(CLIFF_SLOPE - 0.2, CLIFF_SLOPE, point.normal.y);
    const wet = 1 - std.smoothstep(0, 1.2, point.position.y);
    const color = std.mix(std.mix(GRASS, ROCK, cliff), MUD, wet);
    return materials.glossy(color, ROUGHNESS);
  },
});

/** Bakes island, lake bed and surrounding hills into a mesh once, lit by the scene's lighting. */
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
        const dx =
          heightAt(place.x + NORMAL_STEP_METRES, place.y) -
          heightAt(place.x - NORMAL_STEP_METRES, place.y);
        const dz =
          heightAt(place.x, place.y + NORMAL_STEP_METRES) -
          heightAt(place.x, place.y - NORMAL_STEP_METRES);
        return std.normalize(d.vec3f(-dx, NORMAL_STEP_METRES * 2, -dz));
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
