import { srgbToLinear } from '@typegpu/color';
import { instances, materials, meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { perlin3d } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';
import { TANK, type Water } from './water.ts';

const SAND = srgbToLinear(d.vec3f(0.75, 0.66, 0.48));
const SHADE = srgbToLinear(d.vec3f(0.45, 0.38, 0.28));
/** Ripple frequency per world unit along x and z. */
const RIPPLES = d.vec2f(0.8, 2.4);

const Sand = instances.define('Sand', {
  material: (point) => {
    'use gpu';
    const at = point.position;
    const ripples = perlin3d.sample(d.vec3f(at.xz * RIPPLES, 5)) * 0.5 + 0.5;

    return materials.matte(std.mix(SHADE, SAND, ripples));
  },
});

/**
 * Lays the sand as one plane instance. Nothing lies below it, so add it to scene passes without
 * a shadow pass.
 */
export function createFloor(root: TgpuRoot, water: Water) {
  const shape = meshes.plane({ width: TANK.x, depth: TANK.z });
  const pool = meshPool.create(root, meshPool.roomFor([shape]));
  const sand = instances.create(root, pool, { kind: Sand, capacity: 1 });
  sand.add(pool.add(shape));
  return {
    renderer: sand.renderer().pipe(water.inject()),
    flush: sand.flush,
    destroy() {
      sand.destroy();
      pool.destroy();
    },
  };
}
