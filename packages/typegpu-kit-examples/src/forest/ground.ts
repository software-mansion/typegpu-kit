import { instances, materials, type meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { perlin2d } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';

const MOSS = srgbToLinear(d.vec3f(0.24, 0.26, 0.1));
/** Leaf litter showing through the moss in patches. */
const LITTER = srgbToLinear(d.vec3f(0.42, 0.33, 0.17));

const Moss = instances.define('Moss', {
  material: (point) => {
    'use gpu';
    const at = point.position.xz;
    const patch = perlin2d.sample(at * 0.07) + perlin2d.sample(at * 0.3) * 0.5;
    return materials.matte(std.mix(MOSS, LITTER, std.smoothstep(0, 0.8, patch) * 0.7));
  },
});

/** Lays a mossy plain `size` metres across. */
export function createGround(root: TgpuRoot, pool: meshPool.MeshPool, options: { size: number }) {
  const { size } = options;
  const plain = instances.create(root, pool, { kind: Moss, capacity: 1 });
  plain.add(pool.add(meshes.plane({ width: size, depth: size })));
  plain.flush();
  return plain;
}
