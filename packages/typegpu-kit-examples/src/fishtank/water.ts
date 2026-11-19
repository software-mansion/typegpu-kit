import { cameras, shading, type lighting } from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { perlin3d } from '@typegpu/noise';
import { d, std, type TgpuRoot, type TgpuUniform } from 'typegpu';

/** Deep-water and clear colour in linear light. */
export const DEEP = srgbToLinear(d.vec3f(0.01, 0.045, 0.07));
/** Surface-water colour in linear light. */
const SHALLOW = srgbToLinear(d.vec3f(0.07, 0.24, 0.3));
/** Extinction per world unit. One unit transmits `exp(-MURK)` of the light. */
const MURK = 0.06;

/** Tank dimensions in world units. x and z centre on the origin, and y rises from the sand. */
export const TANK = d.vec3f(16, 8, 10);

/** Caustic noise scale per world unit, drift per second, sharpness and maximum brightness. */
const CAUSTICS = { scale: 0.45, drift: 0.3, sharpness: 10, brightness: 2.2 };
/** The share of the sun's light that comes through between the caustics. */
const BETWEEN = 0.3;

/** Moving light patterns focused by the rippling water surface. */
export const caustics = (world: d.v3f, time: number) => {
  'use gpu';
  const p = d.vec3f(world.xz * CAUSTICS.scale, time * CAUSTICS.drift);
  const ridges = 1 - std.abs(perlin3d.sample(p) + perlin3d.sample(p * 2.1 + 7) * 0.5);

  return std.pow(std.saturate(ridges), CAUSTICS.sharpness) * CAUSTICS.brightness;
};

/** Distance fog tending toward brighter water near the surface. */
export const murk = (distance: number, height: number) => {
  'use gpu';
  const water = std.mix(DEEP, SHALLOW, std.saturate(height / TANK.y));
  const through = std.exp(-distance * MURK);

  return shading.Air({ added: water * (1 - through), through: d.vec3f(through) });
};

/**
 * Water shading for `lighting.create`. Caustics modulate sunlight, and fog uses the water
 * colour. Pipelines using this shading or `caustics` must pipe in `water.inject()`.
 */
export function createWater(root: TgpuRoot, time: TgpuUniform<d.F32>) {
  const noise = perlin3d.staticCache({ root, size: d.vec3u(16, 16, 32) });

  const water: lighting.Shading = {
    sunShadow: (point) => {
      'use gpu';
      return shading.sunShadowed(point) * (BETWEEN + caustics(point.position, time.$));
    },

    fog: (point) => {
      'use gpu';
      return murk(std.length(cameras.current.$.position - point.position), point.position.y);
    },
  };

  return {
    shading: water,
    inject: () => noise.inject(),
    destroy() {
      noise.destroy();
    },
  };
}

export type Water = ReturnType<typeof createWater>;
