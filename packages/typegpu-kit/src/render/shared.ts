import * as bindings from '../bindings.ts';
import { d, std, tgpu } from 'typegpu';
import type { Material, Point } from './shading.ts';

// Keep shader slots, schemas and shadow sampling here, so shader modules import neither the
// lighting factory nor its tables

/**
 * This schema packs the sun, sky and fog into 16-byte rows. Each `vec3f` shares a row with a
 * scalar, and `fogStart` starts a row of its own.
 */
export const LightSettings = d.struct({
  /** Points towards the sun. It can have any length; `shading` normalizes it. */
  sunDirection: d.vec3f,
  /** Multiplies `sunColor`. */
  sunIntensity: d.f32,
  /** Sets the sun colour in linear colour space. */
  sunColor: d.vec3f,
  /** Multiplies `skyTop` and `skyBottom`. */
  skyIntensity: d.f32,
  /** Sets the linear sky colour from directly above. */
  skyTop: d.vec3f,
  /** Sets fog density per world unit at y = 0. */
  fogDensity: d.f32,
  /** Sets the linear ground colour from directly below. */
  skyBottom: d.vec3f,
  /**
   * Sets how fog density falls with height. Density falls by a factor of e every `1 /
   * fogFalloff` world units. A value of 0 gives uniform fog.
   */
  fogFalloff: d.f32,
  /** Sets the fog colour in linear colour space. */
  fogColor: d.vec3f,
  /**
   * Scales sky reflections in addition to `skyIntensity`. Values above 1 brighten reflections at
   * the edges of glass and clear plastic without increasing diffuse sky light.
   */
  skyReflection: d.f32,
  /** Sets the distance from the eye in world units where fog begins. */
  fogStart: d.f32,
});

export const lightSettings = tgpu.accessor(LightSettings, () => {
  throw new Error(
    'The pipeline is missing its lighting settings. Give the pass description `beneath(light.inject())` or create the pipeline from `root.pipe(light.inject())`. Use the lighting instance from `lighting.create` as `light`.',
  );
});

// Pack position and range into one 16-byte row for the range test
export const pointLightFields = {
  position: d.vec3f,
  /** Sets the distance in world units where light intensity reaches zero. */
  range: d.f32,
  /** Sets the light colour in linear colour space. */
  color: d.vec3f,
  /** Multiplies `color`. */
  intensity: d.f32,
};

/** This schema holds a point light with a position, range, linear colour and intensity. */
export const Light = d.struct(pointLightFields);
export type Light = d.InferGPU<typeof Light>;

// The cone shares a 16-byte row with nothing, so a spot light's record is 64 bytes
export const spotLightFields = {
  ...pointLightFields,
  /** Points the cone's axis away from the light. It can have any length. */
  aim: d.vec3f,
  /**
   * Holds the cone's scale and offset from `lighting.cone(inner, outer)`. The default `[0, 1]`
   * lights every direction.
   */
  cone: d.vec2f,
};

/** This schema holds a spot light: a point light with an aim and a cone. */
export const Spot = d.struct(spotLightFields);
export type Spot = d.InferGPU<typeof Spot>;

/**
 * Holds the point-light loop for a table or a light function, reflected by `reflect`, or null
 * without point lights.
 */
export const pointLights = tgpu.slot<((material: Material, point: Point) => d.v3f) | null>(null);

/** Holds the point-light loop that sums the light arriving at a point, or null without one. */
export const pointsArriving = tgpu.slot<((point: Point) => d.v3f) | null>(null);

/** Holds the share of a light's cone towards `-toLight`, or null when no light has a cone. */
export const beams = tgpu.slot<((toLight: d.v3f, slot: number) => number) | null>(null);

/** Holds the share of a light that reaches a point past its shadow, or null without shadows. */
export const pointShadowing = tgpu.slot<
  ((point: Point, toLight: d.v3f, distance: number, slot: number) => number) | null
>(null);

interface ShadowShape {
  /** Sets the number of comparisons along each side of the filter. It is 1, 2 or 3. */
  readonly side: number;
  /** Sets how far to move a shaded point along its normal before comparison, in texels. */
  readonly lift: number;
  /** Sets the number of cascades, 1 to 4, each a layer of the map. */
  readonly cascades: number;
}

/**
 * Configures the sun's shadow filter and cascades at shader compile time so their loops can
 * unroll, or null without a shadow, which shaders then leave out. Shaders read where each cascade
 * lies from `shadowView`, which can change without creating pipelines.
 */
export const shadowShape = tgpu.slot<ShadowShape | null>(null);

/**
 * Places one cascade of the sun's shadow. A point `p` in light space lies at `p * scale + offset`
 * in the cascade's clip space, with depth from 0 towards the sun to 1 away from it.
 */
const Cascade = d.struct({
  scale: d.vec3f,
  /** Holds the width of one texel in world units. */
  texel: d.f32,
  offset: d.vec3f,
});

/** Holds the sun's rotation into light space and the place of each cascade. */
export const SunView = d.struct({ light: d.mat4x4f, cascades: d.arrayOf(Cascade, 4) });

/** Provides the sun's light space and its cascades. */
export const shadowView = tgpu.accessor(SunView);
/**
 * Binds the depth map from the sun, a layer per cascade. An app can share these bindings in its
 * own layout.
 */
export const shadowMap = bindings.layout({ map: { texture: d.textureDepth2dArray() } });
export const shadowSampler = tgpu.accessor(d.comparisonSampler());

/**
 * Returns the filter averaging `side × side` comparisons of a shadow map, `step` apart around
 * `uv`, each of which the sampler filters over four texels. Call it in a shader with a side known
 * at resolution: `pcf(side)(map, uv, layer, depth, step)`.
 */
export const pcf = /* @__PURE__ */ tgpu.comptime((side: number) => {
  const percentageCloser = (
    map: d.textureDepth2dArray,
    uv: d.v2f,
    layer: number,
    depth: number,
    step: number,
  ) => {
    'use gpu';
    let lit = d.f32(0);
    for (const y of tgpu.unroll(std.range(side))) {
      for (const x of tgpu.unroll(std.range(side))) {
        const offset = (d.vec2f(x, y) - (side - 1) / 2) * step;
        lit += std.textureSampleCompareLevel(map, shadowSampler.$, uv + offset, layer, depth);
      }
    }
    return lit / (side * side);
  };
  return percentageCloser;
});
