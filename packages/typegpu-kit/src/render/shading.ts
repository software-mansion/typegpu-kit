/**
 * These `'use gpu'` functions shade surfaces with the lighting injected into their pipeline, and
 * read its shadows.
 *
 * @module
 */

import * as cameras from '../cameras.ts';

import { d, std, tgpu } from 'typegpu';
import {
  beams,
  LightSettings,
  lightSettings,
  pcf,
  pointLights,
  pointsArriving,
  pointShadowing,
  shadowMap,
  shadowShape,
  shadowView,
} from './shared.ts';

/**
 * This schema describes a surface with the glTF metallic-roughness model. New fields are only
 * appended, each off at zero. Make one with a `materials` preset or copy one with `Material(m)`,
 * then assign the fields you set: a struct literal stops compiling when a field is added.
 */
export const Material = d.struct({
  /**
   * Sets the diffuse colour of a dielectric or the reflection colour of a metal, in linear
   * colour space.
   */
  color: d.vec3f,
  /** Sets roughness from 0 for a mirror to 1 for chalk. */
  roughness: d.f32,
  /** Sets the emitted light in linear colour space. Shading adds it after lighting. */
  emission: d.vec3f,
  /** Sets metallic content from 0 for a dielectric to 1 for a metal. */
  metallic: d.f32,
  /**
   * Sets the share of ambient light that reaches the surface, from 0 in a crevice to 1 in the
   * open, such as glTF's occlusion texture. `hemisphere` and an environment's `ambient` apply it.
   * The `materials` functions set 1.
   */
  occlusion: d.f32,
});

export type Material = d.InferGPU<typeof Material>;

/**
 * This schema describes a point being shaded in world space. New fields are only appended, each
 * off at zero. Make one with `pointAt`, a copy `Point(p)` or a zero `Point()`, then assign the
 * fields you set: a struct literal stops compiling when a field is added.
 */
export const Point = d.struct({
  position: d.vec3f,
  /** Holds a unit surface normal. */
  normal: d.vec3f,
  /** Holds a unit direction from the point towards the eye. */
  view: d.vec3f,
  /**
   * Holds texture coordinates at the point. A point created with `pointAt` has zero coordinates.
   */
  uv: d.vec2f,
  /** Holds the second uv set at the point, a mesh's `uvs1`, or zero. */
  uv1: d.vec2f,
});

export type Point = d.InferGPU<typeof Point>;

/** The largest finite half float, which `shade` keeps every channel at or below. */
const BRIGHTEST = 65504;

/** Creates a point at `position` with surface normal `normal`, viewed from the kit camera. */
export const pointAt = (position: d.v3f, normal: d.v3f) => {
  'use gpu';
  return Point({
    position,
    normal: std.normalize(normal),
    view: cameras.towards(cameras.current.$, position),
    uv: d.vec2f(),
    uv1: d.vec2f(),
  });
};

/**
 * Returns the sun, sky and fog settings for a custom `reflect`, `ambient` or `fog` function.
 * These are the values in `light.settings`. The sun direction can have any length, and
 * intensities are separate from colours.
 */
export const settings = () => {
  'use gpu';
  return LightSettings(lightSettings.$);
};

/**
 * Returns a unit direction towards the sun. A zero sun direction returns zero and provides no
 * light.
 */
export const towardsSun = () => {
  'use gpu';
  const direction = lightSettings.$.sunDirection;
  return direction / std.max(std.length(direction), 1e-6);
};

/**
 * Returns point-light attenuation at `distance` using inverse-square falloff. The factor `(1 -
 * (distance / range)^4)^2` eases it to zero at `range`, as in glTF and Unreal. Both distances
 * use world units. It runs on the CPU and in shaders.
 */
export const falloff = (distance: number, range: number) => {
  'use gpu';
  const ratio = distance / range;
  const edge = std.saturate(1 - ratio * ratio * ratio * ratio);

  return (edge * edge) / std.max(distance * distance, 1e-4);
};

/**
 * Returns the share of a spot light that reaches along `toLight`, a unit direction from the lit
 * point towards the light. The light aims along `aim`, of any length, with `ramp` from
 * `lighting.cone`. The share falls from 1 inside the inner half-angle to 0 outside the outer, as
 * glTF's spot lights do. A ramp of `[0, 1]` lights every direction. It runs on the CPU and in
 * shaders.
 */
export const beam = (toLight: d.v3f, aim: d.v3f, ramp: d.v2f) => {
  'use gpu';
  const axis = aim / std.max(std.length(aim), 1e-6);
  const share = std.saturate(std.dot(toLight, axis) * -ramp.x + ramp.y);
  return share * share;
};

/**
 * Returns the share of the light in `slot` that its cone sends towards `-toLight`: `beam` with the
 * aim and cone of a spot light's row or `points.spot`, and 1 for point lights.
 */
export const beamOf = (toLight: d.v3f, slot: number) => {
  'use gpu';
  if (beams.$ === null) return d.f32(1);
  return beams.$(toLight, slot);
};

/**
 * Computes reflected light with the glTF metallic-roughness model, using GGX specular and
 * diffuse lobes. Both lobes omit the 1/π factor, so a white matte surface facing a light with
 * radiance 1 has a brightness of about 1.
 */
export const pbr = (material: Material, point: Point, direction: d.v3f, radiance: d.v3f) => {
  'use gpu';
  const cosine = std.dot(point.normal, direction);
  if (cosine <= 0) return d.vec3f();

  const halfway = std.normalize(direction + point.view);
  const facing = std.max(std.dot(point.normal, point.view), 1e-4);
  const nh = std.saturate(std.dot(point.normal, halfway));
  const lh = std.saturate(std.dot(direction, halfway));

  const alpha = std.max(material.roughness * material.roughness, 0.002);
  const alpha2 = alpha * alpha;

  // Use the Trowbridge-Reitz distribution and height-correlated Smith visibility from Filament
  const spread = (nh * alpha2 - nh) * nh + 1;
  const distribution = alpha2 / (spread * spread);
  const visibility =
    0.5 /
    (cosine * std.sqrt(facing * facing * (1 - alpha2) + alpha2) +
      facing * std.sqrt(cosine * cosine * (1 - alpha2) + alpha2));

  const f0 = std.mix(d.vec3f(0.04), material.color, material.metallic);
  const fresnel = f0 + (d.vec3f(1) - f0) * std.pow(1 - lh, 5);

  const diffuse = material.color * (1 - material.metallic);
  const specular = fresnel * (distribution * visibility);

  return (diffuse + specular) * radiance * cosine;
};

/**
 * Computes diffuse reflection without highlights. Every surface, including metal, reflects light
 * in its own colour. It costs less than `pbr` and suits phones and flat-shaded art.
 */
export const lambert = (material: Material, point: Point, direction: d.v3f, radiance: d.v3f) => {
  'use gpu';
  const cosine = std.dot(point.normal, direction);
  // Keep this branch to skip lights that face away from the surface
  if (cosine <= 0) return d.vec3f();

  return material.color * radiance * cosine;
};

/**
 * Returns the split-sum reflectance using the Karis approximation. `f0` is reflectance at normal
 * incidence. `facing` is the cosine between the surface normal and view direction. Use this in a
 * custom `ambient` function, such as one that samples an environment image.
 */
export const reflectance = (f0: d.v3f, roughness: number, facing: number) => {
  'use gpu';
  const r = d.vec4f(-1, -0.0275, -0.572, 0.022) * roughness + d.vec4f(1, 0.0425, 1.04, -0.04);
  const bias = std.min(r.x * r.x, std.exp2(-9.28 * facing)) * r.x + r.y;
  const scale = d.vec2f(-1.04, 1.04) * bias + r.zw;

  return f0 * scale.x + scale.y;
};

/**
 * Returns the share of reflected ambient light that reaches the eye past an occlusion from 0 to 1,
 * from Lagarde's fit. `facing` is the cosine between the normal and the view. Use it beside
 * `reflectance` in a custom `ambient` function.
 */
export const specularOcclusion = (facing: number, occlusion: number, roughness: number) => {
  'use gpu';
  const alpha = roughness * roughness;
  return std.saturate(std.pow(facing + occlusion, std.exp2(-16 * alpha - 1)) - 1 + occlusion);
};

/**
 * Returns the direction to sample reflected light along for `surrounded`. It is the mirror
 * direction, leaned towards the normal as `roughness` grows, after Lagarde.
 */
export const reflectionOf = (point: Point, roughness: number) => {
  'use gpu';
  const alpha = roughness * roughness;
  const mirrored = std.reflect(point.view * -1, point.normal);
  const gloss = 1 - alpha;
  return std.normalize(std.mix(point.normal, mirrored, gloss * (std.sqrt(gloss) + alpha)));
};

/**
 * Shades a point by the light around it, for an ambient hook.
 *
 * `irradiance` is the light a white matte surface at the point reflects, as `environment.irradiance`
 * and a diffuse lightmap give it. It is used as given, so multiply it by `material.occlusion` if the
 * light doesn't already include occlusion.
 *
 * `reflection` is the light arriving along `reflectionOf`, as `environment.reflection` gives it.
 * It is weighted by `reflectance` and darkened by `specularOcclusion` of the material's
 * `occlusion`.
 */
export const surrounded = (
  material: Material,
  point: Point,
  irradiance: d.v3f,
  reflection: d.v3f,
) => {
  'use gpu';
  const facing = std.max(std.dot(point.normal, point.view), 1e-4);
  const f0 = std.mix(d.vec3f(0.04), material.color, material.metallic);
  const diffuse = material.color * (1 - material.metallic) * irradiance;
  const seen = specularOcclusion(facing, material.occlusion, material.roughness);
  return diffuse + reflectance(f0, material.roughness, facing) * reflection * seen;
};

/**
 * Computes ambient light by blending sky and ground colours according to the surface normal.
 * Reflections sample the sky in the reflection direction. Roughness blends them towards diffuse
 * light, and `skyReflection` scales them. The material's `occlusion` darkens both.
 */
export const hemisphere = (material: Material, point: Point) => {
  'use gpu';
  const env = lightSettings.$;
  const top = env.skyTop * env.skyIntensity;
  const bottom = env.skyBottom * env.skyIntensity;
  const reflected = std.reflect(point.view * -1, point.normal);

  const diffuseLight = std.mix(bottom, top, point.normal.y * 0.5 + 0.5);
  const mirrored = std.mix(bottom, top, std.saturate(reflected.y * 0.5 + 0.5));
  const specularLight = std.mix(mirrored, diffuseLight, material.roughness);

  return surrounded(
    material,
    point,
    diffuseLight * material.occlusion,
    specularLight * env.skyReflection,
  );
};

/**
 * Returns how much of the view fog hides between the camera and the point, from 0 to 1, from the
 * lighting's `fogDensity`, `fogFalloff` and `fogStart`. Density falls by a factor of e every `1 /
 * fogFalloff` world units above y = 0, and the fog begins `fogStart` world units from the eye.
 * A custom `fog` adds a colour of its own by it and lets `1 - haze` through.
 */
export const haze = (point: Point) => {
  'use gpu';
  const env = lightSettings.$;
  const rate = std.max(env.fogFalloff, 0);
  const camera = cameras.current.$.position;
  const sight = point.position - camera;
  const distance = std.length(sight);
  const start = std.max(env.fogStart, 0);
  if (distance <= start) return d.f32(0);
  // The fog begins `start` along the line of sight
  const eye = camera + sight * (start / distance);
  const ray = point.position - eye;

  // Measure the ray rise in fog heights to integrate density along it
  const climb = ray.y * rate;
  const thinning = std.select(
    (1 - std.exp(-climb)) / climb,
    1 - climb * 0.5,
    std.abs(climb) < 1e-3,
  );
  const depth = env.fogDensity * std.exp(-eye.y * rate) * std.length(ray) * thinning;
  return 1 - std.exp(-depth);
};

/**
 * This schema holds what the air between the eye and a point does to the point's light: `added`
 * is the light the air adds, and `through` the share of the point's light it lets through, per
 * channel. A colour `c` seen through it is `c * through + added`.
 */
export const Air = d.struct({ added: d.vec3f, through: d.vec3f });

export type Air = d.InferGPU<typeof Air>;

/**
 * Adds the lighting's `fogColor` by `haze` and lets the rest of the point's light through. This is
 * the default `fog`.
 */
export const heightFog = (point: Point) => {
  'use gpu';
  const hidden = haze(point);
  return Air({ added: lightSettings.$.fogColor * hidden, through: d.vec3f(1 - hidden) });
};

/** This schema holds the cascade a point lies in, -1 outside every one, and how near its edge. */
export const CascadeAt = d.struct({
  index: d.i32,
  /** Rises from 0 to 1 over the outer 10% of the cascade's square. */
  edge: d.f32,
});

/**
 * Returns the first cascade of the sun's shadow whose square holds the point, inset by the
 * filter's half width and the shadow's `lift`, and how near its edge the point lies. Index -1
 * means no cascade holds it, or the shadow is off.
 */
export const cascadeOf = (point: Point) => {
  'use gpu';
  if (shadowShape.$ === null) return CascadeAt({ index: -1, edge: 0 });
  const sun = shadowView.$;
  const inLight = (sun.light * d.vec4f(point.position, 1)).xyz;

  for (const k of tgpu.unroll(std.range(shadowShape.$.cascades))) {
    const cascade = sun.cascades[k];
    const clip = inLight * cascade.scale + cascade.offset;
    // A texel spans `texel * scale.x` in clip space. The filter reaches half its width out, from
    // the point lifted by up to `lift` texels
    const reach = (shadowShape.$.side + 1) * 0.5 + shadowShape.$.lift;
    const inside = 1 - reach * cascade.texel * cascade.scale.x;
    const across = std.max(std.abs(clip.x), std.abs(clip.y));
    if (across < inside && clip.z >= 0 && clip.z <= 1) {
      return CascadeAt({ index: k, edge: std.saturate((across / inside - 0.9) * 10) });
    }
  }
  return CascadeAt({ index: -1, edge: 0 });
};

/**
 * Returns the share of sunlight that passes the filter in cascade `index` of the sun's shadow. A
 * grid of depth comparisons gives 0 in full shadow and 1 in full light. For a point outside the
 * cascade's square or depth, an index without a cascade, or while the shadow is off, this returns
 * 1.
 */
export const cascadeShadowed = (point: Point, index: number) => {
  'use gpu';
  if (shadowShape.$ === null) return d.f32(1);
  if (index < 0 || index >= shadowShape.$.cascades) return d.f32(1);
  const cascade = shadowView.$.cascades[index];

  // Move by `lift` texels along the normal to prevent self-shadowing
  const lifted = point.position + point.normal * (cascade.texel * shadowShape.$.lift);
  const clip = (shadowView.$.light * d.vec4f(lifted, 1)).xyz * cascade.scale + cascade.offset;
  if (std.max(std.abs(clip.x), std.abs(clip.y)) > 1 || clip.z < 0 || clip.z > 1) return d.f32(1);

  const uv = clip.xy * d.vec2f(0.5, -0.5) + 0.5;
  const step = cascade.texel * cascade.scale.x * 0.5;
  return pcf(shadowShape.$.side)(shadowMap.$.map, uv, index, clip.z, step);
};

/**
 * Returns the share of sunlight that passes the shadow's filter, in the first cascade that holds
 * the point. Cascades switch without a blend. Without a shadow, outside its cascades or while it
 * is off, this returns 1. This is the default `sunShadow`.
 */
export const sunShadowed = (point: Point) => {
  'use gpu';
  return cascadeShadowed(point, cascadeOf(point).index);
};

/**
 * Returns the share of the light in `slot` that its shadow lets reach `point`, from 0 in shadow to
 * 1 in light. It is 1 for a light without a shadow and in a lighting without `pointShadow`.
 */
export const lampShadowed = (point: Point, toLight: d.v3f, distance: number, slot: number) => {
  'use gpu';
  if (pointShadowing.$ === null) return d.f32(1);
  return pointShadowing.$(point, toLight, distance, slot);
};

/** Computes reflection from one light direction, as `pbr` and `lambert` do. */
export type ReflectHook = (
  material: Material,
  point: Point,
  direction: d.v3f,
  radiance: d.v3f,
) => d.v3f;

/** Computes ambient light, as `hemisphere` does. */
export type AmbientHook = (material: Material, point: Point) => d.v3f;

/** Returns the air between the eye and the point, as `heightFog` does. */
export type FogHook = (point: Point) => Air;

/**
 * Returns the share of sunlight that reaches a point, as `sunShadowed` does. A value of 0
 * means full shadow and 1 means full light. Values above 1 can model focused light.
 */
export type SunShadowHook = (point: Point) => number;

/**
 * Returns the share of point-light intensity that reaches `point`, as `reaching` does. `toLight`
 * is a unit direction towards the light, and `distance` and `range` use world units, as in
 * `falloff`. `slot` identifies the light, so a hook can read data of its own for it.
 */
export type AttenuationHook = (
  point: Point,
  toLight: d.v3f,
  distance: number,
  range: number,
  slot: number,
) => number;

/**
 * Returns the share of a point or spot light that reaches `point`: `falloff` times `beamOf`, times
 * `lampShadowed` where that is above 0. This is the default `attenuation`.
 */
export const reaching = (
  point: Point,
  toLight: d.v3f,
  distance: number,
  range: number,
  slot: number,
) => {
  'use gpu';
  const share = falloff(distance, range) * beamOf(toLight, slot);
  if (share <= 0) return d.f32(0);
  return share * lampShadowed(point, toLight, distance, slot);
};

/**
 * This hook computes reflection from the sun and each point light. `shade` also calls `ambient`
 * once and `fog` last. The `sunShadow` and `attenuation` hooks control how much sunlight and point
 * light reach the surface.
 *
 * Lighting fills hooks named in its `shading` options. Override a hook after injecting lighting,
 * for example with `root.pipe(light.inject()).with(shading.reflect, shading.lambert)`.
 */
export const reflect = /* @__PURE__ */ tgpu.slot<ReflectHook>(pbr);
export const ambient = /* @__PURE__ */ tgpu.slot<AmbientHook>(hemisphere);
export const fog = /* @__PURE__ */ tgpu.slot<FogHook>(heightFog);
export const sunShadow = /* @__PURE__ */ tgpu.slot<SunShadowHook>(sunShadowed);
export const attenuation = /* @__PURE__ */ tgpu.slot<AttenuationHook>(reaching);

/**
 * Returns the share of sunlight that reaches a point through the `sunShadow` hook. A value of 0
 * means full shadow and 1 means full light. The default hook is `sunShadowed`. Use this in
 * shaders that do not call `shade`.
 */
export const sunShare = (point: Point) => {
  'use gpu';
  return sunShadow.$(point);
};

/** Computes sunlight reflected at the point through `sunShare` and `reflect`. */
export const sun = (material: Material, point: Point) => {
  'use gpu';
  const env = lightSettings.$;
  const radiance = env.sunColor * (env.sunIntensity * sunShare(point));
  return reflect.$(material, point, towardsSun(), radiance);
};

/**
 * Computes reflected light from the sun and point lights within range. This calls `reflect` and
 * omits ambient light, emission and fog.
 */
export const direct = (material: Material, point: Point) => {
  'use gpu';
  let color = sun(material, point);
  if (pointLights.$ !== null) color += pointLights.$(material, point);
  return color;
};

/** Computes light before fog, including ambient light, emission, sunlight and point lights. */
export const unfogged = (material: Material, point: Point) => {
  'use gpu';
  return ambient.$(material, point) + material.emission + direct(material, point);
};

/**
 * Computes the final surface colour. It combines ambient light, shadowed sunlight, live point
 * lights and emission, then applies fog. Every channel stays at or below 65 504, the largest half
 * float. Create the pipeline from `root.pipe(light.inject())` after injecting the kit camera.
 */
export const shade = (material: Material, point: Point) => {
  'use gpu';
  const lit = unfogged(material, point);
  const air = fog.$(point);
  return std.min(lit * air.through + air.added, d.vec3f(BRIGHTEST));
};

/**
 * Shades a clear surface such as glass or clear plastic. The result is premultiplied for
 * `pipelines.blend.over`. `opacity` scales diffuse light, emission and the light the fog adds.
 * Reflections, including highlights and sky light, contribute at full strength. A metal has only
 * reflected light, so `opacity` scales only its fog. Every channel stays at or below 65 504.
 */
export const shadeClear = (material: Material, point: Point, opacity: number) => {
  'use gpu';
  const covered = d.f32(opacity);
  // Scaling dielectric colour changes diffuse light without changing reflectance
  // Metal colour is its reflectance, so keep it unchanged
  const clear = Material(material);
  clear.color = material.color * std.mix(covered, 1, material.metallic);
  clear.emission = material.emission * covered;
  const lit = unfogged(clear, point);
  const air = fog.$(point);
  return d.vec4f(std.min(lit * air.through + air.added * covered, d.vec3f(BRIGHTEST)), covered);
};

/**
 * Returns the light arriving at `position` from every direction, for a particle or a volume lit
 * the same from all sides: the sun through `sunShadow`, each point light through `attenuation`,
 * and the mean of the sky and ground colours. Multiply a colour by it. Hooks receive a point with
 * a zero normal and view, so a lift along the normal does nothing. It runs in any stage, kernels
 * included, of a pipeline filled with `light.inject()`. Fill `sunShadow` with 0 on that pipeline to
 * leave the sun out.
 */
export const incoming = (position: d.v3f) => {
  'use gpu';
  const env = lightSettings.$;
  const point = Point();
  point.position = d.vec3f(position);
  const sky = (env.skyTop + env.skyBottom) * (0.5 * env.skyIntensity);
  let light = env.sunColor * (env.sunIntensity * sunShare(point)) + sky;
  if (pointsArriving.$ !== null) light += pointsArriving.$(point);
  return light;
};
