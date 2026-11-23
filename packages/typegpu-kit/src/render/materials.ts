/**
 * These `'use gpu'` functions create materials for `shading.shade`, with colours or mesh
 * textures.
 *
 * @module
 */

import { d, std } from 'typegpu';
import { Material, type Point } from './shading.ts';
import { keepsSecondSet, layout, type TextureUvs } from './textures.ts';

// Cast whole-number literals to prevent i32 inference in GPU calls

/** The variance of normals across a pixel, and the most it widens α², as Filament's defaults. */
const SPREAD = 0.15;
const WIDEST = 0.2;

/** Creates a diffuse material, such as chalk, clay or cloth. */
export const matte = (color: d.v3f) => {
  'use gpu';
  return Material({ color, roughness: 1, emission: d.vec3f(), metallic: 0, occlusion: 1 });
};

/** Creates a dielectric with specular highlights, such as paint, plastic or wet stone. */
export const glossy = (color: d.v3f, roughness: number) => {
  'use gpu';
  return Material({
    color,
    roughness: d.f32(roughness),
    emission: d.vec3f(),
    metallic: 0,
    occlusion: 1,
  });
};

/** Creates a metal that tints its reflections and has no diffuse colour. */
export const metal = (color: d.v3f, roughness: number) => {
  'use gpu';
  return Material({
    color,
    roughness: d.f32(roughness),
    emission: d.vec3f(),
    metallic: 1,
    occlusion: 1,
  });
};

/**
 * Creates an emissive material that gives off `strength` times `color` and reflects almost
 * nothing.
 */
export const glow = (color: d.v3f, strength: number) => {
  'use gpu';
  return Material({
    color: d.vec3f(),
    roughness: 1,
    emission: color * d.f32(strength),
    metallic: 0,
    occlusion: 1,
  });
};

/**
 * Gives the coordinates a texture reads: its uv set, `uv` or `uv1`, moved by its transform. Take a
 * mesh texture's from the reading, such as `meshPool.textureLayout.$.reading.uvs.color`. Only
 * pipelines piped with a pool made with `uvs1` read `uv1`; the rest compile the choice out.
 */
export const uvOf = (uvs: d.Infer<typeof TextureUvs>, uv: d.v2f, uv1: d.v2f) => {
  'use gpu';
  let picked = d.vec2f(uv);
  if (keepsSecondSet.$) picked = std.select(uv, uv1, uvs.uvSet === 1);
  const at = d.vec3f(picked, 1);
  return d.vec2f(std.dot(uvs.u, at), std.dot(uvs.v, at));
};

/**
 * Samples the mesh colour texture where its reading says, once for both `masked` and `textured`.
 *
 * @hidden
 */
export const colorAt = (uv: d.v2f, uv1: d.v2f) => {
  'use gpu';
  const at = uvOf(layout.$.reading.uvs.color, uv, uv1);
  return std.textureSample(layout.$.color, layout.$.filtering, at);
};

/**
 * Returns whether colour texture alpha `alpha` times `opacity` falls below the alpha cutoff.
 *
 * @hidden
 */
export const cutBelow = (alpha: number, opacity: number) => {
  'use gpu';
  return alpha * opacity < layout.$.reading.alphaCutoff;
};

/**
 * Applies mesh textures to `material` as `textured` does, given the colour sample.
 *
 * @hidden
 */
export const texturedBy = (material: Material, point: Point, colored: d.v4f) => {
  'use gpu';
  const uvs = layout.$.reading.uvs;
  const surface = std.textureSample(
    layout.$.metallicRoughness,
    layout.$.filtering,
    uvOf(uvs.metallicRoughness, point.uv, point.uv1),
  );
  const emitted = std.textureSample(
    layout.$.emission,
    layout.$.filtering,
    uvOf(uvs.emission, point.uv, point.uv1),
  );
  const sampled = std.textureSample(
    layout.$.occlusion,
    layout.$.filtering,
    uvOf(uvs.occlusion, point.uv, point.uv1),
  ).r;
  const open = 1 + layout.$.reading.occlusionStrength * (sampled - 1);

  const made = Material(material);
  made.color = material.color * colored.rgb;
  made.roughness = material.roughness * surface.g;
  made.emission = material.emission * emitted.rgb;
  made.metallic = material.metallic * surface.b;
  made.occlusion = material.occlusion * open;
  return made;
};

/**
 * Applies mesh textures to `material` at `point`, each where `uvOf` says. Colour, roughness,
 * metallic, emission and occlusion textures multiply their corresponding factors. Occlusion is
 * the red channel of its texture, which glTF often packs into the metallic-roughness image, eased
 * towards 1 by `occlusionStrength` below 1. Call this only in a fragment shader with
 * `pool.texturesOf(index)` bound for the mesh.
 */
export const textured = (material: Material, point: Point) => {
  'use gpu';
  return texturedBy(material, point, colorAt(point.uv, point.uv1));
};

/**
 * Returns `material` with its roughness widened where `normal` turns within a pixel, so a smooth
 * surface's highlight covers whole pixels and holds steady as it moves, as Filament's specular
 * antialiasing does. Kit renderers apply it unless `specularAntialiasing` is false. Call it only in
 * a fragment shader, in uniform control flow, with the normal the shading reads.
 */
export const antialiased = (material: Material, normal: d.v3f) => {
  'use gpu';
  const across = std.dpdx(normal);
  const down = std.dpdy(normal);
  const variance = SPREAD * (std.dot(across, across) + std.dot(down, down));
  const alpha = material.roughness * material.roughness;
  const widened = std.saturate(alpha * alpha + std.min(variance * 2, WIDEST));
  const smoothed = Material(material);
  smoothed.roughness = std.sqrt(std.sqrt(widened));
  return smoothed;
};

/**
 * Returns whether a pixel falls below the alpha cutoff for a glTF `alpha: 'mask'` material. It
 * compares the colour texture alpha where `uvOf` says, from `uv` and `uv1`, multiplied by
 * `opacity`, with the cutoff. Other materials return false. Discard the pixel when this returns
 * true. Call this only in a fragment shader with `pool.texturesOf(index)` bound for the mesh.
 */
export const masked = (uv: d.v2f, uv1: d.v2f, opacity: number) => {
  'use gpu';
  return cutBelow(colorAt(uv, uv1).a, opacity);
};

/**
 * Applies the mesh normal texture to `normal`, scaled by `normalScale`. Derivatives of
 * `position` and the coordinates the texture reads, from `uv` and `uv1` as `uvOf` says, form the
 * tangent frame, so the mesh needs no tangents. Call this only in a fragment shader with
 * `pool.texturesOf(index)` bound for the mesh.
 */
export const mappedNormal = (normal: d.v3f, position: d.v3f, uv: d.v2f, uv1: d.v2f) => {
  'use gpu';
  // The pool writes a scale of 0 for a material without a normal texture, which bends nothing
  const strength = layout.$.reading.normalScale;
  if (strength === 0) return std.normalize(normal);

  // Build the tangent frame from the coordinates the normal texture reads, which its transform turns
  const at = uvOf(layout.$.reading.uvs.normal, uv, uv1);
  const sampled = std.textureSample(layout.$.normal, layout.$.filtering, at).xyz * 2 - 1;
  const bent = sampled * d.vec3f(strength, strength, 1);

  // Use the cotangent frame from Schüler
  // WebGPU screen y and glTF v increase downwards, but glTF normal +y points up the image
  const perpX = std.cross(normal, std.dpdx(position));
  const perpY = std.cross(std.dpdy(position), normal);
  const tangent = perpY * std.dpdx(at).x + perpX * std.dpdy(at).x;
  const bitangent = perpY * std.dpdx(at).y + perpX * std.dpdy(at).y;
  const scale = std.inverseSqrt(
    std.max(std.max(std.dot(tangent, tangent), std.dot(bitangent, bitangent)), 1e-20),
  );

  return std.normalize((bitangent * bent.y - tangent * bent.x) * scale + normal * bent.z);
};
