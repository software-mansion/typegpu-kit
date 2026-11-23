import { cameras, images, instances, shading } from '@typegpu/kit';
import { unfilled, type Inject } from '@typegpu/kit/inject';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Picture } from '../contract.ts';

/** Ground-tile width and gravel depth in metres. */
const TILE = 2.5;
const DEPTH = 0.05;
/** Maximum ray steps, one per crossed texel. */
const STEPS = 32;
/** Minimum step in texture tiles, equal to one of 1024 texels. */
const FINEST = 1 / 1024;
/**
 * Minimum downward component of the unit view ray. Prevents grazing rays marching to the
 * horizon.
 */
const GRAZING = 0.05;
/** Flattening start and end distances in metres from the camera. Skip marching beyond the end. */
const MARCHED = { from: 12, to: 30 };
/** Sand-variation frequency in noise cells per metre and shade variation amplitude. */
const PATCHES = 0.03;
const PATCHINESS = 0.25;
/**
 * Samples at one of `OFFSETS` texture offsets. Noise at `SHUFFLE` cells per metre selects the
 * offset, blending through the middle `SEAM` of each transition to hide repetition.
 */
const SHUFFLE = 0.08;
const OFFSETS = 8;
const SEAM = 0.2;
/** Cached noise-grid side length in cells. Repeats beyond the grid. */
const CACHE = 128;
/** Stony and sandy ground presets with sand tint and gravel depth. */
const STONY = { tint: d.vec3f(0.74, 0.76, 0.8), depth: 1.4 };
const SANDY = { tint: d.vec3f(1.1, 1.0, 0.88), depth: 0.4 };

const missing = unfilled("the ground's texture", 'gravel', 'a `createGround`');
/** RGB albedo and alpha height, with 1 at the gravel top. */
const sand = tgpu.accessor(d.texture2d(d.f32), missing);
/** Anisotropic sampler for colour. Height uses trilinear sampling to limit samples per ray step. */
const colorSampler = tgpu.accessor(d.sampler(), missing);
const heightSampler = tgpu.accessor(d.sampler(), missing);

/** Two texture offsets and the blend fraction between them. */
const Shuffle = d.struct({ first: d.vec2f, second: d.vec2f, blend: d.f32 });

/** Precomputed texture offsets. */
const offsetOf = (index: number) => {
  'use gpu';
  randf.seed(index);
  return d.vec2f(randf.sample(), randf.sample());
};

const shuffleAt = (place: d.v2f) => {
  'use gpu';
  const pick = (perlin2d.sample(place * SHUFFLE) * 0.5 + 0.5) * OFFSETS;
  const first = std.floor(pick);
  return Shuffle({
    first: offsetOf(first),
    second: offsetOf(first + 1),
    blend: std.smoothstep(0.5 - SEAM / 2, 0.5 + SEAM / 2, pick - first),
  });
};

/** Samples `uv` through `sampler` at one offset, or blends two at a transition. */
const shuffled = (
  sampler: d.sampler,
  uv: d.v2f,
  dx: d.v2f,
  dy: d.v2f,
  shuffle: d.Infer<typeof Shuffle>,
) => {
  'use gpu';
  let at = uv + shuffle.first;
  if (shuffle.blend >= 1) at = uv + shuffle.second;
  const one = std.textureSampleGrad(sand.$, sampler, at, dx, dy);
  if (shuffle.blend <= 0 || shuffle.blend >= 1) return one;
  const other = std.textureSampleGrad(sand.$, sampler, uv + shuffle.second, dx, dy);
  return std.mix(one, other, shuffle.blend);
};

const heightAt = (uv: d.v2f, dx: d.v2f, dy: d.v2f, shuffle: d.Infer<typeof Shuffle>) => {
  'use gpu';
  return shuffled(heightSampler.$, uv, dx, dy, shuffle).a;
};

/** Ground type from -1, stony, to 1, sandy. Varies over metres to hundreds of metres. */
const strewn = (place: d.v2f) => {
  'use gpu';
  return (
    perlin2d.sample(place * 0.006) * 0.4 +
    perlin2d.sample(place * 0.022 + 17) * 0.25 +
    perlin2d.sample(place * 0.08 + 41) * 0.18 +
    perlin2d.sample(place * 0.3 + 73) * 0.17
  );
};

/**
 * Draws unrotated, unscaled ground at y = 0. Each pixel marches through the height map along
 * the view ray to find the gravel surface.
 */
export const Ground = instances.define('Ground', {
  surface: (pixel, base) => {
    'use gpu';
    const point = pixel.point;
    const start = point.position.xz / TILE;
    const dx = std.dpdx(start);
    const dy = std.dpdy(start);
    const texel = std.max(FINEST, std.max(std.length(dx), std.length(dy)));
    const shuffle = shuffleAt(point.position.xz);
    const sandy = std.saturate(strewn(point.position.xz) * 1.5 + 0.5);
    const gravel = DEPTH * std.mix(STONY.depth, SANDY.depth, sandy);

    // March below y = 0 through the gravel depth, moving by `shift` across the texture. Flatten
    // depth with distance
    const view = point.view;
    const away = std.distance(cameras.current.$.position, point.position);
    const relief = std.saturate((MARCHED.to - away) / (MARCHED.to - MARCHED.from));
    const shift = (view.xz / std.max(view.y, GRAZING)) * ((gravel * relief) / TILE);
    const steps = d.i32(std.min(std.ceil(std.length(shift) / texel), STEPS));

    let uv = d.vec2f(start);
    let depth = d.f32(0);
    let below = 1 - heightAt(uv, dx, dy, shuffle);
    let above = d.f32(below);
    const step = 1 / d.f32(std.max(steps, 1));
    for (let i = d.i32(0); i < steps; i++) {
      if (depth >= below) break;
      above = below - depth;
      uv -= shift * step;
      depth += step;
      below = 1 - heightAt(uv, dx, dy, shuffle);
    }
    // Interpolate the intersection between the last step above the gravel and the first below
    // it
    const after = below - depth;
    const met = std.saturate(
      std.select(d.f32(0), after / (after - above), std.abs(after - above) > 1e-5),
    );
    uv += shift * step * met;
    const height = std.select(1 - below, 1 - depth + step * met, steps > 0);

    const across = heightAt(uv + d.vec2f(texel, 0), dx, dy, shuffle) - height;
    const along = heightAt(uv + d.vec2f(0, texel), dx, dy, shuffle) - height;
    const rise = gravel / (texel * TILE);
    const normal = std.normalize(d.vec3f(-across * rise, 1, -along * rise));

    const texture = shuffled(colorSampler.$, uv, dx, dy, shuffle);
    const flat = uv * TILE;
    const position = d.vec3f(flat.x, (height - 1) * gravel * relief, flat.y);
    const patch = 1 + perlin2d.sample(position.xz * PATCHES) * PATCHINESS;
    const tint = std.mix(STONY.tint, SANDY.tint, sandy);
    const material = shading.Material(base);
    material.color = base.color * texture.rgb * tint * patch;

    const moved = shading.Point(point);
    moved.position = d.vec3f(position);
    moved.normal = d.vec3f(normal);
    moved.uv = d.vec2f(uv);
    return instances.Shaded({ point: moved, material });
  },
});

/** Creates a ground texture, mipmaps and sampler from an image spanning `TILE` metres. */
export function createGround(root: TgpuRoot, image: Picture) {
  const texture = images.create(root, image);
  const view = texture.createView(d.texture2d(d.f32));
  const forColor = images.sampler(root, { anisotropy: 16 });
  const forHeight = images.sampler(root);
  // Cache gradients because each pixel samples six noise octaves, twice with multisampling
  const noise = perlin2d.staticCache({ root, size: d.vec2u(CACHE, CACHE) });

  return {
    inject: (): Inject => (cfg) =>
      noise
        .inject()(cfg)
        .with(sand, () => view.$)
        .with(colorSampler, () => forColor.$)
        .with(heightSampler, () => forHeight.$),
    destroy() {
      texture.destroy();
      noise.destroy();
    },
  };
}
