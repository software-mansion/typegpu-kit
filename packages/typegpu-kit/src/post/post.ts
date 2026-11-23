/**
 * These types and functions read a picture another pass rendered, as every post-processing
 * `source` does.
 *
 * @module
 */

import { std, type d } from 'typegpu';
import type { FloatTexture, SizeSource } from '../targets.ts';

export type { FloatTexture } from '../targets.ts';

/**
 * This type is a picture another pass rendered: a texture and the part of it that holds the
 * picture, from its top-left corner. Bloom, TAA, depth of field and AO provide one, and every
 * post-processing `source` accepts one. Readers take `output` and `region` together once at every
 * recording, and use them for all of it.
 */
export interface Rendered {
  readonly output: FloatTexture;
  /** Selects the part of `output` that holds the picture. Default the whole texture. */
  readonly region?: SizeSource | undefined;
}

/**
 * This type is what a `source` option accepts: a function returning the texture to read, such as
 * `() => screen.textures.hdr`, or a picture another module rendered, which brings its region.
 */
export type Source = (() => FloatTexture) | Rendered;

/**
 * Turns a `source` option into a function returning its texture. Call it once at creation and
 * the function at every recording, since a resize replaces the texture.
 *
 * @param source a `source` option
 * @returns a function returning the texture to read now
 */
export const textureOf = (source: Source): (() => FloatTexture) =>
  typeof source === 'function' ? source : () => source.output;

/**
 * Picks the `region` option, else the region of a rendered `source`.
 *
 * @param source a `source` option
 * @param region a `region` option
 * @returns the part of the texture that holds the picture, or undefined for all of it
 */
export const regionOf = (source: Source, region: SizeSource | undefined): SizeSource | undefined =>
  region ?? (typeof source === 'function' ? undefined : source.region);

/**
 * Stores the size of the part of `texture` that `region` selects, at most the texture, in `size`.
 *
 * @param size where the width and height in texels go
 * @param texture the texture a source returned
 * @param region the region `regionOf` picked
 * @returns whether the size changed
 */
export const fitRegion = (
  size: number[],
  texture: FloatTexture,
  region: SizeSource | undefined,
): boolean => {
  const [width, height = 1] = texture.props.size;
  const fitWidth = Math.min(width, region?.width ?? width);
  const fitHeight = Math.min(height, region?.height ?? height);
  const changed = size[0] !== fitWidth || size[1] !== fitHeight;
  size[0] = fitWidth;
  size[1] = fitHeight;
  return changed;
};

/** Clamps texture coordinates to a region in texels, so filtering reads nothing beyond its edge. */
export const within = (at: d.v2f, region: d.v2f, size: d.v2f) => {
  'use gpu';
  return std.clamp(at, 0.5 / size, (region - 0.5) / size);
};
