/**
 * Fills a texture's mip levels in your encoder, each texel combining a 2×2 block of the level
 * above with a `'use gpu'` function: a mean for sampling, the farthest depth for occlusion.
 *
 * @module
 */

import {
  d,
  std,
  tgpu,
  type SampledFlag,
  type TgpuCommandEncoder,
  type TgpuComputePass,
  type TgpuRoot,
  type TgpuTexture,
} from 'typegpu';
import { fill, floatLayers, one } from './fill.ts';

/** This type combines four texels, left to right and then top to bottom, into one. */
export type Combine = (a: d.v4f, b: d.v4f, c: d.v4f, e: d.v4f) => d.v4f;

/** Averages the four texels, a box filter for textures you sample. */
export const mean: Combine = (a, b, c, e) => {
  'use gpu';
  return (a + b + c + e) * 0.25;
};

/** Keeps the largest of the four per channel, such as the farthest depth over a block. */
export const largest: Combine = (a, b, c, e) => {
  'use gpu';
  return std.max(std.max(a, b), std.max(c, e));
};

/** Keeps the smallest of the four per channel, such as the nearest depth over a block. */
export const smallest: Combine = (a, b, c, e) => {
  'use gpu';
  return std.min(std.min(a, b), std.min(c, e));
};

export interface Options {
  /** Combines a 2×2 block of the level above into one texel. @gpu */
  combine: Combine;
  /**
   * Fills level 0 from this texture's 2×2 blocks too, such as the depth a pyramid of farthest
   * depths starts from. It has as many array layers, and blocks past its edge repeat its last
   * texels. Default leaving level 0 to you.
   */
  source?: TgpuTexture & SampledFlag;
}

/** This type fills a texture's mip levels each time it records. */
export interface Mipmaps {
  /**
   * Records the levels in order. A texture with `storage` usage fills in a compute pass, the
   * pass given or one of its own on the encoder. Another fills in render passes on the encoder.
   */
  record(target: TgpuCommandEncoder | TgpuComputePass): void;
  /** Compiles the pipeline. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Returns mipmaps of another texture of this one's format and usage, filled through the same
   * pipeline from its own `source` or level 0, with their bind groups made now: the other texture
   * of a pair, or the one a resize puts in this one's place.
   */
  of(texture: TgpuTexture & SampledFlag, source?: TgpuTexture & SampledFlag): Mipmaps;
}

/** Throws unless the levels can read `from`, which `what` names in the message. */
function sampled(from: TgpuTexture, what: string) {
  if (!from.usableAsSampled) throw new Error(`${what} requires 'sampled' usage.`);
}

/**
 * Creates what fills a texture's mip levels, every array layer of each, from the level above. A
 * level of odd size leaves the last texels of the one above out.
 *
 * @owns a pipeline, a bind group per level: the kernel or the draw that fills a level, and the
 * views each level reads and writes
 * @records `record` after whatever writes level 0 or `source`, before what reads the levels
 * @param texture a 2D float colour texture with `storage` or `render` usage, and `sampled` usage
 * @returns mipmaps that record into the texture
 */
export function create(
  root: TgpuRoot,
  texture: TgpuTexture & SampledFlag,
  options: Options,
): Mipmaps {
  const { combine, source } = options;
  const read = tgpu.bindGroupLayout({
    above: { texture: d.texture2dArray(d.f32), sampleType: 'unfilterable-float' },
  });
  const above = (from: TgpuTexture & SampledFlag, view = {}) =>
    root.createBindGroup(read, {
      above: from.createView(d.texture2dArray(d.f32), {
        ...view,
        sampleType: 'unfilterable-float',
        // A depth-stencil format binds its depth alone
        ...(from.props.format.includes('stencil') && { aspect: 'depth-only' as const }),
      }),
    });
  const block = (texel: d.v2u, layer: number) => {
    'use gpu';
    const last = d.vec2i(std.textureDimensions(read.$.above)) - 1;
    const near = std.min(d.vec2i(texel) * 2, last);
    const far = std.min(d.vec2i(texel) * 2 + 1, last);
    return combine(
      std.textureLoad(read.$.above, near, layer, 0),
      std.textureLoad(read.$.above, d.vec2i(far.x, near.y), layer, 0),
      std.textureLoad(read.$.above, d.vec2i(near.x, far.y), layer, 0),
      std.textureLoad(read.$.above, far, layer, 0),
    );
  };
  const filling = fill(root, texture, block);
  // Level k reads level k - 1, and level 0 reads `source`
  const of = (target: TgpuTexture & SampledFlag, from?: TgpuTexture & SampledFlag): Mipmaps => {
    const filled = filling(target, (layers) => {
      sampled(target, 'A texture whose mip levels fill');
      const steps = [];
      if (from) {
        const count = floatLayers(from, 'A mipmaps source');
        sampled(from, 'A mipmaps source');
        if (count !== layers) {
          throw new Error(
            `The mipmaps source has ${count} array layers. It must match the destination texture, which has ${layers}.`,
          );
        }
        steps.push({ level: 0, group: above(from) });
      }
      const levels = target.props.mipLevelCount ?? 1;
      for (let level = 1; level < levels; level++) {
        steps.push({ level, group: above(target, one(level - 1)) });
      }
      return steps;
    });
    return { ...filled, of };
  };
  return of(texture, source);
}
