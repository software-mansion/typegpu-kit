/**
 * Bakes a `'use gpu'` function of a texel's position into a texture, in your encoder or compute
 * pass: a procedural picture, a distance field, a lookup table.
 *
 * @module
 */

import { d, type TgpuCommandEncoder, type TgpuRoot, type TgpuTexture } from 'typegpu';
import { fill } from './fill.ts';
import type { AnyComputePass } from './passes.ts';

export interface Options {
  /**
   * Returns the texel at `uv`, from 0 to 1 across the texture from its top-left corner, at texel
   * centres, and of array layer `layer`.
   * @gpu
   */
  texel: (uv: d.v2f, layer: number) => d.v4f;
}

/** This type writes a texture's texels each time it records. */
export interface Baker {
  /**
   * Records the texels. A texture with `storage` usage fills in a compute pass, the pass given or
   * one of its own on the encoder. Another fills in a render pass per array layer on the encoder.
   */
  record(target: TgpuCommandEncoder | AnyComputePass): void;
  /** Compiles the pipeline. Calling it is optional. */
  initAsync(): Promise<void>;
}

/**
 * Creates what writes `texel` into the first mip level of a texture, every array layer of it.
 *
 * @owns a pipeline, 0 or 1 bind groups: the kernel or the draw that writes the texels, and the
 * storage view a kernel writes them through
 * @records `record` before what reads the texture, each time what `texel` reads changes
 * @param texture a 2D float colour texture with `storage` usage, or `render` usage
 * @returns a baker that records into the texture
 */
export function create(root: TgpuRoot, texture: TgpuTexture, options: Options): Baker {
  const { texel } = options;
  const size: readonly number[] = texture.props.size;
  const [width, height = 1] = size;
  const centred = (at: d.v2u, layer: number) => {
    'use gpu';
    return texel((d.vec2f(at) + 0.5) / d.vec2f(width, height), layer);
  };
  return fill(root, texture, centred)(texture, () => [{ level: 0 }]);
}
