/**
 * The render module format: what `KitGlslGenerator` writes into a shader module's code and what the
 * device reads back. The device depends only on the types here and on `parse`.
 */

/** A buffer binding a shader reads through its data texture. */
export interface BufferBinding {
  readonly group: number;
  readonly binding: number;
  readonly access: 'uniform' | 'readonly' | 'mutable';
  /** The `usampler2D` uniform that holds the buffer's words. */
  readonly texture: string;
  /** The `uvec2` uniform: `x` is the first word of the bound range, `y` its size in bytes. */
  readonly range: string;
}

/** One GLSL sampler uniform: a texture with the sampler a shader reads it through. */
export interface TextureBinding {
  readonly group: number;
  readonly binding: number;
  /** The GLSL sampler uniform. */
  readonly uniform: string;
  /** The sampler's binding. Absent when the shader only loads texels. */
  readonly sampler?: { readonly group: number; readonly binding: number };
  readonly dimension: '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';
  readonly sampleType: 'f32' | 'i32' | 'u32' | 'depth';
  /** True for a shadow sampler, which the device binds with a comparison function. */
  readonly comparison: boolean;
  readonly multisampled: boolean;
}

export interface RenderModule {
  readonly kind: 'render';
  /** Complete GLSL ES 3.00 vertex source, `#version` line included. */
  readonly vertex: string;
  /** Complete GLSL ES 3.00 fragment source. Absent for a depth-only pipeline's module. */
  readonly fragment: string | undefined;
  readonly buffers: readonly BufferBinding[];
  readonly textures: readonly TextureBinding[];
}

/** Uniforms with fixed names that the device sets when a program declares them. */
export const FIRST_INSTANCE = 'kit_firstInstance';
export const BASE_VERTEX = 'kit_baseVertex';
/** Defined by the device at the top of both stages when `EXT_clip_control` gives a 0 to 1 depth range. */
export const CLIP_ZERO_TO_ONE = 'KIT_CLIP_ZERO_TO_ONE';

export interface ParseOptions {
  /** Lines the device adds after `#version`, such as `#define KIT_CLIP_ZERO_TO_ONE`. Default none. */
  readonly defines?: readonly string[];
}

/** Reports whether `code` is a module `KitGlslGenerator` wrote. */
export function isRenderModule(code: string): boolean {
  return code.includes('//#kit-gl render');
}

/** Splits a render module into its stages and reads its bindings. */
export function parse(_code: string, _options: ParseOptions = {}): RenderModule {
  throw new Error('@typegpu/kit-gl: render modules are not implemented yet');
}
