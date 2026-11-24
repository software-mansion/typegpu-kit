/** Samplers as `WebGLSampler` objects, each with a variant that filters nothing. */

import { type Ctx, IMPL, implOf } from './context.ts';

export interface SamplerImpl {
  readonly descriptor: GPUSamplerDescriptor;
  readonly comparison: boolean;
  readonly filtering: boolean;
  readonly sampler: WebGLSampler;
  /** The same sampler with nearest filtering, for textures that cannot filter. */
  unfiltered(): WebGLSampler;
}

export function samplerImpl(sampler: GPUSampler): SamplerImpl {
  return implOf<SamplerImpl>(sampler, 'sampler');
}

export function compareFunc(gl: WebGL2RenderingContext, compare: GPUCompareFunction): number {
  switch (compare) {
    case 'never':
      return gl.NEVER;
    case 'less':
      return gl.LESS;
    case 'equal':
      return gl.EQUAL;
    case 'less-equal':
      return gl.LEQUAL;
    case 'greater':
      return gl.GREATER;
    case 'not-equal':
      return gl.NOTEQUAL;
    case 'greater-equal':
      return gl.GEQUAL;
    case 'always':
      return gl.ALWAYS;
  }
}

function wrap(gl: WebGL2RenderingContext, mode: GPUAddressMode | undefined): number {
  if (mode === 'repeat') return gl.REPEAT;
  if (mode === 'mirror-repeat') return gl.MIRRORED_REPEAT;
  return gl.CLAMP_TO_EDGE;
}

function minFilter(gl: WebGL2RenderingContext, min: boolean, mip: boolean): number {
  if (min) return mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR_MIPMAP_NEAREST;
  return mip ? gl.NEAREST_MIPMAP_LINEAR : gl.NEAREST_MIPMAP_NEAREST;
}

function make(ctx: Ctx, descriptor: GPUSamplerDescriptor, filter: boolean): WebGLSampler {
  const { gl, ext } = ctx;
  const sampler = gl.createSampler();
  gl.samplerParameteri(sampler, gl.TEXTURE_WRAP_S, wrap(gl, descriptor.addressModeU));
  gl.samplerParameteri(sampler, gl.TEXTURE_WRAP_T, wrap(gl, descriptor.addressModeV));
  gl.samplerParameteri(sampler, gl.TEXTURE_WRAP_R, wrap(gl, descriptor.addressModeW));
  const mag = filter && descriptor.magFilter === 'linear';
  const min = filter && descriptor.minFilter === 'linear';
  const mip = filter && descriptor.mipmapFilter === 'linear';
  gl.samplerParameteri(sampler, gl.TEXTURE_MAG_FILTER, mag ? gl.LINEAR : gl.NEAREST);
  gl.samplerParameteri(sampler, gl.TEXTURE_MIN_FILTER, minFilter(gl, min, mip));
  gl.samplerParameterf(sampler, gl.TEXTURE_MIN_LOD, descriptor.lodMinClamp ?? 0);
  gl.samplerParameterf(sampler, gl.TEXTURE_MAX_LOD, descriptor.lodMaxClamp ?? 32);
  if (descriptor.compare) {
    gl.samplerParameteri(sampler, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.samplerParameteri(sampler, gl.TEXTURE_COMPARE_FUNC, compareFunc(gl, descriptor.compare));
  }
  const anisotropy = descriptor.maxAnisotropy ?? 1;
  if (filter && anisotropy > 1 && ext.anisotropic) {
    const max = gl.getParameter(ext.anisotropic.MAX_TEXTURE_MAX_ANISOTROPY_EXT) as number;
    gl.samplerParameterf(
      sampler,
      ext.anisotropic.TEXTURE_MAX_ANISOTROPY_EXT,
      Math.min(anisotropy, max),
    );
  }
  return sampler;
}

export function createSampler(ctx: Ctx, descriptor: GPUSamplerDescriptor = {}): GPUSampler {
  const { gl } = ctx;
  const filtering =
    descriptor.magFilter === 'linear' ||
    descriptor.minFilter === 'linear' ||
    descriptor.mipmapFilter === 'linear';
  const sampler = make(ctx, descriptor, true);
  let unfiltered: WebGLSampler | undefined = filtering ? undefined : sampler;
  const impl: SamplerImpl = {
    descriptor: { ...descriptor },
    comparison: descriptor.compare !== undefined,
    filtering,
    sampler,
    unfiltered() {
      return (unfiltered ??= make(ctx, descriptor, false));
    },
  };
  ctx.track(() => {
    gl.deleteSampler(sampler);
    if (unfiltered && unfiltered !== sampler) gl.deleteSampler(unfiltered);
  });
  return {
    [IMPL]: impl,
    __brand: 'GPUSampler',
    label: descriptor.label ?? '',
  } as unknown as GPUSampler;
}
