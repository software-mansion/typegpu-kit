/** WebGPU's limits and features as the GL context and its extensions allow. */

import type { Extensions } from './extensions.ts';
import type { FormatTable } from './formats.ts';

export function limitsOf(gl: WebGL2RenderingContext): GPUSupportedLimits {
  const get = (name: number) => gl.getParameter(name) as number;
  const textureSize = get(gl.MAX_TEXTURE_SIZE);
  const units = get(gl.MAX_TEXTURE_IMAGE_UNITS);
  const colors = Math.min(get(gl.MAX_DRAW_BUFFERS), get(gl.MAX_COLOR_ATTACHMENTS), 8);
  // Buffers are data textures of up to 1024 texels a row, 16 bytes a texel
  const bufferSize = Math.min(2 ** 28, 1024 * 16 * textureSize);
  const limits = {
    __brand: 'GPUSupportedLimits',
    maxTextureDimension1D: textureSize,
    maxTextureDimension2D: textureSize,
    maxTextureDimension3D: get(gl.MAX_3D_TEXTURE_SIZE),
    maxTextureArrayLayers: get(gl.MAX_ARRAY_TEXTURE_LAYERS),
    maxBindGroups: 4,
    maxBindGroupsPlusVertexBuffers: 24,
    maxBindingsPerBindGroup: 1000,
    maxDynamicUniformBuffersPerPipelineLayout: 8,
    maxDynamicStorageBuffersPerPipelineLayout: 4,
    maxSampledTexturesPerShaderStage: units,
    maxSamplersPerShaderStage: units,
    // Kernels run on the CPU or as fragment passes, so storage bindings cost no WebGPU slots: a
    // program that needs more texture units or uniform blocks than the context has fails to link
    // with a message naming the counts, and a GPU form that does not fit runs on the CPU
    maxStorageBuffersPerShaderStage: 16,
    maxStorageBuffersInVertexStage: 16,
    maxStorageBuffersInFragmentStage: 16,
    maxStorageTexturesPerShaderStage: 8,
    maxStorageTexturesInVertexStage: 0,
    maxStorageTexturesInFragmentStage: 0,
    maxUniformBuffersPerShaderStage: Math.max(
      12,
      Math.min(get(gl.MAX_VERTEX_UNIFORM_BLOCKS), get(gl.MAX_FRAGMENT_UNIFORM_BLOCKS)),
    ),
    maxUniformBufferBindingSize: 65536,
    maxStorageBufferBindingSize: bufferSize,
    minUniformBufferOffsetAlignment: 256,
    minStorageBufferOffsetAlignment: 256,
    maxVertexBuffers: 8,
    maxBufferSize: bufferSize,
    maxVertexAttributes: Math.min(get(gl.MAX_VERTEX_ATTRIBS), 16),
    maxVertexBufferArrayStride: 2048,
    maxInterStageShaderVariables: Math.min(get(gl.MAX_VARYING_VECTORS), 16),
    maxInterStageShaderComponents: Math.min(get(gl.MAX_VARYING_COMPONENTS), 64),
    maxColorAttachments: colors,
    maxColorAttachmentBytesPerSample: 32,
    maxComputeWorkgroupStorageSize: 16384,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupsPerDimension: 65535,
    maxImmediateSize: 0,
    maxSubgroupSize: 0,
    minSubgroupSize: 0,
  };
  return Object.freeze(limits) as unknown as GPUSupportedLimits;
}

/** `timed` says whether the context can time spans of commands, which `timestamps.ts` needs. */
export function featuresOf(
  ext: Extensions,
  formats: FormatTable,
  timed: boolean,
): GPUSupportedFeatures {
  const features = new Set<string>(['indirect-first-instance', 'depth32float-stencil8']);
  if (timed) features.add('timestamp-query');
  if (ext.textureFloatLinear) features.add('float32-filterable');
  if (ext.floatBlend && formats.get('rgba32float')?.renderable) features.add('float32-blendable');
  if (formats.get('rg11b10ufloat')?.renderable) features.add('rg11b10ufloat-renderable');
  return features as unknown as GPUSupportedFeatures;
}
