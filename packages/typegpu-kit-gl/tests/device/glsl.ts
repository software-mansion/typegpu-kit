/**
 * Hand-written render modules for device tests, following the conventions in ARCHITECTURE.md:
 * buffers as RGBA32UI data textures, a y-flipping vertex epilogue and 0 to 1 depth.
 */

import type {
  BlockBinding,
  BufferBinding,
  RenderModule,
  TextureBinding,
} from '../../src/glsl/renderModule.ts';
import { shaderModuleFrom } from '../../src/device/index.ts';

const PRECISION = `precision highp float;
precision highp int;
precision highp usampler2D;
precision highp sampler2D;
precision highp sampler2DArray;
precision highp samplerCube;
precision highp sampler3D;
precision highp sampler2DShadow;`;

function header(device: GPUDevice): string {
  const gl = (device as unknown as { gl: WebGL2RenderingContext }).gl;
  const clip = gl.getSupportedExtensions()?.includes('EXT_clip_control');
  return `#version 300 es\n${clip ? '#define KIT_CLIP_ZERO_TO_ONE\n' : ''}${PRECISION}`;
}

/** Reads word `i` of a buffer binding named `name`. */
export function bufferReader(name: string): string {
  return `uniform usampler2D ${name};
uniform uvec2 ${name}_range;
uint ${name}_word(uint i) {
  uint w = ${name}_range.x + i;
  int width = textureSize(${name}, 0).x;
  int texel = int(w / 4u);
  return texelFetch(${name}, ivec2(texel % width, texel / width), 0)[w % 4u];
}
float ${name}_float(uint i) { return uintBitsToFloat(${name}_word(i)); }`;
}

export function bufferBinding(
  name: string,
  group: number,
  binding: number,
  access: BufferBinding['access'] = 'readonly',
): BufferBinding {
  return { group, binding, access, texture: name, range: `${name}_range` };
}

export function textureBinding(
  uniform: string,
  group: number,
  binding: number,
  rest: Partial<TextureBinding> = {},
): TextureBinding {
  return {
    group,
    binding,
    uniform,
    dimension: '2d',
    sampleType: 'f32',
    comparison: false,
    multisampled: false,
    ...rest,
  };
}

export interface ModuleSource {
  readonly vertex: string;
  readonly vertexMain: string;
  readonly fragment?: string;
  readonly fragmentMain?: string;
  readonly buffers?: readonly BufferBinding[];
  readonly blocks?: readonly BlockBinding[];
  readonly textures?: readonly TextureBinding[];
}

export function renderModule(device: GPUDevice, source: ModuleSource): RenderModule {
  const top = header(device);
  return {
    kind: 'render',
    vertex: `${top}
uniform int kit_firstInstance;
uniform int kit_baseVertex;
${source.vertex}
void main() {
${source.vertexMain}
  gl_Position.y = -gl_Position.y;
#ifndef KIT_CLIP_ZERO_TO_ONE
  gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;
#endif
}
`,
    fragment:
      source.fragmentMain === undefined
        ? undefined
        : `${top}
${source.fragment ?? ''}
void main() {
${source.fragmentMain}
}
`,
    buffers: source.buffers ?? [],
    blocks: source.blocks ?? [],
    textures: source.textures ?? [],
    levels: [],
  };
}

export function module(device: GPUDevice, source: ModuleSource): GPUShaderModule {
  return shaderModuleFrom(device, renderModule(device, source));
}
