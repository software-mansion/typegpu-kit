/**
 * Shader modules. A module's code is a render module from the GLSL generator, a compute module
 * from the CPU generator, or one of the few WGSL shaders TypeGPU compiles itself, which the device
 * replaces with its own GLSL.
 */

import { compile, isCpuModule } from '../cpu/index.ts';
import type { CpuModule } from '../cpu/types.ts';
import { type ComputeModule, cpuOnlyReason, parseCompute } from '../glsl/computeModule.ts';
import {
  type BlockBinding,
  type BufferBinding,
  isRenderModule,
  type LevelCountBinding,
  parse,
  type RenderModule,
  type TextureBinding,
} from '../glsl/renderModule.ts';
import { type Ctx, IMPL } from './context.ts';

/** What a pipeline takes from a module: stage sources and the bindings they read. */
export interface Stages {
  readonly vertex: string | undefined;
  readonly fragment: string | undefined;
  readonly buffers: readonly BufferBinding[];
  readonly blocks: readonly BlockBinding[];
  readonly textures: readonly TextureBinding[];
  readonly levels: readonly LevelCountBinding[];
}

export type ModuleImpl =
  | { readonly kind: 'render'; readonly stages: Stages; readonly label: string }
  | {
      readonly kind: 'compute';
      readonly module: CpuModule;
      /** The kernel as fragment passes, when the generator could write it so. */
      readonly gpu: ComputeModule | undefined;
      /** Why the GPU form failed to parse, or why the generator wrote none. */
      readonly gpuError?: string;
      readonly label: string;
    }
  | { readonly kind: 'error'; readonly message: string; readonly label: string };

/** A descriptor key that carries a module built by hand instead of `code`. */
/** The dispatch conditions `runGpu` checks, so GPU forms that need them parse. */
const CONDITIONS = ['limits', 'single'] as const;

export const PREBUILT: unique symbol = Symbol('@typegpu/kit-gl/prebuilt');

export function moduleImpl(module: GPUShaderModule): ModuleImpl {
  return (module as unknown as { [IMPL]: ModuleImpl })[IMPL];
}

function header(ctx: Ctx): string {
  return ['#version 300 es', ...ctx.defines].join('\n');
}

/** TypeGPU's fullscreen triangle for mipmaps and resampling writes, `textureUtils.js`. */
function isBlitVertex(code: string): boolean {
  return (
    code.includes('fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput') &&
    code.includes('const pos = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));')
  );
}

const SAMPLE_FRAGMENT = /return textureSample\(src, samp, uv\)(?:\.([rgba])\1\1\1)?;/;
const GATHER_FRAGMENT = 'let r = textureGather(0, src, samp, uv);';

function blitVertex(ctx: Ctx): string {
  return `${header(ctx)}
out vec2 kit_uv;
void main() {
  vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  vec2 uvs[3] = vec2[3](vec2(0.0, 1.0), vec2(2.0, 1.0), vec2(0.0, -1.0));
  kit_uv = uvs[gl_VertexID];
  gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
  gl_Position.y = -gl_Position.y;
}
`;
}

function blitFragment(ctx: Ctx, body: string): string {
  return `${header(ctx)}
precision highp float;
precision highp sampler2D;
uniform sampler2D kit_source;
in vec2 kit_uv;
layout(location = 0) out vec4 kit_color;
void main() {
${body}
}
`;
}

const blitTexture: TextureBinding = {
  group: 0,
  binding: 0,
  uniform: 'kit_source',
  sampler: { group: 0, binding: 1 },
  dimension: '2d',
  sampleType: 'f32',
  comparison: false,
  multisampled: false,
};

const { sampler: _sampler, ...loadTexture } = blitTexture;

/** The GLSL for one of TypeGPU's own WGSL shaders, or undefined for any other WGSL. */
function builtin(ctx: Ctx, code: string): Stages | undefined {
  if (isBlitVertex(code)) {
    return {
      vertex: blitVertex(ctx),
      fragment: undefined,
      buffers: [],
      blocks: [],
      textures: [],
      levels: [],
    };
  }
  const sample = SAMPLE_FRAGMENT.exec(code);
  if (sample && code.includes('@group(0) @binding(0) var src: texture_2d<f32>;')) {
    const swizzle = sample[1] ? `.${sample[1].repeat(4)}` : '';
    return {
      vertex: undefined,
      fragment: blitFragment(ctx, `  kit_color = texture(kit_source, kit_uv)${swizzle};`),
      buffers: [],
      blocks: [],
      textures: [blitTexture],
      levels: [],
    };
  }
  if (code.includes(GATHER_FRAGMENT)) {
    // textureGather is GLSL ES 3.1, so average the same 2x2 footprint with texel loads
    return {
      vertex: undefined,
      fragment: blitFragment(
        ctx,
        `  ivec2 size = textureSize(kit_source, 0);
  ivec2 at = ivec2(floor(kit_uv * vec2(size) - 0.5));
  ivec2 high = size - 1;
  vec4 sum = texelFetch(kit_source, clamp(at, ivec2(0), high), 0)
    + texelFetch(kit_source, clamp(at + ivec2(1, 0), ivec2(0), high), 0)
    + texelFetch(kit_source, clamp(at + ivec2(0, 1), ivec2(0), high), 0)
    + texelFetch(kit_source, clamp(at + ivec2(1, 1), ivec2(0), high), 0);
  kit_color = sum * 0.25;`,
      ),
      buffers: [],
      blocks: [],
      textures: [loadTexture],
      levels: [],
    };
  }
  return undefined;
}

function stagesOf(module: RenderModule): Stages {
  return {
    vertex: module.vertex,
    fragment: module.fragment,
    buffers: module.buffers,
    blocks: module.blocks ?? [],
    textures: module.textures,
    levels: module.levels ?? [],
  };
}

/** A module built by hand: a render module, or a CPU module with an optional GPU form. */
export type Prebuilt =
  | RenderModule
  | CpuModule
  | { readonly cpu: CpuModule; readonly gpu: ComputeModule };

function build(
  ctx: Ctx,
  descriptor: GPUShaderModuleDescriptor & { [PREBUILT]?: Prebuilt },
): ModuleImpl {
  const label = descriptor.label ?? '';
  const prebuilt = descriptor[PREBUILT];
  if (prebuilt) {
    if ('cpu' in prebuilt)
      return { kind: 'compute', module: prebuilt.cpu, gpu: prebuilt.gpu, label };
    return prebuilt.kind === 'render'
      ? { kind: 'render', stages: stagesOf(prebuilt), label }
      : { kind: 'compute', module: prebuilt, gpu: undefined, label };
  }
  const { code } = descriptor;
  try {
    if (isRenderModule(code)) {
      return { kind: 'render', stages: stagesOf(parse(code, { defines: ctx.defines })), label };
    }
    if (isCpuModule(code)) {
      const module = compile(code);
      try {
        const gpu = parseCompute(code, { defines: ctx.defines, conditions: CONDITIONS });
        const reason = cpuOnlyReason(code);
        return {
          kind: 'compute',
          module,
          gpu,
          ...(!gpu && reason !== undefined && { gpuError: `CPU only: ${reason}` }),
          label,
        };
      } catch (error) {
        return {
          kind: 'compute',
          module,
          gpu: undefined,
          gpuError: (error as Error).message,
          label,
        };
      }
    }
  } catch (error) {
    return {
      kind: 'error',
      message: `The shader module ${label ? `'${label}' ` : ''}is invalid: ${(error as Error).message}`,
      label,
    };
  }
  const stages = builtin(ctx, code);
  if (stages) return { kind: 'render', stages, label };
  return {
    kind: 'error',
    message: `The shader module ${label ? `'${label}' ` : ''}is WGSL that the WebGL fallback cannot run. Shaders reach WebGL 2 only through a root made with the fallback's shader generator.\n${code.slice(0, 400)}`,
    label,
  };
}

export function createShaderModule(
  ctx: Ctx,
  descriptor: GPUShaderModuleDescriptor & { [PREBUILT]?: Prebuilt },
): GPUShaderModule {
  const impl = build(ctx, descriptor);
  if (impl.kind === 'error') ctx.errors.report('validation', impl.message);
  const messages =
    impl.kind === 'error'
      ? [{ type: 'error', message: impl.message, lineNum: 0, linePos: 0, offset: 0, length: 0 }]
      : [];
  return {
    [IMPL]: impl,
    __brand: 'GPUShaderModule',
    label: impl.label,
    getCompilationInfo: () => Promise.resolve({ messages }),
  } as unknown as GPUShaderModule;
}
