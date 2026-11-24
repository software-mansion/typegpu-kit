import { recordingRoot, type RecordingRoot } from '@typegpu/kit/testing';
import { tgpu, type TgpuRoot } from 'typegpu';
import {
  WgslGenerator,
  type FunctionDefinitionOptions,
  type ResolutionCtx,
  type ShaderGenerator,
} from 'typegpu/~internal';
import { KitGenerator } from '../../src/generator.ts';
import { KitGlslGenerator } from '../../src/glsl/generator.ts';
import { isRenderModule, parse, type RenderModule } from '../../src/glsl/renderModule.ts';

type GeneratorClass = NonNullable<
  Parameters<typeof tgpu.initFromDevice>[0]['unstable_shaderGeneratorClass']
>;

/**
 * Picks backends as `KitGenerator` does, but leaves compute pipelines in WGSL, so GLSL tests do not
 * depend on what the CPU backend supports yet.
 */
class RenderGenerator implements ShaderGenerator {
  #ctx: ResolutionCtx | undefined;
  #backend: ShaderGenerator | undefined;
  readonly languageKey = 'wgsl' as const;

  #use(compute = false): ShaderGenerator {
    if (!this.#backend) {
      const scope = (this.#ctx as { topFunctionScope?: { functionType?: string } } | undefined)
        ?.topFunctionScope;
      this.#backend =
        compute || scope?.functionType === 'compute' ? new WgslGenerator() : new KitGlslGenerator();
      if (this.#ctx) this.#backend.initGenerator(this.#ctx);
    }
    return this.#backend;
  }

  initGenerator(ctx: ResolutionCtx) {
    this.#ctx = ctx;
    // A compute entry written as a WGSL string never reaches `functionDefinition`, so this reads
    // the pipeline's kind from the call that started the resolution
    const limit = Error.stackTraceLimit;
    Error.stackTraceLimit = 50;
    const stack = new Error().stack ?? '';
    Error.stackTraceLimit = limit;
    if (stack.includes('computePipeline.js')) this.#use(true);
    else this.#backend?.initGenerator(ctx);
  }

  functionDefinition(options: FunctionDefinitionOptions) {
    return this.#use(options.functionType === 'compute').functionDefinition(options);
  }

  declareGlobalConst: ShaderGenerator['declareGlobalConst'] = (...args) =>
    this.#use().declareGlobalConst(...args);
  declareGlobalVar: ShaderGenerator['declareGlobalVar'] = (...args) =>
    this.#use().declareGlobalVar(...args);
  typeInstantiation: ShaderGenerator['typeInstantiation'] = (...args) =>
    this.#use().typeInstantiation(...args);
  numericLiteral: ShaderGenerator['numericLiteral'] = (...args) =>
    this.#use().numericLiteral(...args);
  call: ShaderGenerator['call'] = (...args) => this.#use().call(...args);
  emitTypeAnnotation: ShaderGenerator['emitTypeAnnotation'] = (...args) =>
    this.#use().emitTypeAnnotation(...args);
  emitCall: ShaderGenerator['emitCall'] = (...args) => this.#use().emitCall(...args);
  emitBinaryOp: ShaderGenerator['emitBinaryOp'] = (...args) => this.#use().emitBinaryOp(...args);
}

export interface GlslRoot {
  /** A root that resolves its render pipelines to GLSL on a recording device. */
  readonly root: TgpuRoot;
  readonly recording: RecordingRoot;
  /** The code of every render module created so far. */
  codes(): string[];
  /** Every render module created so far, parsed. */
  modules(): RenderModule[];
}

export interface GlslRootOptions {
  /** Resolves compute pipelines with the CPU backend too, as the fallback does. Default false. */
  readonly compute?: boolean;
  readonly features?: readonly string[];
}

export function glslRoot(options: GlslRootOptions = {}): GlslRoot {
  const recording = recordingRoot(options.features ? { features: options.features } : {});
  const root = tgpu.initFromDevice({
    device: recording.root.device,
    unstable_shaderGeneratorClass: (options.compute
      ? KitGenerator
      : RenderGenerator) as unknown as GeneratorClass,
  });
  const codes = () => recording.log.shaders.filter(isRenderModule);
  return { root, recording, codes, modules: () => codes().map((code) => parse(code)) };
}
