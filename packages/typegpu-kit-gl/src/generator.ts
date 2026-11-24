import type {
  BinaryOperator,
  ConstantDefinitionOptions,
  FunctionDefinitionOptions,
  ResolutionCtx,
  ResolvedSnippet,
  ShaderGenerator,
  Snippet,
  VariableDefinitionOptions,
} from 'typegpu/~internal';
import { CpuGenerator } from './cpu/generator.ts';
import { KitGlslGenerator } from './glsl/generator.ts';

type Call = Parameters<ShaderGenerator['call']>;

/**
 * The generator a fallback root resolves every pipeline with. TypeGPU creates one per pipeline and
 * says nothing about the pipeline's kind, so this waits for the entry function: a compute entry
 * goes to the CPU generator, anything else to GLSL.
 */
export class KitGenerator implements ShaderGenerator {
  #ctx: ResolutionCtx | undefined;
  #backend: ShaderGenerator | undefined;

  /**
   * TypeGPU types a root's generator as one for WGSL, the only language its own device accepts.
   * This device accepts what these backends write, so the key only satisfies that type.
   */
  readonly languageKey = 'wgsl' as const;

  /**
   * Picks the backend on the first call. TypeGPU may declare a global or a type before it defines
   * the entry function, so read the kind from the function being resolved, the entry at that point.
   */
  #use(): ShaderGenerator {
    if (!this.#backend) {
      const scope = (this.#ctx as { topFunctionScope?: { functionType?: string } } | undefined)
        ?.topFunctionScope;
      this.#backend =
        scope?.functionType === 'compute' ? new CpuGenerator() : new KitGlslGenerator();
      if (this.#ctx) this.#backend.initGenerator(this.#ctx);
    }
    return this.#backend;
  }

  initGenerator(ctx: ResolutionCtx): void {
    this.#ctx = ctx;
    this.#backend?.initGenerator(ctx);
  }

  functionDefinition(options: FunctionDefinitionOptions): string {
    if (!this.#backend && options.functionType === 'compute') {
      this.#backend = new CpuGenerator();
      if (this.#ctx) this.#backend.initGenerator(this.#ctx);
    }
    return this.#use().functionDefinition(options);
  }

  declareGlobalConst(options: ConstantDefinitionOptions): ResolvedSnippet {
    return this.#use().declareGlobalConst(options);
  }

  declareGlobalVar(options: VariableDefinitionOptions): ResolvedSnippet {
    return this.#use().declareGlobalVar(options);
  }

  typeInstantiation(...args: Parameters<ShaderGenerator['typeInstantiation']>): ResolvedSnippet {
    return this.#use().typeInstantiation(...args);
  }

  numericLiteral(...args: Parameters<ShaderGenerator['numericLiteral']>): ResolvedSnippet {
    return this.#use().numericLiteral(...args);
  }

  call(...args: Call): Snippet {
    return this.#use().call(...args);
  }

  emitTypeAnnotation(...args: Parameters<ShaderGenerator['emitTypeAnnotation']>): string {
    return this.#use().emitTypeAnnotation(...args);
  }

  emitCall(...args: Parameters<ShaderGenerator['emitCall']>): string {
    return this.#use().emitCall(...args);
  }

  emitBinaryOp(lhs: Snippet, op: BinaryOperator, rhs: Snippet): string {
    return this.#use().emitBinaryOp(lhs, op, rhs);
  }
}
