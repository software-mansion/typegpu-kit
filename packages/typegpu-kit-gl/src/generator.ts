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

  get languageKey(): string {
    return this.#backend?.languageKey ?? 'glsl';
  }

  #use(kind: 'compute' | 'render'): ShaderGenerator {
    if (!this.#backend) {
      this.#backend = kind === 'compute' ? new CpuGenerator() : new KitGlslGenerator();
      if (this.#ctx) this.#backend.initGenerator(this.#ctx);
    }
    return this.#backend;
  }

  initGenerator(ctx: ResolutionCtx): void {
    this.#ctx = ctx;
    this.#backend?.initGenerator(ctx);
  }

  functionDefinition(options: FunctionDefinitionOptions): string {
    return this.#use(options.functionType === 'compute' ? 'compute' : 'render').functionDefinition(
      options,
    );
  }

  declareGlobalConst(options: ConstantDefinitionOptions): ResolvedSnippet {
    return this.#use('render').declareGlobalConst(options);
  }

  declareGlobalVar(options: VariableDefinitionOptions): ResolvedSnippet {
    return this.#use('render').declareGlobalVar(options);
  }

  typeInstantiation(...args: Parameters<ShaderGenerator['typeInstantiation']>): ResolvedSnippet {
    return this.#use('render').typeInstantiation(...args);
  }

  numericLiteral(...args: Parameters<ShaderGenerator['numericLiteral']>): ResolvedSnippet {
    return this.#use('render').numericLiteral(...args);
  }

  call(...args: Call): Snippet {
    return this.#use('render').call(...args);
  }

  emitTypeAnnotation(...args: Parameters<ShaderGenerator['emitTypeAnnotation']>): string {
    return this.#use('render').emitTypeAnnotation(...args);
  }

  emitCall(...args: Parameters<ShaderGenerator['emitCall']>): string {
    return this.#use('render').emitCall(...args);
  }

  emitBinaryOp(lhs: Snippet, op: BinaryOperator, rhs: Snippet): string {
    return this.#use('render').emitBinaryOp(lhs, op, rhs);
  }
}
