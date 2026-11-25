import {
  WgslGenerator,
  type FunctionDefinitionOptions,
  type ResolutionCtx,
  type VariableDefinitionOptions,
} from 'typegpu/~internal';
import { layoutOf } from '../../src/cpu/memory.ts';
import { CpuGenerator } from '../../src/cpu/generator.ts';

class ObservedCpuGenerator extends CpuGenerator {
  override declareGlobalVar(options: VariableDefinitionOptions) {
    if (['uniform', 'readonly', 'mutable'].includes(options.scope))
      this.ctx.addDeclaration(
        `//#test-layout ${JSON.stringify({ group: options.group, binding: options.binding, layout: layoutOf(options.dataType) })}`,
      );
    return super.declareGlobalVar(options);
  }
}

export class SweepGenerator extends WgslGenerator {
  override languageKey = 'wgsl' as const;
  constructor() {
    super();
    let backend: WgslGenerator | undefined;
    let context: ResolutionCtx;
    return new Proxy(this, {
      get(_target, key) {
        if (key === 'initGenerator')
          return (ctx: ResolutionCtx) => {
            context = ctx;
          };
        if (key === 'functionDefinition')
          return (options: FunctionDefinitionOptions) => {
            if (!backend) {
              backend =
                options.functionType === 'compute'
                  ? new ObservedCpuGenerator()
                  : new WgslGenerator();
              backend.initGenerator(context);
            }
            return backend.functionDefinition(options);
          };
        if (key === 'languageKey') return backend?.languageKey ?? 'wgsl';
        if (!backend) {
          backend =
            context.topFunctionScope?.functionType === 'compute'
              ? new ObservedCpuGenerator()
              : new WgslGenerator();
          backend.initGenerator(context);
        }
        const member = Reflect.get(backend, key);
        return typeof member === 'function' ? member.bind(backend) : member;
      },
    });
  }
}
