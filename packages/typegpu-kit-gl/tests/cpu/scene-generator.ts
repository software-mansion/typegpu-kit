import {
  WgslGenerator,
  type FunctionDefinitionOptions,
  type ResolutionCtx,
} from 'typegpu/~internal';
import { CpuGenerator } from '../../src/cpu/generator.ts';

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
                options.functionType === 'compute' ? new CpuGenerator() : new WgslGenerator();
              backend.initGenerator(context);
            }
            return backend.functionDefinition(options);
          };
        if (key === 'languageKey') return backend?.languageKey ?? 'wgsl';
        if (!backend) {
          backend =
            context.topFunctionScope?.functionType === 'compute'
              ? new CpuGenerator()
              : new WgslGenerator();
          backend.initGenerator(context);
        }
        const member = Reflect.get(backend, key);
        return typeof member === 'function' ? member.bind(backend) : member;
      },
    });
  }
}
