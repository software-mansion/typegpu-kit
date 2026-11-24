import { CrossShaderStageState, GlslGenerator } from '@typegpu/gl';

/**
 * Emits a render module: both stages of a pipeline as GLSL ES 3.00, with the directives
 * `renderModule.ts` reads.
 */
export class KitGlslGenerator extends GlslGenerator {
  constructor() {
    super('neutral', new CrossShaderStageState());
  }
}
