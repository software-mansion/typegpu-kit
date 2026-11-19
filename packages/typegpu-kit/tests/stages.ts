import type { d, TgpuFragmentFn, TgpuVertexFn, WithBinding } from 'typegpu';
import type { passes } from '../src/index.ts';

/** The stages and state a description hands `createRenderPipeline` for a drawing. */
export interface Stages {
  readonly vertex: TgpuVertexFn;
  /** Absent for a drawing into depth alone that writes no depth of its own and cuts nothing out. */
  readonly fragment: TgpuFragmentFn<Record<string, d.BaseData>, Record<string, d.AnyWgslData>>;
  readonly targets: Record<string, GPUColorTargetState>;
  readonly depthStencil: GPUDepthStencilState | undefined;
  readonly multisample: GPUMultisampleState;
  readonly primitive: GPUPrimitiveState | undefined;
}

/** Returns what `into` builds a pipeline of `drawing` from, without building it. */
export function stagesOf<V extends d.WgslStruct<{ position: d.Vec4f }>, P>(
  into: passes.Into,
  drawing: passes.Shaders<V, P>,
): Stages {
  let stages: Stages | undefined;
  const from = {
    pipe: () => from,
    createRenderPipeline(given: Stages) {
      stages = given;
      return { $name: () => undefined };
    },
  };
  into.pipeline(drawing, from as unknown as WithBinding);
  return stages!;
}
