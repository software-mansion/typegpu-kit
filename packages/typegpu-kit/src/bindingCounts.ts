// Counts bindings per stage as WebGPU does: every entry of every bind group layout in a pipeline's
// layout, against each stage its visibility names. Shared by `pipelines.bindingsOf` and the census.

import type { Binding, BindingKind, BindingsOptions, Overflow, ShaderStage } from './pipelines.ts';

/** The kinds a stage has a limit for, with the device limit that holds it. */
export const limitNames = {
  uniform: 'maxUniformBuffersPerShaderStage',
  storage: 'maxStorageBuffersPerShaderStage',
  texture: 'maxSampledTexturesPerShaderStage',
  sampler: 'maxSamplersPerShaderStage',
  storageTexture: 'maxStorageTexturesPerShaderStage',
} as const satisfies Record<Overflow['kind'], keyof GPUSupportedLimits>;

const guaranteed = {
  maxUniformBuffersPerShaderStage: 12,
  maxStorageBuffersPerShaderStage: 8,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16,
  maxStorageTexturesPerShaderStage: 4,
};

const limited = Object.keys(limitNames) as Overflow['kind'][];

/** How many of each limited kind a binding takes. An external texture takes four planes' worth. */
const weights: Record<BindingKind, Partial<Record<Overflow['kind'], number>>> = {
  uniform: { uniform: 1 },
  storage: { storage: 1 },
  texture: { texture: 1 },
  sampler: { sampler: 1 },
  storageTexture: { storageTexture: 1 },
  externalTexture: { texture: 4, sampler: 1, uniform: 1 },
};

export const nouns: Record<Overflow['kind'], string> = {
  uniform: 'uniform buffers',
  storage: 'storage buffers',
  texture: 'textures',
  sampler: 'samplers',
  storageTexture: 'storage textures',
};

/** How much of `kind`'s limit a binding takes. */
export const weight = (binding: Binding, kind: Overflow['kind']) =>
  weights[binding.kind][kind] ?? 0;

/** Lists each stage and kind over its limit. */
export function overflowing(
  bindings: readonly Binding[],
  pipelineStages: readonly ShaderStage[],
  limits: BindingsOptions['limits'] = {},
): Overflow[] {
  const over: Overflow[] = [];
  for (const stage of pipelineStages) {
    for (const kind of limited) {
      const limit = limits[limitNames[kind]] ?? guaranteed[limitNames[kind]];
      let count = 0;
      for (const b of bindings) if (b.stages.includes(stage)) count += weight(b, kind);
      if (count > limit) over.push({ stage, kind, count, limit });
    }
  }
  return over;
}

const declared = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<[^>]*>)?\s+(\w+)\s*:/g;

/** Reads the WGSL variable names of bindings, by `group:binding`. */
export function declaredNames(wgsl: string, into = new Map<string, string>()) {
  for (const [, group, binding, name] of wgsl.matchAll(declared)) {
    into.set(`${group}:${binding}`, name!);
  }
  return into;
}

/** Turns a visibility bit set into the pipeline's stages it names. */
export function stagesIn(
  visibility: number,
  pipelineStages: readonly ShaderStage[],
): ShaderStage[] {
  return pipelineStages.filter(
    (stage) => (visibility & (stage === 'vertex' ? 1 : stage === 'fragment' ? 2 : 4)) !== 0,
  );
}
