/**
 * Counts pipelines, their bindings per stage, and late compiles.
 *
 * @module
 * @unstable
 */

import type { TgpuRoot } from 'typegpu';
import { declaredNames, overflowing, stagesIn } from '../bindingCounts.ts';
import type { Binding, BindingKind, BindingsOptions, Overflow, ShaderStage } from '../pipelines.ts';
import { patches } from './patch.ts';

export interface CensusEntry {
  readonly label: string;
  readonly kind: 'render' | 'compute';
  /** Reports whether an asynchronous call created the pipeline, as warm-up does. */
  readonly async: boolean;
  /**
   * Reports whether the pipeline was created synchronously after `warm()`, during a frame instead
   * of warm-up.
   */
  readonly late: boolean;
  readonly wgslBytes: number;
  /** Counts the bind groups in the pipeline's layout, including TypeGPU's automatic group. */
  readonly groups: number;
  /**
   * Lists every entry of the pipeline's bind group layouts with the stages its visibility names,
   * as `pipelines.bindingsOf` does.
   */
  readonly bindings: readonly Binding[];
  /** Lists each stage and kind over its limit. Empty when the pipeline fits. */
  readonly over: readonly Overflow[];
}

export interface Options {
  /**
   * Sets the limits to compare with, such as `root.device.limits`. Default the minimums WebGPU
   * guarantees: 4 bind groups, 8 storage buffers, 12 uniform buffers, 16 textures, 16 samplers and
   * 4 storage textures per stage.
   */
  readonly limits?: BindingsOptions['limits'] & { readonly maxBindGroups?: number };
}

export interface Census extends Disposable {
  /** Lists every pipeline created so far, in creation order. */
  readonly pipelines: readonly CensusEntry[];
  /** Counts the shader modules created so far. */
  readonly modules: number;
  /** Sums the WGSL length of every shader module created so far, in characters. */
  readonly wgslBytes: number;
  /** Lists pipelines created synchronously after `warm()`. */
  readonly late: readonly CensusEntry[];
  /** Lists pipelines with a stage over its limits, or more bind groups than allowed. */
  readonly overBudget: readonly CensusEntry[];
  /** Marks warm-up complete. Later synchronous compilations count as `late`. */
  warm(): void;
  /** Restores the device's methods. */
  destroy(): void;
}

function kindOf(entry: GPUBindGroupLayoutEntry): BindingKind | undefined {
  if (entry.buffer) return (entry.buffer.type ?? 'uniform') === 'uniform' ? 'uniform' : 'storage';
  if (entry.storageTexture) return 'storageTexture';
  if (entry.texture) return 'texture';
  if (entry.sampler) return 'sampler';
  if (entry.externalTexture) return 'externalTexture';
  return undefined;
}

/**
 * Wraps the device's creation methods to count shader modules, pipelines and the bindings in each
 * pipeline's layout, until destroyed. Create it right after the root, before anything compiles, so
 * it sees every layout.
 *
 * @returns counts that grow as pipelines compile, until you destroy it
 */
export function create(root: TgpuRoot, options: Options = {}): Census {
  const { limits = {} } = options;
  const maxBindGroups = limits.maxBindGroups ?? 4;
  const { device } = root;
  const { patch, restore } = patches();

  const code = new WeakMap<GPUShaderModule, string>();
  const groupLayouts = new WeakMap<GPUBindGroupLayout, readonly GPUBindGroupLayoutEntry[]>();
  const pipelineLayouts = new WeakMap<
    GPUPipelineLayout,
    readonly (GPUBindGroupLayout | null | undefined)[]
  >();
  const pipelines: CensusEntry[] = [];
  let modules = 0;
  let wgslBytes = 0;
  let warmed = false;

  const record = (
    async: boolean,
    descriptor: GPURenderPipelineDescriptor | GPUComputePipelineDescriptor,
  ) => {
    const render = 'vertex' in descriptor ? descriptor : undefined;
    const used = new Set(
      'vertex' in descriptor
        ? [descriptor.vertex.module, descriptor.fragment?.module]
        : [descriptor.compute.module],
    );
    const pipelineStages: ShaderStage[] = !render
      ? ['compute']
      : render.fragment
        ? ['vertex', 'fragment']
        : ['vertex'];

    let bytes = 0;
    const declared = new Map<string, string>();
    for (const module of used) {
      const wgsl = (module && code.get(module)) ?? '';
      bytes += wgsl.length;
      declaredNames(wgsl, declared);
    }

    const layouts =
      descriptor.layout === 'auto' ? [] : (pipelineLayouts.get(descriptor.layout) ?? []);
    const bindings: Binding[] = [];
    for (const [group, layout] of layouts.entries()) {
      for (const entry of (layout && groupLayouts.get(layout)) ?? []) {
        const bindingKind = kindOf(entry);
        if (!bindingKind) continue;
        bindings.push({
          group,
          binding: entry.binding,
          name: declared.get(`${group}:${entry.binding}`) ?? `${layout!.label}.${entry.binding}`,
          kind: bindingKind,
          stages: stagesIn(entry.visibility, pipelineStages),
        });
      }
    }

    pipelines.push({
      label: descriptor.label ?? '<unnamed>',
      kind: render ? 'render' : 'compute',
      async,
      late: warmed && !async,
      wgslBytes: bytes,
      groups: layouts.length,
      bindings,
      over: overflowing(bindings, pipelineStages, limits),
    });
  };

  patch(device, 'createShaderModule', (_self, [descriptor], next) => {
    const module = next(descriptor);
    code.set(module, descriptor.code);
    modules += 1;
    wgslBytes += descriptor.code.length;
    return module;
  });

  // Read one-shot iterables once, and hand the device the arrays read
  patch(device, 'createBindGroupLayout', (_self, [descriptor], next) => {
    const entries = [...descriptor.entries];
    const layout = next({ ...descriptor, entries });
    groupLayouts.set(layout, entries);
    return layout;
  });

  patch(device, 'createPipelineLayout', (_self, [descriptor], next) => {
    const bindGroupLayouts = [...descriptor.bindGroupLayouts];
    const layout = next({ ...descriptor, bindGroupLayouts });
    pipelineLayouts.set(layout, bindGroupLayouts);
    return layout;
  });

  patch(device, 'createRenderPipeline', (_self, [descriptor], next) => {
    record(false, descriptor);
    return next(descriptor);
  });
  patch(device, 'createRenderPipelineAsync', (_self, [descriptor], next) => {
    record(true, descriptor);
    return next(descriptor);
  });
  patch(device, 'createComputePipeline', (_self, [descriptor], next) => {
    record(false, descriptor);
    return next(descriptor);
  });
  patch(device, 'createComputePipelineAsync', (_self, [descriptor], next) => {
    record(true, descriptor);
    return next(descriptor);
  });

  let destroyed = false;
  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    restore();
  };

  return {
    pipelines,

    get modules() {
      return modules;
    },

    get wgslBytes() {
      return wgslBytes;
    },

    get late() {
      return pipelines.filter((entry) => entry.late);
    },

    get overBudget() {
      return pipelines.filter((entry) => entry.over.length > 0 || entry.groups > maxBindGroups);
    },

    warm() {
      warmed = true;
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
