/**
 * Counts pipelines, their bindings per stage, and late compiles.
 *
 * @module
 * @unstable
 */

import type { TgpuRoot } from 'typegpu';
import { declaredNames, overflowing, stagesIn } from '../bindingCounts.ts';
import type { Binding, BindingKind, BindingsOptions, Overflow, ShaderStage } from '../pipelines.ts';

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

const names = [
  'createShaderModule',
  'createBindGroupLayout',
  'createPipelineLayout',
  'createRenderPipeline',
  'createRenderPipelineAsync',
  'createComputePipeline',
  'createComputePipelineAsync',
] as const;

type Name = (typeof names)[number];

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

  type Method = (descriptor: never) => unknown;
  const device = root.device as unknown as Record<Name, Method>;
  const originals = {} as Record<Name, Method>;
  for (const name of names) originals[name] = device[name];
  const own = new Set(names.filter((name) => Object.hasOwn(device, name)));

  const code = new WeakMap<object, string>();
  const groupLayouts = new WeakMap<object, readonly GPUBindGroupLayoutEntry[]>();
  const pipelineLayouts = new WeakMap<object, readonly (GPUBindGroupLayout | null | undefined)[]>();
  const pipelines: CensusEntry[] = [];
  let modules = 0;
  let wgslBytes = 0;
  let warmed = false;

  const call = (name: Name, descriptor: unknown) =>
    (originals[name] as (descriptor: unknown) => unknown).call(device, descriptor);

  const record = (
    kind: CensusEntry['kind'],
    async: boolean,
    descriptor: GPURenderPipelineDescriptor | GPUComputePipelineDescriptor,
  ) => {
    const render = kind === 'render' ? (descriptor as GPURenderPipelineDescriptor) : undefined;
    const used = new Set(
      render
        ? [render.vertex.module, render.fragment?.module]
        : [(descriptor as GPUComputePipelineDescriptor).compute.module],
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

    const layouts = pipelineLayouts.get(descriptor.layout as object) ?? [];
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
      kind,
      async,
      late: warmed && !async,
      wgslBytes: bytes,
      groups: layouts.length,
      bindings,
      over: overflowing(bindings, pipelineStages, limits),
    });
  };

  device.createShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
    const module = call('createShaderModule', descriptor) as GPUShaderModule;
    code.set(module, descriptor.code);
    modules += 1;
    wgslBytes += descriptor.code.length;
    return module;
  };

  device.createBindGroupLayout = (descriptor: GPUBindGroupLayoutDescriptor) => {
    const layout = call('createBindGroupLayout', descriptor) as GPUBindGroupLayout;
    groupLayouts.set(layout, [...descriptor.entries]);
    return layout;
  };

  device.createPipelineLayout = (descriptor: GPUPipelineLayoutDescriptor) => {
    const layout = call('createPipelineLayout', descriptor) as GPUPipelineLayout;
    pipelineLayouts.set(layout, [...descriptor.bindGroupLayouts]);
    return layout;
  };

  for (const [name, kind, async] of [
    ['createRenderPipeline', 'render', false],
    ['createRenderPipelineAsync', 'render', true],
    ['createComputePipeline', 'compute', false],
    ['createComputePipelineAsync', 'compute', true],
  ] as const) {
    device[name] = (descriptor: GPURenderPipelineDescriptor | GPUComputePipelineDescriptor) => {
      record(kind, async, descriptor);
      return call(name, descriptor);
    };
  }

  let destroyed = false;
  const destroy = () => {
    if (destroyed) return;
    destroyed = true;

    for (const name of names) {
      if (own.has(name)) device[name] = originals[name];
      else delete device[name];
    }
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
