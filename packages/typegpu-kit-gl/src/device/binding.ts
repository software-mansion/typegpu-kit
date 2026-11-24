/** Bind group layouts, pipeline layouts and bind groups: plain records of what a draw binds. */

import { type BufferImpl, bufferImpl } from './buffer.ts';
import { type Ctx, IMPL } from './context.ts';
import { type SamplerImpl, samplerImpl } from './sampler.ts';
import { makeView, textureImpl, type ViewImpl, viewImpl } from './texture.ts';

export interface LayoutImpl {
  readonly entries: readonly GPUBindGroupLayoutEntry[];
  /** Bindings with a dynamic offset, in binding order, as `setBindGroup` takes the offsets. */
  readonly dynamic: readonly number[];
}

export type Entry =
  | {
      readonly kind: 'buffer';
      readonly buffer: BufferImpl;
      readonly offset: number;
      readonly size: number;
    }
  | { readonly kind: 'texture'; readonly view: ViewImpl }
  | { readonly kind: 'sampler'; readonly sampler: SamplerImpl };

export interface GroupImpl {
  readonly layout: LayoutImpl;
  readonly entries: ReadonlyMap<number, Entry>;
}

/** A bind group as a pass bound it, with its dynamic offsets. */
export interface Bound {
  readonly group: GroupImpl;
  readonly offsets: ReadonlyMap<number, number> | undefined;
}

export function layoutImpl(layout: GPUBindGroupLayout): LayoutImpl {
  return (layout as unknown as { [IMPL]: LayoutImpl })[IMPL];
}

export function makeLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
  const entries = Array.from(descriptor.entries);
  const dynamic = entries
    .filter((entry) => entry.buffer?.hasDynamicOffset)
    .map((entry) => entry.binding)
    .toSorted((a, b) => a - b);
  return {
    [IMPL]: { entries, dynamic } satisfies LayoutImpl,
    __brand: 'GPUBindGroupLayout',
    label: descriptor.label ?? '',
  } as unknown as GPUBindGroupLayout;
}

export function makePipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
  return {
    [IMPL]: { groups: Array.from(descriptor.bindGroupLayouts) },
    __brand: 'GPUPipelineLayout',
    label: descriptor.label ?? '',
  } as unknown as GPUPipelineLayout;
}

export function pipelineLayoutGroups(
  layout: GPUPipelineLayout,
): (GPUBindGroupLayout | null | undefined)[] {
  return (layout as unknown as { [IMPL]: { groups: (GPUBindGroupLayout | null | undefined)[] } })[
    IMPL
  ].groups;
}

function entryOf(resource: GPUBindingResource): Entry {
  const record = resource as unknown as { [IMPL]?: unknown; buffer?: GPUBuffer };
  const impl = record[IMPL] as Record<string, unknown> | undefined;
  if (!impl && record.buffer) {
    const binding = resource as GPUBufferBinding;
    const buffer = bufferImpl(binding.buffer);
    const offset = binding.offset ?? 0;
    return { kind: 'buffer', buffer, offset, size: binding.size ?? buffer.size - offset };
  }
  if (impl && 'bytes' in impl) {
    const buffer = impl as unknown as BufferImpl;
    return { kind: 'buffer', buffer, offset: 0, size: buffer.size };
  }
  if (impl && 'baseMipLevel' in impl) return { kind: 'texture', view: impl as unknown as ViewImpl };
  if (impl && 'comparison' in impl) {
    return { kind: 'sampler', sampler: samplerImpl(resource as GPUSampler) };
  }
  if (impl && 'mipLevelCount' in impl) {
    return { kind: 'texture', view: viewImpl(makeView(textureImpl(resource as GPUTexture))) };
  }
  throw new Error('@typegpu/kit-gl: a bind group entry holds a resource of another device.');
}

export function makeGroup(ctx: Ctx, descriptor: GPUBindGroupDescriptor): GPUBindGroup {
  const layout = layoutImpl(descriptor.layout);
  const entries = new Map<number, Entry>();
  for (const entry of descriptor.entries) {
    try {
      entries.set(entry.binding, entryOf(entry.resource));
    } catch (error) {
      ctx.errors.report('validation', (error as Error).message);
    }
  }
  return {
    [IMPL]: { layout, entries } satisfies GroupImpl,
    __brand: 'GPUBindGroup',
    label: descriptor.label ?? '',
  } as unknown as GPUBindGroup;
}

export function groupImpl(group: GPUBindGroup): GroupImpl {
  return (group as unknown as { [IMPL]: GroupImpl })[IMPL];
}

/** Pairs a bind group with its dynamic offsets, as `setBindGroup` gives them. */
export function bind(
  group: GPUBindGroup,
  offsets: Iterable<number> | Uint32Array | undefined,
  start = 0,
  length?: number,
): Bound {
  const impl = groupImpl(group);
  if (!offsets || impl.layout.dynamic.length === 0) return { group: impl, offsets: undefined };
  const values = Array.from(offsets as Iterable<number>);
  const used = length === undefined ? values.slice(start) : values.slice(start, start + length);
  const map = new Map<number, number>();
  impl.layout.dynamic.forEach((binding, i) => map.set(binding, used[i] ?? 0));
  return { group: impl, offsets: map };
}

/** The buffer range a binding reads, with its dynamic offset applied. */
export function bufferEntry(
  bound: Bound | undefined,
  binding: number,
): { buffer: BufferImpl; offset: number; size: number } | undefined {
  const entry = bound?.group.entries.get(binding);
  if (entry?.kind !== 'buffer') return undefined;
  const extra = bound!.offsets?.get(binding) ?? 0;
  return { buffer: entry.buffer, offset: entry.offset + extra, size: entry.size };
}
