/** Running compute dispatches with the CPU module over the CPU copies of bound resources. */

import type { CpuResources, CpuTexture, CpuTextureLevel } from '../cpu/types.ts';
import { type Bound, bufferEntry } from './binding.ts';
import { type BufferImpl, written } from './buffer.ts';
import type { Ctx } from './context.ts';
import { runGpu } from './gpuCompute.ts';
import { beginProbe, endProbe, ranOnCpu, ranOnGpu, runsOnGpu } from './placement.ts';
import type { ComputePipelineImpl } from './pipeline.ts';
import { cpuWritten, type TextureImpl, type ViewImpl } from './texture.ts';
import { cpuLevels } from './transfer.ts';

export interface ComputePass {
  pipeline: ComputePipelineImpl | undefined;
  groups: (Bound | undefined)[];
}

function cpuTexture(ctx: Ctx, view: ViewImpl): CpuTexture {
  const levels = cpuLevels(ctx, view.texture);
  const { info } = view.texture;
  const is3d = view.texture.dimension === '3d';
  return {
    format: view.format,
    dimension: view.dimension,
    levels: levels
      .slice(view.baseMipLevel, view.baseMipLevel + view.mipLevelCount)
      .map((level): CpuTextureLevel => {
        if (
          is3d ||
          (view.baseArrayLayer === 0 && view.arrayLayerCount === level.depthOrArrayLayers)
        ) {
          return level;
        }
        const layer = level.width * level.height * info.components;
        return {
          width: level.width,
          height: level.height,
          depthOrArrayLayers: view.arrayLayerCount,
          data: level.data.subarray(
            view.baseArrayLayer * layer,
            (view.baseArrayLayer + view.arrayLayerCount) * layer,
          ),
        };
      }),
  };
}

/** A kernel's stats, with the invocations its dispatches ran, which profiles read. */
interface KernelStats {
  ms: number;
  dispatches: number;
  invocations: number;
}

function count(ctx: Ctx, label: string, ms: number, invocations: number): void {
  const kernel = ctx.stats.kernels.get(label) as KernelStats | undefined;
  if (kernel) {
    kernel.ms += ms;
    kernel.dispatches++;
    kernel.invocations += invocations;
  } else {
    const fresh: KernelStats = { ms, dispatches: 1, invocations };
    ctx.stats.kernels.set(label, fresh);
  }
}

/**
 * Converts a workgroup count as WebIDL converts an `[EnforceRange] GPUSize32`, throwing its
 * `TypeError`, so the GPU form and the CPU module run the same dispatch.
 */
export function workgroupCount(value: unknown): number {
  // Unary plus is ToNumber, which throws on a BigInt or a Symbol
  const number = +(value as number);
  if (!Number.isFinite(number)) {
    throw new TypeError(`dispatchWorkgroups: the count ${String(value)} is not a finite number.`);
  }
  const whole = Math.trunc(number) + 0;
  if (whole < 0 || whole > 0xffffffff) {
    throw new TypeError(`dispatchWorkgroups: the count ${String(value)} is outside a GPUSize32.`);
  }
  return whole;
}

export function dispatch(ctx: Ctx, pass: ComputePass, x: number, y: number, z: number): void {
  const pipeline = pass.pipeline;
  const module = pipeline?.module;
  if (!module) {
    if (!pipeline) ctx.errors.report('validation', 'dispatchWorkgroups without a pipeline.');
    return;
  }
  if (x === 0 || y === 0 || z === 0) return;
  const size = pipeline.gpu?.module.workgroupSize ?? module.workgroupSize;
  const invocations = x * y * z * size[0] * size[1] * size[2];
  const gpu = pipeline.gpu;
  if (gpu && runsOnGpu(ctx, pipeline, pipeline.label, module, pass.groups, invocations)) {
    const start = performance.now();
    // A GPU form replaces the data texture of each buffer it writes, the passes' targets and the
    // counters it takes from alike: the replaced ones are what this dispatch wrote
    const before = new Map<BufferImpl, WebGLTexture | undefined>();
    for (const bound of pass.groups) {
      for (const entry of bound?.group.entries.values() ?? []) {
        if (entry.kind === 'buffer') before.set(entry.buffer, entry.buffer.data?.value.texture);
      }
    }
    const probe = beginProbe(ctx, pipeline, pipeline.label, invocations);
    let reason: string | undefined;
    let ran = false;
    try {
      reason = runGpu(ctx, gpu, pass.groups, x, y, z);
      ran = reason === undefined;
    } finally {
      endProbe(ctx, pipeline, probe, ran);
    }
    if (reason === undefined) {
      const ms = performance.now() - start;
      count(ctx, `${pipeline.label} (gpu)`, ms, invocations);
      const replaced = [...before]
        .filter(([buffer, texture]) => buffer.data?.value.texture !== texture)
        .map(([buffer]) => buffer);
      ranOnGpu(ctx, pipeline, pipeline.label, gpu, pass.groups, ms, invocations, replaced);
      return;
    }
    ctx.stats.fallbacks.set(pipeline.label, reason);
  }
  const buffers = new Map<string, { buffer: BufferImpl; offset: number; size: number }>();
  const textures = new Map<string, TextureImpl>();

  const resources: CpuResources = {
    buffer(group, binding) {
      const entry = bufferEntry(pass.groups[group], binding);
      if (!entry) throw new Error(`no buffer at @group(${group}) @binding(${binding})`);
      buffers.set(`${group}:${binding}`, entry);
      entry.buffer.sync();
      return {
        bytes: entry.buffer.words.buffer as ArrayBuffer,
        offset: entry.offset,
        size: entry.size,
      };
    },
    texture(group, binding) {
      const entry = pass.groups[group]?.group.entries.get(binding);
      if (entry?.kind !== 'texture')
        throw new Error(`no texture at @group(${group}) @binding(${binding})`);
      textures.set(`${group}:${binding}`, entry.view.texture);
      return cpuTexture(ctx, entry.view);
    },
    sampler(group, binding) {
      const entry = pass.groups[group]?.group.entries.get(binding);
      if (entry?.kind !== 'sampler')
        throw new Error(`no sampler at @group(${group}) @binding(${binding})`);
      return entry.sampler.descriptor;
    },
  };

  try {
    const start = performance.now();
    module.dispatch(x, y, z, resources);
    const ms = performance.now() - start;
    count(ctx, pipeline.label, ms, invocations);
    if (gpu) ranOnCpu(ctx, pipeline, ms);
  } catch (error) {
    ctx.errors.report(
      'internal',
      `The compute pipeline '${pipeline.label}' failed on the CPU: ${(error as Error).message}`,
    );
  }
  // A binding the module does not describe may have been written
  const writes = (key: string) => {
    const use = module.bindings.find((b) => `${b.group}:${b.binding}` === key);
    return use === undefined || use.writes;
  };
  for (const [key, buffer] of buffers) {
    if (!writes(key)) continue;
    // Only the words the module stored to, where it can tell, so GL uploads no more than changed
    const [group, binding] = key.split(':').map(Number) as [number, number];
    const words = module.writtenWords?.(group, binding);
    const [from, to] = words ?? [0, Infinity];
    const start = buffer.offset + Math.max(0, from) * 4;
    const end = buffer.offset + Math.min(buffer.size, to * 4);
    written(buffer.buffer, start, end);
  }
  for (const [key, texture] of textures) {
    const use = module.bindings.find((b) => `${b.group}:${b.binding}` === key);
    if (use?.kind === 'storageTexture' && use.writes) cpuWritten(texture);
  }
}

export function dispatchIndirect(
  ctx: Ctx,
  pass: ComputePass,
  buffer: BufferImpl,
  offset: number,
): void {
  const at = offset >>> 2;
  buffer.sync();
  dispatch(ctx, pass, buffer.words[at]!, buffer.words[at + 1]!, buffer.words[at + 2]!);
}
