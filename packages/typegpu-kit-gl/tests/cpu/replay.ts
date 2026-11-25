import type { CpuBindingUse, CpuBuffer, CpuResources, CpuTexture } from '../../src/cpu/types.ts';
import { compile } from '../../src/cpu/index.ts';
import type { DispatchSnapshot } from './device.ts';
import type { observeScene, ObservedDispatch, Resource } from './scene-observer.ts';
import {
  compareCaptures,
  readBuffer,
  readTexture,
  type CapturedResource,
} from './example-comparison.ts';

interface HeldBinding {
  use: CpuBindingUse;
  key: string;
  resource?: Resource;
  buffer?: CpuBuffer;
  texture?: CpuTexture;
  sampler?: GPUSamplerDescriptor;
}
export interface ReplaySnapshot {
  code: string;
  label: string;
  workgroups: [number, number, number];
  bindings: HeldBinding[];
}
export function snapshotDispatch(
  code: string,
  label: string,
  dispatch: DispatchSnapshot,
  observer: ReturnType<typeof observeScene>,
): ReplaySnapshot {
  const copied = new Map<ArrayBuffer, ArrayBuffer>();
  const clone = (bytes: ArrayBuffer) => {
    let result = copied.get(bytes);
    if (!result) {
      result = bytes.slice(0);
      copied.set(bytes, result);
    }
    return result;
  };
  return {
    code,
    label,
    workgroups: dispatch.workgroups,
    bindings: dispatch.module.bindings.map((use) => {
      const group = dispatch.groups.get(use.group)!;
      const entry = Array.from(group.entries).find((item) => item.binding === use.binding)!;
      const native = entry.resource as GPUBufferBinding;
      const resource = observer.resourceOf(native.buffer ?? entry.resource);
      const base = {
        use,
        key: resource?.key ?? `sampler:${use.group}:${use.binding}`,
        ...(resource ? { resource: { ...resource } } : {}),
      };
      if (use.kind === 'buffer') {
        const value = dispatch.resources.buffer(use.group, use.binding);
        return { ...base, buffer: { ...value, bytes: clone(value.bytes) } };
      }
      if (use.kind === 'sampler')
        return { ...base, sampler: { ...dispatch.resources.sampler(use.group, use.binding) } };
      const texture = dispatch.resources.texture(use.group, use.binding);
      return {
        ...base,
        texture: {
          ...texture,
          levels: texture.levels.map((level) => {
            const bytes = level.data.buffer as ArrayBuffer;
            const constructor = level.data.constructor as new (
              buffer: ArrayBuffer,
              offset: number,
              count: number,
            ) => Uint8Array;
            return {
              ...level,
              data: new constructor(
                clone(bytes),
                level.data.byteOffset,
                (level.data as Uint8Array).length,
              ),
            };
          }),
        },
      };
    }),
  };
}

function captured(binding: HeldBinding, levels: CapturedResource['levels']): CapturedResource {
  const { handle: _handle, ...metadata } = binding.resource!;
  return { ...metadata, reasons: [], order: [], levels };
}

/** Runs one compiled CPU kernel and its WGSL counterpart from the same captured bytes. */
export async function replay(
  device: GPUDevice,
  events: ObservedDispatch[],
  snapshot: ReplaySnapshot,
) {
  const keys = snapshot.bindings
    .filter((binding) => binding.resource)
    .map((binding) => binding.key)
    .toSorted();
  const candidates = events.filter(
    (event) =>
      event.label === snapshot.label &&
      JSON.stringify(event.bindings.map((binding) => binding.resource.key).toSorted()) ===
        JSON.stringify(keys),
  );
  const event = candidates[0];
  if (!event)
    throw new Error(
      `No WGSL dispatch matches ${snapshot.label}: ${keys.join(', ')}; available ${[...new Set(events.map((item) => item.label))].join(', ')}`,
    );
  const groups = new Map<number, GPUBindGroupEntry[]>();
  const append = (group: number, binding: number, resource: GPUBindingResource) => {
    const entries = groups.get(group) ?? [];
    entries.push({ binding, resource });
    groups.set(group, entries);
  };
  const buffers = new Map<ArrayBuffer, GPUBuffer>(),
    textures = new Map<HeldBinding, GPUTexture>();
  const find = (group: number, binding: number) =>
    snapshot.bindings.find((item) => item.use.group === group && item.use.binding === binding)!;
  const resources: CpuResources = {
    buffer: (group, binding) => find(group, binding).buffer!,
    texture: (group, binding) => find(group, binding).texture!,
    sampler: (group, binding) => find(group, binding).sampler!,
  };
  try {
    for (const binding of snapshot.bindings) {
      if (binding.sampler) continue;
      const gpuBinding =
        event.bindings.find(
          (item) => item.resource.key === binding.key && item.writes === binding.use.writes,
        ) ?? event.bindings.find((item) => item.resource.key === binding.key);
      if (!gpuBinding) throw new Error(`Missing WGSL binding ${binding.key}`);
      if (binding.buffer) {
        const source = binding.buffer;
        let buffer = buffers.get(source.bytes);
        if (!buffer) {
          buffer = device.createBuffer({
            size: source.bytes.byteLength,
            usage:
              GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST |
              GPUBufferUsage.UNIFORM |
              GPUBufferUsage.STORAGE,
          });
          device.queue.writeBuffer(buffer, 0, source.bytes);
          buffers.set(source.bytes, buffer);
        }
        append(gpuBinding.group, gpuBinding.binding, {
          buffer,
          offset: source.offset,
          size: source.size,
        });
      } else {
        const source = binding.texture!,
          first = source.levels[0]!;
        const texture = device.createTexture({
          size: [first.width, first.height, first.depthOrArrayLayers],
          mipLevelCount: source.levels.length,
          format: source.format,
          dimension: source.dimension === '3d' ? '3d' : '2d',
          usage:
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.TEXTURE_BINDING |
            (binding.use.kind === 'storageTexture' ? GPUTextureUsage.STORAGE_BINDING : 0),
        });
        textures.set(binding, texture);
        for (const [level, data] of source.levels.entries()) {
          const row = data.data.byteLength / data.height / data.depthOrArrayLayers;
          device.queue.writeTexture(
            { texture, mipLevel: level },
            data.data as Uint8Array<ArrayBuffer>,
            { bytesPerRow: row, rowsPerImage: data.height },
            [data.width, data.height, data.depthOrArrayLayers],
          );
        }
        append(
          gpuBinding.group,
          gpuBinding.binding,
          texture.createView({ dimension: source.dimension }),
        );
      }
    }
    for (const sampler of event.samplers)
      append(sampler.group, sampler.binding, device.createSampler(sampler.descriptor));
    const encoder = device.createCommandEncoder(),
      pass = encoder.beginComputePass();
    pass.setPipeline(event.pipeline);
    for (const [group, entries] of groups)
      pass.setBindGroup(
        group,
        device.createBindGroup({ layout: event.pipeline.getBindGroupLayout(group), entries }),
      );
    pass.dispatchWorkgroups(...snapshot.workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    compile(snapshot.code).dispatch(...snapshot.workgroups, resources);
    await device.queue.onSubmittedWorkDone();
    const cpu: CapturedResource[] = [],
      gpu: CapturedResource[] = [];
    for (const binding of snapshot.bindings.filter((item) => item.use.writes && item.resource)) {
      if (binding.buffer) {
        const bytes = binding.buffer.bytes;
        cpu.push(captured(binding, [{ width: 0, height: 0, depth: 0, bytes }]));
        gpu.push(
          captured(binding, [
            { width: 0, height: 0, depth: 0, bytes: await readBuffer(device, buffers.get(bytes)!) },
          ]),
        );
      } else {
        cpu.push(
          captured(
            binding,
            binding.texture!.levels.map((level) => ({
              width: level.width,
              height: level.height,
              depth: level.depthOrArrayLayers,
              bytes: new Uint8Array(
                level.data.buffer,
                level.data.byteOffset,
                level.data.byteLength,
              ).slice().buffer,
            })),
          ),
        );
        gpu.push(captured(binding, await readTexture(device, textures.get(binding)!)));
      }
      const encoded = event.bindings.find((item) => item.resource.key === binding.key)?.encoded;
      if (encoded) gpu.at(-1)!.encoded = encoded;
    }
    return {
      pipeline: snapshot.label,
      workgroups: snapshot.workgroups,
      ...compareCaptures(
        { resources: gpu, skipped: [], frames: 0 },
        { resources: cpu, skipped: [], frames: 0 },
      ),
    };
  } finally {
    for (const buffer of buffers.values()) buffer.destroy();
    for (const texture of textures.values()) texture.destroy();
  }
}
