import { backends } from '@typegpu/kit';
import { textureDevice } from './texture-device.ts';
import { recordingRoot } from '@typegpu/kit/testing';
import { tgpu } from 'typegpu';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile, isCpuModule, type CpuModule, type CpuResources } from '../../src/cpu/index.ts';

const noop = () => {};
export const cpuBackend: backends.Backend = {
  kind: 'webgl',
  compute: 'cpu',
  preferredCanvasFormat: 'rgba8unorm',
  configure() {
    throw new Error('CPU test device has no canvas');
  },
};

type Command = () => void;
export interface DispatchSnapshot {
  module: CpuModule;
  resources: CpuResources;
  workgroups: [number, number, number];
}
interface Pipeline {
  label?: string;
  compute: { module: CpuModule };
}

export function cpuRoot(
  options: {
    generator?: NonNullable<
      Parameters<typeof tgpu.initFromDevice>[0]['unstable_shaderGeneratorClass']
    >;
    onModule?: (code: string) => void;
    onDispatch?: (
      code: string,
      milliseconds: number,
      invocations: number,
      label: string,
      snapshot: DispatchSnapshot,
    ) => void;
  } = {},
) {
  const recording = recordingRoot();
  const original = recording.root.device;
  const sources = new WeakMap<CpuModule, string>();
  const texture = textureDevice(original);
  const limits = { ...original.limits, maxStorageBuffersPerShaderStage: 16 };
  const queue = new Proxy(original.queue, {
    get(target, key) {
      if (key === 'writeTexture') return texture.writeTexture;
      if (key === 'submit')
        return (commands: { commands: Command[] }[]) => {
          for (const buffer of commands) for (const command of buffer.commands) command();
        };
      const value = Reflect.get(target, key);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  const device = new Proxy(original, {
    get(target, key) {
      if (key === backends.brand) return cpuBackend;
      if (key === 'limits') return limits;
      if (key === 'queue') return queue;
      if (key === 'createTexture') return texture.createTexture;
      if (key === 'createSampler') return texture.createSampler;
      if (key === 'createShaderModule')
        return (descriptor: GPUShaderModuleDescriptor) => {
          options.onModule?.(descriptor.code);
          if (!isCpuModule(descriptor.code))
            return { ...original.createShaderModule(descriptor), code: descriptor.code };
          const module = {
            ...compile(descriptor.code),
            getCompilationInfo: async () => ({ messages: [] }),
          };
          sources.set(module, descriptor.code);
          return module;
        };
      if (key === 'createComputePipeline')
        return (descriptor: GPUComputePipelineDescriptor) => descriptor;
      if (key === 'createComputePipelineAsync')
        return async (descriptor: GPUComputePipelineDescriptor) => descriptor;
      if (key === 'createCommandEncoder')
        return () => {
          const commands: Command[] = [];
          return {
            beginRenderPass(descriptor: GPURenderPassDescriptor) {
              const depth = descriptor.depthStencilAttachment;
              if (depth?.depthLoadOp === 'clear')
                commands.push(() => {
                  (texture.textures.get(depth.view)!.levels[0]!.data as Float32Array).fill(
                    depth.depthClearValue ?? 1,
                  );
                });
              return {
                setPipeline: noop,
                setBindGroup: noop,
                setVertexBuffer: noop,
                setIndexBuffer: noop,
                draw: noop,
                drawIndexed: noop,
                drawIndirect: noop,
                drawIndexedIndirect: noop,
                executeBundles: noop,
                setViewport: noop,
                setScissorRect: noop,
                setBlendConstant: noop,
                setStencilReference: noop,
                pushDebugGroup: noop,
                popDebugGroup: noop,
                insertDebugMarker: noop,
                end: noop,
              };
            },
            beginComputePass() {
              let pipeline: Pipeline;
              const groups = new Map<number, GPUBindGroupDescriptor>();
              const dispatch = (x: number, y: number, z: number) => {
                const module = pipeline.compute.module;
                const label = pipeline.label ?? '';
                if (!module.dispatch)
                  throw new Error(
                    `CPU test device: raw compute shader ${(module as unknown as { code: string }).code}`,
                  );
                const bindings = new Map(groups);
                commands.push(() => {
                  const start = performance.now();
                  const resources: CpuResources = {
                    buffer(group, binding) {
                      const resource = Array.from(bindings.get(group)!.entries).find(
                        (entry) => entry.binding === binding,
                      )!.resource as GPUBufferBinding;
                      const data = recording.bytes(resource.buffer);
                      return {
                        bytes: data.buffer as ArrayBuffer,
                        offset: data.byteOffset + (resource.offset ?? 0),
                        size: resource.size ?? data.byteLength - (resource.offset ?? 0),
                      };
                    },
                    texture(group, binding) {
                      const resource = Array.from(bindings.get(group)!.entries).find(
                        (entry) => entry.binding === binding,
                      )!.resource;
                      return texture.textures.get(resource)!;
                    },
                    sampler(group, binding) {
                      const resource = Array.from(bindings.get(group)!.entries).find(
                        (entry) => entry.binding === binding,
                      )!.resource;
                      return texture.samplers.get(resource)!;
                    },
                  };
                  module.dispatch(x, y, z, resources);
                  options.onDispatch?.(
                    sources.get(module)!,
                    performance.now() - start,
                    x * y * z * module.workgroupSize.reduce((a, b) => a * b, 1),
                    label,
                    { module, resources, workgroups: [x, y, z] },
                  );
                });
              };
              return {
                setPipeline(value: Pipeline) {
                  pipeline = value;
                },
                setBindGroup(index: number, group: GPUBindGroupDescriptor) {
                  groups.set(index, group);
                },
                dispatchWorkgroups(x: number, y = 1, z = 1) {
                  dispatch(x, y, z);
                },
                dispatchWorkgroupsIndirect(buffer: GPUBuffer, offset: number) {
                  const saved = commands.length;
                  dispatch(0, 0, 0);
                  commands.splice(saved, 1);
                  const heldPipeline = pipeline;
                  const heldGroups = new Map(groups);
                  commands.push(() => {
                    const data = recording.bytes(buffer);
                    const counts = new Uint32Array(data.buffer, data.byteOffset + offset, 3);
                    pipeline = heldPipeline;
                    groups.clear();
                    heldGroups.forEach((value, index) => groups.set(index, value));
                    const before = commands.length;
                    dispatch(counts[0]!, counts[1]!, counts[2]!);
                    commands.pop()!();
                    if (commands.length !== before)
                      throw new Error('Unexpected indirect command mutation');
                  });
                },
                end() {},
              };
            },
            copyBufferToBuffer(
              source: GPUBuffer,
              sourceOffset: number,
              destination: GPUBuffer,
              targetOffset: number,
              size: number,
            ) {
              commands.push(() =>
                recording
                  .bytes(destination)
                  .set(
                    recording.bytes(source).slice(sourceOffset, sourceOffset + size),
                    targetOffset,
                  ),
              );
            },
            clearBuffer(buffer: GPUBuffer, offset = 0, size?: number) {
              commands.push(() =>
                recording
                  .bytes(buffer)
                  .fill(0, offset, size === undefined ? undefined : offset + size),
              );
            },
            finish() {
              return { commands };
            },
          };
        };
      const value = Reflect.get(target, key);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  return tgpu.initFromDevice({
    device,
    unstable_shaderGeneratorClass:
      options.generator ??
      (CpuGenerator as unknown as NonNullable<
        Parameters<typeof tgpu.initFromDevice>[0]['unstable_shaderGeneratorClass']
      >),
  });
}
