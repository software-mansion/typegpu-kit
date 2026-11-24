import { textureDevice } from './texture-device.ts';
import { recordingRoot } from '@typegpu/kit/testing';
import { tgpu } from 'typegpu';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile, type CpuModule } from '../../src/cpu/index.ts';

type Command = () => void;
interface Pipeline {
  compute: { module: CpuModule };
}

export function cpuRoot() {
  const recording = recordingRoot();
  const original = recording.root.device;
  const texture = textureDevice(original);
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
      if (key === 'queue') return queue;
      if (key === 'createTexture') return texture.createTexture;
      if (key === 'createSampler') return texture.createSampler;
      if (key === 'createShaderModule')
        return ({ code }: GPUShaderModuleDescriptor) => ({
          ...compile(code),
          getCompilationInfo: async () => ({ messages: [] }),
        });
      if (key === 'createComputePipeline')
        return (descriptor: GPUComputePipelineDescriptor) => descriptor;
      if (key === 'createComputePipelineAsync')
        return async (descriptor: GPUComputePipelineDescriptor) => descriptor;
      if (key === 'createCommandEncoder')
        return () => {
          const commands: Command[] = [];
          return {
            beginComputePass() {
              let pipeline: Pipeline;
              const groups = new Map<number, GPUBindGroupDescriptor>();
              const dispatch = (x: number, y: number, z: number) => {
                const module = pipeline.compute.module;
                const bindings = new Map(groups);
                commands.push(() =>
                  module.dispatch(x, y, z, {
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
                  }),
                );
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
    unstable_shaderGeneratorClass: CpuGenerator as unknown as NonNullable<
      Parameters<typeof tgpu.initFromDevice>[0]['unstable_shaderGeneratorClass']
    >,
  });
}
