import { encodedFields, type EncodedField } from './encoded-fields.ts';
import type { Layout } from '../../src/cpu/memory.ts';
import type { CpuTexture } from '../../src/cpu/types.ts';

export interface Resource {
  key: string;
  label: string;
  kind: 'buffer' | 'texture';
  handle: GPUBuffer | GPUTexture;
  written: boolean;
  destroyed: boolean;
  reasons: Set<string>;
  order: Set<string>;
  halfStoreLinks?: number;
  approximate: boolean;
  quotient: boolean;
  layout?: Layout | undefined;
  layoutOffset?: number;
  format?: GPUTextureFormat;
  encoded?: EncodedField[];
}
export interface Binding {
  encoded?: EncodedField[];
  group: number;
  binding: number;
  writes: boolean;
  layout?: Layout | undefined;
}
interface Shader {
  code: string;
  bindings: Binding[];
}
interface Pipeline extends Shader {
  label: string;
}

function origin(): string {
  const line = (new Error().stack ?? '')
    .split('\n')
    .find(
      (item) => /\/typegpu-kit(?:-examples)?\/src\//.test(item) && !item.includes('/targets.ts'),
    );
  return /\/(typegpu-kit(?:-examples)?\/src\/[^?:)]+)/.exec(line ?? '')?.[1] ?? 'host';
}

function shader(code: string): Shader {
  const layouts = [...code.matchAll(/^\/\/#test-layout (.+)$/gm)].map(
    (match) => JSON.parse(match[1]!) as Binding,
  );
  if (code.includes('//#kit-gl compute'))
    return {
      code,
      bindings: [...code.matchAll(/^\/\/#binding (.+)$/gm)].map((match) => {
        const use = JSON.parse(match[1]!) as Binding;
        return {
          ...use,
          layout: layouts.find(
            (item) => Number(item.group) === Number(use.group) && item.binding === use.binding,
          )?.layout,
        };
      }),
    };
  const bindings: Binding[] = [];
  for (const match of code.matchAll(
    /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?\s+(\w+)\s*:\s*([^;]+);/g,
  ))
    bindings.push({
      group: Number(match[1]),
      binding: Number(match[2]),
      encoded: encodedFields(code, match[4]!),
      writes:
        /read_write/.test(match[3] ?? '') ||
        /texture_storage.*(?:write|read_write)/.test(match[5]!),
    });
  return { code, bindings };
}

function wrap<T extends object>(target: T, overrides: Record<string, unknown>): T {
  return new Proxy(target, {
    get(object, key) {
      if (typeof key === 'string' && key in overrides) return overrides[key];
      const value: unknown = Reflect.get(object, key);
      return typeof value === 'function' ? value.bind(object) : value;
    },
  });
}

export interface ObservedDispatch {
  pipeline: GPUComputePipeline;
  label: string;
  code: string;
  bindings: (Binding & { resource: Resource })[];
  samplers: { group: number; binding: number; descriptor: GPUSamplerDescriptor }[];
}

export function observeScene() {
  const resources: Resource[] = [];
  const dispatches: ObservedDispatch[] = [];
  const handles = new WeakMap<object, Resource>();
  const groups = new WeakMap<object, GPUBindGroupDescriptor>();
  const samplers = new WeakMap<object, GPUSamplerDescriptor>();
  const shaders = new WeakMap<object, Shader>();
  const pipelines = new WeakMap<object, Pipeline>();
  const counts = new Map<string, number>();
  const commands = new WeakMap<object, (() => void)[]>();
  const cpuTextures = new WeakMap<object, CpuTexture>();
  const add = (
    handle: GPUBuffer | GPUTexture,
    descriptor: GPUBufferDescriptor | GPUTextureDescriptor,
    kind: Resource['kind'],
  ) => {
    const base = `${kind}:${descriptor.label ?? '<unnamed>'}:${origin()}`;
    const count = counts.get(base) ?? 0;
    counts.set(base, count + 1);
    const item: Resource = {
      key: `${base}#${count}`,
      label: descriptor.label ?? '<unnamed>',
      handle,
      kind,
      written: false,
      destroyed: false,
      reasons: new Set(),
      order: new Set(),
      approximate: false,
      quotient: false,
    };
    if (
      kind === 'buffer' &&
      descriptor.usage & (GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE)
    )
      item.reasons.add(
        'host mapping buffer without COPY_SRC; shader output is compared at its source',
      );
    resources.push(item);
    handles.set(handle, item);
    const destroy = handle.destroy.bind(handle);
    handle.destroy = () => {
      item.destroyed = true;
      destroy();
    };
    if (kind === 'texture') {
      const texture = handle as GPUTexture;
      item.format = texture.format;
      const create = texture.createView.bind(texture);
      return wrap(texture, {
        destroy: handle.destroy,
        createView: (view?: GPUTextureViewDescriptor) => {
          const made = create(view);
          handles.set(made, item);
          return made;
        },
      });
    }
    return handle;
  };
  const inherit = (outputs: Resource[], inputs: Resource[]) => {
    for (const output of outputs)
      for (const input of inputs) {
        for (const reason of input.reasons) output.reasons.add(reason);
        for (const order of input.order) output.order.add(order);
        output.halfStoreLinks = Math.max(output.halfStoreLinks ?? 0, input.halfStoreLinks ?? 0);
        output.approximate ||= input.approximate;
        output.quotient ||= input.quotient;
      }
  };
  const device = (original: GPUDevice): GPUDevice =>
    wrap(original, {
      createBuffer: (descriptor: GPUBufferDescriptor) => {
        const mapped = descriptor.usage & (GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE);
        return add(
          original.createBuffer({
            ...descriptor,
            usage: descriptor.usage | (mapped ? 0 : GPUBufferUsage.COPY_SRC),
          }),
          descriptor,
          'buffer',
        );
      },
      createTexture: (descriptor: GPUTextureDescriptor) => {
        const handle = original.createTexture({
          ...descriptor,
          usage:
            descriptor.usage | ((descriptor.sampleCount ?? 1) === 1 ? GPUTextureUsage.COPY_SRC : 0),
        });
        const result = add(handle, descriptor, 'texture');
        handles.set(result, handles.get(handle)!);
        return result;
      },
      createSampler: (descriptor: GPUSamplerDescriptor = {}) => {
        const made = original.createSampler(descriptor);
        samplers.set(made, descriptor);
        return made;
      },
      createBindGroup: (descriptor: GPUBindGroupDescriptor) => {
        const made = original.createBindGroup(descriptor);
        groups.set(made, descriptor);
        return made;
      },
      createShaderModule: (descriptor: GPUShaderModuleDescriptor) => {
        const made = original.createShaderModule(descriptor);
        shaders.set(made, shader(descriptor.code));
        return made;
      },
      createComputePipeline: (descriptor: GPUComputePipelineDescriptor) => {
        const made = original.createComputePipeline(descriptor);
        pipelines.set(made, {
          ...shaders.get(descriptor.compute.module)!,
          label: descriptor.label ?? descriptor.compute.entryPoint ?? '<unnamed>',
        });
        return made;
      },
      createComputePipelineAsync: async (descriptor: GPUComputePipelineDescriptor) => {
        const made = await original.createComputePipelineAsync(descriptor);
        pipelines.set(made, {
          ...shaders.get(descriptor.compute.module)!,
          label: descriptor.label ?? descriptor.compute.entryPoint ?? '<unnamed>',
        });
        return made;
      },
      queue: wrap(original.queue, {
        writeTexture: (
          destination: GPUTexelCopyTextureInfo,
          data: GPUAllowSharedBufferSource,
          layout: GPUTexelCopyBufferLayout,
          size: GPUExtent3D,
        ) =>
          original.queue.writeTexture(
            {
              ...destination,
              texture: (handles.get(destination.texture)?.handle ??
                destination.texture) as GPUTexture,
            },
            data,
            layout,
            size,
          ),
        submit: (buffers: Iterable<GPUCommandBuffer>) => {
          const list = Array.from(buffers);
          for (const buffer of list) for (const command of commands.get(buffer) ?? []) command();
          original.queue.submit(list);
        },
      }),
      createCommandEncoder: (descriptor?: GPUCommandEncoderDescriptor) => {
        const encoder = original.createCommandEncoder(descriptor);
        const recorded: (() => void)[] = [];
        return wrap(encoder, {
          beginRenderPass: (description: GPURenderPassDescriptor) => {
            const targets = [
              ...Array.from(description.colorAttachments).flatMap((attachment) =>
                attachment ? [attachment.view, attachment.resolveTarget].filter(Boolean) : [],
              ),
              description.depthStencilAttachment?.view,
            ].filter(Boolean);
            recorded.push(() => {
              for (const target of targets)
                handles.get(target as object)?.reasons.add('written by a render pass');
            });
            return encoder.beginRenderPass(description);
          },
          beginComputePass: (description?: GPUComputePassDescriptor) => {
            const pass = encoder.beginComputePass(description);
            let selected: Pipeline | undefined;
            let pipelineHandle: GPUComputePipeline;
            const bound = new Map<number, { group: GPUBindGroup; offsets: number[] }>();
            const dispatch = () => {
              const pipeline = selected!;
              const used = pipeline.bindings.flatMap((binding) => {
                const group = bound.get(Number(binding.group));
                const entry =
                  group &&
                  Array.from(groups.get(group.group)!.entries).find(
                    (item) => item.binding === binding.binding,
                  );
                if (!entry) return [];
                const buffer = entry.resource as GPUBufferBinding;
                const resource = handles.get(buffer.buffer ?? entry.resource);
                if (!resource) return [];
                if (binding.encoded?.length) resource.encoded = binding.encoded;
                if (binding.layout) {
                  resource.layout = binding.layout;
                  resource.layoutOffset = buffer.offset ?? 0;
                }
                return [{ ...binding, resource }];
              });
              const observed = {
                pipeline: pipelineHandle,
                label: pipeline.label,
                code: pipeline.code,
                bindings: used,
                samplers: pipeline.bindings.flatMap((binding) => {
                  const group = bound.get(Number(binding.group));
                  const entry =
                    group &&
                    Array.from(groups.get(group.group)!.entries).find(
                      (item) => item.binding === binding.binding,
                    );
                  const samplerDescriptor = entry && samplers.get(entry.resource);
                  return samplerDescriptor
                    ? [
                        {
                          group: Number(binding.group),
                          binding: binding.binding,
                          descriptor: samplerDescriptor,
                        },
                      ]
                    : [];
                }),
              };
              recorded.push(() => {
                dispatches.push(observed);
                const outputs = used.filter((item) => item.writes).map((item) => item.resource);
                inherit(
                  outputs,
                  used.map((item) => item.resource),
                );
                const halfDepth = Math.max(
                  0,
                  ...used.map((item) => item.resource.halfStoreLinks ?? 0),
                );
                for (const output of outputs) {
                  if (output.format?.includes('16float')) output.halfStoreLinks = halfDepth + 1;
                  output.written = true;
                  output.quotient ||= /[/%]/.test(pipeline.code.replace(/\/\/[^\n]*/g, ''));
                  output.approximate ||=
                    /\b(?:sin|cos|pow|exp|log|sqrt|normalize|inverseSqrt|atan2?)\s*\(/.test(
                      pipeline.code,
                    );
                  if (/\batomic(?:Add|Sub|Exchange|CompareExchangeWeak)\s*\(/.test(pipeline.code))
                    output.order.add(
                      `${pipeline.label}: atomic allocation or exchange; invocation ordering needs a module-specific invariant`,
                    );
                }
              });
            };
            return wrap(pass, {
              setPipeline: (value: GPUComputePipeline) => {
                pipelineHandle = value;
                selected = pipelines.get(value);
                pass.setPipeline(value);
              },
              setBindGroup: (index: number, group: GPUBindGroup, offsets?: Iterable<number>) => {
                bound.set(index, { group, offsets: Array.from(offsets ?? []) });
                pass.setBindGroup(index, group, offsets);
              },
              dispatchWorkgroups: (x: number, y?: number, z?: number) => {
                if (x === 0 || y === 0 || z === 0) return;
                dispatch();
                pass.dispatchWorkgroups(x, y, z);
              },
              dispatchWorkgroupsIndirect: (buffer: GPUBuffer, offset: number) => {
                dispatch();
                pass.dispatchWorkgroupsIndirect(buffer, offset);
              },
            });
          },
          copyBufferToBuffer: (
            source: GPUBuffer,
            from: number,
            target: GPUBuffer,
            to: number,
            size: number,
          ) => {
            recorded.push(() => {
              const a = handles.get(source),
                b = handles.get(target);
              if (a && b) {
                inherit([b], [a]);
                b.written ||= a.written;
                b.layout ??= a.layout;
                b.layoutOffset ??= (a.layoutOffset ?? 0) + to - from;
              }
            });
            encoder.copyBufferToBuffer(source, from, target, to, size);
          },
          copyTextureToTexture: (
            source: GPUTexelCopyTextureInfo,
            target: GPUTexelCopyTextureInfo,
            size: GPUExtent3D,
          ) => {
            recorded.push(() => {
              const a = handles.get(source.texture),
                b = handles.get(target.texture);
              if (a && b) {
                inherit([b], [a]);
                b.written ||= a.written;
              }
            });
            encoder.copyTextureToTexture(
              {
                ...source,
                texture: (handles.get(source.texture)?.handle ?? source.texture) as GPUTexture,
              },
              {
                ...target,
                texture: (handles.get(target.texture)?.handle ?? target.texture) as GPUTexture,
              },
              size,
            );
          },
          copyTextureToBuffer: (
            source: GPUTexelCopyTextureInfo,
            target: GPUTexelCopyBufferInfo,
            size: GPUExtent3D,
          ) => {
            recorded.push(() => {
              const a = handles.get(source.texture),
                b = handles.get(target.buffer);
              if (a && b) {
                inherit([b], [a]);
                b.written ||= a.written;
              }
            });
            encoder.copyTextureToBuffer(
              {
                ...source,
                texture: (handles.get(source.texture)?.handle ?? source.texture) as GPUTexture,
              },
              target,
              size,
            );
          },
          finish: (description?: GPUCommandBufferDescriptor) => {
            const made = encoder.finish(description);
            commands.set(made, recorded);
            return made;
          },
        });
      },
    });
  return {
    device,
    resources,
    cpuTextures,
    dispatches,
    resourceOf: (handle: object) => handles.get(handle),
  };
}
