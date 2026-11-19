import type { Patch } from './patch.ts';

export type Access = 'read' | 'write' | 'readwrite' | 'unknown';

export interface GpuResource {
  id: number;
  kind: 'buffer' | 'texture' | 'canvas';
  label: string;
}

export interface GpuResourceUse {
  resource: GpuResource;
  access: Access;
}

type Pipeline = GPUComputePipeline | GPURenderPipeline;

const storageTextureAccess = {
  'write-only': 'write',
  'read-only': 'read',
  'read-write': 'readwrite',
} as const satisfies Record<GPUStorageTextureAccess, Access>;

const resources = new WeakMap<GPUBuffer | GPUTexture, GpuResource>();
const viewTextures = new WeakMap<GPUTextureView, GPUTexture>();
const canvasTextures = new WeakSet<GPUTexture>();
const bindingAccess = new WeakMap<GPUBindGroupLayout, Map<number, Access>>();
const bindGroupUses = new WeakMap<GPUBindGroup, GpuResourceUse[]>();
const shaderCode = new WeakMap<GPUShaderModule, string>();
const pipelineNames = new WeakMap<Pipeline, string>();
const pipelineCode = new WeakMap<Pipeline, string[]>();
let nextId = 0;

export function meaningfulLabel(label: string | undefined): string | undefined {
  return label && label !== '<unnamed>' ? label : undefined;
}

function resourceOf(target: GPUBuffer | GPUTexture): GpuResource {
  let resource = resources.get(target);
  if (resource) return resource;

  const kind =
    target instanceof GPUBuffer ? 'buffer' : canvasTextures.has(target) ? 'canvas' : 'texture';
  const id = nextId++;
  resource = { id, kind, label: target.label || (kind === 'canvas' ? kind : `${kind} ${id}`) };
  resources.set(target, resource);

  return resource;
}

export function usesOf(
  target: GPUBuffer | GPUTexture | GPUTextureView | null | undefined,
  access: Access,
): GpuResourceUse[] {
  const resolved = target instanceof GPUTextureView ? viewTextures.get(target) : target;
  return resolved ? [{ resource: resourceOf(resolved), access }] : [];
}

export function usesOfBindGroup(group: GPUBindGroup | null | undefined): GpuResourceUse[] {
  return (group && bindGroupUses.get(group)) ?? [];
}

export function usesOfAttachments(descriptor: GPURenderPassDescriptor): GpuResourceUse[] {
  const depth = descriptor.depthStencilAttachment;
  const depthReadOnly = depth?.depthReadOnly && (depth.stencilReadOnly ?? true);

  return [
    ...[...descriptor.colorAttachments].flatMap((attachment) =>
      attachment
        ? [...usesOf(attachment.view, 'write'), ...usesOf(attachment.resolveTarget, 'write')]
        : [],
    ),
    ...usesOf(depth?.view, depthReadOnly ? 'read' : 'write'),
  ];
}

/** Names a pipeline from its label or entry point, or from its label alone if created earlier. */
export function pipelineName(pipeline: Pipeline): string {
  return pipelineNames.get(pipeline) ?? meaningfulLabel(pipeline.label) ?? 'pipeline';
}

/** The WGSL of each shader module a pipeline uses, or none if created before patching. */
export function pipelineWgsl(pipeline: Pipeline): readonly string[] {
  return pipelineCode.get(pipeline) ?? [];
}

function accessOf(entry: GPUBindGroupLayoutEntry): Access {
  if (entry.buffer) return entry.buffer.type === 'storage' ? 'readwrite' : 'read';
  if (entry.storageTexture)
    return storageTextureAccess[entry.storageTexture.access ?? 'write-only'];
  return 'read';
}

function bindingUses(resource: GPUBindingResource, access: Access): GpuResourceUse[] {
  if (
    resource instanceof GPUBuffer ||
    resource instanceof GPUTexture ||
    resource instanceof GPUTextureView
  ) {
    return usesOf(resource, access);
  }

  return 'buffer' in resource ? usesOf(resource.buffer, access) : [];
}

function entryName(
  stage: GPUProgrammableStage | undefined,
  kind: 'compute' | 'vertex' | 'fragment',
): string | undefined {
  if (!stage) return undefined;

  const code = shaderCode.get(stage.module);
  return stage.entryPoint ?? code?.match(new RegExp(`@${kind}\\b[^{]*?\\bfn\\s+(\\w+)`))?.[1];
}

function computeEntry(descriptor: GPUComputePipelineDescriptor): string | undefined {
  return entryName(descriptor.compute, 'compute');
}

function renderEntry(descriptor: GPURenderPipelineDescriptor): string | undefined {
  return entryName(descriptor.fragment, 'fragment') ?? entryName(descriptor.vertex, 'vertex');
}

function namePipeline<P extends Pipeline>(
  pipeline: P,
  entry: string | undefined,
  stages: (GPUProgrammableStage | undefined)[],
): P {
  const name = meaningfulLabel(pipeline.label) ?? entry;
  if (name) pipelineNames.set(pipeline, name);
  const code = stages.flatMap((stage) => (stage ? (shaderCode.get(stage.module) ?? []) : []));
  pipelineCode.set(pipeline, [...new Set(code)]);
  return pipeline;
}

/** Tracks textures behind views, pipeline names and bind group contents for the panel. */
export function trackMetadata(patch: Patch) {
  // React Native worker runtimes may lack the class
  if (typeof GPUCanvasContext !== 'undefined') {
    patch(GPUCanvasContext.prototype, 'getCurrentTexture', (_self, args, next) => {
      const texture = next(...args);
      canvasTextures.add(texture);
      return texture;
    });
  }

  patch(GPUTexture.prototype, 'createView', (self, args, next) => {
    const view = next(...args);
    viewTextures.set(view, self);
    return view;
  });

  patch(GPUDevice.prototype, 'createShaderModule', (_self, [descriptor], next) => {
    const module = next(descriptor);
    shaderCode.set(module, descriptor.code);
    return module;
  });

  patch(GPUDevice.prototype, 'createComputePipeline', (_self, [descriptor], next) =>
    namePipeline(next(descriptor), computeEntry(descriptor), [descriptor.compute]),
  );

  patch(GPUDevice.prototype, 'createComputePipelineAsync', (_self, [descriptor], next) =>
    next(descriptor).then((pipeline) =>
      namePipeline(pipeline, computeEntry(descriptor), [descriptor.compute]),
    ),
  );

  patch(GPUDevice.prototype, 'createRenderPipeline', (_self, [descriptor], next) =>
    namePipeline(next(descriptor), renderEntry(descriptor), [
      descriptor.vertex,
      descriptor.fragment,
    ]),
  );

  patch(GPUDevice.prototype, 'createRenderPipelineAsync', (_self, [descriptor], next) =>
    next(descriptor).then((pipeline) =>
      namePipeline(pipeline, renderEntry(descriptor), [descriptor.vertex, descriptor.fragment]),
    ),
  );

  patch(GPUDevice.prototype, 'createBindGroupLayout', (_self, [descriptor], next) => {
    const layout = next(descriptor);
    bindingAccess.set(
      layout,
      new Map(Array.from(descriptor.entries, (entry) => [entry.binding, accessOf(entry)])),
    );
    return layout;
  });

  patch(GPUDevice.prototype, 'createBindGroup', (_self, [descriptor], next) => {
    const group = next(descriptor);
    const access = bindingAccess.get(descriptor.layout);
    bindGroupUses.set(
      group,
      Array.from(descriptor.entries).flatMap((entry) =>
        bindingUses(entry.resource, access?.get(entry.binding) ?? 'unknown'),
      ),
    );
    return group;
  });
}
