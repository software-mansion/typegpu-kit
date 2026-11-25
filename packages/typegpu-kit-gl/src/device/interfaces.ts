/**
 * Stand-ins for WebGPU's interface objects, such as `GPUCommandEncoder`, for the device's own
 * objects. A tool that patches WebGPU's prototypes, as `@typegpu/kit/debug` does, patches these
 * too when given them: the device's devices, queues, encoders, passes and textures take their
 * methods from these prototypes, which `instanceof` checks.
 */

const METHODS = {
  GPUDevice: [
    'createBuffer',
    'createTexture',
    'createSampler',
    'importExternalTexture',
    'createBindGroupLayout',
    'createPipelineLayout',
    'createBindGroup',
    'createShaderModule',
    'createComputePipeline',
    'createComputePipelineAsync',
    'createRenderPipeline',
    'createRenderPipelineAsync',
    'createCommandEncoder',
    'createRenderBundleEncoder',
    'createQuerySet',
    'pushErrorScope',
    'popErrorScope',
    'destroy',
  ],
  GPUQueue: [
    'submit',
    'writeBuffer',
    'writeTexture',
    'copyExternalImageToTexture',
    'onSubmittedWorkDone',
  ],
  GPUCommandEncoder: [
    'beginRenderPass',
    'beginComputePass',
    'copyBufferToBuffer',
    'copyBufferToTexture',
    'copyTextureToBuffer',
    'copyTextureToTexture',
    'clearBuffer',
    'resolveQuerySet',
    'writeTimestamp',
    'pushDebugGroup',
    'popDebugGroup',
    'insertDebugMarker',
    'finish',
  ],
  GPUComputePassEncoder: [
    'setPipeline',
    'setBindGroup',
    'dispatchWorkgroups',
    'dispatchWorkgroupsIndirect',
    'pushDebugGroup',
    'popDebugGroup',
    'insertDebugMarker',
    'end',
  ],
  GPURenderPassEncoder: [
    'setPipeline',
    'setBindGroup',
    'setVertexBuffer',
    'setIndexBuffer',
    'draw',
    'drawIndexed',
    'drawIndirect',
    'drawIndexedIndirect',
    'setViewport',
    'setScissorRect',
    'setBlendConstant',
    'setStencilReference',
    'beginOcclusionQuery',
    'endOcclusionQuery',
    'executeBundles',
    'pushDebugGroup',
    'popDebugGroup',
    'insertDebugMarker',
    'end',
  ],
  GPUTexture: ['createView', 'destroy'],
} as const;

type Name = keyof typeof METHODS;

/** Where an object keeps the methods its prototype calls, so a patch of the prototype sees them. */
const OWN = Symbol('@typegpu/kit-gl/methods');

type Methods = Record<string, (...args: unknown[]) => unknown>;

/** An interface object: a prototype to patch, and `instanceof` for the device's objects. */
export interface Interface<T> {
  readonly prototype: T;
  [Symbol.hasInstance](value: unknown): boolean;
}

function makeInterface(name: Name): Interface<object> {
  const prototype: Methods = {};
  for (const key of METHODS[name]) {
    // A function expression, so the method takes the receiver its caller gives
    prototype[key] = function (this: { [OWN]: Methods }, ...args: unknown[]) {
      return this[OWN][key]!.apply(this, args);
    };
  }
  return {
    prototype,
    [Symbol.hasInstance]: (value: unknown) =>
      typeof value === 'object' &&
      value !== null &&
      Object.prototype.isPrototypeOf.call(prototype, value),
  };
}

/** WebGPU's interface objects for the device's objects, by WebGPU's names. */
export const interfaces = {
  GPUDevice: makeInterface('GPUDevice') as Interface<GPUDevice>,
  GPUQueue: makeInterface('GPUQueue') as Interface<GPUQueue>,
  GPUCommandEncoder: makeInterface('GPUCommandEncoder') as Interface<GPUCommandEncoder>,
  GPUComputePassEncoder: makeInterface('GPUComputePassEncoder') as Interface<GPUComputePassEncoder>,
  GPURenderPassEncoder: makeInterface('GPURenderPassEncoder') as Interface<GPURenderPassEncoder>,
  GPUTexture: makeInterface('GPUTexture') as Interface<GPUTexture>,
};

/** Moves `object`'s methods of interface `name` behind that interface's prototype. */
export function shaped<T extends object>(name: Name, object: T): T {
  const own: Methods = {};
  const record = object as unknown as Methods;
  for (const key of METHODS[name]) {
    const method = record[key];
    if (typeof method !== 'function') continue;
    own[key] = method;
    delete record[key];
  }
  Object.defineProperty(object, OWN, { value: own });
  Object.setPrototypeOf(object, interfaces[name].prototype);
  return object;
}
