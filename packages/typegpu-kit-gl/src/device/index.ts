/**
 * A `GPUDevice` implemented on a `WebGL2RenderingContext`, for TypeGPU to run on unchanged.
 */

import { CLIP_ZERO_TO_ONE } from '../glsl/renderModule.ts';
import { makeGroup, makeLayout, makePipelineLayout } from './binding.ts';
import { bufferImpl, createBuffer, writeBytes } from './buffer.ts';
import { createPresenter } from './canvas.ts';
import { createStats, type Ctx, type DeviceStats } from './context.ts';

export type { DeviceStats } from './context.ts';
import {
  commandBufferImpl,
  createCommandEncoder,
  createRenderBundleEncoder,
  execute,
} from './encoder.ts';
import { createErrors } from './errors.ts';
import { enableExtensions } from './extensions.ts';
import { formatTable } from './formats.ts';
import { featuresOf, limitsOf } from './limits.ts';
import {
  createComputePipeline,
  createRenderPipeline,
  createRenderPipelineAsync,
} from './pipeline.ts';
import { createSampler } from './sampler.ts';
import { createShaderModule, type Prebuilt, PREBUILT } from './shader.ts';

export type { Prebuilt } from './shader.ts';
import { createState } from './state.ts';
import { createTexture, textureImpl } from './texture.ts';
import { copyExternalImageToTexture, importExternalTexture, writeTexture } from './transfer.ts';

export interface DeviceOptions {
  /**
   * The canvas the WebGL context is created on, which the device then presents to without a copy.
   * Default an `OffscreenCanvas`; every configured canvas then receives its frame as a bitmap.
   */
  readonly canvas?: HTMLCanvasElement | OffscreenCanvas;
  /** Default `'high-performance'`. */
  readonly powerPreference?: WebGLPowerPreference;
  /**
   * Runs compute kernels as fragment passes where their generator wrote a GPU form. False runs
   * every kernel on the CPU, to compare the two. Default true.
   */
  readonly gpuCompute?: boolean;
}

/** How `@typegpu/kit` recognises the fallback: `device[BACKEND]`, read by `backends.of(root)`. */
export const BACKEND: unique symbol = Symbol.for('@typegpu/kit/backend');

/** What the device reports about itself under `BACKEND`. */
export interface BackendInfo {
  readonly kind: 'webgl';
  /** Where compute pipelines run. */
  readonly compute: 'cpu';
  /** The canvas format the device presents without conversion. */
  readonly preferredCanvasFormat: GPUTextureFormat;
  /** Stands in for `canvas.getContext('webgpu')` and `context.configure`. */
  configure(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    configuration: Omit<GPUCanvasConfiguration, 'device'>,
  ): GPUCanvasContext;
}

export interface GLDevice extends GPUDevice {
  readonly gl: WebGL2RenderingContext;
  /** Time spent in CPU kernels, readback and uploads since `stats.reset()`. */
  readonly stats: DeviceStats;
  readonly [BACKEND]: BackendInfo;
}

/** Reports whether this environment can create a WebGL 2 context. */
export function isSupported(): boolean {
  if (typeof OffscreenCanvas === 'undefined') return false;
  return new OffscreenCanvas(1, 1).getContext('webgl2') !== null;
}

/**
 * Creates a shader module from a module built by hand rather than generated, for tests and for
 * shaders written in GLSL or JavaScript directly.
 */
export function shaderModuleFrom(
  device: GPUDevice,
  module: Prebuilt,
  label?: string,
): GPUShaderModule {
  return device.createShaderModule({
    code: '',
    label,
    [PREBUILT]: module,
  } as GPUShaderModuleDescriptor);
}

function typedBytes(data: AllowSharedBufferSource, dataOffset: number, size: number | undefined) {
  const unit = ArrayBuffer.isView(data) ? ((data as Uint8Array).BYTES_PER_ELEMENT ?? 1) : 1;
  const bytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data as ArrayBuffer);
  const start = dataOffset * unit;
  const length = size === undefined ? bytes.byteLength - start : size * unit;
  return bytes.subarray(start, start + length);
}

/** Creates the device. Throws if WebGL 2 is unavailable. */
export function createDevice(options: DeviceOptions = {}): GLDevice {
  const canvas = options.canvas ?? new OffscreenCanvas(1, 1);
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: true,
    preserveDrawingBuffer: false,
    powerPreference: options.powerPreference ?? 'high-performance',
  }) as WebGL2RenderingContext | null;
  if (!gl) throw new Error('@typegpu/kit-gl: WebGL 2 is not available.');

  const ext = enableExtensions(gl);
  const defines: string[] = [];
  if (ext.clipControl) {
    ext.clipControl.clipControlEXT(ext.clipControl.LOWER_LEFT_EXT, ext.clipControl.ZERO_TO_ONE_EXT);
    defines.push(`#define ${CLIP_ZERO_TO_ONE}`);
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.PACK_ALIGNMENT, 1);

  const frees = new Set<() => void>();
  let uncaptured: ((event: Event) => void) | null = null;
  const errors = createErrors(() => uncaptured);
  const formats = formatTable(gl, ext);
  const state = createState(gl, ext);
  const nearest = gl.createSampler();
  gl.samplerParameteri(nearest, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.samplerParameteri(nearest, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.samplerParameteri(nearest, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.samplerParameteri(nearest, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.samplerParameteri(nearest, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  let nextId = 1;
  let presentRecord: ((record: object) => void) | undefined;
  // Frees what an owner held once it is collected, unless the device was destroyed first
  const collected = new FinalizationRegistry<() => void>((free) => {
    if (frees.delete(free) && !ctx.lost) free();
  });
  const ctx: Ctx = {
    gl,
    canvas,
    ext,
    state,
    formats,
    errors,
    defines,
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
    maxTextureUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) as number,
    uniformAlignment: gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT) as number,
    maxUniformBindings: gl.getParameter(gl.MAX_UNIFORM_BUFFER_BINDINGS) as number,
    nearest,
    scratch: { read: gl.createFramebuffer(), draw: gl.createFramebuffer() },
    id: () => nextId++,
    track(free, owner) {
      frees.add(free);
      if (owner) collected.register(owner, free, free);
      return () => {
        frees.delete(free);
        if (owner) collected.unregister(free);
      };
    },
    presented: (record) => presentRecord?.(record),
    stats: createStats(),
    gpuCompute: options.gpuCompute ?? true,
    lost: false,
  };

  const make = (descriptor: GPUTextureDescriptor) => createTexture(ctx, descriptor);
  const presenter = createPresenter(ctx, make);
  presentRecord = presenter.presented;
  const externals = new WeakMap<object, GPUTexture>();

  let resolveLost: ((info: GPUDeviceLostInfo) => void) | undefined;
  const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
    resolveLost = resolve;
  });
  const lose = (reason: GPUDeviceLostReason, message: string) => {
    if (ctx.lost) return;
    ctx.lost = true;
    resolveLost?.({ reason, message, __brand: 'GPUDeviceLostInfo' } as GPUDeviceLostInfo);
  };
  const onContextLost = () => lose('unknown', 'The WebGL context was lost.');
  canvas.addEventListener('webglcontextlost', onContextLost as EventListener);

  const queue = {
    __brand: 'GPUQueue',
    label: '',
    submit(buffers: Iterable<GPUCommandBuffer>) {
      if (ctx.lost) return;
      for (const buffer of buffers) execute(ctx, commandBufferImpl(buffer));
    },
    writeBuffer(
      buffer: GPUBuffer,
      offset: number,
      data: AllowSharedBufferSource,
      dataOffset = 0,
      size?: number,
    ) {
      const impl = bufferImpl(buffer);
      const bytes = typedBytes(data, dataOffset, size);
      if (offset + bytes.byteLength > impl.size) {
        errors.report(
          'validation',
          `writeBuffer of ${bytes.byteLength} bytes at ${offset} overruns a ${impl.size}-byte buffer.`,
        );
        return;
      }
      writeBytes(impl, offset, bytes);
    },
    writeTexture(
      destination: GPUTexelCopyTextureInfo,
      data: AllowSharedBufferSource,
      layout: GPUTexelCopyBufferLayout,
      size: GPUExtent3DStrict,
    ) {
      try {
        writeTexture(ctx, destination, data, layout, size);
        const impl = textureImpl(destination.texture);
        if (impl.canvas) ctx.presented(impl.canvas);
      } catch (error) {
        errors.report('validation', (error as Error).message);
      }
    },
    copyExternalImageToTexture(
      source: GPUCopyExternalImageSourceInfo,
      destination: GPUCopyExternalImageDestInfo,
      size: GPUExtent3DStrict,
    ) {
      try {
        copyExternalImageToTexture(ctx, source, destination, size);
      } catch (error) {
        errors.report('validation', (error as Error).message);
      }
    },
    onSubmittedWorkDone(): Promise<undefined> {
      if (ctx.lost) return Promise.resolve(undefined);
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      return new Promise((resolve) => {
        const poll = () => {
          if (!sync || ctx.lost) {
            resolve(undefined);
            return;
          }
          const status = gl.clientWaitSync(sync, 0, 0);
          if (status === gl.TIMEOUT_EXPIRED) {
            setTimeout(poll, 1);
            return;
          }
          gl.deleteSync(sync);
          resolve(undefined);
        };
        setTimeout(poll, 0);
      });
    },
  };

  const debug = ext.debugRendererInfo;
  const adapterInfo = {
    __brand: 'GPUAdapterInfo',
    vendor: debug ? String(gl.getParameter(debug.UNMASKED_VENDOR_WEBGL)) : '',
    architecture: 'webgl2',
    device: debug ? String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)) : '',
    description: `WebGL 2 (${String(gl.getParameter(gl.VERSION))})`,
    subgroupMinSize: 0,
    subgroupMaxSize: 0,
    isFallbackAdapter: false,
  };

  let device: GLDevice;
  const backend: BackendInfo = {
    kind: 'webgl',
    compute: 'cpu',
    preferredCanvasFormat: 'rgba8unorm',
    configure: (target, configuration) => presenter.configure(target, configuration, device),
  };

  const deviceObject = {
    __brand: 'GPUDevice',
    label: '',
    gl,
    stats: ctx.stats,
    [BACKEND]: backend,
    features: featuresOf(ext, formats),
    limits: limitsOf(gl),
    adapterInfo,
    queue,
    lost,
    get onuncapturederror() {
      return uncaptured;
    },
    set onuncapturederror(handler: ((event: Event) => void) | null) {
      uncaptured = handler;
    },
    createBuffer: (descriptor: GPUBufferDescriptor) => createBuffer(ctx, descriptor),
    createTexture: make,
    createSampler: (descriptor?: GPUSamplerDescriptor) => createSampler(ctx, descriptor),
    importExternalTexture: (descriptor: GPUExternalTextureDescriptor) =>
      importExternalTexture(ctx, make, externals, descriptor),
    createBindGroupLayout: makeLayout,
    createPipelineLayout: makePipelineLayout,
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => makeGroup(ctx, descriptor),
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) =>
      createShaderModule(ctx, descriptor),
    createComputePipeline: (descriptor: GPUComputePipelineDescriptor) =>
      createComputePipeline(ctx, descriptor),
    createComputePipelineAsync: (descriptor: GPUComputePipelineDescriptor) =>
      Promise.resolve(createComputePipeline(ctx, descriptor)),
    createRenderPipeline: (descriptor: GPURenderPipelineDescriptor) =>
      createRenderPipeline(ctx, descriptor),
    createRenderPipelineAsync: (descriptor: GPURenderPipelineDescriptor) =>
      createRenderPipelineAsync(ctx, descriptor),
    createCommandEncoder: (descriptor?: GPUCommandEncoderDescriptor) =>
      createCommandEncoder(ctx, descriptor),
    createRenderBundleEncoder: createRenderBundleEncoder,
    createQuerySet(descriptor: GPUQuerySetDescriptor) {
      return {
        __brand: 'GPUQuerySet',
        label: descriptor.label ?? '',
        type: descriptor.type,
        count: descriptor.count,
        destroy() {},
      } as unknown as GPUQuerySet;
    },
    pushErrorScope: (filter: GPUErrorFilter) => errors.push(filter),
    popErrorScope: () => errors.pop(),
    addEventListener: errors.target.addEventListener.bind(errors.target),
    removeEventListener: errors.target.removeEventListener.bind(errors.target),
    dispatchEvent: errors.target.dispatchEvent.bind(errors.target),
    destroy() {
      if (ctx.lost) return;
      presenter.destroy();
      for (const free of frees) free();
      frees.clear();
      gl.deleteSampler(nearest);
      gl.deleteFramebuffer(ctx.scratch.read);
      gl.deleteFramebuffer(ctx.scratch.draw);
      canvas.removeEventListener('webglcontextlost', onContextLost as EventListener);
      lose('destroyed', 'The device was destroyed.');
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
  device = deviceObject as unknown as GLDevice;
  return device;
}
