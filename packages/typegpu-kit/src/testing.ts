/**
 * Runs scene frames in tests without a GPU. Provides a root that records WebGPU calls, a surface
 * without a canvas and a schedule you step by hand.
 *
 * @module
 * @kind tool
 * @flat
 * @owns a texture: an offscreen surface's, replaced when it resizes
 */

import { isBuffer, tgpu, type d, type TgpuBuffer, type TgpuRoot } from 'typegpu';
import type { Schedule } from './animate.ts';
import { checkSize } from './canvases.ts';
import type { FixedSize, Surface } from './targets.ts';

/**
 * Records WebGPU calls and object creation. Object counts exclude `submissions`. Use them to check
 * that steady frames create no objects.
 */
export interface RecordingLog {
  /** Stores the WGSL of every shader module in creation order. */
  readonly shaders: string[];
  /** Lists pass, draw, dispatch and submit calls in recording order. */
  readonly events: string[];
  /**
   * Lists the `setPipeline` and `setBindGroup` calls that passes receive, as `set-pipeline` and
   * `set-bind-group`, in recording order.
   */
  readonly state: string[];
  /** Records every `queue.writeBuffer` in call order, with sizes in bytes. */
  readonly writes: { buffer: GPUBuffer; offset: number; size: number }[];
  buffers: number;
  textures: number;
  views: number;
  bindGroups: number;
  samplers: number;
  querySets: number;
  pipelines: number;
  /**
   * Counts the pipelines created without `…Async`, which hold the thread until they compile. A
   * frame should add none: compile while loading, with `pipelines.compile`.
   */
  blocking: number;
  submissions: number;
  /** Returns counts of created objects. Steady frames should leave these counts unchanged. */
  counts(): RecordingCounts;
  /**
   * Lists buffers, textures and query sets that have not been destroyed, in creation order, as
   * `[Buffer "label"]`. With `uniforms: false`, leaves out uniform buffers. Compare that list
   * before creation and after `destroy()` to check explicit cleanup.
   */
  alive(options?: AliveOptions): string[];
}

export interface AliveOptions {
  /** Lists uniform buffers too. Default true. */
  readonly uniforms?: boolean;
}

/** Counts created objects as returned by `log.counts()`. Excludes submissions. */
export interface RecordingCounts {
  readonly buffers: number;
  readonly textures: number;
  readonly views: number;
  readonly bindGroups: number;
  readonly samplers: number;
  readonly querySets: number;
  readonly pipelines: number;
}

export interface RecordingRoot {
  /** Provides a TypeGPU root on a device that records instead of drawing. */
  readonly root: TgpuRoot;
  /** Records the calls and objects the device sees. */
  readonly log: RecordingLog;
  /**
   * Returns the buffer bytes after queued writes and submitted copies. Changes to the returned
   * array change the buffer.
   */
  bytes(buffer: GPUBuffer | TgpuBuffer<d.BaseData>): Uint8Array;
}

export interface RecordingOptions {
  /**
   * Sets the device features reported by the recording root, such as `'timestamp-query'`. Default
   * none.
   */
  readonly features?: readonly string[];
  /**
   * Overrides device limits, such as a small `maxStorageBufferBindingSize` to reach a size guard.
   * Default the WebGPU default limits.
   */
  readonly limits?: Partial<Record<keyof GPUSupportedLimits, number>>;
}

/** The limits every WebGPU device supports, which a recording root reports. */
const defaultLimits = {
  maxTextureDimension1D: 8192,
  maxTextureDimension2D: 8192,
  maxTextureDimension3D: 2048,
  maxTextureArrayLayers: 256,
  maxBindGroups: 4,
  maxBindGroupsPlusVertexBuffers: 24,
  maxBindingsPerBindGroup: 1000,
  maxDynamicUniformBuffersPerPipelineLayout: 8,
  maxDynamicStorageBuffersPerPipelineLayout: 4,
  maxSampledTexturesPerShaderStage: 16,
  maxSamplersPerShaderStage: 16,
  maxStorageBuffersPerShaderStage: 8,
  maxStorageTexturesPerShaderStage: 4,
  maxUniformBuffersPerShaderStage: 12,
  maxUniformBufferBindingSize: 65536,
  maxStorageBufferBindingSize: 134217728,
  minUniformBufferOffsetAlignment: 256,
  minStorageBufferOffsetAlignment: 256,
  maxVertexBuffers: 8,
  maxBufferSize: 268435456,
  maxVertexAttributes: 16,
  maxVertexBufferArrayStride: 2048,
  maxInterStageShaderVariables: 16,
  maxColorAttachments: 8,
  maxColorAttachmentBytesPerSample: 32,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
};

const webgpuConstants = {
  GPUBufferUsage: {
    MAP_READ: 1,
    MAP_WRITE: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    INDEX: 16,
    VERTEX: 32,
    UNIFORM: 64,
    STORAGE: 128,
    INDIRECT: 256,
    QUERY_RESOLVE: 512,
  },
  GPUTextureUsage: {
    COPY_SRC: 1,
    COPY_DST: 2,
    TEXTURE_BINDING: 4,
    STORAGE_BINDING: 8,
    RENDER_ATTACHMENT: 16,
    TRANSIENT_ATTACHMENT: 32,
  },
  GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  GPUMapMode: { READ: 1, WRITE: 2 },
  GPUColorWrite: { RED: 1, GREEN: 2, BLUE: 4, ALPHA: 8, ALL: 15 },
};

type Usages = Record<string, number>;
const { GPUBufferUsage: BUFFER, GPUTextureUsage: TEXTURE } = webgpuConstants;

const bindingUsage = {
  uniform: ['UNIFORM', BUFFER],
  storage: ['STORAGE', BUFFER],
  'read-only-storage': ['STORAGE', BUFFER],
  texture: ['TEXTURE_BINDING', TEXTURE],
  storageTexture: ['STORAGE_BINDING', TEXTURE],
} as const satisfies Record<string, readonly [string, Usages]>;

const kindOf = (entry: GPUBindGroupLayoutEntry | undefined) =>
  entry?.buffer
    ? (entry.buffer.type ?? 'uniform')
    : entry?.texture
      ? 'texture'
      : entry?.storageTexture
        ? 'storageTexture'
        : undefined;

interface Destroyable {
  readonly label: string;
  readonly kind: 'Buffer' | 'Texture' | 'QuerySet';
}

/** Submitting work that uses a destroyed buffer or texture must throw. */
interface Resource extends Destroyable {
  readonly kind: 'Buffer' | 'Texture';
  readonly usage: number;
  destroyed: boolean;
}

/** Tells a buffer bound only as a uniform, such as one from `root.createUniform`. */
const isUniform = (each: Destroyable) => {
  if (each.kind !== 'Buffer') return false;
  const { usage } = each as Resource;
  const bound = BUFFER.STORAGE | BUFFER.VERTEX | BUFFER.INDEX | BUFFER.INDIRECT;
  return (usage & BUFFER.UNIFORM) !== 0 && (usage & bound) === 0;
};

/** What a render bundle recorded, replayed into the pass that executes it. */
interface Bundle {
  readonly events: string[];
  readonly state: string[];
  readonly used: Set<Resource>;
}

/** Members that code probes on any object, such as `then` when a promise resolves with it. */
const optional = new Set([
  'then',
  'toJSON',
  'constructor',
  'asymmetricMatch',
  'nodeType',
  '$$typeof',
]);

const nameOf = (resource: Destroyable) => `[${resource.kind} "${resource.label}"]`;

const usageNames = (usage: number, usages: Usages) =>
  Object.keys(usages)
    .filter((name) => usage & usages[name])
    .join('|');

function need(resource: Resource, usage: string, usages: Usages, what = nameOf(resource)) {
  if (!(resource.usage & usages[usage])) {
    throw new Error(
      `${what} usage (${usageNames(resource.usage, usages)}) doesn't include ${usage}.`,
    );
  }
}

function live(resource: Resource) {
  if (resource.destroyed)
    throw new Error(`Cannot submit work that uses destroyed ${nameOf(resource)}.`);
}

function fit(what: string, buffer: Resource & { size: number }, offset: number, size: number) {
  if (offset % 4 !== 0) throw new Error(`${what} offset (${offset}) is not a multiple of 4 bytes.`);
  if (size % 4 !== 0) throw new Error(`${what} size (${size}) is not a multiple of 4 bytes.`);
  if (offset + size > buffer.size) {
    throw new Error(
      `${what} range (offset: ${offset}, size: ${size}) does not fit in ${nameOf(buffer)} with size (${buffer.size}).`,
    );
  }
}

/**
 * Creates a recording WebGPU test double under a TypeGPU root. TypeGPU resolves shaders to WGSL.
 * The double keeps the WGSL and records commands without compiling or executing shaders. Use it to
 * check resources, pipelines and recording order without a GPU. Queue writes take effect, and
 * submitted buffer copies and clears update bytes, so tests can check uploads and readbacks.
 *
 * Throws for invalid mapping usage, out-of-bounds or unaligned copies and writes, missing resource
 * usage flags, writes to destroyed buffers, and mapping in the wrong state. Also throws for
 * non-attachment render targets, bind groups with incompatible resource usages, finishing an
 * encoder with an open pass, and submitting destroyed resources. Throws at the call, where a
 * device would report most of these errors later. Reports WebGPU's default limits and records
 * render bundles, which replay into the pass that executes them. A device member it lacks throws
 * an error that names it.
 *
 * @returns the root, a log of what it saw, and the bytes it holds
 */
export function recordingRoot(options: RecordingOptions = {}): RecordingRoot {
  for (const [name, value] of Object.entries(webgpuConstants)) {
    if (!(name in globalThis)) Object.assign(globalThis, { [name]: value });
  }

  const living = new Set<Destroyable>();

  const log: RecordingLog = {
    shaders: [],
    events: [],
    state: [],
    writes: [],
    buffers: 0,
    textures: 0,
    views: 0,
    bindGroups: 0,
    samplers: 0,
    querySets: 0,
    pipelines: 0,
    blocking: 0,
    submissions: 0,

    counts() {
      const { buffers, textures, views, bindGroups, samplers, querySets, pipelines } = log;
      return { buffers, textures, views, bindGroups, samplers, querySets, pipelines };
    },

    alive: ({ uniforms = true } = {}) =>
      Array.from(living)
        .filter((each) => uniforms || !isUniform(each))
        .map(nameOf),
  };

  const contents = new WeakMap<object, Uint8Array>();
  const bytes = (buffer: GPUBuffer | TgpuBuffer<d.BaseData>) => {
    const data = contents.get(isBuffer(buffer) ? buffer.buffer : buffer);
    if (!data) throw new Error('The buffer does not belong to this recording root.');
    return data;
  };

  // Track resources behind views and bind groups for lifetime checks at submission
  const textureOf = new WeakMap<object, Resource>();
  const heldBy = new WeakMap<object, Resource[]>();

  const event = (name: string) => () => {
    log.events.push(name);
  };

  /** Records state, draws and the resources they use, for a pass or a render bundle. */
  const recorder = (used: Set<Resource>, events: string[], state: string[]) => {
    const use = (...resources: unknown[]) => {
      for (const resource of resources) {
        if (contents.has(resource as object)) used.add(resource as Resource);
      }
    };
    const record = (name: string) => () => {
      events.push(name);
    };
    const indirect = (name: string) => (buffer: GPUBuffer) => {
      use(buffer);
      events.push(name);
    };

    return {
      use,
      indirect,
      common: {
        setPipeline() {
          state.push('set-pipeline');
        },
        setBindGroup(_index: number, group: object | null) {
          for (const resource of (group && heldBy.get(group)) ?? []) used.add(resource);
          state.push('set-bind-group');
        },
        pushDebugGroup() {},
        popDebugGroup() {},
        insertDebugMarker() {},
      },
      draws: {
        setVertexBuffer: (_slot: number, buffer: GPUBuffer | null) => use(buffer),
        setIndexBuffer: (buffer: GPUBuffer) => use(buffer),
        draw: record('draw'),
        drawIndexed: record('draw-indexed'),
        drawIndirect: indirect('draw-indirect'),
        drawIndexedIndirect: indirect('draw-indexed-indirect'),
      },
    };
  };

  const compile = () => {
    log.pipelines += 1;
    return { getBindGroupLayout: () => ({}) };
  };

  const block = () => {
    log.blocking += 1;
    return compile();
  };

  const device = {
    label: 'recording device',
    features: new Set<string>(options.features),

    limits: Object.freeze({ ...defaultLimits, ...options.limits }),

    lost: new Promise(() => {}),

    queue: {
      submit(commandBuffers: { copies: (() => void)[]; used: Set<Resource> }[]) {
        for (const { used } of commandBuffers) for (const resource of used) live(resource);

        log.submissions += 1;
        log.events.push('submit');
        for (const buffer of commandBuffers) for (const copy of buffer.copies) copy();
      },

      writeBuffer(
        buffer: GPUBuffer & Resource,
        offset: number,
        data: AllowSharedBufferSource,
        dataOffset = 0,
        size?: number,
      ) {
        // Typed-array offsets and sizes count elements. Other inputs count bytes
        const unit = ArrayBuffer.isView(data) ? ((data as Uint8Array).BYTES_PER_ELEMENT ?? 1) : 1;
        const source = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
          : new Uint8Array(data);

        const start = dataOffset * unit;
        const length = size === undefined ? source.byteLength - start : size * unit;

        live(buffer);
        need(buffer, 'COPY_DST', BUFFER);
        fit('Write', buffer, offset, length);

        bytes(buffer).set(source.subarray(start, start + length), offset);
        log.writes.push({ buffer, offset, size: length });
      },

      writeTexture() {},

      copyExternalImageToTexture(
        _source: unknown,
        destination: { texture: GPUTexture & Resource },
      ) {
        live(destination.texture);
        need(destination.texture, 'COPY_DST', TEXTURE);
        need(destination.texture, 'RENDER_ATTACHMENT', TEXTURE);
      },

      onSubmittedWorkDone: async () => {},
    },

    createBuffer(descriptor: GPUBufferDescriptor) {
      const { size, usage, label = '' } = descriptor;

      for (const [map, other] of [
        ['MAP_READ', 'COPY_DST'],
        ['MAP_WRITE', 'COPY_SRC'],
      ] as const) {
        if (usage & BUFFER[map] && usage & ~(BUFFER[map] | BUFFER[other])) {
          throw new Error(
            `Buffer usages ${usageNames(usage, BUFFER)} are invalid. ${map} permits only ${other} alongside it.`,
          );
        }
      }

      log.buffers += 1;
      const data = new ArrayBuffer(size);
      let pending: { reject(error: Error): void } | undefined;

      const buffer = {
        kind: 'Buffer' as const,
        label,
        size,
        usage,
        mapState: (descriptor.mappedAtCreation ? 'mapped' : 'unmapped') as GPUBufferMapState,
        destroyed: false,

        getMappedRange() {
          if (buffer.mapState !== 'mapped') throw new Error(`${nameOf(buffer)} is not mapped.`);
          return data;
        },

        mapAsync(mode: GPUMapModeFlags) {
          if (buffer.mapState === 'pending') {
            throw new Error(
              `${nameOf(buffer)} already has a pending map. Wait for it before mapping again.`,
            );
          }
          if (buffer.mapState === 'mapped') throw new Error(`${nameOf(buffer)} is already mapped.`);
          need(buffer, mode & GPUMapMode.READ ? 'MAP_READ' : 'MAP_WRITE', BUFFER);

          buffer.mapState = 'pending';

          return new Promise<void>((resolve, reject) => {
            const map = { reject };
            pending = map;

            queueMicrotask(() => {
              if (pending !== map) return;
              pending = undefined;
              buffer.mapState = 'mapped';
              resolve();
            });
          });
        },

        unmap() {
          pending?.reject(
            new DOMException(
              `${nameOf(buffer)} was unmapped or destroyed before mapping was resolved.`,
              'AbortError',
            ),
          );
          pending = undefined;
          buffer.mapState = 'unmapped';
        },

        destroy() {
          buffer.unmap();
          buffer.destroyed = true;
          living.delete(buffer);
        },
      };
      contents.set(buffer, new Uint8Array(data));
      living.add(buffer);

      return buffer;
    },

    createTexture(descriptor: GPUTextureDescriptor) {
      log.textures += 1;
      const size = descriptor.size as number[];

      const texture = {
        kind: 'Texture' as const,
        label: descriptor.label ?? '',
        format: descriptor.format,
        usage: descriptor.usage,
        dimension: descriptor.dimension ?? '2d',
        width: size[0],
        height: size[1] ?? 1,
        depthOrArrayLayers: size[2] ?? 1,
        mipLevelCount: descriptor.mipLevelCount ?? 1,
        sampleCount: descriptor.sampleCount ?? 1,
        destroyed: false,

        createView(viewDescriptor: GPUTextureViewDescriptor = {}) {
          log.views += 1;
          const view = { ...viewDescriptor };
          textureOf.set(view, texture);
          return view;
        },

        destroy() {
          texture.destroyed = true;
          living.delete(texture);
        },
      };
      living.add(texture);

      return texture;
    },

    createSampler() {
      log.samplers += 1;
      return {};
    },

    createQuerySet(descriptor: GPUQuerySetDescriptor) {
      log.querySets += 1;

      const set = {
        kind: 'QuerySet' as const,
        label: descriptor.label ?? '',
        destroy() {
          living.delete(set);
        },
      };
      living.add(set);

      return set;
    },

    createBindGroup(descriptor: GPUBindGroupDescriptor) {
      log.bindGroups += 1;
      const layout = Array.from(
        (descriptor.layout as Partial<GPUBindGroupLayoutDescriptor>).entries ?? [],
      );
      const held: Resource[] = [];

      for (const { binding, resource } of descriptor.entries) {
        const buffer = 'buffer' in resource ? resource.buffer : resource;
        const backing = (contents.has(buffer) ? buffer : textureOf.get(resource)) as Resource;
        if (!backing) continue;
        held.push(backing);

        const kind = kindOf(layout.find((entry) => entry.binding === binding));
        if (!kind) continue;

        const [usage, usages]: readonly [string, Usages] = bindingUsage[kind];
        if (!(backing.usage & usages[usage])) {
          throw new Error(
            `Binding usage (${usageNames(backing.usage, usages)}) of ${nameOf(backing)} doesn't match expected usage (${usage}).`,
          );
        }
      }

      const group = { ...descriptor };
      heldBy.set(group, held);
      return group;
    },

    createBindGroupLayout: (descriptor: unknown) => descriptor,
    createPipelineLayout: (descriptor: unknown) => descriptor,

    createShaderModule(descriptor: GPUShaderModuleDescriptor) {
      log.shaders.push(descriptor.code);
      return { getCompilationInfo: async () => ({ messages: [] }) };
    },

    createComputePipeline: block,
    createComputePipelineAsync: async () => compile(),
    createRenderPipeline: block,
    createRenderPipelineAsync: async () => compile(),

    createRenderBundleEncoder() {
      const bundle: Bundle = { events: [], state: [], used: new Set() };
      const { common, draws } = recorder(bundle.used, bundle.events, bundle.state);
      return { ...common, ...draws, finish: () => bundle };
    },

    createCommandEncoder() {
      // Apply copies and clears at submission. Shaders do not execute
      const copies: (() => void)[] = [];
      const used = new Set<Resource>();
      let open: 'render' | 'compute' | undefined;
      const { use, indirect, common, draws } = recorder(used, log.events, log.state);

      const attach = (view: GPUTextureView | undefined, what: string) => {
        const texture = view && textureOf.get(view);
        if (!texture) return;
        need(texture, 'RENDER_ATTACHMENT', TEXTURE, `The ${what} ${nameOf(texture)}`);
        used.add(texture);
      };

      return {
        beginRenderPass(descriptor: GPURenderPassDescriptor) {
          for (const attachment of descriptor.colorAttachments) {
            attach(attachment?.view as GPUTextureView | undefined, 'color attachment');
            attach(attachment?.resolveTarget as GPUTextureView | undefined, 'resolve target');
          }
          attach(
            descriptor.depthStencilAttachment?.view as GPUTextureView | undefined,
            'depth stencil attachment',
          );
          log.events.push('begin-render');
          open = 'render';

          return {
            ...common,
            ...draws,
            setViewport() {},
            setScissorRect() {},
            setBlendConstant() {},
            setStencilReference() {},

            // Replays what each bundle recorded, as the pass would run it
            executeBundles(bundles: Iterable<Bundle>) {
              log.events.push('bundles');
              for (const bundle of bundles) {
                for (const resource of bundle.used) used.add(resource);
                log.state.push(...bundle.state);
                log.events.push(...bundle.events);
              }
            },
            end() {
              open = undefined;
              log.events.push('end-render');
            },
          };
        },

        beginComputePass() {
          log.events.push('begin-compute');
          open = 'compute';

          return {
            ...common,
            dispatchWorkgroups: event('dispatch'),
            dispatchWorkgroupsIndirect: indirect('dispatch-indirect'),
            end() {
              open = undefined;
              log.events.push('end-compute');
            },
          };
        },

        copyBufferToBuffer(
          source: GPUBuffer & Resource,
          sourceOffset: number,
          destination: GPUBuffer & Resource,
          destinationOffset: number,
          size: number,
        ) {
          need(source, 'COPY_SRC', BUFFER);
          need(destination, 'COPY_DST', BUFFER);
          fit('Copy source', source, sourceOffset, size);
          fit('Copy destination', destination, destinationOffset, size);
          use(source, destination);

          log.events.push('copy-buffer');
          copies.push(() =>
            bytes(destination).set(
              bytes(source).subarray(sourceOffset, sourceOffset + size),
              destinationOffset,
            ),
          );
        },

        copyTextureToTexture: event('copy-texture'),

        clearBuffer(buffer: GPUBuffer & Resource, offset = 0, size = buffer.size - offset) {
          need(buffer, 'COPY_DST', BUFFER);
          fit('Clear', buffer, offset, size);
          use(buffer);

          log.events.push('clear-buffer');
          copies.push(() => bytes(buffer).fill(0, offset, offset + size));
        },

        resolveQuerySet(_set: GPUQuerySet, _first: number, _count: number, buffer: GPUBuffer) {
          use(buffer);
          log.events.push('resolve-queries');
        },

        finish() {
          if (open) {
            throw new Error(
              `The encoder was finished while a ${open} pass was open; end it first.`,
            );
          }

          return { copies, used };
        },
      };
    },

    pushErrorScope() {},
    popErrorScope: async () => null,
    addEventListener() {},
    removeEventListener() {},
    destroy() {},
  };

  // Name the method a test reaches that the double lacks, rather than fail with a TypeError
  const strict = new Proxy(device, {
    get(target, key, receiver) {
      if (typeof key === 'symbol' || key in target) return Reflect.get(target, key, receiver);
      if (optional.has(key)) return undefined;
      throw new Error(
        `recordingRoot doesn't implement GPUDevice.${key}; run this test on gpuRoot from @typegpu/kit/testing/dawn.`,
      );
    },
  }) as unknown as GPUDevice;

  return { root: tgpu.initFromDevice({ device: strict }), log, bytes };
}

export interface OffscreenSurfaceOptions {
  /** Sets the texture width in pixels. */
  readonly width: number;
  /** Sets the texture height in pixels. */
  readonly height: number;
  /** Sets the texture format, as a canvas preferred format would. */
  readonly format: GPUTextureFormat;
  /** Sets the reported `pixelRatio`, as for a canvas on a phone. Default 1. */
  readonly pixelRatio?: number;
}

export interface OffscreenSurface extends Surface {
  /**
   * Provides the texture that frames draw into, a WebGPU texture and not a TypeGPU one, so a
   * `source` option does not take it. Supports copying so GPU tests can read its pixels.
   */
  readonly texture: GPUTexture;
  /**
   * Reports whether the surface is paused, true between `pause` and `resume`. Hosts use this state
   * to pause frames.
   */
  readonly paused: boolean;
  /**
   * Replaces the texture with one of the given size. Reports the supplied `pixelRatio`, or 1 if
   * omitted.
   */
  setSize(size: FixedSize): void;
  /** Pauses the surface, setting `paused`. */
  pause(): void;
  /** Resumes the surface, clearing `paused`. */
  resume(): void;
}

/**
 * Creates a surface backed by a texture for running scene frames without a canvas. Accepts a
 * recording root or a real device. `view` returns one view per texture. `destroy` destroys the
 * texture.
 *
 * @param root any root, recording or real
 * @returns a surface you draw into and destroy
 */
export function offscreenSurface(
  root: TgpuRoot,
  options: OffscreenSurfaceOptions,
): OffscreenSurface {
  const { format } = options;
  const maxSize = root.device.limits.maxTextureDimension2D;
  let { pixelRatio } = checkSize(options, maxSize, 'offscreenSurface');

  const create = (width: number, height: number) =>
    root.device.createTexture({
      label: 'offscreen surface',
      size: [width, height],
      format,
      usage: TEXTURE.RENDER_ATTACHMENT | TEXTURE.COPY_SRC,
    });

  let texture = create(options.width, options.height);
  let view = texture.createView();
  let paused = false;

  // `setSize` replaces the texture, so free the current one
  const destroy = () => texture.destroy();

  return {
    format,

    get pixelRatio() {
      return pixelRatio;
    },

    get width() {
      return texture.width;
    },

    get height() {
      return texture.height;
    },

    get texture() {
      return texture;
    },

    get paused() {
      return paused;
    },

    view: () => view,

    setSize(size) {
      const { width, height, pixelRatio: ratio } = checkSize(size, maxSize, 'offscreenSurface');
      pixelRatio = ratio;
      if (width === texture.width && height === texture.height) return;

      texture.destroy();
      texture = create(width, height);
      view = texture.createView();
    },

    pause() {
      paused = true;
    },

    resume() {
      paused = false;
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}

/** Provides animation frames that a test steps by hand, for `animate` or a host's `schedule`. */
export interface SteppedSchedule extends Schedule {
  /**
   * Runs a requested frame for each given time, in milliseconds. Does nothing for times with no
   * requested frame.
   */
  step(...milliseconds: number[]): void;
  /** Reports whether a requested frame is waiting to run or be cancelled. */
  readonly waiting: boolean;
}

/**
 * Creates a schedule stepped by hand. Each time passed to `step` runs the last requested animation
 * frame, as `requestAnimationFrame` would at that time.
 *
 * @returns a schedule to give `animate` or a host
 */
export function steppedSchedule(): SteppedSchedule {
  let pending: ((milliseconds: number) => void) | undefined;
  let requests = 0;

  return {
    request(callback) {
      pending = callback;
      return ++requests;
    },

    cancel(request) {
      if (request === requests) pending = undefined;
    },

    step(...milliseconds) {
      for (const time of milliseconds) {
        const run = pending;
        pending = undefined;
        run?.(time);
      }
    },

    get waiting() {
      return pending !== undefined;
    },
  };
}
