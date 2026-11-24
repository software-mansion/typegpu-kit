/**
 * Textures and views. A texture's GL storage is made on first use, with the target its views ask
 * for; a view GL cannot sample directly, such as one layer of an array or a cube of a texture
 * stored as an array, samples a copy refreshed when the texture changed.
 */

import { type Ctx, IMPL, implOf } from './context.ts';
import { type FormatInfo, formatInfo } from './formats.ts';
import type { TexelArray } from './packing.ts';

export interface CpuLevelCopy {
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers: number;
  /** The level's texels, read back from GL when first touched. */
  readonly data: TexelArray;
  /** Whether `data` exists yet; a level never touched is still current on GL alone. */
  readonly loaded: boolean;
}

interface Derived {
  readonly target: number;
  readonly texture: WebGLTexture;
  readonly layers: number;
  version: number;
  levels: number;
}

export interface TextureImpl {
  readonly id: number;
  readonly label: string;
  readonly format: GPUTextureFormat;
  readonly info: FormatInfo;
  readonly dimension: GPUTextureDimension;
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers: number;
  readonly mipLevelCount: number;
  readonly sampleCount: number;
  readonly usage: number;
  /** The GL target, or 0 before the storage exists or for a renderbuffer. */
  target: number;
  texture: WebGLTexture | null;
  renderbuffer: WebGLRenderbuffer | null;
  /** The view dimension the texture is stored for when a choice remains. */
  hint: 'cube' | 'array' | undefined;
  /** Bumped by every write to the GL copy. */
  version: number;
  written: boolean;
  /** Which copy is current: GL's, the CPU's, or both. */
  home: 'gl' | 'cpu' | 'both';
  cpu: CpuLevelCopy[] | undefined;
  /** `TEXTURE_BASE_LEVEL * 256 + TEXTURE_MAX_LEVEL` last set on the GL texture. */
  levelRange: number;
  readonly derived: Map<string, Derived>;
  readonly framebuffers: Set<string>;
  destroyed: boolean;
  /** Set for a canvas context's texture, which a submit presents. */
  canvas: object | undefined;
}

export interface ViewImpl {
  readonly texture: TextureImpl;
  readonly format: GPUTextureFormat;
  readonly dimension: GPUTextureViewDimension;
  readonly aspect: GPUTextureAspect;
  readonly baseMipLevel: number;
  readonly mipLevelCount: number;
  readonly baseArrayLayer: number;
  readonly arrayLayerCount: number;
}

export function textureImpl(texture: GPUTexture): TextureImpl {
  return implOf<TextureImpl>(texture, 'texture');
}

export function viewImpl(view: GPUTextureView): ViewImpl {
  return implOf<ViewImpl>(view, 'texture view');
}

export function mipSize(texture: TextureImpl, level: number): [number, number, number] {
  return [
    Math.max(1, texture.width >> level),
    Math.max(1, texture.height >> level),
    texture.dimension === '3d'
      ? Math.max(1, texture.depthOrArrayLayers >> level)
      : texture.depthOrArrayLayers,
  ];
}

export function attachmentPoint(gl: WebGL2RenderingContext, info: FormatInfo, color = 0): number {
  if (info.hasStencil) return gl.DEPTH_STENCIL_ATTACHMENT;
  if (info.hasDepth) return gl.DEPTH_ATTACHMENT;
  return gl.COLOR_ATTACHMENT0 + color;
}

function targetFor(gl: WebGL2RenderingContext, dimension: GPUTextureViewDimension): number {
  switch (dimension) {
    case '1d':
    case '2d':
      return gl.TEXTURE_2D;
    case '2d-array':
      return gl.TEXTURE_2D_ARRAY;
    case 'cube':
      return gl.TEXTURE_CUBE_MAP;
    case '3d':
      return gl.TEXTURE_3D;
    case 'cube-array':
      throw new Error('@typegpu/kit-gl: cube-array textures do not exist on WebGL 2.');
  }
}

function chooseTarget(ctx: Ctx, texture: TextureImpl): number {
  const { gl } = ctx;
  if (texture.dimension === '3d') return gl.TEXTURE_3D;
  const layers = texture.depthOrArrayLayers;
  if (
    texture.hint === 'cube' &&
    layers === 6 &&
    texture.width === texture.height &&
    !texture.info.hasStencil
  ) {
    return gl.TEXTURE_CUBE_MAP;
  }
  if (layers > 1 || texture.hint === 'array') return gl.TEXTURE_2D_ARRAY;
  return gl.TEXTURE_2D;
}

function samplesFor(ctx: Ctx, info: FormatInfo, requested: number): number {
  const { gl } = ctx;
  const supported = gl.getInternalformatParameter(
    gl.RENDERBUFFER,
    info.internal,
    gl.SAMPLES,
  ) as Int32Array | null;
  if (!supported || supported.length === 0) return 0;
  const counts = Array.from(supported).toSorted((a, b) => a - b);
  return counts.find((count) => count >= requested) ?? counts[counts.length - 1]!;
}

function freeFramebuffers(ctx: Ctx, texture: TextureImpl): void {
  for (const key of texture.framebuffers) {
    const fbo = framebuffersOf(ctx).get(key);
    if (fbo) {
      ctx.gl.deleteFramebuffer(fbo);
      framebuffersOf(ctx).delete(key);
    }
  }
  texture.framebuffers.clear();
}

function freeStorage(ctx: Ctx, texture: TextureImpl): void {
  const { gl } = ctx;
  freeFramebuffers(ctx, texture);
  if (texture.texture) gl.deleteTexture(texture.texture);
  if (texture.renderbuffer) gl.deleteRenderbuffer(texture.renderbuffer);
  for (const derived of texture.derived.values()) gl.deleteTexture(derived.texture);
  texture.derived.clear();
  texture.texture = null;
  texture.renderbuffer = null;
  texture.target = 0;
  texture.levelRange = -1;
}

function allocate(ctx: Ctx, texture: TextureImpl): void {
  const { gl, state } = ctx;
  const { info, width, height, depthOrArrayLayers, mipLevelCount } = texture;
  if (texture.sampleCount > 1) {
    const samples = samplesFor(ctx, info, texture.sampleCount);
    const renderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, info.internal, width, height);
    texture.renderbuffer = renderbuffer;
    return;
  }
  const target = chooseTarget(ctx, texture);
  const value = gl.createTexture();
  state.scratchTexture(target, value);
  if (target === gl.TEXTURE_2D || target === gl.TEXTURE_CUBE_MAP) {
    gl.texStorage2D(target, mipLevelCount, info.internal, width, height);
  } else {
    gl.texStorage3D(target, mipLevelCount, info.internal, width, height, depthOrArrayLayers);
  }
  texture.texture = value;
  texture.target = target;
  texture.levelRange = -1;
}

/**
 * Makes the GL copy current: creates the storage, and uploads what a compute kernel wrote.
 * Returns the texture for chaining.
 */
export function onGl(ctx: Ctx, texture: TextureImpl): TextureImpl {
  if (texture.destroyed) throw new Error('@typegpu/kit-gl: the texture was destroyed.');
  if (!texture.texture && !texture.renderbuffer) allocate(ctx, texture);
  if (texture.home === 'cpu') {
    uploadCpu(ctx, texture);
    texture.home = 'both';
    texture.version++;
  }
  return texture;
}

/** Records a write to the GL copy, which leaves the CPU copy stale. */
export function glWritten(texture: TextureImpl): void {
  texture.version++;
  texture.written = true;
  texture.home = 'gl';
}

/** Records a compute kernel's write to the CPU copy. */
export function cpuWritten(texture: TextureImpl): void {
  texture.written = true;
  texture.home = 'cpu';
}

let uploadCpu: (ctx: Ctx, texture: TextureImpl) => void = () => {
  throw new Error('@typegpu/kit-gl: CPU texture upload is not installed.');
};

/** Lets `transfer.ts` provide the upload of CPU copies without an import cycle. */
export function installCpuUpload(upload: (ctx: Ctx, texture: TextureImpl) => void): void {
  uploadCpu = upload;
}

/** Attaches one level and layer of a texture to the bound framebuffer at `point`. */
export function attachLevel(
  ctx: Ctx,
  framebufferTarget: number,
  point: number,
  texture: TextureImpl,
  level: number,
  layer: number,
): void {
  const { gl } = ctx;
  if (texture.renderbuffer) {
    gl.framebufferRenderbuffer(framebufferTarget, point, gl.RENDERBUFFER, texture.renderbuffer);
  } else if (texture.target === gl.TEXTURE_2D) {
    gl.framebufferTexture2D(framebufferTarget, point, gl.TEXTURE_2D, texture.texture, level);
  } else if (texture.target === gl.TEXTURE_CUBE_MAP) {
    gl.framebufferTexture2D(
      framebufferTarget,
      point,
      gl.TEXTURE_CUBE_MAP_POSITIVE_X + layer,
      texture.texture,
      level,
    );
  } else {
    gl.framebufferTextureLayer(framebufferTarget, point, texture.texture, level, layer);
  }
}

/** One attachment of a render pass's framebuffer. */
export interface Attachment {
  readonly texture: TextureImpl;
  readonly level: number;
  readonly layer: number;
}

const framebufferCaches = new WeakMap<Ctx, Map<string, WebGLFramebuffer>>();

function framebuffersOf(ctx: Ctx): Map<string, WebGLFramebuffer> {
  let cache = framebufferCaches.get(ctx);
  if (!cache) {
    cache = new Map();
    framebufferCaches.set(ctx, cache);
  }
  return cache;
}

const keyOf = (a: Attachment | null) => (a ? `${a.texture.id}.${a.level}.${a.layer}` : '-');

/** A framebuffer with these attachments, made once and kept until a texture in it is freed. */
export function framebuffer(
  ctx: Ctx,
  colors: readonly (Attachment | null)[],
  depth: Attachment | null,
): WebGLFramebuffer {
  const { gl, state } = ctx;
  const key = `${colors.map(keyOf).join(',')}|${keyOf(depth)}`;
  const cache = framebuffersOf(ctx);
  const cached = cache.get(key);
  if (cached) return cached;

  const fbo = gl.createFramebuffer();
  state.drawFramebuffer(fbo);
  const buffers: number[] = [];
  colors.forEach((color, i) => {
    if (!color) {
      buffers.push(gl.NONE);
      return;
    }
    onGl(ctx, color.texture);
    attachLevel(
      ctx,
      gl.DRAW_FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0 + i,
      color.texture,
      color.level,
      color.layer,
    );
    buffers.push(gl.COLOR_ATTACHMENT0 + i);
  });
  if (depth) {
    onGl(ctx, depth.texture);
    attachLevel(
      ctx,
      gl.DRAW_FRAMEBUFFER,
      attachmentPoint(gl, depth.texture.info),
      depth.texture,
      depth.level,
      depth.layer,
    );
  }
  gl.drawBuffers(buffers.length > 0 ? buffers : [gl.NONE]);
  const status = gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    ctx.errors.report(
      'validation',
      `The render pass attachments (${[...colors, depth]
        .filter((a): a is Attachment => a !== null)
        .map((a) => a.texture.format)
        .join(', ')}) do not make a complete WebGL framebuffer (status 0x${status.toString(16)}).`,
    );
  }
  cache.set(key, fbo);
  for (const attachment of [...colors, depth]) attachment?.texture.framebuffers.add(key);
  return fbo;
}

/** Clears the masks and scissor that WebGL applies to clears and `blitFramebuffer` too. */
export function writeEverything(ctx: Ctx): void {
  const { gl, state } = ctx;
  state.enable(gl.SCISSOR_TEST, false);
  state.enable(gl.RASTERIZER_DISCARD, false);
  state.colorMaskAll(0xf);
  state.depthMask(true);
  state.stencilWriteMask(0xff);
}

/** Copies one level and layer between textures with `blitFramebuffer`, both on GL. */
export function blitLevel(
  ctx: Ctx,
  source: { texture: TextureImpl; level: number; layer: number; x: number; y: number },
  destination: { texture: TextureImpl; level: number; layer: number; x: number; y: number },
  width: number,
  height: number,
): void {
  const { gl, state, scratch } = ctx;
  const { info } = source.texture;
  const point = attachmentPoint(gl, info);
  const mask = info.hasStencil
    ? gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT
    : info.hasDepth
      ? gl.DEPTH_BUFFER_BIT
      : gl.COLOR_BUFFER_BIT;
  state.readFramebuffer(scratch.read);
  state.drawFramebuffer(scratch.draw);
  attachLevel(ctx, gl.READ_FRAMEBUFFER, point, source.texture, source.level, source.layer);
  attachLevel(
    ctx,
    gl.DRAW_FRAMEBUFFER,
    point,
    destination.texture,
    destination.level,
    destination.layer,
  );
  if (point === gl.COLOR_ATTACHMENT0) {
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  }
  writeEverything(ctx);
  gl.blitFramebuffer(
    source.x,
    source.y,
    source.x + width,
    source.y + height,
    destination.x,
    destination.y,
    destination.x + width,
    destination.y + height,
    mask,
    gl.NEAREST,
  );
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, point, gl.TEXTURE_2D, null, 0);
  gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, point, gl.TEXTURE_2D, null, 0);
}

function setLevels(
  ctx: Ctx,
  target: number,
  texture: WebGLTexture,
  current: number,
  base: number,
  max: number,
): number {
  const range = base * 256 + max;
  if (current === range) return range;
  const { gl, state } = ctx;
  state.scratchTexture(target, texture);
  gl.texParameteri(target, gl.TEXTURE_BASE_LEVEL, base);
  gl.texParameteri(target, gl.TEXTURE_MAX_LEVEL, max);
  return range;
}

export interface Sampled {
  readonly target: number;
  readonly texture: WebGLTexture;
}

/**
 * The GL texture a shader samples `view` through, current, with its level range set to the
 * view's, so a pass may render into one level while sampling another.
 */
export function sampled(ctx: Ctx, view: ViewImpl): Sampled {
  const { gl } = ctx;
  const texture = view.texture;
  if (texture.sampleCount > 1) {
    throw new Error('@typegpu/kit-gl: multisampled textures cannot be sampled on WebGL 2.');
  }
  const want = targetFor(gl, view.dimension);
  if (!texture.texture) {
    texture.hint ??=
      view.dimension === 'cube' ? 'cube' : view.dimension === '2d-array' ? 'array' : undefined;
  }
  onGl(ctx, texture);
  const layers = texture.dimension === '3d' ? 1 : texture.depthOrArrayLayers;
  const whole = view.baseArrayLayer === 0 && view.arrayLayerCount === layers;

  if (texture.target !== want && !texture.written && whole && texture.home !== 'cpu') {
    // Nothing was stored yet, so store the texture as this view wants it
    texture.hint =
      view.dimension === 'cube' ? 'cube' : view.dimension === '2d-array' ? 'array' : undefined;
    freeStorage(ctx, texture);
    allocate(ctx, texture);
  }

  const last = view.baseMipLevel + view.mipLevelCount - 1;
  if (texture.target === want && whole) {
    texture.levelRange = setLevels(
      ctx,
      want,
      texture.texture!,
      texture.levelRange,
      view.baseMipLevel,
      last,
    );
    return { target: want, texture: texture.texture! };
  }

  const derived = derivedCopy(ctx, texture, view, want);
  derived.levels = setLevels(ctx, want, derived.texture, derived.levels, view.baseMipLevel, last);
  return { target: want, texture: derived.texture };
}

function derivedCopy(ctx: Ctx, texture: TextureImpl, view: ViewImpl, target: number): Derived {
  const { gl, state } = ctx;
  const key = `${view.dimension}:${view.baseArrayLayer}:${view.arrayLayerCount}`;
  let derived = texture.derived.get(key);
  if (!derived) {
    if (!texture.info.renderable) {
      throw new Error(
        `@typegpu/kit-gl: a ${view.dimension} view of part of a '${texture.format}' texture cannot be sampled on WebGL 2.`,
      );
    }
    const value = gl.createTexture();
    state.scratchTexture(target, value);
    const { info, width, height, mipLevelCount } = texture;
    if (target === gl.TEXTURE_2D || target === gl.TEXTURE_CUBE_MAP) {
      gl.texStorage2D(target, mipLevelCount, info.internal, width, height);
    } else {
      gl.texStorage3D(target, mipLevelCount, info.internal, width, height, view.arrayLayerCount);
    }
    derived = { target, texture: value, layers: view.arrayLayerCount, version: -1, levels: -1 };
    texture.derived.set(key, derived);
  }
  if (derived.version !== texture.version) {
    const copy: TextureImpl = {
      ...texture,
      target,
      texture: derived.texture,
      renderbuffer: null,
    };
    for (let level = 0; level < texture.mipLevelCount; level++) {
      const [width, height] = mipSize(texture, level);
      for (let layer = 0; layer < derived.layers; layer++) {
        blitLevel(
          ctx,
          { texture, level, layer: view.baseArrayLayer + layer, x: 0, y: 0 },
          { texture: copy, level, layer, x: 0, y: 0 },
          width,
          height,
        );
      }
    }
    derived.version = texture.version;
  }
  return derived;
}

function viewOf(texture: TextureImpl, descriptor: GPUTextureViewDescriptor = {}): ViewImpl {
  const layers = texture.dimension === '3d' ? 1 : texture.depthOrArrayLayers;
  const baseArrayLayer = descriptor.baseArrayLayer ?? 0;
  const baseMipLevel = descriptor.baseMipLevel ?? 0;
  const dimension =
    descriptor.dimension ??
    (texture.dimension === '3d'
      ? '3d'
      : texture.dimension === '1d'
        ? '1d'
        : layers > 1
          ? '2d-array'
          : '2d');
  const arrayLayerCount =
    descriptor.arrayLayerCount ??
    (dimension === '2d' || dimension === '1d' || dimension === '3d'
      ? 1
      : dimension === 'cube'
        ? 6
        : layers - baseArrayLayer);
  return {
    texture,
    format: descriptor.format ?? texture.format,
    dimension,
    aspect: descriptor.aspect ?? 'all',
    baseMipLevel,
    mipLevelCount: descriptor.mipLevelCount ?? texture.mipLevelCount - baseMipLevel,
    baseArrayLayer,
    arrayLayerCount,
  };
}

export function makeView(
  texture: TextureImpl,
  descriptor?: GPUTextureViewDescriptor,
): GPUTextureView {
  const impl = viewOf(texture, descriptor);
  if (!texture.texture && !texture.renderbuffer && !texture.hint) {
    if (impl.dimension === 'cube') texture.hint = 'cube';
    else if (impl.dimension === '2d-array') texture.hint = 'array';
  }
  return {
    [IMPL]: impl,
    __brand: 'GPUTextureView',
    label: descriptor?.label ?? '',
  } as unknown as GPUTextureView;
}

export function createTexture(ctx: Ctx, descriptor: GPUTextureDescriptor): GPUTexture {
  const size = descriptor.size;
  const [width, height = 1, depthOrArrayLayers = 1] =
    Symbol.iterator in size
      ? Array.from(size as Iterable<number>)
      : [size.width, size.height, size.depthOrArrayLayers];
  const info = formatInfo(ctx.formats, descriptor.format);
  const dimension = descriptor.dimension ?? '2d';
  const viewDimension = (descriptor as { textureBindingViewDimension?: GPUTextureViewDimension })
    .textureBindingViewDimension;

  const impl: TextureImpl = {
    id: ctx.id(),
    format: descriptor.format,
    info,
    dimension,
    width: width!,
    height: height ?? 1,
    depthOrArrayLayers: depthOrArrayLayers ?? 1,
    mipLevelCount: descriptor.mipLevelCount ?? 1,
    sampleCount: descriptor.sampleCount ?? 1,
    usage: descriptor.usage,
    target: 0,
    texture: null,
    renderbuffer: null,
    hint: viewDimension === 'cube' ? 'cube' : viewDimension === '2d-array' ? 'array' : undefined,
    version: 0,
    written: false,
    home: 'gl',
    cpu: undefined,
    levelRange: -1,
    derived: new Map(),
    framebuffers: new Set(),
    destroyed: false,
    canvas: undefined,
    label: descriptor.label ?? '',
  };
  if (impl.width > ctx.maxTextureSize || impl.height > ctx.maxTextureSize) {
    ctx.errors.report(
      'validation',
      `Texture size ${impl.width}x${impl.height} exceeds maxTextureDimension2D ${ctx.maxTextureSize}.`,
    );
  }
  const untrack = ctx.track(() => freeStorage(ctx, impl));

  const texture = {
    [IMPL]: impl,
    __brand: 'GPUTexture',
    label: descriptor.label ?? '',
    width: impl.width,
    height: impl.height,
    depthOrArrayLayers: impl.depthOrArrayLayers,
    mipLevelCount: impl.mipLevelCount,
    sampleCount: impl.sampleCount,
    dimension,
    format: descriptor.format,
    usage: descriptor.usage,
    textureBindingViewDimension: viewDimension,
    createView: (viewDescriptor?: GPUTextureViewDescriptor) => makeView(impl, viewDescriptor),
    destroy() {
      if (impl.destroyed) return;
      impl.destroyed = true;
      impl.cpu = undefined;
      freeStorage(ctx, impl);
      untrack();
    },
  };
  return texture as unknown as GPUTexture;
}
