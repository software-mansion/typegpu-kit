/**
 * Moving texels: uploads from the CPU and from images, copies between textures and buffers,
 * readback, and the CPU copies of textures that compute kernels use.
 */

import { type BufferImpl, written } from './buffer.ts';
import { type Ctx, IMPL } from './context.ts';
import { link } from './program.ts';
import { packTexels, swapRedBlue, type TexelArray, texelArray, viewAs } from './packing.ts';
import {
  attachLevel,
  blitLevel,
  type CpuLevelCopy,
  glWritten,
  installCpuUpload,
  mipSize,
  onGl,
  type TextureImpl,
  textureImpl,
} from './texture.ts';

interface Region {
  readonly level: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly width: number;
  readonly height: number;
  readonly layers: number;
}

function extent(size: GPUExtent3D | GPUExtent3DStrict): [number, number, number] {
  if (Symbol.iterator in size) {
    const [w = 1, h = 1, d = 1] = Array.from(size as Iterable<number>);
    return [w, h, d];
  }
  return [size.width, size.height ?? 1, size.depthOrArrayLayers ?? 1];
}

function origin(value: GPUOrigin3D | undefined): [number, number, number] {
  if (!value) return [0, 0, 0];
  if (Symbol.iterator in value) {
    const [x = 0, y = 0, z = 0] = Array.from(value as Iterable<number>);
    return [x, y, z];
  }
  return [value.x ?? 0, value.y ?? 0, value.z ?? 0];
}

function origin2d(value: GPUOrigin2D | undefined): [number, number] {
  if (!value) return [0, 0];
  if (Symbol.iterator in value) {
    const [x = 0, y = 0] = Array.from(value as Iterable<number>);
    return [x, y];
  }
  return [value.x ?? 0, value.y ?? 0];
}

function bytesOf(data: AllowSharedBufferSource): Uint8Array {
  if (ArrayBuffer.isView(data))
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Uint8Array(data as ArrayBuffer);
}

/**
 * Uploads rows of texels in the format's own layout into the GL copy. `rowLength` and
 * `imageHeight` count texels and rows between the starts of consecutive rows and images.
 */
function upload(
  ctx: Ctx,
  texture: TextureImpl,
  region: Region,
  source: Uint8Array,
  rowLength: number,
  imageHeight: number,
): void {
  const { gl, state } = ctx;
  const { info } = texture;
  onGl(ctx, texture);
  if (!texture.texture) {
    throw new Error('@typegpu/kit-gl: multisampled textures cannot be written from the CPU.');
  }
  const bytes = info.bgra ? swapRedBlue(source) : source;
  const data = viewAs(info, bytes);
  const perTexel = info.bytes / (data.BYTES_PER_ELEMENT * 1);
  state.scratchTexture(texture.target, texture.texture);
  state.unpack(false, false, rowLength === region.width ? 0 : rowLength, imageHeight);
  const { level, x, y, z, width, height, layers } = region;
  if (texture.target === gl.TEXTURE_2D) {
    gl.texSubImage2D(gl.TEXTURE_2D, level, x, y, width, height, info.upload, info.type, data, 0);
  } else if (texture.target === gl.TEXTURE_CUBE_MAP) {
    for (let i = 0; i < layers; i++) {
      gl.texSubImage2D(
        gl.TEXTURE_CUBE_MAP_POSITIVE_X + z + i,
        level,
        x,
        y,
        width,
        height,
        info.upload,
        info.type,
        data,
        i * rowLength * imageHeight * perTexel,
      );
    }
  } else {
    gl.texSubImage3D(
      texture.target,
      level,
      x,
      y,
      z,
      width,
      height,
      layers,
      info.upload,
      info.type,
      data,
      0,
    );
  }
  glWritten(texture);
}

/** Uploads bytes laid out as `layout` says, as `writeTexture` and `copyBufferToTexture` do. */
function uploadLaidOut(
  ctx: Ctx,
  destination: GPUTexelCopyTextureInfo,
  bytes: Uint8Array,
  layout: GPUTexelCopyBufferLayout,
  size: GPUExtent3DStrict,
): void {
  const texture = textureImpl(destination.texture);
  const { info } = texture;
  const [width, height, layers] = extent(size);
  if (width === 0 || height === 0 || layers === 0) return;
  const [x, y, z] = origin(destination.origin);
  const offset = layout.offset ?? 0;
  const tight = width * info.bytes;
  const bytesPerRow = layout.bytesPerRow ?? tight;
  const rowsPerImage = layout.rowsPerImage ?? height;
  const region: Region = { level: destination.mipLevel ?? 0, x, y, z, width, height, layers };
  if (bytesPerRow % info.bytes === 0 && offset % 4 === 0) {
    const length = (layers - 1) * bytesPerRow * rowsPerImage + (height - 1) * bytesPerRow + tight;
    upload(
      ctx,
      texture,
      region,
      bytes.subarray(offset, offset + length),
      bytesPerRow / info.bytes,
      rowsPerImage,
    );
    return;
  }
  const packed = new Uint8Array(tight * height * layers);
  for (let layer = 0; layer < layers; layer++)
    for (let row = 0; row < height; row++) {
      const from = offset + layer * bytesPerRow * rowsPerImage + row * bytesPerRow;
      packed.set(bytes.subarray(from, from + tight), (layer * height + row) * tight);
    }
  upload(ctx, texture, region, packed, width, height);
}

export function writeTexture(
  ctx: Ctx,
  destination: GPUTexelCopyTextureInfo,
  data: AllowSharedBufferSource,
  layout: GPUTexelCopyBufferLayout,
  size: GPUExtent3DStrict,
): void {
  uploadLaidOut(ctx, destination, bytesOf(data), layout, size);
}

export function copyBufferToTexture(
  ctx: Ctx,
  source: BufferImpl,
  layout: GPUTexelCopyBufferLayout,
  destination: GPUTexelCopyTextureInfo,
  size: GPUExtent3DStrict,
): void {
  source.sync();
  uploadLaidOut(ctx, destination, source.bytes, layout, size);
}

const READBACK_VERTEX = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

const READBACK_FRAGMENT = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D source;
uniform ivec2 origin;
out vec4 color;
void main() {
  color = texelFetch(source, ivec2(gl_FragCoord.xy) + origin, 0);
}
`;

/** Reads a region of one level and layer as RGBA floats by drawing it into an RGBA32F target. */
function readThroughShader(
  ctx: Ctx,
  texture: TextureImpl,
  level: number,
  layer: number,
  x: number,
  y: number,
  width: number,
  height: number,
): { values: Float32Array; stride: number } {
  const { gl, state } = ctx;
  if (!ctx.ext.colorBufferFloat) {
    throw new Error(
      `@typegpu/kit-gl: reading '${texture.format}' back needs EXT_color_buffer_float.`,
    );
  }
  let source = texture;
  let sourceLevel = level;
  let temporary: WebGLTexture | null = null;
  if (texture.target !== gl.TEXTURE_2D) {
    const [w, h] = mipSize(texture, level);
    temporary = gl.createTexture();
    state.scratchTexture(gl.TEXTURE_2D, temporary);
    gl.texStorage2D(gl.TEXTURE_2D, 1, texture.info.internal, w, h);
    source = { ...texture, target: gl.TEXTURE_2D, texture: temporary, levelRange: -1 };
    blitLevel(
      ctx,
      { texture, level, layer, x: 0, y: 0 },
      { texture: source, level: 0, layer: 0, x: 0, y: 0 },
      w,
      h,
    );
    sourceLevel = 0;
  }
  const linked = link(ctx, READBACK_VERTEX, READBACK_FRAGMENT, 'the readback program');
  if (!linked.ok) throw new Error(linked.message);
  const { program, uniforms } = linked.program;

  const target = gl.createTexture();
  state.scratchTexture(gl.TEXTURE_2D, target);
  // One channel reads back a quarter of the bytes where the driver reads R32F as RED
  const single = texture.info.components === 1 && texture.info.format !== 'rgb9e5ufloat';
  gl.texStorage2D(gl.TEXTURE_2D, 1, single ? gl.R32F : gl.RGBA32F, width, height);
  state.scratchTexture(gl.TEXTURE_2D, source.texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, sourceLevel);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, sourceLevel);
  texture.levelRange = -1;
  state.sampler(state.scratchUnit, ctx.nearest);

  state.drawFramebuffer(ctx.scratch.draw);
  gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  state.program(program);
  gl.uniform1i(uniforms.get('source')!, state.scratchUnit);
  gl.uniform2i(uniforms.get('origin')!, x, y);
  state.vertexArray(null);
  state.viewport(0, 0, width, height);
  for (const cap of [
    gl.SCISSOR_TEST,
    gl.BLEND,
    gl.DEPTH_TEST,
    gl.STENCIL_TEST,
    gl.CULL_FACE,
    gl.RASTERIZER_DISCARD,
    gl.SAMPLE_ALPHA_TO_COVERAGE,
    gl.POLYGON_OFFSET_FILL,
  ])
    state.enable(cap, false);
  state.blend(0, {
    enabled: false,
    colorOp: 0,
    colorSrc: 0,
    colorDst: 0,
    alphaOp: 0,
    alphaSrc: 0,
    alphaDst: 0,
  });
  state.colorMask(0, 0xf);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  state.readFramebuffer(ctx.scratch.draw);
  gl.readBuffer(gl.COLOR_ATTACHMENT0);
  const red =
    single &&
    gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) === gl.RED &&
    gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE) === gl.FLOAT;
  const stride = red ? 1 : 4;
  const out = new Float32Array(width * height * stride);
  gl.readPixels(0, 0, width, height, red ? gl.RED : gl.RGBA, gl.FLOAT, out);
  gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  gl.deleteTexture(target);
  if (temporary) gl.deleteTexture(temporary);
  return { values: out, stride };
}

/** Whether the bound read framebuffer reads back as RGBA half floats, which need no conversion. */
function readsHalves(gl: WebGL2RenderingContext): boolean {
  return (
    gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) === gl.RGBA &&
    gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE) === gl.HALF_FLOAT
  );
}

/** Reads a region of one level and layer from GL in the format's own layout. */
function readFromGl(
  ctx: Ctx,
  texture: TextureImpl,
  level: number,
  layer: number,
  x: number,
  y: number,
  width: number,
  height: number,
): TexelArray {
  const { gl, state } = ctx;
  const { info } = texture;
  onGl(ctx, texture);
  if (texture.renderbuffer) {
    throw new Error('@typegpu/kit-gl: multisampled textures cannot be read back on WebGL 2.');
  }
  const out = texelArray(info, width * height);
  if (!info.renderable || info.hasDepth || info.kind === 'snorm') {
    const { values, stride } = readThroughShader(ctx, texture, level, layer, x, y, width, height);
    packTexels(info, values, out, stride);
    return out;
  }
  state.readFramebuffer(ctx.scratch.read);
  attachLevel(ctx, gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, texture, level, layer);
  gl.readBuffer(gl.COLOR_ATTACHMENT0);
  const count = width * height * 4;
  if (info.format === 'rgb10a2unorm') {
    gl.readPixels(x, y, width, height, gl.RGBA, gl.UNSIGNED_INT_2_10_10_10_REV, out);
  } else if (info.kind === 'unorm') {
    const rgba = new Uint8Array(count);
    gl.readPixels(x, y, width, height, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    packTexels(info, rgba, out);
  } else if (info.array === 'f16' && readsHalves(gl)) {
    const rgba = new Uint16Array(count);
    gl.readPixels(x, y, width, height, gl.RGBA, gl.HALF_FLOAT, rgba);
    const n = info.components;
    if (n === 4) out.set(rgba);
    else
      for (let i = 0; i < width * height; i++)
        for (let c = 0; c < n; c++) out[i * n + c] = rgba[i * 4 + c]!;
  } else if (info.kind === 'float') {
    const rgba = new Float32Array(count);
    gl.readPixels(x, y, width, height, gl.RGBA, gl.FLOAT, rgba);
    packTexels(info, rgba, out);
  } else if (info.kind === 'uint') {
    const rgba = new Uint32Array(count);
    gl.readPixels(x, y, width, height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, rgba);
    packTexels(info, rgba, out);
  } else {
    const rgba = new Int32Array(count);
    gl.readPixels(x, y, width, height, gl.RGBA_INTEGER, gl.INT, rgba);
    packTexels(info, rgba, out);
  }
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  return out;
}

/** Reads a region of one level and layer in the format's own layout, from the current copy. */
export function readTexels(
  ctx: Ctx,
  texture: TextureImpl,
  level: number,
  layer: number,
  x: number,
  y: number,
  width: number,
  height: number,
): TexelArray {
  const copy = texture.home === 'cpu' ? texture.cpu?.[level] : undefined;
  if (!copy) return readFromGl(ctx, texture, level, layer, x, y, width, height);
  const n = texture.info.components;
  const out = texelArray(texture.info, width * height);
  for (let row = 0; row < height; row++) {
    const from = ((layer * copy.height + y + row) * copy.width + x) * n;
    out.set(copy.data.subarray(from, from + width * n), row * width * n);
  }
  return out;
}

/** The texture's CPU copy, current: every level, rows top to bottom, layers in order. */
/**
 * One level of a texture's CPU copy, read back from GL only when a kernel first touches its
 * `data`: kernels often read one small level of a mipmapped target.
 */
function lazyLevel(ctx: Ctx, texture: TextureImpl, level: number): CpuLevelCopy {
  const [width, height, layers] = mipSize(texture, level);
  const stored = texture.written;
  let data: TexelArray | undefined;
  return {
    width,
    height,
    depthOrArrayLayers: layers,
    get loaded() {
      return data !== undefined;
    },
    get data() {
      if (data) return data;
      const start = performance.now();
      const made = texelArray(texture.info, width * height * layers);
      if (stored) {
        const layerSize = width * height * texture.info.components;
        for (let layer = 0; layer < layers; layer++) {
          made.set(readFromGl(ctx, texture, level, layer, 0, 0, width, height), layer * layerSize);
        }
      }
      data = made;
      ctx.stats.readback.ms += performance.now() - start;
      ctx.stats.readback.textures++;
      return made;
    },
  };
}

/** The texture's CPU copy, current: every level, rows top to bottom, layers in order. */
export function cpuLevels(ctx: Ctx, texture: TextureImpl): CpuLevelCopy[] {
  if (texture.cpu && texture.home !== 'gl') return texture.cpu;
  const levels = Array.from({ length: texture.mipLevelCount }, (_, level) =>
    lazyLevel(ctx, texture, level),
  );
  texture.cpu = levels;
  texture.home = 'both';
  return levels;
}

installCpuUpload((ctx, texture) => {
  const levels = texture.cpu;
  if (!levels) return;
  const start = performance.now();
  texture.home = 'both';
  levels.forEach((copy, level) => {
    // A level no kernel touched holds on GL what it held before
    if (!copy.loaded) return;
    const bytes = new Uint8Array(copy.data.buffer, copy.data.byteOffset, copy.data.byteLength);
    upload(
      ctx,
      texture,
      {
        level,
        x: 0,
        y: 0,
        z: 0,
        width: copy.width,
        height: copy.height,
        layers: copy.depthOrArrayLayers,
      },
      bytes,
      copy.width,
      copy.height,
    );
  });
  texture.home = 'both';
  ctx.stats.upload.ms += performance.now() - start;
  ctx.stats.upload.textures++;
});

export function copyTextureToBuffer(
  ctx: Ctx,
  source: GPUTexelCopyTextureInfo,
  destination: BufferImpl,
  layout: GPUTexelCopyBufferLayout,
  size: GPUExtent3DStrict,
): void {
  destination.sync();
  const texture = textureImpl(source.texture);
  const [width, height, layers] = extent(size);
  const [x, y, z] = origin(source.origin);
  const level = source.mipLevel ?? 0;
  const tight = width * texture.info.bytes;
  const offset = layout.offset ?? 0;
  const bytesPerRow = layout.bytesPerRow ?? tight;
  const rowsPerImage = layout.rowsPerImage ?? height;
  for (let layer = 0; layer < layers; layer++) {
    const texels = readTexels(ctx, texture, level, z + layer, x, y, width, height);
    let bytes = new Uint8Array(texels.buffer, texels.byteOffset, texels.byteLength);
    if (texture.info.bgra) bytes = swapRedBlue(bytes);
    for (let row = 0; row < height; row++) {
      const at = offset + layer * bytesPerRow * rowsPerImage + row * bytesPerRow;
      destination.bytes.set(bytes.subarray(row * tight, (row + 1) * tight), at);
    }
  }
  written(
    destination,
    offset,
    offset + (layers - 1) * bytesPerRow * rowsPerImage + (height - 1) * bytesPerRow + tight,
  );
}

export function copyTextureToTexture(
  ctx: Ctx,
  source: GPUTexelCopyTextureInfo,
  destination: GPUTexelCopyTextureInfo,
  size: GPUExtent3DStrict,
): void {
  const from = textureImpl(source.texture);
  const to = textureImpl(destination.texture);
  const [width, height, layers] = extent(size);
  const [sx, sy, sz] = origin(source.origin);
  const [dx, dy, dz] = origin(destination.origin);
  const sourceLevel = source.mipLevel ?? 0;
  const destinationLevel = destination.mipLevel ?? 0;
  onGl(ctx, from);
  onGl(ctx, to);
  if (from.info.renderable && to.info.renderable && from.info.internal === to.info.internal) {
    for (let layer = 0; layer < layers; layer++) {
      blitLevel(
        ctx,
        { texture: from, level: sourceLevel, layer: sz + layer, x: sx, y: sy },
        { texture: to, level: destinationLevel, layer: dz + layer, x: dx, y: dy },
        width,
        height,
      );
    }
    glWritten(to);
    return;
  }
  for (let layer = 0; layer < layers; layer++) {
    const texels = readTexels(ctx, from, sourceLevel, sz + layer, sx, sy, width, height);
    upload(
      ctx,
      to,
      { level: destinationLevel, x: dx, y: dy, z: dz + layer, width, height, layers: 1 },
      new Uint8Array(texels.buffer, texels.byteOffset, texels.byteLength),
      width,
      height,
    );
  }
}

type ImageSource = GPUCopyExternalImageSource;

function sourceSize(source: ImageSource): [number, number] {
  if ('videoWidth' in source) return [source.videoWidth, source.videoHeight];
  if ('displayWidth' in source) return [source.displayWidth, source.displayHeight];
  if ('naturalWidth' in source) return [source.naturalWidth, source.naturalHeight];
  return [source.width, source.height];
}

/** Draws an `ImageBitmap` upside down onto a canvas, which WebGL then uploads unflipped. */
function flipped(source: ImageSource, x: number, y: number, width: number, height: number) {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d')!;
  context.translate(0, height);
  context.scale(1, -1);
  context.drawImage(source as CanvasImageSource, x, y, width, height, 0, 0, width, height);
  return canvas;
}

export function copyExternalImageToTexture(
  ctx: Ctx,
  source: GPUCopyExternalImageSourceInfo,
  destination: GPUCopyExternalImageDestInfo,
  size: GPUExtent3DStrict,
): void {
  const { gl, state } = ctx;
  const texture = textureImpl(destination.texture);
  const { info } = texture;
  onGl(ctx, texture);
  if (!texture.texture) {
    throw new Error('@typegpu/kit-gl: multisampled textures cannot take images.');
  }
  const [width, height] = extent(size);
  if (width === 0 || height === 0) return;
  const [sx, sy] = origin2d(source.origin);
  const [x, y, z] = origin(destination.origin);
  const level = destination.mipLevel ?? 0;
  const flipY = source.flipY ?? false;
  const premultiply = destination.premultipliedAlpha ?? false;

  let image: TexImageSource = source.source as TexImageSource;
  let skipX = sx;
  let skipY = sy;
  // WebGL ignores the flip and premultiply settings for an ImageBitmap
  if (flipY && typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) {
    image = flipped(image, sx, sourceSize(image)[1] - sy - height, width, height);
    skipX = 0;
    skipY = 0;
  }
  const domFlip = flipY && image === source.source;

  state.scratchTexture(texture.target, texture.texture);
  state.unpack(domFlip, premultiply, 0, 0);
  gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, skipX);
  gl.pixelStorei(gl.UNPACK_SKIP_ROWS, skipY);
  const type =
    info.type === gl.UNSIGNED_INT_10F_11F_11F_REV || info.array === 'f16' ? gl.FLOAT : info.type;
  if (texture.target === gl.TEXTURE_2D) {
    gl.texSubImage2D(gl.TEXTURE_2D, level, x, y, width, height, info.upload, type, image);
  } else if (texture.target === gl.TEXTURE_CUBE_MAP) {
    gl.texSubImage2D(
      gl.TEXTURE_CUBE_MAP_POSITIVE_X + z,
      level,
      x,
      y,
      width,
      height,
      info.upload,
      type,
      image,
    );
  } else {
    gl.texSubImage3D(texture.target, level, x, y, z, width, height, 1, info.upload, type, image);
  }
  gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
  gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
  glWritten(texture);
}

/**
 * `importExternalTexture`: uploads the video's current frame into an `rgba8unorm` texture kept per
 * source, which a shader then samples as a 2D texture.
 */
export function importExternalTexture(
  ctx: Ctx,
  createTexture: (descriptor: GPUTextureDescriptor) => GPUTexture,
  cache: WeakMap<object, GPUTexture>,
  descriptor: GPUExternalTextureDescriptor,
): GPUExternalTexture {
  const source = descriptor.source as ImageSource;
  const [width, height] = sourceSize(source);
  let texture = cache.get(source);
  if (texture && (texture.width !== width || texture.height !== height)) {
    texture.destroy();
    texture = undefined;
  }
  if (!texture) {
    texture = createTexture({
      size: [Math.max(1, width), Math.max(1, height)],
      format: 'rgba8unorm',
      usage: 0x4 | 0x2 | 0x10,
      label: descriptor.label ?? 'external texture',
    });
    cache.set(source, texture);
  }
  if (width > 0 && height > 0) {
    copyExternalImageToTexture(ctx, { source }, { texture }, [width, height]);
  }
  const view = texture.createView();
  return {
    [IMPL]: (view as unknown as { [IMPL]: unknown })[IMPL],
    __brand: 'GPUExternalTexture',
    label: descriptor.label ?? '',
  } as unknown as GPUExternalTexture;
}
