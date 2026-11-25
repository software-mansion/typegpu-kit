/**
 * Moving texels: uploads from the CPU and from images, copies between textures and buffers,
 * readback, and the CPU copies of textures that compute kernels use.
 */

import { type BufferImpl, written } from './buffer.ts';
import { type Ctx, IMPL } from './context.ts';
import { link } from './program.ts';
import { packTexels, swapRedBlue, type TexelArray, texelArray, viewAs } from './packing.ts';
import { discard, packPixels, type Packed, signalled, unpack } from './pack.ts';
import { readBack as readBackFrom } from './placement.ts';
import {
  attachLevel,
  blitLevel,
  type CpuLevelCopy,
  drawPlainly,
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

/**
 * Fetches texels of a 2d, 2d-array or 3d texture and writes their bits to an RGBA32UI target,
 * which every WebGL 2 renders to: a float target would need EXT_color_buffer_float.
 */
const readbackFragment = (sampler: 'sampler2D' | 'sampler2DArray' | 'sampler3D') => `#version 300 es
precision highp float;
precision highp ${sampler};
uniform ${sampler} source;
uniform ivec2 origin;
uniform int layer;
uniform int level;
out uvec4 color;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy) + origin;
  color = floatBitsToUint(texelFetch(source, ${sampler === 'sampler2D' ? 'at' : 'ivec3(at, layer)'}, level));
}
`;

/** Reads a region of one level and layer as RGBA floats by drawing their bits into an RGBA32UI target. */
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
  let source = texture;
  let sourceLevel = level;
  let temporary: WebGLTexture | null = null;
  // Cube faces have no texelFetch, so a face is copied out first
  if (texture.target === gl.TEXTURE_CUBE_MAP) {
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
  const sampler =
    source.target === gl.TEXTURE_3D
      ? 'sampler3D'
      : source.target === gl.TEXTURE_2D_ARRAY
        ? 'sampler2DArray'
        : 'sampler2D';
  const linked = link(ctx, READBACK_VERTEX, readbackFragment(sampler), 'the readback program');
  if (!linked.ok) throw new Error(linked.message);
  const { program, uniforms } = linked.program;

  const target = gl.createTexture();
  state.scratchTexture(gl.TEXTURE_2D, target);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, width, height);
  state.scratchTexture(source.target, source.texture);
  // The level goes to texelFetch, so the read does not depend on how a driver applies base levels
  gl.texParameteri(source.target, gl.TEXTURE_BASE_LEVEL, 0);
  gl.texParameteri(
    source.target,
    gl.TEXTURE_MAX_LEVEL,
    source === texture ? texture.mipLevelCount - 1 : 0,
  );
  texture.levelRange = -1;
  state.sampler(state.scratchUnit, ctx.nearest);

  state.drawFramebuffer(ctx.scratch.draw);
  gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
  state.drawBuffers([gl.COLOR_ATTACHMENT0]);
  state.program(program);
  gl.uniform1i(uniforms.get('source')!, state.scratchUnit);
  gl.uniform2i(uniforms.get('origin')!, x, y);
  gl.uniform1i(uniforms.get('level')!, sourceLevel);
  const layered = uniforms.get('layer');
  if (layered) gl.uniform1i(layered, source === texture ? layer : 0);
  state.vertexArray(null);
  state.viewport(0, 0, width, height);
  drawPlainly(ctx);
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
  state.readBuffer(gl.COLOR_ATTACHMENT0);
  const bits = new Uint32Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, bits);
  const out = new Float32Array(bits.buffer);
  const stride = 4;
  gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  gl.deleteTexture(target);
  if (temporary) gl.deleteTexture(temporary);
  return { values: out, stride };
}

const readFormats = new WeakMap<Ctx, Map<string, boolean>>();

/**
 * Whether the bound read framebuffer, whose attachment has the `internal` format, reads back in
 * `format` and `type`. Asked once per format: the query waits for the GPU in some browsers.
 */
function readsAs(ctx: Ctx, internal: number, format: number, type: number): boolean {
  const { gl } = ctx;
  let known = readFormats.get(ctx);
  if (!known) readFormats.set(ctx, (known = new Map()));
  const key = `${internal}:${format}:${type}`;
  let reads = known.get(key);
  if (reads === undefined) {
    reads =
      gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) === format &&
      gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE) === type;
    known.set(key, reads);
  }
  return reads;
}

/**
 * How the bound read framebuffer gives a texture's texels: `readPixels`' format and type, an array
 * for what it reads, and the conversion into the format's own layout. Undefined for formats only a
 * shader can read.
 */
type Raw = Uint8Array | Uint16Array | Uint32Array | Int32Array | Float32Array;

interface Readout {
  readonly format: number;
  readonly type: number;
  raw(count: number): Raw;
  convert(raw: Raw, out: TexelArray, texels: number): void;
}

/** Whether `readPixels` reads the format; the others go through a shader. */
const readsDirectly = (info: TextureImpl['info']) =>
  info.renderable && !info.hasDepth && info.kind !== 'snorm';

/** The readout of a format `readsDirectly` reads, with the texture attached for reading. */
function readoutOf(ctx: Ctx, info: TextureImpl['info']): Readout {
  const { gl } = ctx;
  const packed = (format: number, type: number, raw: Readout['raw']): Readout => ({
    format,
    type,
    raw,
    convert: (from, out) => packTexels(info, from as Parameters<typeof packTexels>[1], out),
  });
  if (info.format === 'rgb10a2unorm') {
    return {
      format: gl.RGBA,
      type: gl.UNSIGNED_INT_2_10_10_10_REV,
      raw: (count) => new Uint32Array(count / 4),
      convert: (from, out) => out.set(from),
    };
  }
  if (info.kind === 'unorm') return packed(gl.RGBA, gl.UNSIGNED_BYTE, (n) => new Uint8Array(n));
  if (info.array === 'f16' && readsAs(ctx, info.internal, gl.RGBA, gl.HALF_FLOAT)) {
    return {
      format: gl.RGBA,
      type: gl.HALF_FLOAT,
      raw: (count) => new Uint16Array(count),
      convert: (from, out, texels) => {
        const n = info.components;
        if (n === 4) out.set(from);
        else
          for (let i = 0; i < texels; i++)
            for (let c = 0; c < n; c++) out[i * n + c] = from[i * 4 + c]!;
      },
    };
  }
  if (info.kind === 'float') return packed(gl.RGBA, gl.FLOAT, (n) => new Float32Array(n));
  if (info.kind === 'uint') {
    return packed(gl.RGBA_INTEGER, gl.UNSIGNED_INT, (n) => new Uint32Array(n));
  }
  return packed(gl.RGBA_INTEGER, gl.INT, (n) => new Int32Array(n));
}

/** Binds one level and layer of a texture on GL for reading, and returns how to read it. */
function readFrom(ctx: Ctx, texture: TextureImpl, level: number, layer: number) {
  const { gl, state } = ctx;
  onGl(ctx, texture);
  if (texture.renderbuffer) {
    throw new Error('@typegpu/kit-gl: multisampled textures cannot be read back on WebGL 2.');
  }
  if (!readsDirectly(texture.info)) return undefined;
  // Apple's OpenGL driver reads level 0 of a 32-bit float texture whichever level is attached
  if (level > 0 && texture.info.array === 'f32') return undefined;
  state.readFramebuffer(ctx.scratch.read);
  attachLevel(ctx, gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, texture, level, layer);
  state.readBuffer(gl.COLOR_ATTACHMENT0);
  // The f16 check asks the bound framebuffer, so the readout is chosen with it attached
  return readoutOf(ctx, texture.info);
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
  const { gl } = ctx;
  const { info } = texture;
  const out = texelArray(info, width * height);
  const readout = readFrom(ctx, texture, level, layer);
  if (!readout) {
    const { values, stride } = readThroughShader(ctx, texture, level, layer, x, y, width, height);
    packTexels(info, values, out, stride);
    return out;
  }
  const raw = readout.raw(width * height * 4);
  gl.readPixels(x, y, width, height, readout.format, readout.type, raw);
  readout.convert(raw, out, width * height);
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  return out;
}

/**
 * Starts reading a region of one level and layer from GL without waiting for the GPU, or returns
 * undefined where only a synchronous read can. `finish` gives the texels as they were when the
 * read started, in the format's own layout, waiting if the GPU has not finished.
 */
export function readLaterFromGl(
  ctx: Ctx,
  texture: TextureImpl,
  level: number,
  layer: number,
  x: number,
  y: number,
  width: number,
  height: number,
): { readonly packed: Packed; finish(): TexelArray } | undefined {
  const { gl } = ctx;
  const readout = readFrom(ctx, texture, level, layer);
  if (!readout) return undefined;
  const raw = readout.raw(width * height * 4);
  const packed = packPixels(ctx, x, y, width, height, readout.format, readout.type, raw.byteLength);
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  return {
    packed,
    finish() {
      unpack(ctx, packed, raw);
      const out = texelArray(texture.info, width * height);
      readout.convert(raw, out, width * height);
      return out;
    },
  };
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
      const ms = performance.now() - start;
      if (stored) readBackFrom(ctx, texture, ms);
      ctx.stats.readback.ms += ms;
      ctx.stats.readback.textures++;
      const key = `${texture.label} ${texture.format} ${width}x${height}x${layers}`;
      const seen = ctx.stats.readbacks.get(key);
      if (seen) {
        seen.ms += ms;
        seen.levels++;
      } else {
        ctx.stats.readbacks.set(key, { ms, levels: 1 });
      }
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

/**
 * Starts `copyTextureToBuffer` into a mappable buffer without waiting for the GPU, where the texture
 * is current only on GL and its format reads with `readPixels`. Returns whether it did; the caller
 * copies synchronously otherwise.
 */
export function copyTextureToBufferLater(
  ctx: Ctx,
  source: GPUTexelCopyTextureInfo,
  destination: BufferImpl,
  layout: GPUTexelCopyBufferLayout,
  size: GPUExtent3DStrict,
): boolean {
  const texture = textureImpl(source.texture);
  if (texture.home === 'cpu' || texture.renderbuffer || destination.destroyed) return false;
  const [width, height, layers] = extent(size);
  const [x, y, z] = origin(source.origin);
  const level = source.mipLevel ?? 0;
  const reads: NonNullable<ReturnType<typeof readLaterFromGl>>[] = [];
  for (let layer = 0; layer < layers; layer++) {
    const read = readLaterFromGl(ctx, texture, level, z + layer, x, y, width, height);
    if (!read) {
      for (const each of reads) discard(ctx, each.packed);
      return false;
    }
    reads.push(read);
  }
  const tight = width * texture.info.bytes;
  const offset = layout.offset ?? 0;
  const bytesPerRow = layout.bytesPerRow ?? tight;
  const rowsPerImage = layout.rowsPerImage ?? height;
  destination.reads.push({
    ready: () => reads.every((read) => signalled(ctx, read.packed)),
    finish() {
      reads.forEach((read, layer) => {
        const texels = read.finish();
        let bytes = new Uint8Array(texels.buffer, texels.byteOffset, texels.byteLength);
        if (texture.info.bgra) bytes = swapRedBlue(bytes);
        for (let row = 0; row < height; row++) {
          const at = offset + layer * bytesPerRow * rowsPerImage + row * bytesPerRow;
          destination.bytes.set(bytes.subarray(row * tight, (row + 1) * tight), at);
        }
      });
      written(
        destination,
        offset,
        offset + (layers - 1) * bytesPerRow * rowsPerImage + (height - 1) * bytesPerRow + tight,
      );
    },
    cancel: () => {
      for (const read of reads) discard(ctx, read.packed);
    },
  });
  return true;
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

/** Draws a region of an `ImageBitmap` onto a canvas, upside down if `flip`, for WebGL to upload. */
function drawn(
  source: ImageSource,
  x: number,
  y: number,
  width: number,
  height: number,
  flip: boolean,
) {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d')!;
  if (flip) {
    context.translate(0, height);
    context.scale(1, -1);
  }
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
  // WebGL ignores the flip and premultiply settings for an ImageBitmap, so it goes through a canvas,
  // which WebGL premultiplies. WebGPU flips within the region, whose origin counts from the top
  if (
    (flipY || premultiply) &&
    typeof ImageBitmap !== 'undefined' &&
    image instanceof ImageBitmap
  ) {
    image = drawn(image, sx, sy, width, height, flipY);
    skipX = 0;
    skipY = 0;
  }
  const domFlip = flipY && image === source.source;
  // WebGL flips the whole source before it skips rows
  if (domFlip) skipY = sourceSize(image as ImageSource)[1] - sy - height;

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
