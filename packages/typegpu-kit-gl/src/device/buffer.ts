/**
 * Buffers. The CPU copy is current unless a compute kernel that ran as fragment passes replaced the
 * data texture, the one GL copy a shader writes; `sync` reads it back before the CPU copy is used.
 * GL holds lazily made copies for vertex, index and uniform input and an `RGBA32UI` data texture for
 * bindings, each with the byte range the CPU changed since its last upload.
 */

import { type BufferStats, type Ctx, IMPL, implOf, MAP_MODE } from './context.ts';

/** Bytes per data-texture row is `ROW_TEXELS * 16` unless the buffer needs wider rows. */
const ROW_TEXELS = 1024;

function statsOf(ctx: Ctx, buffer: BufferImpl): BufferStats {
  const key = `${buffer.label} ${buffer.size}`;
  let stats = ctx.stats.buffers.get(key);
  if (!stats) {
    stats = { readMs: 0, reads: 0, uploadMs: 0, uploads: 0, uploadBytes: 0 };
    ctx.stats.buffers.set(key, stats);
  }
  return stats;
}

/** Counts an upload of `bytes` that began at `start`. */
function uploaded(ctx: Ctx, buffer: BufferImpl, start: number, bytes: number): void {
  const stats = statsOf(ctx, buffer);
  stats.uploadMs += performance.now() - start;
  stats.uploads++;
  stats.uploadBytes += bytes;
}

interface Copy<T> {
  value: T;
  start: number;
  end: number;
}

export interface DataTexture {
  readonly texture: WebGLTexture;
  readonly width: number;
  readonly height: number;
}

export interface BufferImpl {
  readonly id: number;
  /** The descriptor's label, for device stats. */
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  /** The buffer's bytes, padded to whole 16-byte texels. */
  readonly bytes: Uint8Array;
  readonly words: Uint32Array;
  destroyed: boolean;
  /** The data texture holds newer contents than the CPU copy. */
  gpuNewer: boolean;
  /** Makes the CPU copy current. Call before reading or writing `bytes`. */
  sync(): void;
  vertex: Copy<WebGLBuffer> | undefined;
  index: Copy<WebGLBuffer> | undefined;
  data: Copy<DataTexture> | undefined;
  uniform: Copy<WebGLBuffer> | undefined;
  /** Uniform copies of ranges the whole copy cannot bind, by first byte and size. */
  readonly slices: Map<
    string,
    Copy<WebGLBuffer> & { readonly offset: number; readonly size: number }
  >;
  free(): void;
}

export function bufferImpl(buffer: GPUBuffer): BufferImpl {
  return implOf<BufferImpl>(buffer, 'buffer');
}

/** Records that the CPU copy changed in `[start, end)`, for every GL copy to upload. */
export function written(buffer: BufferImpl, start: number, end: number): void {
  if (end <= start) return;
  for (const copy of [buffer.vertex, buffer.index, buffer.data, buffer.uniform]) {
    if (!copy) continue;
    copy.start = Math.min(copy.start, start);
    copy.end = Math.max(copy.end, end);
  }
  for (const slice of buffer.slices.values()) {
    slice.start = Math.min(slice.start, start);
    slice.end = Math.max(slice.end, end);
  }
}

/** The GL buffer for vertex input, current. Binds it to `ARRAY_BUFFER`. */
export function vertexBuffer(ctx: Ctx, buffer: BufferImpl): WebGLBuffer {
  const { gl, state } = ctx;
  buffer.sync();
  let copy = buffer.vertex;
  if (!copy) {
    const value = gl.createBuffer();
    state.arrayBuffer(value);
    gl.bufferData(gl.ARRAY_BUFFER, buffer.bytes, gl.DYNAMIC_DRAW);
    copy = buffer.vertex = { value, start: Infinity, end: 0 };
    return value;
  }
  state.arrayBuffer(copy.value);
  if (copy.start < copy.end) {
    const start = performance.now();
    gl.bufferSubData(gl.ARRAY_BUFFER, copy.start, buffer.bytes, copy.start, copy.end - copy.start);
    uploaded(ctx, buffer, start, copy.end - copy.start);
    copy.start = Infinity;
    copy.end = 0;
  }
  return copy.value;
}

/**
 * The GL buffer for index input, current. Binds it to `ELEMENT_ARRAY_BUFFER` of the vertex array
 * that is bound, since WebGL never lets an index buffer bind anywhere else.
 */
export function indexBuffer(ctx: Ctx, buffer: BufferImpl): WebGLBuffer {
  const { gl } = ctx;
  buffer.sync();
  let copy = buffer.index;
  if (!copy) {
    const value = gl.createBuffer();
    ctx.state.elementBuffer(value);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, buffer.bytes, gl.DYNAMIC_DRAW);
    copy = buffer.index = { value, start: Infinity, end: 0 };
    return value;
  }
  ctx.state.elementBuffer(copy.value);
  if (copy.start < copy.end) {
    const start = performance.now();
    gl.bufferSubData(
      gl.ELEMENT_ARRAY_BUFFER,
      copy.start,
      buffer.bytes,
      copy.start,
      copy.end - copy.start,
    );
    uploaded(ctx, buffer, start, copy.end - copy.start);
    copy.start = Infinity;
    copy.end = 0;
  }
  return copy.value;
}

/**
 * Binds `size` bytes of the buffer from `offset` to a uniform block binding point, current. A range
 * GL cannot bind in place, at an offset off `UNIFORM_BUFFER_OFFSET_ALIGNMENT` or past the end,
 * binds a copy of its own.
 */
export function bindUniformRange(
  ctx: Ctx,
  buffer: BufferImpl,
  point: number,
  offset: number,
  size: number,
): void {
  const { gl } = ctx;
  buffer.sync();
  const upload = (copy: Copy<WebGLBuffer>, base: number, length: number) => {
    if (copy.start >= copy.end) return;
    gl.bindBuffer(gl.UNIFORM_BUFFER, copy.value);
    const start = Math.max(copy.start, base);
    const end = Math.min(copy.end, base + length, buffer.bytes.length);
    if (start < end) {
      const began = performance.now();
      gl.bufferSubData(gl.UNIFORM_BUFFER, start - base, buffer.bytes, start, end - start);
      uploaded(ctx, buffer, began, end - start);
    }
    copy.start = Infinity;
    copy.end = 0;
  };
  if (offset % ctx.uniformAlignment === 0 && offset + size <= buffer.bytes.length) {
    let copy = buffer.uniform;
    if (!copy) {
      const value = gl.createBuffer();
      gl.bindBuffer(gl.UNIFORM_BUFFER, value);
      gl.bufferData(gl.UNIFORM_BUFFER, buffer.bytes, gl.DYNAMIC_DRAW);
      copy = buffer.uniform = { value, start: Infinity, end: 0 };
    } else {
      upload(copy, 0, buffer.bytes.length);
    }
    ctx.state.uniformRange(point, copy.value, offset, size);
    return;
  }
  const key = `${offset}:${size}`;
  let slice = buffer.slices.get(key);
  if (!slice) {
    const value = gl.createBuffer();
    gl.bindBuffer(gl.UNIFORM_BUFFER, value);
    gl.bufferData(gl.UNIFORM_BUFFER, size, gl.DYNAMIC_DRAW);
    slice = { value, offset, size, start: 0, end: buffer.bytes.length };
    buffer.slices.set(key, slice);
  }
  upload(slice, offset, size);
  ctx.state.uniformRange(point, slice.value, 0, size);
}

/** The data texture a shader reads the buffer through, current. */
export function dataTexture(ctx: Ctx, buffer: BufferImpl): DataTexture {
  const { gl, state } = ctx;
  let copy = buffer.data;
  const texels = buffer.words.length / 4;
  if (!copy) {
    const width = Math.max(
      1,
      Math.min(texels, Math.max(ROW_TEXELS, Math.ceil(texels / ctx.maxTextureSize))),
    );
    const height = Math.max(1, Math.ceil(texels / width));
    if (width > ctx.maxTextureSize || height > ctx.maxTextureSize) {
      throw new Error(
        `@typegpu/kit-gl: a ${buffer.size}-byte buffer is too large to bind on WebGL 2.`,
      );
    }
    const texture = gl.createTexture();
    state.scratchTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, width, height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    copy = buffer.data = { value: { texture, width, height }, start: 0, end: buffer.bytes.length };
  }
  if (copy.start < copy.end) {
    const began = performance.now();
    const bytes = Math.min(copy.end, buffer.bytes.length) - copy.start;
    const { texture, width } = copy.value;
    state.scratchTexture(gl.TEXTURE_2D, texture);
    state.unpack(false, false, 0, 0);
    const first = Math.floor(copy.start / 16);
    const last = Math.min(texels, Math.ceil(copy.end / 16));
    const firstRow = Math.floor(first / width);
    const lastRow = Math.floor((last - 1) / width);
    const upload = (x: number, y: number, w: number, h: number) =>
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        x,
        y,
        w,
        h,
        gl.RGBA_INTEGER,
        gl.UNSIGNED_INT,
        buffer.words,
        (y * width + x) * 4,
      );
    if (firstRow === lastRow) {
      upload(first - firstRow * width, firstRow, last - first, 1);
    } else {
      upload(0, firstRow, width, lastRow - firstRow);
      upload(0, lastRow, last - lastRow * width, 1);
    }
    uploaded(ctx, buffer, began, bytes);
    copy.start = Infinity;
    copy.end = 0;
  }
  return copy.value;
}

/** Reads the data texture back into the CPU copy, which every other GL copy then uploads. */
function readBack(ctx: Ctx, buffer: BufferImpl): void {
  const { gl, state } = ctx;
  const began = performance.now();
  const data = buffer.data!.value;
  const all = new Uint32Array(data.width * data.height * 4);
  state.readFramebuffer(ctx.scratch.read);
  gl.framebufferTexture2D(
    gl.READ_FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    data.texture,
    0,
  );
  state.readBuffer(gl.COLOR_ATTACHMENT0);
  gl.readPixels(0, 0, data.width, data.height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, all);
  gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
  buffer.words.set(all.subarray(0, buffer.words.length));
  buffer.gpuNewer = false;
  written(buffer, 0, buffer.bytes.length);
  buffer.data!.start = Infinity;
  buffer.data!.end = 0;
  const stats = statsOf(ctx, buffer);
  stats.readMs += performance.now() - began;
  stats.reads++;
}

/**
 * Replaces the data texture with one a fragment pass wrote, of the same size, which leaves every
 * other copy stale. Returns the texture it replaced, for the caller to reuse or delete.
 */
export function replaceData(buffer: BufferImpl, texture: WebGLTexture): WebGLTexture {
  const copy = buffer.data!;
  const old = copy.value.texture;
  copy.value = { ...copy.value, texture };
  copy.start = Infinity;
  copy.end = 0;
  buffer.gpuNewer = true;
  return old;
}

export function createBuffer(ctx: Ctx, descriptor: GPUBufferDescriptor): GPUBuffer {
  const { gl } = ctx;
  const { size, usage } = descriptor;
  const padded = Math.max(16, Math.ceil(size / 16) * 16);
  const storage = new ArrayBuffer(padded);

  const impl: BufferImpl = {
    id: ctx.id(),
    label: descriptor.label ?? '',
    size,
    usage,
    bytes: new Uint8Array(storage),
    words: new Uint32Array(storage),
    destroyed: false,
    gpuNewer: false,
    sync() {
      if (impl.gpuNewer && impl.data && !impl.destroyed) readBack(ctx, impl);
    },
    vertex: undefined,
    index: undefined,
    data: undefined,
    uniform: undefined,
    slices: new Map(),
    free() {
      if (impl.vertex) gl.deleteBuffer(impl.vertex.value);
      if (impl.index) gl.deleteBuffer(impl.index.value);
      if (impl.data) gl.deleteTexture(impl.data.value.texture);
      if (impl.uniform) gl.deleteBuffer(impl.uniform.value);
      for (const slice of impl.slices.values()) gl.deleteBuffer(slice.value);
      impl.slices.clear();
      impl.vertex = impl.index = impl.data = impl.uniform = undefined;
    },
  };
  const untrack = ctx.track(() => impl.free());

  let mapState: GPUBufferMapState = descriptor.mappedAtCreation ? 'mapped' : 'unmapped';
  let mapMode = descriptor.mappedAtCreation ? MAP_MODE.WRITE : 0;
  let mapOffset = 0;
  let mapSize = 0;
  let pending: { reject(error: unknown): void } | undefined;
  let ranges: { offset: number; data: ArrayBuffer }[] = [];
  if (descriptor.mappedAtCreation) mapSize = size;

  const buffer = {
    [IMPL]: impl,
    __brand: 'GPUBuffer' as const,
    label: descriptor.label ?? '',
    get size() {
      return size;
    },
    get usage() {
      return usage;
    },
    get mapState() {
      return mapState;
    },

    mapAsync(mode: GPUMapModeFlags, offset = 0, rangeSize?: number): Promise<undefined> {
      if (mapState !== 'unmapped' || impl.destroyed) {
        const reason = impl.destroyed ? 'destroyed' : `already ${mapState}`;
        ctx.errors.report('validation', `mapAsync on a buffer that is ${reason}.`);
        return Promise.reject(new DOMException(`The buffer is ${reason}.`, 'OperationError'));
      }
      mapState = 'pending';
      return new Promise<undefined>((resolve, reject) => {
        const map = { reject };
        pending = map;
        queueMicrotask(() => {
          if (pending !== map) return;
          pending = undefined;
          impl.sync();
          mapState = 'mapped';
          mapMode = mode;
          mapOffset = offset;
          mapSize = rangeSize ?? size - offset;
          resolve(undefined);
        });
      });
    },

    getMappedRange(offset = mapOffset, rangeSize?: number): ArrayBuffer {
      if (mapState !== 'mapped') {
        throw new DOMException('getMappedRange on a buffer that is not mapped.', 'OperationError');
      }
      const length = rangeSize ?? mapOffset + mapSize - offset;
      if (offset === 0 && length === storage.byteLength) {
        ranges.push({ offset, data: storage });
        return storage;
      }
      const data = storage.slice(offset, offset + length);
      ranges.push({ offset, data });
      return data;
    },

    unmap(): void {
      if (pending) {
        pending.reject(new DOMException('The buffer was unmapped before mapping.', 'AbortError'));
        pending = undefined;
      }
      if (mapState === 'mapped' && mapMode & MAP_MODE.WRITE) {
        for (const range of ranges) {
          if (range.data !== storage) impl.bytes.set(new Uint8Array(range.data), range.offset);
          written(impl, range.offset, range.offset + range.data.byteLength);
        }
      }
      ranges = [];
      mapState = 'unmapped';
    },

    destroy(): void {
      if (impl.destroyed) return;
      buffer.unmap();
      impl.destroyed = true;
      impl.free();
      untrack();
    },
  };
  return buffer as unknown as GPUBuffer;
}

/** Copies `size` bytes from the start of `data` into the buffer at `offset`. */
export function writeBytes(buffer: BufferImpl, offset: number, data: Uint8Array): void {
  buffer.sync();
  buffer.bytes.set(data, offset);
  written(buffer, offset, offset + data.byteLength);
}
