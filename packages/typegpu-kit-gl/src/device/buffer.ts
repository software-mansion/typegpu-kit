/**
 * Buffers. The CPU copy is the only one anything writes, since WebGL 2 shaders cannot write
 * buffers: GL holds lazily made copies for vertex and index input and an `RGBA32UI` data texture
 * for bindings, each with the byte range the CPU changed since its last upload.
 */

import { BUFFER_USAGE, type Ctx, IMPL, implOf, MAP_MODE } from './context.ts';

/** Bytes per data-texture row is `ROW_TEXELS * 16` unless the buffer needs wider rows. */
const ROW_TEXELS = 1024;

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
  readonly size: number;
  readonly usage: number;
  /** The buffer's bytes, padded to whole 16-byte texels. */
  readonly bytes: Uint8Array;
  readonly words: Uint32Array;
  destroyed: boolean;
  vertex: Copy<WebGLBuffer> | undefined;
  index: Copy<WebGLBuffer> | undefined;
  data: Copy<DataTexture> | undefined;
  free(): void;
}

export function bufferImpl(buffer: GPUBuffer): BufferImpl {
  return implOf<BufferImpl>(buffer, 'buffer');
}

/** Records that the CPU copy changed in `[start, end)`, for every GL copy to upload. */
export function written(buffer: BufferImpl, start: number, end: number): void {
  if (end <= start) return;
  for (const copy of [buffer.vertex, buffer.index, buffer.data]) {
    if (!copy) continue;
    copy.start = Math.min(copy.start, start);
    copy.end = Math.max(copy.end, end);
  }
}

/** The GL buffer for vertex input, current. Binds it to `ARRAY_BUFFER`. */
export function vertexBuffer(ctx: Ctx, buffer: BufferImpl): WebGLBuffer {
  const { gl, state } = ctx;
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
    gl.bufferSubData(gl.ARRAY_BUFFER, copy.start, buffer.bytes, copy.start, copy.end - copy.start);
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
  let copy = buffer.index;
  if (!copy) {
    const value = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, value);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, buffer.bytes, gl.DYNAMIC_DRAW);
    copy = buffer.index = { value, start: Infinity, end: 0 };
    return value;
  }
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, copy.value);
  if (copy.start < copy.end) {
    gl.bufferSubData(
      gl.ELEMENT_ARRAY_BUFFER,
      copy.start,
      buffer.bytes,
      copy.start,
      copy.end - copy.start,
    );
    copy.start = Infinity;
    copy.end = 0;
  }
  return copy.value;
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
    copy.start = Infinity;
    copy.end = 0;
  }
  return copy.value;
}

export function createBuffer(ctx: Ctx, descriptor: GPUBufferDescriptor): GPUBuffer {
  const { gl } = ctx;
  const { size, usage } = descriptor;
  const padded = Math.max(16, Math.ceil(size / 16) * 16);
  const storage = new ArrayBuffer(padded);

  const impl: BufferImpl = {
    id: ctx.id(),
    size,
    usage,
    bytes: new Uint8Array(storage),
    words: new Uint32Array(storage),
    destroyed: false,
    vertex: undefined,
    index: undefined,
    data: undefined,
    free() {
      if (impl.vertex) gl.deleteBuffer(impl.vertex.value);
      if (impl.index) gl.deleteBuffer(impl.index.value);
      if (impl.data) gl.deleteTexture(impl.data.value.texture);
      impl.vertex = impl.index = impl.data = undefined;
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
  buffer.bytes.set(data, offset);
  written(buffer, offset, offset + data.byteLength);
}

export function isIndexOrVertex(buffer: BufferImpl): boolean {
  return (buffer.usage & (BUFFER_USAGE.INDEX | BUFFER_USAGE.VERTEX)) !== 0;
}
