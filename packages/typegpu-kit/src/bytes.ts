// Reuse typed views and field offsets to avoid allocations during scalar and vector writes. Unlike
// TypeGPU's writer, this writer leaves omitted struct fields unchanged
import { d, writeToArrayBuffer } from 'typegpu';

export interface ByteViews {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly f32: Float32Array<ArrayBuffer>;
  readonly u32: Uint32Array<ArrayBuffer>;
  readonly i32: Int32Array<ArrayBuffer>;
}

export function byteViews(buffer: ArrayBuffer): ByteViews {
  return {
    bytes: new Uint8Array(buffer),
    f32: new Float32Array(buffer),
    u32: new Uint32Array(buffer),
    i32: new Int32Array(buffer),
  };
}

export type Scalar = 'f32' | 'u32' | 'i32';

/** Returns the byte offset of each struct field. */
export function offsetsOf(struct: d.WgslStruct): Record<string, number> {
  const offsets: Record<string, number> = {};
  for (const key of Object.keys(struct.propTypes)) {
    offsets[key] = d.memoryLayoutOf(struct, (s) => s[key]).offset;
  }

  return offsets;
}

/** Returns the scalar type of a schema, or undefined for structs, arrays, matrices and `f16`. */
export function scalarOf(schema: d.BaseData): Scalar | undefined {
  const { type } = schema;
  if (type === 'f32' || type === 'u32' || type === 'i32') return type;
  if (!d.isWgslData(schema)) return undefined;
  if (type.startsWith('vec') && 'primitive' in schema) return scalarOf(schema.primitive);
  if (d.isAtomic(schema) || d.isDecorated(schema)) return scalarOf(schema.inner);
  return undefined;
}

/** Returns the component count of a scalar or vector schema. */
export function componentsOf(schema: d.BaseData): number {
  if (!d.isWgslData(schema)) return 1;
  if ('componentCount' in schema) return schema.componentCount;
  if (d.isAtomic(schema) || d.isDecorated(schema)) return componentsOf(schema.inner);
  return 1;
}

export function writeScalar(views: ByteViews, scalar: Scalar, offset: number, value: number) {
  if (scalar === 'f32') views.f32[offset >>> 2] = value;
  else if (scalar === 'u32') views.u32[offset >>> 2] = value;
  else views.i32[offset >>> 2] = value;
}

export function readScalar(views: ByteViews, scalar: Scalar, offset: number): number {
  if (scalar === 'f32') return views.f32[offset >>> 2];
  if (scalar === 'u32') return views.u32[offset >>> 2];
  return views.i32[offset >>> 2];
}

/**
 * Writes `value` in GPU memory layout at the byte `offset`. Leaves struct fields missing from
 * `value` unchanged.
 */
export type Writer = (views: ByteViews, offset: number, value: unknown) => void;

/**
 * Creates a writer for `schema` and calculates its offsets once. Scalar and vector writes allocate
 * nothing. Throws with the field name for types the CPU writer does not support, such as `bool`
 * and `f16`.
 */
export function writerOf(schema: d.BaseData, name: string): Writer {
  const scalar = scalarOf(schema);

  if (scalar) {
    const count = componentsOf(schema);
    if (count === 1) {
      return (views, offset, value) => writeScalar(views, scalar, offset, value as number);
    }

    return (views, offset, value) => {
      const vector = value as ArrayLike<number>;
      for (let i = 0; i < count; i++) writeScalar(views, scalar, offset + i * 4, vector[i]);
    };
  }

  if (d.isWgslStruct(schema)) {
    const offsets = offsetsOf(schema);
    const keys = Object.keys(schema.propTypes);
    const starts = keys.map((key) => offsets[key]);
    const writers = keys.map((key) => writerOf(schema.propTypes[key], `${name}.${key}`));

    return (views, offset, value) => {
      const fields = value as Record<string, unknown>;
      for (let i = 0; i < keys.length; i++) {
        const field = fields[keys[i]];
        if (field !== undefined) writers[i](views, offset + starts[i], field);
      }
    };
  }

  if (d.isWgslArray(schema) && d.isData(schema.elementType)) {
    const element = schema.elementType;
    // Elements sit at their alignment, so a vec3f takes 16 bytes
    const align = d.alignmentOf(element);
    const stride = Math.ceil(d.sizeOf(element) / align) * align;
    const length = schema.elementCount;
    const write = writerOf(element, `${name}[]`);

    return (views, offset, value) => {
      const items = value as ArrayLike<unknown>;
      const count = Math.min(items.length, length);
      for (let i = 0; i < count; i++) write(views, offset + i * stride, items[i]);
    };
  }

  if (d.isDecorated(schema)) return writerOf(schema.inner, name);

  if (d.isWgslData(schema) && schema.type.startsWith('mat')) {
    // Plain arrays omit the padding between mat3x3f columns
    const size = d.sizeOf(schema);
    const range = { startOffset: 0, endOffset: 0 };

    return (views, offset, value) => {
      range.startOffset = offset;
      range.endOffset = offset + size;
      writeToArrayBuffer(views.bytes.buffer, schema, value as never, range);
    };
  }

  throw new TypeError(
    `Field '${name}' has type ${schema.type}, which tables can't write from the CPU.`,
  );
}
