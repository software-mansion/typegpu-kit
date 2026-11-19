// Reuse typed views and field offsets to avoid allocations during scalar and vector writes. Unlike
// TypeGPU's writer, this writer leaves omitted struct fields unchanged
import { d, writeToArrayBuffer } from 'typegpu';

export interface ByteViews {
  readonly bytes: Uint8Array;
  readonly f32: Float32Array;
  readonly u32: Uint32Array;
  readonly i32: Int32Array;
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

type Schema = { readonly type: string } & Record<string, unknown>;

/** Returns the byte offset of each struct field. */
export function offsetsOf(struct: d.WgslStruct): Record<string, number> {
  const offsets: Record<string, number> = {};
  for (const key of Object.keys(struct.propTypes)) {
    offsets[key] = d.memoryLayoutOf(struct, (s) => (s as Record<string, never>)[key]).offset;
  }

  return offsets;
}

/** Returns the scalar type of a schema, or undefined for structs, arrays, matrices and `f16`. */
export function scalarOf(schema: d.AnyWgslData): Scalar | undefined {
  const s = schema as unknown as Schema;
  if (s.type === 'f32' || s.type === 'u32' || s.type === 'i32') return s.type;
  if (s.type.startsWith('vec')) return scalarOf(s.primitive as d.AnyWgslData);
  if (s.type === 'atomic' || s.type === 'decorated') return scalarOf(s.inner as d.AnyWgslData);
  return undefined;
}

/** Returns the component count of a scalar or vector schema. */
export function componentsOf(schema: d.AnyWgslData): number {
  const s = schema as unknown as Schema;
  if (s.type.startsWith('vec')) return s.componentCount as number;
  if (s.type === 'atomic' || s.type === 'decorated') return componentsOf(s.inner as d.AnyWgslData);
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
export function writerOf(schema: d.AnyWgslData, name: string): Writer {
  const s = schema as unknown as Schema;
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

  switch (s.type) {
    case 'struct': {
      const struct = schema as d.WgslStruct;
      const offsets = offsetsOf(struct);
      const keys = Object.keys(struct.propTypes);
      const starts = keys.map((key) => offsets[key]);
      const writers = keys.map((key) =>
        writerOf(struct.propTypes[key] as d.AnyWgslData, `${name}.${key}`),
      );

      return (views, offset, value) => {
        const fields = value as Record<string, unknown>;
        for (let i = 0; i < keys.length; i++) {
          const field = fields[keys[i]];
          if (field !== undefined) writers[i](views, offset + starts[i], field);
        }
      };
    }

    case 'array': {
      const element = s.elementType as d.AnyWgslData;
      const stride = d.sizeOf(element);
      const length = s.elementCount as number;
      const write = writerOf(element, `${name}[]`);

      return (views, offset, value) => {
        const items = value as ArrayLike<unknown>;
        const count = Math.min(items.length, length);
        for (let i = 0; i < count; i++) write(views, offset + i * stride, items[i]);
      };
    }

    case 'decorated':
      return writerOf(s.inner as d.AnyWgslData, name);
  }

  if (s.type.startsWith('mat')) {
    // Plain arrays omit the padding between mat3x3f columns
    const size = d.sizeOf(schema);
    const range = { startOffset: 0, endOffset: 0 };

    return (views, offset, value) => {
      range.startOffset = offset;
      range.endOffset = offset + size;
      writeToArrayBuffer(views.bytes.buffer as ArrayBuffer, schema, value as never, range);
    };
  }

  throw new TypeError(`Field '${name}' has type ${s.type}, which tables can't write from the CPU.`);
}
