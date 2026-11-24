import { d } from 'typegpu';
import type { CpuBuffer } from './types.ts';

export interface Layout {
  type: string;
  size: number;
  count?: number;
  stride?: number;
  element?: Layout;
  fields?: Record<string, { offset: number; layout: Layout }>;
}

export function layoutOf(schema: d.BaseData): Layout {
  if (d.isDecorated(schema)) return { ...layoutOf(schema.inner), size: d.sizeOf(schema) };
  const type = schema.type;
  if (['f32', 'i32', 'u32'].includes(type)) return { type, size: 4 };
  if (d.isWgslStruct(schema)) {
    const fields: NonNullable<Layout['fields']> = {};
    let offset = 0;
    for (const [key, member] of Object.entries(schema.propTypes)) {
      const alignment = d.alignmentOf(member as d.AnyData);
      offset = Math.ceil(offset / alignment) * alignment;
      fields[key] = { offset, layout: layoutOf(member) };
      offset += d.sizeOf(member as d.AnyData);
    }
    return { type, fields, size: d.sizeOf(schema) };
  }
  if (d.isWgslArray(schema)) {
    const alignment = d.alignmentOf(schema.elementType as d.AnyData);
    const stride = Math.ceil(d.sizeOf(schema.elementType as d.AnyData) / alignment) * alignment;
    return {
      type,
      size: schema.elementCount * stride,
      count: schema.elementCount,
      stride,
      element: layoutOf(schema.elementType),
    };
  }
  const vector = /^vec([234])([fui])$/.exec(type);
  if (vector)
    return {
      type,
      size: Number(vector[1]) * 4,
      count: Number(vector[1]),
      stride: 4,
      element: { type: vector[2] === 'f' ? 'f32' : vector[2] === 'i' ? 'i32' : 'u32', size: 4 },
    };
  const matrix = /^mat([234])x([234])f$/.exec(type);
  if (matrix) {
    const rows = Number(matrix[2]);
    const stride = rows === 2 ? 8 : 16;
    return {
      type,
      size: Number(matrix[1]) * stride,
      count: Number(matrix[1]),
      stride,
      element: layoutOf(d[`vec${rows}f` as 'vec2f']),
    };
  }
  throw new Error(`@typegpu/kit-gl CPU: unsupported buffer element ${type}`);
}

export function bufferValue(buffer: CpuBuffer, layout: Layout): unknown {
  const words = {
    u32: new Uint32Array(buffer.bytes, buffer.offset, buffer.size / 4),
    i32: new Int32Array(buffer.bytes, buffer.offset, buffer.size / 4),
    f32: new Float32Array(buffer.bytes, buffer.offset, buffer.size / 4),
  };
  function read(type: Layout, offset: number): unknown {
    if (type.type in words) return words[type.type as keyof typeof words][offset / 4] ?? 0;
    const count = type.count || Math.floor((buffer.size - offset) / (type.stride ?? 1));
    const cache = new Map<string, unknown>();
    return new Proxy(
      {},
      {
        get(_target, key) {
          if (key === 'length') return count;
          if (typeof key !== 'string') return undefined;
          let member = type.fields?.[key];
          if (!member && type.element) {
            const index =
              'xyzw'.includes(key) && key.length === 1 ? 'xyzw'.indexOf(key) : Number(key);
            if (!Number.isInteger(index) || index < 0 || index >= count) return undefined;
            member = { offset: index * type.stride!, layout: type.element };
          }
          if (!member) return undefined;
          if (member.layout.type in words) return read(member.layout, offset + member.offset);
          if (!cache.has(key)) cache.set(key, read(member.layout, offset + member.offset));
          return cache.get(key);
        },
        set(_target, key, value) {
          if (typeof key !== 'string') return false;
          let member = type.fields?.[key];
          if (!member && type.element) {
            const index =
              'xyzw'.includes(key) && key.length === 1 ? 'xyzw'.indexOf(key) : Number(key);
            if (!Number.isInteger(index) || index < 0 || index >= count) return true;
            member = { offset: index * type.stride!, layout: type.element };
          }
          if (member) write(member.layout, offset + member.offset, value);
          return true;
        },
      },
    );
  }
  function write(type: Layout, offset: number, value: unknown): void {
    if (type.type in words) {
      words[type.type as keyof typeof words][offset / 4] = value as number;
    } else if (type.fields) {
      for (const [key, member] of Object.entries(type.fields))
        write(member.layout, offset + member.offset, (value as Record<string, unknown>)[key]);
    } else if (type.element) {
      for (let i = 0; i < type.count!; i++)
        write(type.element, offset + i * type.stride!, (value as Record<number, unknown>)[i]);
    }
  }
  return read(layout, 0);
}
