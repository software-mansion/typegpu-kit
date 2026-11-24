import { d } from 'typegpu';

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
  if (d.isAtomic(schema)) return layoutOf(schema.inner);
  if (['f32', 'i32', 'u32', 'bool'].includes(type)) return { type, size: 4 };
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
  const vector = /^vec([234])([fui]|<bool>)$/.exec(type);
  if (vector)
    return {
      type,
      size: Number(vector[1]) * 4,
      count: Number(vector[1]),
      stride: 4,
      element: {
        type:
          vector[2] === 'f'
            ? 'f32'
            : vector[2] === 'i'
              ? 'i32'
              : vector[2] === '<bool>'
                ? 'bool'
                : 'u32',
        size: 4,
      },
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
