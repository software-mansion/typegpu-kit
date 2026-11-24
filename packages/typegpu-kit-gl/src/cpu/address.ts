import type { Layout } from './memory.ts';

export interface Address {
  layout: Layout;
  base: string;
  offset: string;
}

export function readAddress(address: Address): string {
  const { layout, base, offset } = address;
  if (['u32', 'i32', 'f32'].includes(layout.type)) return `${base}_${layout.type}[(${offset})/4]`;
  if (layout.fields)
    return `({${Object.entries(layout.fields)
      .map(
        ([key, field]) =>
          `${JSON.stringify(key)}:${readAddress({ layout: field.layout, base, offset: `(${offset})+${field.offset}` })}`,
      )
      .join(',')}})`;
  const elements = Array.from({ length: layout.count! }, (_, i) =>
    readAddress({ layout: layout.element!, base, offset: `(${offset})+${i * layout.stride!}` }),
  );
  return layout.type.startsWith('vec')
    ? `$vector([${elements.join(',')}])`
    : `[${elements.join(',')}]`;
}

export function writeAddress(address: Address, value: string): string {
  const { layout, base, offset } = address;
  if (['u32', 'i32', 'f32'].includes(layout.type))
    return `${base}_${layout.type}[(${offset})/4]=${value};`;
  if (layout.fields)
    return Object.entries(layout.fields)
      .map(([key, field]) =>
        writeAddress(
          { layout: field.layout, base, offset: `(${offset})+${field.offset}` },
          `${value}[${JSON.stringify(key)}]`,
        ),
      )
      .join('');
  return Array.from({ length: layout.count! }, (_, i) =>
    writeAddress(
      { layout: layout.element!, base, offset: `(${offset})+${i * layout.stride!}` },
      `${value}[${i}]`,
    ),
  ).join('');
}

export function referenceAddress(address: Address): string {
  const make = (at: Address): string => {
    const { layout, base, offset } = at;
    if (!layout.fields && !layout.element) return readAddress(at);
    if (layout.type === 'array' && !layout.count)
      throw new Error('@typegpu/kit-gl CPU: unsupported pointer to runtime-sized array');
    const fields = layout.fields
      ? Object.entries(layout.fields).map(([key, field]) => ({
          key,
          at: { layout: field.layout, base, offset: `(${offset})+${field.offset}` },
        }))
      : Array.from({ length: layout.count! }, (_, i) => ({
          key: String(i),
          at: { layout: layout.element!, base, offset: `(${offset})+${i * layout.stride!}` },
        }));
    const children = fields.filter((field) => field.at.layout.fields || field.at.layout.element);
    const parameters = children.map((_, i) => `child${i}`);
    const properties = fields.map((field) => {
      const index = children.indexOf(field);
      return `get ${JSON.stringify(field.key)}(){return ${index < 0 ? readAddress(field.at) : parameters[index]};},set ${JSON.stringify(field.key)}(value){${writeAddress(field.at, 'value')}}`;
    });
    if (!layout.fields) properties.push(`length:${layout.count}`);
    const value = `({${properties.join(',')}})`;
    return children.length
      ? `((${parameters.join(',')})=>${value})(${children.map((field) => make(field.at)).join(',')})`
      : value;
  };
  const local = { ...address, offset: 'offset' };
  return `((offset)=>((view)=>({get $(){return view},set $(value){${writeAddress(local, 'value')}}}))(${make(local)}))(${address.offset})`;
}
