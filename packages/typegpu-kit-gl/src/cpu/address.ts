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
