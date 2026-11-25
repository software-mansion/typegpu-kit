import type { Layout } from './memory.ts';

type Value = number | boolean | Value[] | { [key: string]: Value };
export function vector(values: number[]): number[] {
  return values;
}

export function construct(layout: Layout, ...args: Value[]): Value {
  if (layout.type === 'bool') return Boolean(args[0]);
  if (layout.type === 'u32') return Number(args[0] ?? 0) >>> 0;
  if (layout.type === 'i32') return Number(args[0] ?? 0) | 0;
  if (layout.type === 'f32') return Math.fround(Number(args[0] ?? 0));
  if (layout.fields) {
    const value: Record<string, Value> = {};
    Object.entries(layout.fields).forEach(([key, field], index) => {
      const input =
        args.length === 1 && typeof args[0] === 'object'
          ? (args[0] as Record<string, Value>)[key]
          : args[index];
      value[key] = construct(field.layout, ...(input === undefined ? [] : [input]));
    });
    return value;
  }
  if (layout.type.startsWith('vec')) {
    const components: number[] = [];
    for (const arg of args) {
      if (typeof arg === 'number') components.push(arg);
      else
        for (let i = 0; i < Number((arg as Record<string, Value>).length); i++)
          components.push((arg as number[])[i]!);
    }
    return vector(
      Array.from({ length: layout.count! }, (_, i) =>
        Number(
          construct(
            layout.element!,
            components.length === 1 ? components[0]! : (components[i] ?? 0),
          ),
        ),
      ),
    );
  }
  if (layout.type.startsWith('mat') && typeof args[0] === 'number') {
    const rows = layout.element!.count!;
    return Array.from({ length: layout.count! }, (_, column) =>
      vector(Array.from({ length: rows }, (_row, row) => Number(args[column * rows + row] ?? 0))),
    );
  }
  const result: Value[] = [];
  for (let i = 0; i < layout.count!; i++) {
    const input = args.length === 1 ? ((args[0] ?? []) as Value[])[i] : args[i];
    result.push(construct(layout.element!, ...(input === undefined ? [] : [input])));
  }
  return result;
}
