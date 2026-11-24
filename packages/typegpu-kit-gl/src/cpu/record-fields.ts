import { callArguments } from './references.ts';

export function completeCall(
  expression: string,
): { name: string; arguments: string[] } | undefined {
  const call = /^([A-Za-z_$][\w$]*)\(/.exec(expression);
  if (!call || !expression.endsWith(')')) return undefined;
  let depth = 0;
  let quote = '';
  for (let i = call[1]!.length; i < expression.length; i++) {
    const char = expression[i]!;
    if (quote) {
      if (char === '\\') i++;
      else if (char === quote) quote = '';
    } else if (char === '"' || char === "'") quote = char;
    else if (char === '(') depth++;
    else if (char === ')' && --depth === 0 && i !== expression.length - 1) return undefined;
  }
  return depth === 0 ? { name: call[1]!, arguments: callArguments(expression) } : undefined;
}

export function usedRecordFields(body: string, parameter: string): string[] | undefined {
  const identifier = `(?<![\\w$])${parameter.replace(/[$]/g, '\\$')}(?![\\w$])`;
  const fields = new Set<string>();
  const rest = body.replace(
    new RegExp(`${identifier}\\.([A-Za-z_$][\\w$]*)`, 'g'),
    (_match, field: string) => {
      fields.add(field);
      return '';
    },
  );
  return new RegExp(identifier).test(rest) ? undefined : [...fields].toSorted();
}

/** Remove unused fields only from independent stores into a fresh, zero-initialized record. */
export function projectRecordBody(
  body: string,
  fields: readonly string[],
  unwrap: boolean,
  initial: (field: string) => string,
  forward: (expression: string) => string | undefined,
): string | undefined {
  const returned = /^\{([\s\S]*?)\breturn ([^\n]*);\s*}$/.exec(body);
  if (!returned || /\breturn\b/.test(returned[1]!)) return undefined;
  let prefix = returned[1]!;
  const expression = returned[2]!;
  if (expression.startsWith('({') && expression.endsWith('})')) {
    const entries = new Map<string, string>();
    for (const entry of callArguments(`record(${expression.slice(2, -2)})`)) {
      const property = /^"([A-Za-z_$][\w$]*)":([\s\S]*)$/.exec(entry);
      if (!property) return undefined;
      entries.set(property[1]!, property[2]!);
    }
    if (fields.some((field) => !entries.has(field))) return undefined;
    return `{${prefix}return ${unwrap ? entries.get(fields[0]!) : `({${fields.map((field) => `${JSON.stringify(field)}:${entries.get(field)}`).join(',')}})`};}`;
  }
  if (completeCall(expression)) {
    const value = forward(expression);
    return value ? `{${prefix}return ${value};}` : undefined;
  }
  if (!/^[A-Za-z_$][\w$]*$/.test(expression)) return undefined;
  const local = expression.replace(/[$]/g, '\\$');
  const identifier = new RegExp(`(?<![\\w$])${local}(?![\\w$])`);
  const initializer = new RegExp(`^let ${local} = cpuType(?:_\\d+)?\\(\\);$`);
  let initialized = false;
  const lines: string[] = [];
  for (const line of prefix.split('\n')) {
    if (initializer.test(line)) {
      if (initialized) return undefined;
      initialized = true;
      lines.push(
        `let ${expression} = ({${fields.map((field) => `${JSON.stringify(field)}:${initial(field)}`).join(',')}});`,
      );
      continue;
    }
    if (!identifier.test(line)) {
      lines.push(line);
      continue;
    }
    const scalar = new RegExp(`^\\s*\\(${local}\\.([A-Za-z_$][\\w$]*)\\) = ([^;]*);$`).exec(line);
    if (scalar && !identifier.test(scalar[2]!)) {
      if (fields.includes(scalar[1]!)) lines.push(line);
      continue;
    }
    const vector = /^\s*\(\((v\d(?:,v\d)*)\)=>\(([^;]*)\)\)\((.*)\);$/.exec(line);
    if (vector && !identifier.test(vector[3]!)) {
      const stores = vector[2]!
        .split(',')
        .map((store) =>
          new RegExp(`^${local}\\.([A-Za-z_$][\\w$]*)\\[[0-3]\\]=v[0-3]$`).exec(store),
        );
      const field = stores[0]?.[1];
      if (field && stores.every((store) => store?.[1] === field)) {
        if (fields.includes(field)) lines.push(line);
        continue;
      }
    }
    return undefined;
  }
  if (!initialized) return undefined;
  prefix = lines.join('\n');
  return `{${prefix}return ${expression}${unwrap ? `.${fields[0]}` : ''};}`;
}
