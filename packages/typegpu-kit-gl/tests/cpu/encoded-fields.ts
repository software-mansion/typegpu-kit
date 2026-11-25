export type Encoding =
  | 'ordered-f32'
  | 'octahedral-2x16snorm'
  | '2x16snorm'
  | '2x16unorm'
  | '2x16float'
  | '4x8snorm'
  | '4x8unorm';
export interface EncodedField {
  component: number;
  components: number;
  encoding: Encoding;
}

function argsOf(expression: string): { name: string; args: string[] } | undefined {
  const match = /^([\w]+)(?:<[^>]+>)?\s*\(/.exec(expression.trim());
  if (!match) return undefined;
  const text = expression.trim();
  let depth = 0,
    begin = match[0].length;
  const args: string[] = [];
  for (let i = begin; i < text.length; i++) {
    const character = text[i];
    if (character === '(' || character === '[') depth++;
    if (character === ']' || character === ')') {
      if (depth === 0) {
        args.push(text.slice(begin, i).trim());
        return { name: match[1]!, args };
      }
      depth--;
    }
    if (character === ',' && depth === 0) {
      args.push(text.slice(begin, i).trim());
      begin = i + 1;
    }
  }
  return undefined;
}

function functionBodies(code: string): { from: number; to: number; name: string; body: string }[] {
  const functions: ReturnType<typeof functionBodies> = [];
  for (const match of code.matchAll(/\bfn\s+(\w+)\s*\([^{}]*\)[^{]*\{/g)) {
    const from = match.index + match[0].length;
    let depth = 1,
      to = from;
    for (; to < code.length && depth > 0; to++) {
      if (code[to] === '{') depth++;
      if (code[to] === '}') depth--;
    }
    functions.push({ name: match[1]!, from, to, body: code.slice(from, to - 1) });
  }
  return functions;
}

/** Recovers only explicit packing operations; ordinary u32 values remain exact. */
export function encodedFields(code: string, binding: string): EncodedField[] {
  if (!/^\w+$/.test(binding)) return [];
  const functions = functionBodies(code);
  const ordered = new Set(
    functions
      .filter(
        (fn) =>
          /bitcast<u32>/.test(fn.body) &&
          /(?:2147483648|0x80000000)/i.test(fn.body) &&
          /(?:4294967295|0xffffffff)/i.test(fn.body),
      )
      .map((fn) => fn.name),
  );
  const octahedral = new Set(
    functions
      .filter(
        (fn) =>
          /\.xy/.test(fn.body) &&
          /\.yx/.test(fn.body) &&
          /select\(/.test(fn.body) &&
          /\.z\s*<\s*0/.test(fn.body) &&
          /abs\(/.test(fn.body),
      )
      .map((fn) => fn.name),
  );
  const result: EncodedField[] = [];
  for (const fn of functions) {
    const aliases = new Map<string, string>();
    for (const statement of fn.body.matchAll(
      /\b(?:let|var|const)\s+(\w+)(?:\s*:[^=;]+)?\s*=\s*([^;]+);/g,
    )) {
      const name = statement[1]!;
      if (aliases.has(name)) aliases.set(name, '');
      else aliases.set(name, statement[2]!.trim());
    }
    const encodingOf = (source: string): Encoding | undefined => {
      let expression = source.trim();
      const seen = new Set<string>();
      while (aliases.has(expression) && !seen.has(expression)) {
        seen.add(expression);
        expression = aliases.get(expression)!;
      }
      const call = argsOf(expression);
      if (!call) return undefined;
      const pack = /^pack(2x16(?:snorm|unorm|float)|4x8(?:snorm|unorm))$/.exec(call.name);
      const packedArgument = call.args[0] && argsOf(call.args[0]);
      if (pack?.[1] === '2x16snorm' && packedArgument && octahedral.has(packedArgument.name))
        return 'octahedral-2x16snorm';
      return pack ? (pack[1] as Encoding) : ordered.has(call.name) ? 'ordered-f32' : undefined;
    };
    for (const store of fn.body.matchAll(
      new RegExp(`\\b${binding}\\s*\\[[^;\\n]+?\\]\\s*=\\s*([^;]+);`, 'g'),
    )) {
      const expression = store[1]!.trim();
      const vector = argsOf(expression);
      if (vector && /^vec[234]u?$/.test(vector.name)) {
        const components = Number(vector.name[3]);
        if (vector.args.length !== components) continue;
        for (let component = 0; component < components; component++) {
          const encoding = encodingOf(vector.args[component]!);
          if (encoding) result.push({ component, components, encoding });
        }
      } else {
        const encoding = encodingOf(expression);
        if (encoding) result.push({ component: 0, components: 1, encoding });
      }
    }
    for (const call of fn.body.matchAll(/\batomic(?:Min|Max)\s*\(/g)) {
      const parsed = argsOf(fn.body.slice(call.index));
      if (!parsed || !new RegExp(`^&?\\s*${binding}\\s*\\[`).test(parsed.args[0] ?? '')) continue;
      const encoding = encodingOf(parsed.args[1] ?? '');
      if (encoding) result.push({ component: 0, components: 1, encoding });
    }
  }
  return [...new Map(result.map((field) => [JSON.stringify(field), field])).values()];
}

export function packedEqual(
  a: number,
  b: number,
  encoding: Exclude<Encoding, 'ordered-f32'>,
): boolean {
  if (encoding === 'octahedral-2x16snorm') {
    if (packedEqual(a, b, '2x16snorm')) return true;
    const left = unfold(a),
      right = unfold(b);
    const magnitude = Math.max(...left.map(Math.abs), ...right.map(Math.abs));
    return left.every(
      (value, i) => Math.abs(value - right[i]!) <= Math.max(1e-6, magnitude * 1e-5),
    );
  }
  const width = encoding.startsWith('2x16') ? 16 : 8;
  const mask = (1 << width) - 1;
  for (let at = 0; at < 32; at += width) {
    let left = (a >>> at) & mask,
      right = (b >>> at) & mask;
    if (encoding.endsWith('snorm')) {
      left = (left << (32 - width)) >> (32 - width);
      right = (right << (32 - width)) >> (32 - width);
    }
    if (encoding.endsWith('float') && (left & 0x7fff) === 0 && (right & 0x7fff) === 0) continue;
    if (Math.abs(left - right) > 1) return false;
  }
  return true;
}

function unfold(bits: number): number[] {
  let x = Math.max(-1, ((bits << 16) >> 16) / 32767),
    y = Math.max(-1, (bits >> 16) / 32767);
  const z = 1 - Math.abs(x) - Math.abs(y),
    fold = Math.min(1, Math.max(0, -z));
  x -= (x >= 0 ? 1 : -1) * fold;
  y -= (y >= 0 ? 1 : -1) * fold;
  const magnitude = Math.hypot(x, y, z);
  return [x / magnitude, y / magnitude, z / magnitude];
}
