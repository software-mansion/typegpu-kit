export function addressParts(expression: string): { object: string; key: string } | undefined {
  if (expression.endsWith(']')) {
    let depth = 0;
    for (let i = expression.length - 1; i >= 0; i--) {
      if (expression[i] === ']') depth++;
      if (expression[i] === '[' && --depth === 0)
        return { object: expression.slice(0, i), key: expression.slice(i + 1, -1) };
    }
  }
  const field = /^(.*)\.([A-Za-z_]\w*)$/.exec(expression);
  return field ? { object: field[1]!, key: JSON.stringify(field[2]) } : undefined;
}

export function reference(expression: string): string {
  if (expression.endsWith(']')) {
    let depth = 0;
    for (let i = expression.length - 1; i >= 0; i--) {
      if (expression[i] === ']') depth++;
      if (expression[i] === '[' && --depth === 0)
        return `$ref(${expression.slice(0, i)},${expression.slice(i + 1, -1)})`;
    }
  }
  const member = /^(.*)\.([A-Za-z_]\w*)$/.exec(expression);
  if (member) return `$ref(${member[1]},${JSON.stringify(member[2])})`;
  return `({get $(){return ${expression}},set $(v){${expression}=v}})`;
}

export function lowerReferences(source: string): string {
  let code = source.replace(/\(\*([A-Za-z_]\w*)\)/g, '$1.$');
  let start = code.indexOf('(&');
  while (start !== -1) {
    let depth = 1;
    let end = start + 2;
    for (; end < code.length && depth; end++) {
      if (code[end] === '(') depth++;
      if (code[end] === ')') depth--;
    }
    if (depth) throw new Error('@typegpu/kit-gl CPU: unbalanced reference expression');
    code = code.slice(0, start) + reference(code.slice(start + 2, end - 1)) + code.slice(end);
    start = code.indexOf('(&');
  }
  return code;
}

export function ref(object: Record<string | number, unknown>, key: string | number): object {
  return {
    get $() {
      return object[key];
    },
    set $(value: unknown) {
      object[key] = value;
    },
  };
}
