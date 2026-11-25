import { completeCall } from './record-fields.ts';

const NUMERIC_LITERAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

function unparenthesized(value: string): string {
  let result = value.trim();
  while (result.startsWith('(') && completeCall(`value${result}`)?.arguments.length === 1)
    result = result.slice(1, -1).trim();
  return result;
}

export function f32Literal(value: string): string {
  const literal = unparenthesized(value);
  if (!NUMERIC_LITERAL.test(literal)) return value;
  const rounded = Math.fround(Number(literal));
  return Object.is(rounded, -0) ? '-0' : String(rounded);
}

/** Round constants while generating code and runtime results at each f32 operation. */
export function f32(value: string): string {
  value = unparenthesized(value);
  const literal = f32Literal(value);
  if (literal !== value || NUMERIC_LITERAL.test(value)) return literal;
  if (value.startsWith('Math.fround(') && completeCall(value.replace('Math.fround', 'fround')))
    return value;
  return `Math.fround(${value})`;
}

export function f32op(left: string, op: string, right: string): string {
  // Non-literal operands are already typed f32 values; only abstract literals need conversion.
  left = f32Literal(left);
  right = f32Literal(right);
  if (NUMERIC_LITERAL.test(left) && NUMERIC_LITERAL.test(right)) {
    const a = Number(left),
      b = Number(right);
    const result =
      op === '+' ? a + b : op === '-' ? a - b : op === '*' ? a * b : op === '/' ? a / b : a % b;
    return f32Literal(Object.is(result, -0) ? '-0' : String(result));
  }
  return f32(`(${left})${op}(${right})`);
}

export function f32sum(values: string[]): string {
  return values.reduce((sum, value) => f32op(sum, '+', value));
}

export function f32dot(left: string[], right: string[]): string {
  return f32sum(left.map((value, i) => f32op(value, '*', right[i]!)));
}
