/** Round constants while generating code and runtime results at each f32 operation. */
export function f32(value: string): string {
  if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())) {
    const rounded = Math.fround(Number(value));
    return Object.is(rounded, -0) ? '-0' : String(rounded);
  }
  if (value.startsWith('Math.fround(') && completeCall(value.replace('Math.fround', 'fround')))
    return value;
  return `Math.fround(${value})`;
}

export function f32op(left: string, op: string, right: string): string {
  return f32(`(${f32(left)})${op}(${f32(right)})`);
}

export function f32sum(values: string[]): string {
  return values.reduce((sum, value) => f32op(sum, '+', value));
}

export function f32dot(left: string[], right: string[]): string {
  return f32sum(left.map((value, i) => f32op(value, '*', right[i]!)));
}
import { completeCall } from './record-fields.ts';
