import { tgpu } from 'typegpu';

/** Wraps a `'use gpu'` function so that WGSL names it `label`. */
export const named = <F extends (...args: never[]) => unknown>(label: string, fn: F) =>
  tgpu.fn(fn).$name(label);
