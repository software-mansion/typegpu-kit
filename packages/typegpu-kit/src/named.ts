import { tgpu } from 'typegpu';

/** Wraps a `'use gpu'` function so that WGSL names it `label`. */
// oxlint-disable-next-line typescript/no-explicit-any -- a function of any signature
export const named = <F extends (...args: any[]) => any>(label: string, fn: F) =>
  tgpu.fn(fn).$name(label);
