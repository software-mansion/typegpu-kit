import { afterEach, expect } from 'vitest';

export interface WarningOptions {
  /**
   * Allows matching warnings without failing tests. Each expression tests the whole message.
   * Default none.
   */
  readonly allow?: readonly RegExp[];
}

let watching = false;

/**
 * Fails tests that emit TypeGPU warnings, such as deprecated-operator warnings. It patches
 * `console.warn` until disposed. TypeGPU warnings start with `⚠️ [type]`. Call this from a
 * Vitest setup file, or import `@typegpu/kit/testing/warnings` to use the defaults. Throws if
 * another call is still active.
 *
 * @returns the patch, which disposing undoes
 */
export function failOnWarnings(options: WarningOptions = {}): Disposable {
  if (watching) {
    throw new Error('Dispose of the active failOnWarnings instance before creating another.');
  }
  watching = true;
  const { allow = [] } = options;
  const warnings: string[] = [];
  const warn = console.warn;
  let done = false;

  console.warn = (...args: unknown[]) => {
    const message = args.join(' ');
    const typegpu = typeof args[0] === 'string' && args[0].startsWith('⚠️ [');
    // `search` starts at 0 and keeps `lastIndex`, so a global or sticky pattern answers the same
    // for every repeat.
    if (typegpu && !allow.some((pattern) => message.search(pattern) !== -1)) {
      warnings.push(message);
    }
    warn(...args);
  };

  afterEach(() => {
    if (done) return;
    const seen = warnings.splice(0);
    expect(seen, 'TypeGPU warned').toEqual([]);
  });

  return {
    [Symbol.dispose]() {
      if (done) return;
      done = true;
      watching = false;
      console.warn = warn;
    },
  };
}
