import { describe, it, vi } from 'vitest';

// The setup file already watches through this module, so take a fresh copy of it and keep the
// setup's watcher from seeing these warnings.
vi.resetModules();
const { failOnWarnings } = await import('../src/testing/failOnWarnings.ts');

describe('failOnWarnings', () => {
  const outer = console.warn;
  console.warn = () => {};
  failOnWarnings({ allow: [/repeated/g] });
  const warn = console.warn;
  console.warn = outer;

  it('allows every repeat of a warning a global pattern matches', () => {
    warn('⚠️ [typegpu] repeated');
    warn('⚠️ [typegpu] repeated');
  });
});
