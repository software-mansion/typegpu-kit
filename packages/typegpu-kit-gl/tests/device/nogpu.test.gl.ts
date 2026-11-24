import { afterAll, beforeAll, expect, test } from 'vitest';

/**
 * The fallback where it is needed: a page without WebGPU. Removes `navigator.gpu` and every
 * `GPU*` global before anything that reads them is imported, then runs an example on WebGL 2.
 */

const removed = new Map<string, PropertyDescriptor>();
let gpu: PropertyDescriptor | undefined;

beforeAll(() => {
  gpu = Object.getOwnPropertyDescriptor(Navigator.prototype, 'gpu');
  delete (Navigator.prototype as { gpu?: unknown }).gpu;
  for (const name of Object.getOwnPropertyNames(globalThis)) {
    if (!name.startsWith('GPU')) continue;
    removed.set(name, Object.getOwnPropertyDescriptor(globalThis, name)!);
    delete (globalThis as Record<string, unknown>)[name];
  }
});

afterAll(() => {
  for (const [name, descriptor] of removed) Object.defineProperty(globalThis, name, descriptor);
  if (gpu) Object.defineProperty(Navigator.prototype, 'gpu', gpu);
});

test('runs an example in a page without WebGPU', { timeout: 120_000 }, async () => {
  expect('gpu' in navigator).toBe(false);
  expect('GPUBufferUsage' in globalThis).toBe(false);
  const { webgl } = await import('../../src/index.ts');
  const { runScene } = await import('./scenes.ts');
  const root = await webgl.init();
  try {
    const run = await runScene(root, 'raymarch', { steady: 3 });
    expect(run.errors).toEqual([]);
    const bytes = run.frame.bytes;
    let lit = 0;
    for (let i = 0; i < bytes.length; i += 4)
      if (bytes[i]! + bytes[i + 1]! + bytes[i + 2]! > 30) lit++;
    expect(lit / (bytes.length / 4)).toBeGreaterThan(0.2);
  } finally {
    root.destroy();
  }
});
