import { tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { compare, glRoot, hasWebGpu } from './compare.ts';
import { runScene } from './scenes.ts';

/**
 * Examples that ship no files and run quickly with CPU compute, as a smoke test of the device under
 * whole scenes. `scripts/examples.ts` runs every example in its page.
 */
const EXAMPLES = ['blockfall', 'lattice', 'raymarch', 'robot-arm'];

let gpu: TgpuRoot;
beforeAll(async () => {
  if (!hasWebGpu) return;
  gpu = await tgpu.init();
});
afterAll(() => gpu?.destroy());

describe.skipIf(!hasWebGpu)('whole examples on the WebGL device', () => {
  for (const key of EXAMPLES) {
    test(key, { timeout: 120_000 }, async () => {
      const expected = await runScene(gpu, key);
      const gl = glRoot();
      const actual = await runScene(gl, key);
      gl.destroy();
      const result = compare(expected.frame, actual.frame, 24 / 255);
      console.info(
        JSON.stringify({
          key,
          mismatched: Number(result.mismatched.toFixed(4)),
          gpuMs: Number(expected.frameTime.toFixed(1)),
          glMs: Number(actual.frameTime.toFixed(1)),
          errors: [...new Set(actual.errors)].map((e) => e.slice(0, 600)),
          gpuErrors: expected.errors.length,
        }),
      );
      expect(actual.errors).toEqual([]);
    });
  }
});
