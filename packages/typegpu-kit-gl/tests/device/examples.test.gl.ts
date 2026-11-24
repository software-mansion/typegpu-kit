import { tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { compare } from './compare.ts';
import { glRoot } from './compare.ts';
import { runScene } from './scenes.ts';

/** Examples that ship no files, so they load without a page around them. */
const EXAMPLES = (import.meta.env.EXAMPLES as string | undefined)?.split(',') ?? [
  'benthos', 'blockfall', 'censer', 'dominoes', 'ember-mirror', 'falling-sand', 'fishtank',
  'flow-field', 'forest', 'jellies', 'jump-flood', 'lattice', 'ocean', 'platformer', 'raymarch',
  'robot-arm', 'slime-mould', 'sph-sculpt', 'spider', 'survey-model',
];

let gpu: TgpuRoot;
beforeAll(async () => {
  gpu = await tgpu.init();
});
afterAll(() => gpu?.destroy());

describe('whole examples on the WebGL device', () => {
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
