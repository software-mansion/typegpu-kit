import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { createScene, LANTERNS } from '../src/cloister/scene.ts';
import { BLOOM, describeExample, pass, recorded } from './utils/examples.ts';

/** The cloister's stone, plaster, paving, panels, lawn, planting and props: 17 meshes. */
const masonry = Array<string>(17).fill('draw-indexed');
/** Eight lanterns, a cube map of six faces each. */
const SHADOW_PASSES = 8 * 6;
const frame = (shadows: boolean) => [
  ...Array.from({ length: shadows ? SHADOW_PASSES : 0 }, () => pass(...masonry)).flat(),
  // The masonry, the lanterns' iron and glass, then the sky.
  ...pass(...masonry, 'draw-indexed', 'draw-indexed', 'draw'),
  // TAA, then keeping the camera for the next frame's motion, the bloom and the tone curve.
  ...pass('draw'),
  'copy-buffer',
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'cloister',
  createScene,
  (build) => {
    it('draws the lanterns’ shadows once, and again when they come back on', async () => {
      const { log, scene, run } = await build();
      for (const [shadows, drawn] of [
        [true, true],
        [true, false],
        [false, false],
        [true, true],
      ]) {
        scene.settings.values.shadows = shadows;
        await scene.settings.write!('shadows');
        expect(await recorded(log, () => run(1))).toEqual(frame(drawn));
      }
    });

    it('shows the occlusion alone or leaves it out by a write, creating nothing', async () => {
      const { log, scene, run } = await build();
      await run(3);
      const counts = log.counts();
      for (const shown of ['alone', 'off', 'baked'] as const) {
        const writes = log.writes.length;
        scene.settings.values.occlusion = shown;
        await scene.settings.write!('occlusion');
        expect(log.writes.length - writes).toBe(1);
      }
      await run(1);
      expect(log.counts()).toEqual(counts);
    });
  },
  { pipelines: 15 },
);

it('hangs a light in each of the model’s lanterns', () => {
  const glb = readFileSync(new URL('../src/cloister/models/cloister.glb', import.meta.url));
  const json = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)).toString()) as {
    nodes: { name?: string; translation?: number[] }[];
  };
  const empties = json.nodes
    .filter((node) => /^Lantern\.\d+$/.test(node.name ?? ''))
    .toSorted((a, b) => a.name!.localeCompare(b.name!))
    .map((node) => node.translation!.map((x) => Math.round(x * 100) / 100));
  expect(empties).toEqual(LANTERNS);
});
