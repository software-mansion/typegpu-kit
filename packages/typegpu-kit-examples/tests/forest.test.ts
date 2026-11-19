import { expect, it } from 'vitest';
import { createScene } from '../src/forest/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/**
 * The shadow of the trunks and both crowns, indirect when culled; the scene, with the ground first
 * and the sky last; then the bloom's eleven passes and the final pass.
 */
const drawn = (draw: string) => [
  ...pass(...Array<string>(3).fill(draw)),
  ...pass('draw-indexed', ...Array<string>(3).fill(draw), 'draw'),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];
const culledDraws = drawn('draw-indexed-indirect');

/** Two kernels culling for the camera, and two for the sun. */
const culled = compute(4);

describeExample(
  'forest',
  createScene,
  (build) => {
    it('culls for the camera and the sun, then draws the shadow and the scene', async () => {
      const { log, next } = await build();
      const loaded = log.events.length;
      next();
      next();
      expect(log.events.slice(loaded)).toEqual([
        ...culled,
        ...culledDraws,
        ...culled,
        ...culledDraws,
      ]);
    });

    it('draws every tree without culling, and stops the gusts without wind', async () => {
      const { log, scene, next } = await build();
      next();

      scene.settings.values.cull = false;
      scene.settings.values.wind = 0;
      const writes = log.writes.length;
      const loaded = log.events.length;
      next();
      expect(log.events.slice(loaded)).toEqual(drawn('draw-indexed'));
      expect(log.writes.length).toBe(writes);
    });
  },
  {
    options: { trees: 40 },
    features: ['indirect-first-instance'],
    gap: 0.05,
    // A bake for each shape while loading, then a culling renderer's colour and shadow pipelines
    // and four kernels, a plain one's pair, the ground, the sky, three bloom passes and the final.
    pipelines: 4 + 6 + 2 + 1 + 1 + 3 + 1,
  },
);
