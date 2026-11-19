import { expect, it } from 'vitest';
import { createScene } from '../src/survey-model/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

const casting = pass(...Array.from({ length: 4 }, () => 'draw-indexed'));

/**
 * The two shadow cascades, each drawing the plinth's slab, the legend's strips and arrowhead, and
 * the rings, the scene, which draws the graticule too, the resolve and its jitter, the bloom's
 * eleven passes, the final pass.
 */
const drawing = [
  ...casting,
  ...casting,
  ...pass(...Array.from({ length: 5 }, () => 'draw-indexed')),
  ...pass('draw'),
  'copy-buffer',
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'survey-model',
  createScene,
  (build) => {
    it('traces the rings while they rise, then draws the shadow and the scene with one draw for every ring', async () => {
      const { log, next } = await build();
      const loaded = log.events.length;

      // Loading traced the rings at rest, so the first frame draws them as they are
      next();
      expect(log.events.slice(loaded)).toEqual(drawing);

      const rising = log.events.length;
      next(1);
      expect(log.events.slice(rising)).toEqual([...compute(2), ...drawing]);

      // Risen after three seconds, the rings stay where they are and trace no more
      next(3);
      const risen = log.events.length;
      next();
      expect(log.events.slice(risen)).toEqual(drawing);
    });
  },
  {
    // A bake for each of the plinth's three meshes, three scene and two shadow pipelines, the trace
    // and the measure, the resolve, the bloom's three and the final pass
    pipelines: 3 + 3 + 2 + 2 + 1 + 3 + 1,
  },
);
