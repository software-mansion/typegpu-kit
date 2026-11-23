import { expect, it } from 'vitest';
import { createScene, SETTLE_UPDATES } from '../src/falling-sand/scene.ts';
import { compute, describeExample, pass, recorded } from './utils/examples.ts';

/** Steps a frame runs at most, which a frame 0.04 seconds after the last owes. */
const STEPS = 4;
/** A tick, a step of every cell, six passes of liquids along the rows, and the embers' step. */
const PER_STEP = 1 + 1 + 6 + 1;
/** The embers' liveLists, then the cells' relief and what falls and drifts through them. */
const PER_FRAME = 2 + 1 + 1;
/** A stroke painted into the cells. */
const PAINT = 1;
/** The medium's fill and the holographic runner's dispatches over 320 × 180 cells. */
const FIELD = 1 + 73;
/** Five bloom levels down and four up. */
const BLOOM = 5 + 4;

const frame = ({ painted = false, steps = STEPS, field = true } = {}) => [
  ...(painted ? compute(PAINT) : []),
  ...compute(steps * PER_STEP + PER_FRAME),
  // The embers' light into the field's emission, then the field.
  ...(field ? [...pass('draw-indirect'), ...compute(FIELD)] : []),
  // The cave and its embers, the bloom, then the tone curve into the surface.
  ...pass('draw', 'draw-indirect'),
  ...Array.from({ length: BLOOM }, () => pass('draw')).flat(),
  ...pass('draw'),
  'submit',
];

describeExample(
  'falling-sand',
  createScene,
  (build) => {
    it('steps the cells, gathers the light and draws the cave with one submit', async () => {
      const { log, run } = await build();
      await run(3);
      expect(await recorded(log, () => run(1))).toEqual(frame());
    });

    it('paints a stroke in its own pass before the steps', async () => {
      const { scene, log, run, next } = await build();
      await run(3);
      scene.pointing.drag(0.45, 0.5, true);
      scene.pointing.drag(0.6, 0.5, false);
      expect(recorded(log, next)).toEqual(frame({ painted: true }));
    });

    it('holds the cells and the field while paused, and settles the field after an edit', async () => {
      const { scene, log, run, next } = await build();
      await run(3);
      scene.settings.values.paused = true;
      await scene.settings.write!('paused');
      await run(SETTLE_UPDATES);
      expect(recorded(log, next)).toEqual(frame({ steps: 0, field: false }));
      scene.pointing.drag(0.5, 0.5, true);
      expect(recorded(log, next)).toEqual(frame({ painted: true, steps: 0 }));
      await run(SETTLE_UPDATES - 1);
      expect(recorded(log, next)).toEqual(frame({ steps: 0, field: false }));
    });
  },
  { pipelines: 69 },
);
