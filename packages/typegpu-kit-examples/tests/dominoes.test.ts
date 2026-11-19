import { expect, it } from 'vitest';
import { createScene } from '../src/dominoes/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/**
 * A frame's events after its compute pass: the dominoes' shadow, the floor's and the dominoes'
 * half-resolution depth, the occlusion's trace and denoise, the scene (the floor, the dominoes,
 * the dust), the bloom's eleven passes and the final pass.
 */
const shadow = pass('draw-indexed');
const drawings = pass('draw-indexed', 'draw-indexed', 'draw-indirect');
const post = [...BLOOM, ...pass('draw'), 'submit'];
const drawn = [
  ...shadow,
  ...pass('draw-indexed', 'draw-indexed'),
  ...pass('draw'),
  ...pass('draw'),
  ...drawings,
  ...post,
];
/** Without time passing, nothing steps; the dust is only compacted, in two dispatches. */
const compacted = compute(2);

/** Two dispatches of the fall and one of the dust a step, then the dust's live lists. */
const steps = (count: number) => count * 3 + 2;

/** The dispatches in each compute pass of `events`. */
const dispatches = (events: string[]) => {
  const passes: number[] = [];
  for (const event of events) {
    if (event === 'begin-compute') passes.push(0);
    if (event === 'dispatch') passes.push(passes.pop()! + 1);
  }
  return passes;
};

describeExample(
  'dominoes',
  createScene,
  (build) => {
    it('steps the fall and the dust, draws the shadow and the scene, blooms and finishes', async () => {
      const { log, scene, next } = await build();
      scene.settings.values.auto = false;

      const loaded = log.events.length;
      next(0);
      next(0);
      // The first frame draws the sunset and paints the portrait's backs first, and reads the
      // fallen counter back.
      expect(log.events.slice(loaded)).toEqual([
        ...compute(4),
        'copy-buffer',
        ...drawn,
        ...compacted,
        ...drawn,
      ]);
    });

    it('shades the floor and the dominoes through the occlusion in the shared group', async () => {
      const { log } = await build();
      const lit = log.shaders.filter((code) => code.includes('fn hemisphere'));
      // The floor and the dominoes
      expect(lit).toHaveLength(2);
      for (const code of lit) {
        expect(code.match(/var ao_occlusion: texture_2d<f32>/g)).toHaveLength(1);
        expect(code).not.toContain('var<storage, read_write>');
      }
    });

    it('turned off, clears the occlusion once and skips its depth', async () => {
      const { log, scene, next } = await build();
      scene.settings.values.auto = false;
      next(0);
      scene.settings.values.occlusion = false;
      await scene.settings.write!('occlusion');

      const from = log.events.length;
      next(0);
      next(0);
      expect(log.events.slice(from)).toEqual([
        // The occlusion clears once
        ...compacted,
        ...shadow,
        ...pass(),
        ...drawings,
        ...post,
        ...compacted,
        ...shadow,
        ...drawings,
        ...post,
      ]);
    });

    it('lays the portrait out without creating anything, and paints its backs once', async () => {
      const { log, scene, next, run } = await build();
      scene.settings.values.auto = false;
      await run(3);

      const counts = log.counts();
      scene.settings.values.layout = 'portrait';
      await scene.settings.write!('layout');
      const from = log.events.length;
      next(0);
      next(0);
      // The paint, then the dust's liveLists; the next frame compacts alone.
      expect(dispatches(log.events.slice(from))).toEqual([1 + 2, 2]);
      await run(4);
      expect(log.counts()).toEqual(counts);
    });

    it('dispatches each step of the fall, and a pick and a push for each tap', async () => {
      const { log, scene, next } = await build();
      next(0);
      const frame = (dt: number) => {
        const from = log.events.length;
        next(dt);
        return dispatches(log.events.slice(from));
      };

      const ghost = [frame(1 / 60), frame(0.5), frame(0.5)];
      scene.pointing.press(0.5, 0.5);
      const pressed = [frame(0), frame(0), frame(0.5)];
      expect(scene.settings.values.auto).toBe(false);
      expect([...ghost, ...pressed]).toEqual([
        // A 1/60 s frame takes two steps, and half a second eight, the most a frame does
        [steps(2)],
        [steps(8)],
        // A second in, the ghost finger picks and pushes first
        [1 + 1 + steps(8)],
        // A press picks and pushes in its frame alone, and the ghost stops
        [1 + 1 + steps(0)],
        [steps(0)],
        [steps(8)],
      ]);
    });
  },
  {
    // The fall, the place, the pick and the push, the sunset and the portrait's paint, the dust's
    // and the bloom's, the occlusion's two and its view, and the drawings with their depth for it.
    pipelines: 25,
  },
);
