import { expect, it } from 'vitest';
import { createScene } from '../src/solar-system/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/**
 * A frame's events: a comet step a fixed step, the planets' move and a pick if the pointer moved,
 * then the scene (the bodies' two meshes, the sky, the belt, the orbits, the tails and the labels),
 * the bloom and the final pass.
 */
const frame = (steps: number, picked = false) => [
  ...compute(steps + (picked ? 2 : 1)),
  ...(picked ? ['copy-buffer'] : []),
  ...pass('draw-indexed', 'draw-indexed', 'draw', 'draw', 'draw-indexed', 'draw-indexed', 'draw'),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'solar-system',
  createScene,
  (build) => {
    it('steps the comets, moves the planets, picks only in a frame the pointer moved in and draws', async () => {
      const { log, scene, next } = await build();
      log.events.length = 0;
      next();
      next(0.05);
      scene.pointing.move!(0.5, 0.5);
      next(0.05);
      next(0.05);
      expect(log.events).toEqual([...frame(0), ...frame(3), ...frame(3, true), ...frame(3)]);
    });

    it('chooses Earth after 1.8 s with one write, and keeps it when a pick finds nothing', async () => {
      const { log, scene, next, run } = await build();
      const writes = (gap: number) => {
        const from = log.writes.length;
        next(gap);
        return log.writes.slice(from).map(({ size }) => size);
      };
      next();
      next(1);
      const steady = writes(0.4).length;
      // Earth's choice is one planet's time and state, and the orbits' focus.
      expect(writes(0.5)).toHaveLength(steady + 2);
      expect(writes(0.1)).toHaveLength(steady);

      scene.pointing.move!(0.5, 0.5);
      await run(8, 0.1);
      expect(writes(0.1)).toHaveLength(steady);
    });
  },
  {
    // The sky, the bodies, the belt, the orbits, the tails and the labels, the comets' and the
    // planets' kernels, the pick, the bloom's three and the final pass, and the two meshes' bakes.
    pipelines: 15,
  },
);
