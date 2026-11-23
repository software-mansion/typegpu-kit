import { expect, it } from 'vitest';
import { createScene } from '../src/benthos/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/**
 * A frame's events: the snow's work in one compute pass, the water and the snow, the bloom's six
 * levels down and five up, and the final pass.
 */
const frame = (dispatches: number) => [
  ...compute(dispatches),
  ...pass('draw', 'draw-indirect'),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'benthos',
  createScene,
  (build) => {
    it('spawns the snow in its first frame, then steps it', async () => {
      const { log, next } = await build();

      // The noise table fills once, at creation. The first frame spawns the snow, which `drift`
      // sizes and colours as it is born, and both frames compact it in two dispatches.
      next();
      next();
      expect(log.events).toEqual([...compute(), 'submit', ...frame(1 + 2), ...frame(2 + 2)]);
    });

    it('writes the pointer while it moves and until its stir eases out, then stops', async () => {
      const { log, scene, next } = await build();
      const writes = () => {
        const before = log.writes.length;
        next();
        return log.writes.length - before;
      };

      // A steady frame writes only the time and the time since the last step, and a pointer that
      // hasn't moved adds nothing.
      writes();
      scene.pointing.move!(0.5, 0.5);
      expect(writes()).toBe(2);

      scene.pointing.move!(0.6, 0.5);
      expect(writes()).toBe(3);

      let easing = 0;
      while (writes() > 2) easing += 1;
      expect(easing).toBeLessThan(30);
      expect(writes()).toBe(2);
    });
  },
  {
    // The noise table's fill, the water, the snow's spawn, step, two compacting kernels and draw,
    // three bloom passes and the final pass.
    pipelines: 11,
  },
);
