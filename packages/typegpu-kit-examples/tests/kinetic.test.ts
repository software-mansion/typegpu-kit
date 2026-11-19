import { expect, it } from 'vitest';
import { createScene } from '../src/kinetic/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/** The mobile's two shapes, discs and a ball, each one draw. */
const shapes = ['draw-indexed', 'draw-indexed'];

/** The exposure's meter, one workgroup. */
const meter = compute();

/**
 * A frame: two cascades of the pieces' shadow, the pieces, the backdrop and the wires, bloom, the
 * exposure's meter and the final pass.
 */
const frame = [
  ...pass(...shapes),
  ...pass(...shapes),
  ...pass(...shapes, 'draw', 'draw-indexed'),
  ...BLOOM,
  ...meter,
  ...pass('draw'),
  'submit',
];

/** A bake: the source, 8 halvings, 5 rougher reflection levels and the harmonics. */
const bake = compute(15);

describeExample(
  'kinetic',
  createScene,
  (build) => {
    it('bakes the sky in its first frame, then draws without baking', async () => {
      const { log, next } = await build();
      const loaded = log.events.length;
      next();
      expect(log.events.slice(loaded)).toEqual([...bake, ...frame]);
      const steady = log.events.length;
      next();
      expect(log.events.slice(steady)).toEqual(frame);
    });

    it('re-bakes when the sky changes, and while the time of day moves', async () => {
      const { log, scene, next, run } = await build();
      next();
      // A bake is the only compute pass of more than one dispatch
      const bakes = () =>
        log.events.filter(
          (event, i) => event === 'begin-compute' && log.events[i + 2] !== 'end-compute',
        ).length;
      const before = bakes();
      scene.settings.values.sky = 'venice';
      scene.settings.write!('sky');
      await run(3);
      expect(bakes() - before).toBe(1);

      scene.settings.values.sky = 'daylight';
      scene.settings.write!('sky');
      for (const hour of [16, 17, 18, 19]) {
        scene.settings.values.hour = hour;
        scene.settings.write!('hour');
        next(0.1);
      }
      expect(bakes() - before).toBe(5);
    });
  },
  { pipelines: 18 },
);
