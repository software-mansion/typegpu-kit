import { expect, it } from 'vitest';
import { createScene } from '../src/star-map/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/**
 * A frame's events: the pick and its copy, the stars, the bloom's eleven passes, and the final pass
 * with the label's callout and text over it.
 */
const frame = [
  ...compute(),
  'copy-buffer',
  ...pass('draw'),
  ...BLOOM,
  ...pass('draw', 'draw', 'draw'),
  'submit',
];

describeExample(
  'star-map',
  createScene,
  (build) => {
    it('picks, draws the stars, blooms and finishes', async () => {
      const { scene, log, next } = await build();
      scene.pointing.move(0.5, 0.5);
      next();
      next();
      expect(log.events).toEqual([...frame, ...frame]);
    });

    it('reads out what the pick read back, the frame it picked in and the frame drawn now', async () => {
      const { scene, next, run } = await build();
      expect(scene.pointing.reach).toBe(24);
      scene.pointing.move(0.5, 0.5);
      next();
      const read = () =>
        Object.fromEntries(
          Object.entries(scene.settings.readouts!).map(([name, value]) => [name, value()]),
        );
      expect(read()).toEqual({
        result: 'none yet',
        'picked in frame': 'pending',
        'drawn in frame': 0,
      });

      await run(8);
      expect(read()).toMatchObject({ result: 'null', 'drawn in frame': 8 });
      expect(read()['picked in frame']).toBeLessThanOrEqual(8);
    });

    it('switches nearest through the uniform its rank reads, building nothing', async () => {
      const { scene, log, run } = await build();
      await run(2);
      const pipelines = log.pipelines;
      const writes = log.writes.length;

      scene.settings.values.nearest = 'eye';
      scene.settings.write!('nearest');
      await run(2);
      expect(log.pipelines).toBe(pipelines);
      expect(log.writes.slice(writes).some(({ size }) => size === 4)).toBe(true);
    });

    it('labels the star it starts with, types the label on, then writes no more than unlabelled', async () => {
      const labelled = await build({ choose: [3, 0, 0] });
      const plain = await build();
      /** The writes of the scene's next four frames. */
      const writes = async ({ log, run }: typeof plain) => {
        const before = log.writes.length;
        await run(4);
        return log.writes.length - before;
      };

      expect(await writes(labelled)).toBeGreaterThan(await writes(plain));
      await labelled.run(56);
      await plain.run(56);
      expect(await writes(labelled)).toBe(await writes(plain));
      expect(labelled.scene.choice).not.toBeNull();
    });

    it('chooses nothing before a pick comes back', async () => {
      const { scene, next } = await build();
      scene.pointing.move(0.5, 0.5);
      next();
      scene.pointing.press(0.5, 0.5);
      expect(scene.choice).toBeNull();
    });
  },
  { options: { count: 256 }, gap: 0.05, pipelines: 8 },
);
