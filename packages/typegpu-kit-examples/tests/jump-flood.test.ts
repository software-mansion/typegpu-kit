import { expect, it } from 'vitest';
import { createScene } from '../src/jump-flood/scene.ts';
import { compute, describeExample, pass, recorded } from './utils/examples.ts';

/** A frame's events: clearing, placing, flooding and finding neighbours in one compute pass, then the draw. */
const frame = (passes: number) => [...compute(3 + passes), ...pass('draw'), 'submit'];
/** The map fills its noise gradients once, before the first frame. */
const gradients = [...compute(1), 'submit'];

describeExample(
  'jump-flood',
  createScene,
  (build) => {
    it('records the requested flood steps with a bind group per step in one compute pass', async () => {
      const { log, scene, next } = await build();
      expect(scene.settings.values.immediates).toBe(false);

      const floodShader = log.shaders.find((code) => code.includes('floodKernel'));
      expect(floodShader).toMatch(/var<uniform> \w+: Pass/);
      expect(floodShader).not.toContain('var<immediate>');

      // A 512-cell grid floods with steps 256, 128, … 1 and 1 again.
      next();
      next(1 / 60);
      expect(scene.settings.values.passes).toBe(10);
      expect(log.events).toEqual([...gradients, ...frame(10), ...frame(10)]);

      scene.settings.values.passes = 3;
      expect(recorded(log, next)).toEqual(frame(3));
    });

    it('records fewer flood steps from the first frame when set before it', async () => {
      const { log, scene, next } = await build();
      scene.settings.values.passes = 3;
      next();
      expect(log.events).toEqual([...gradients, ...frame(3)]);
    });
  },
  { pipelines: 6 },
);
