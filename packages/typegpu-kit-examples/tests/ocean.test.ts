import { expect, it } from 'vitest';
import { createScene } from '../src/ocean/scene.ts';
import { compute, describeExample, pass, recorded } from './utils/examples.ts';

/** A steady frame's kernels: the clouds' march and blend, three FFT steps, three maps' mips. */
const KERNELS = 2 + 3 + 3 * 3;
/** The sea's plate and rings, the sky, bloom's five passes, metering and the final pass. */
const draws = [
  ...pass('draw-indexed', 'draw-indexed', 'draw'),
  ...Array.from({ length: 5 }, () => pass('draw')).flat(),
  ...compute(1),
  ...pass('draw'),
  'submit',
];

describeExample(
  'ocean',
  createScene,
  (build) => {
    it('steps the waves, foam and clouds, then draws the sea under the sky', async () => {
      const { log, next } = await build();
      next();
      expect(recorded(log, next)).toEqual([...compute(KERNELS), ...draws]);
    });

    it('rebakes the sky and rebuilds the spectrum only when settings change', async () => {
      const { log, scene, next } = await build();
      next();
      scene.settings.values.sun = 12;
      await scene.settings.write!('sun');
      // The atmosphere's four bakes and the deck's three join the kernels, and the environment
      // rebakes in one pass of its own
      const rebaked = recorded(log, next);
      const kernels = compute(4 + 3 + KERNELS);
      const environment = rebaked.slice(kernels.length, rebaked.length - draws.length);
      expect(rebaked).toEqual([...kernels, ...compute(environment.length - 2), ...draws]);

      // A drag writes the wind many times a frame; the frame rebuilds the sea once
      scene.settings.values.wind = 8;
      await scene.settings.write!('wind');
      scene.settings.values.wind = 9;
      await scene.settings.write!('wind');
      expect(recorded(log, next)).toEqual([...compute(1 + KERNELS), ...draws]);
    });
  },
  {
    // The example's 20: waves 5, atmosphere 4, clouds 6, foam 3, sea and sky 2; the kit's 12 for
    // the environment, bloom, metering and the final pass
    pipelines: 32,
  },
);
