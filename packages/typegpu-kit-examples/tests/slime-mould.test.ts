import { expect, it } from 'vitest';
import { createScene } from '../src/slime-mould/scene.ts';
import { BLOOM, compute, describeExample, pass, recorded } from './utils/examples.ts';

/**
 * A frame: its steps, each moving the agents and spreading the trail across and down, the agar and
 * the mould's light into HDR, the bloom, the final pass.
 */
const frame = ({ first = false, steps = 1 } = {}) => [
  // The agents scattered, after their cells are emptied, and the food painted.
  ...(first ? ['clear-buffer', ...compute(1), ...compute(1)] : []),
  ...Array.from({ length: steps }, () => compute(3)).flat(),
  ...pass('draw', 'draw'),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'slime-mould',
  createScene,
  (build) => {
    it('scatters and paints the food in its first frame, then steps and draws with one submit', async () => {
      const { log, next } = await build();
      const twice = recorded(log, () => {
        next();
        next();
      });
      expect(twice).toEqual([...frame({ first: true, steps: 0 }), ...frame({ steps: 2 })]);
    });

    it('takes no step while paused, and still draws', async () => {
      const { log, scene, next } = await build();
      next();
      scene.settings.values.paused = true;
      scene.settings.write!('paused');
      expect(recorded(log, next)).toEqual(frame({ steps: 0 }));
    });

    it('writes the time and how many are alive as it grows, and an oat where a tap lands', async () => {
      const { log, scene, next } = await build();
      next();
      next();
      log.writes.length = 0;
      next();
      expect(log.writes.map((write) => write.size)).toEqual([4, 4]);

      log.writes.length = 0;
      scene.pointing.press!(0.5, 0.5);
      expect(log.writes.map((write) => write.size)).toEqual([20, 12]);
      expect(recorded(log, next).slice(0, 3)).toEqual(compute(1));
    });
  },
  {
    options: { grid: 64, count: 50_000 },
    pipelines: 11,
  },
);
