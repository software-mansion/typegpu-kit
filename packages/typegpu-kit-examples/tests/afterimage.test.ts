import { expect, it } from 'vitest';
import { createScene } from '../src/afterimage/scene.ts';
import { describeExample, pass, recorded } from './utils/examples.ts';

/** The hall's seven materials and the pendulum's three, a draw each. */
const MESHES = Array<string>(10).fill('draw-indexed');

describeExample(
  'afterimage',
  createScene,
  (build) => {
    it('draws the shadow and hall, then shafts, bloom, feedback and the final picture', async () => {
      const { log, next } = await build();
      next();
      expect(recorded(log, next)).toEqual([
        ...pass(...MESHES),
        ...pass(...MESHES, 'draw'),
        ...pass('draw'),
        ...Array.from({ length: 9 }, () => pass('draw')).flat(),
        ...pass('draw'),
        ...pass('draw'),
        'submit',
      ]);
    });

    it('shows the clean render with the image passes skipped', async () => {
      const { log, scene, next } = await build();
      scene.settings.values.compare = 'clean';
      scene.settings.write!('compare');
      expect(recorded(log, next)).toEqual([
        ...pass(...MESHES),
        ...pass(...MESHES, 'draw'),
        ...pass('draw'),
        'submit',
      ]);
    });

    it('decays feedback by elapsed time and clears it after a resize, camera cut or compare', async () => {
      const { log, scene, surface, bytes, next } = await build();
      const retained = () => {
        const write = log.writes.findLast(({ buffer }) => buffer.label === 'afterimage feedback')!;
        const data = bytes(write.buffer);
        return new DataView(data.buffer, data.byteOffset, data.byteLength).getFloat32(0, true);
      };
      next();
      expect(retained()).toBe(0);
      next(0.04);
      expect(retained()).toBe(0);

      scene.settings.values.feedback = true;
      scene.settings.write!('feedback');
      next();
      expect(retained()).toBe(0);
      next(0.04);
      const one = retained();
      expect(one).toBeGreaterThan(0);
      next(0.08);
      expect(retained()).toBeCloseTo(one * one, 6);

      surface.setSize({ width: 96, height: 54 });
      next();
      expect(retained()).toBe(0);
      next();
      expect(retained()).toBeGreaterThan(0);
      scene.view.goal.yaw += 0.5;
      next();
      expect(retained()).toBe(0);

      scene.settings.values.compare = 'clean';
      scene.settings.write!('compare');
      next();
      scene.settings.values.compare = 'processed';
      scene.settings.write!('compare');
      next();
      expect(retained()).toBe(0);
    });
  },
  { pipelines: 10 },
);
