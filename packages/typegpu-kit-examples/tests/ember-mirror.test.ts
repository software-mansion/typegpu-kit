import { expect, it } from 'vitest';
import { createScene } from '../src/ember-mirror/scene.ts';
import { describeExample } from './utils/examples.ts';

const camera = { texture: {} as GPUExternalTexture, width: 1280, height: 720 };

describeExample(
  'ember-mirror',
  createScene,
  (build) => {
    it('sheds from smoke at 30 Hz, then from every camera frame while one arrives', async () => {
      const { scene, next } = await build();
      const shed = scene.settings.readouts!['pictures shed from'] as () => number;
      const frames = scene.settings.readouts!['camera frames'] as () => number;

      next();
      const before = shed();
      for (let i = 0; i < 6; i++) next(1 / 60);
      expect(shed() - before).toBe(3);

      let serial = 0;
      scene.ui.camera.attach({ latest: () => ({ ...camera, serial }) });
      for (let i = 0; i < 4; i++) {
        serial += 1;
        next(1 / 60);
      }
      expect([frames(), shed() - before]).toEqual([4, 7]);

      // Without new frames the last stays, and smoke returns after a second
      for (let i = 0; i < 30; i++) next(1 / 60);
      expect(shed() - before).toBe(7);
      for (let i = 0; i < 40; i++) next(1 / 60);
      expect(shed() - before).toBeGreaterThan(7);
    });
  },
  { pipelines: 13 },
);
