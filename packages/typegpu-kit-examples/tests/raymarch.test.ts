import { expect, it } from 'vitest';
import { createScene } from '../src/raymarch/scene.ts';
import { describeExample, pass } from './utils/examples.ts';

/** A frame's events: the march into HDR, the final pass, and FXAA into the surface. */
const frame = [...pass('draw'), ...pass('draw'), ...pass('draw'), 'submit'];

describeExample(
  'raymarch',
  createScene,
  (build) => {
    it('marches the field, finishes and smooths it', async () => {
      const { log, next } = await build();
      const march = log.shaders.find((code) => code.includes('sdSphere'));
      expect(march).toContain('oklabToLinearRgb');

      next();
      next();
      expect(log.events).toEqual([...frame, ...frame]);
    });

    it('finishes straight into the surface without FXAA', async () => {
      const { log, scene, next } = await build();
      scene.settings.values.fxaa = false;
      scene.settings.write!('fxaa');

      next();
      expect(log.events).toEqual([...pass('draw'), ...pass('draw'), 'submit']);
    });
  },
  {
    // The march, the final pass into the picture FXAA reads and into the surface, and FXAA
    pipelines: 4,
  },
);
