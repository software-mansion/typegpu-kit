import { expect, it } from 'vitest';
import { createScene } from '../src/quiver-trees/scene.ts';
import { BLOOM, compute, describeExample, pass, recorded } from './utils/examples.ts';

const indirect = (count: number) => Array<string>(count).fill('draw-indexed-indirect');
/** The ground, which isn't culled. */
const ground = ['draw-indexed'];
/** Two kernels culling for the camera and two for the sun, for each culling renderer. */
const culled = (renderers: number) => compute(renderers * 4);
/** The bloom's eleven passes, the final pass. */
const after = [...BLOOM, ...pass('draw'), 'submit'];

/**
 * The near meshes and each model's cards culled; a shadow of the five meshes and the four cards;
 * then the ground, the meshes, and each card's depth and colour.
 */
const withImpostors = [
  ...culled(5),
  ...pass(...indirect(5 + 4)),
  ...pass(...ground, ...indirect(5 + 4 * 2)),
  ...after,
];
/** Every mesh, culled, without the cards. */
const asMeshes = [
  ...culled(1),
  ...pass(...indirect(5)),
  ...pass(...ground, ...indirect(5)),
  ...after,
];

describeExample(
  'quiver-trees',
  createScene,
  (build) => {
    it('hands far things to impostors, culling each set for the camera and the sun', async () => {
      const { log, next } = await build();
      const events = recorded(log, () => {
        next();
        next();
      });
      expect(events).toEqual([...withImpostors, ...withImpostors]);
    });

    it('reads the atlases another way once that way has compiled, compiling it once', async () => {
      const { log, scene, next } = await build();
      const { values } = scene.settings;
      const write = scene.settings.write!;
      next();
      const pipelines = log.pipelines;

      values.sampling = 'nearest';
      const writing = write('sampling');
      next();
      await writing;
      expect(log.pipelines).toBe(pipelines + (7 + 3) * 4);

      expect(recorded(log, next)).toEqual(withImpostors);
      values.sampling = 'blended';
      write('sampling');
      next();
      values.sampling = 'nearest';
      write('sampling');
      next();
      expect(log.pipelines).toBe(pipelines + (7 + 3) * 4);
    });

    it('draws every mesh without impostors', async () => {
      const { log, scene, next } = await build();
      next();

      scene.settings.values.impostors = false;
      scene.settings.write!('impostors');
      expect(recorded(log, next)).toEqual(asMeshes);
    });
  },
  {
    options: { count: 60 },
    features: ['indirect-first-instance'],
    gap: 0.05,
    // While loading, the ground's bake, and four captures, each of four kernels fitting the
    // sphere, one drawing the views and two making the mipmaps; the ground's noise cache; the
    // ground; for the meshes everywhere and for the near ones, colour, shadow and four kernels;
    // for each model's cards those and a depth pass; three bloom passes and the final. The models
    // are two-sided, and every renderer of the pool compiles that look too, beside the cards'
    // one-sided one.
    pipelines: 1 + 4 * (5 + 2) + 1 + 2 + (6 + 2) * 2 + (7 + 3) * 4 + 3 + 1,
  },
);
