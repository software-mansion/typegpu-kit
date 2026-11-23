import { census } from '@typegpu/kit/debug';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';
import { createScene, type SceneOptions } from '../src/skeletons/scene.ts';
import { filesOf } from '../src/hosts.ts';
import { BLOOM, describeExample, fromDisk, pass, SORTING, recorded } from './utils/examples.ts';

/** A draw for each of the 13 meshes the horde holds: four characters of two, and five things in hand. */
const horde = Array<string>(13).fill('draw-indexed-indirect');

/** A draw for each mesh of the walls and braziers: a stone, and a stem, bowl, lip and coals. */
const yard = Array<string>(5).fill('draw-indexed');

/**
 * With `steps` fixed steps, the grid's sort and a step of the machine each; then every pose found,
 * every skeleton moved, and two kernels culling for the camera and two for the sun. Then the
 * fire's and the smoke's `spawns`, a step of each with each fixed step, and their two compacting
 * kernels each.
 */
const moved = (steps: number, spawns: number) => [
  'begin-compute',
  ...(steps > 0 ? SORTING : []),
  ...Array<string>(steps + 6 + spawns + steps * 2 + 4).fill('dispatch'),
  'end-compute',
];

/**
 * The moon's shadow of the horde, the walls and the braziers; the ground, the walls and braziers,
 * the horde, the sky, the smoke and the flames; the bloom's eleven passes and the final pass.
 */
const drawn = [
  ...pass(...horde, ...yard),
  ...pass('draw-indexed', ...yard, ...horde, 'draw', 'draw-indirect', 'draw-indirect'),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'skeletons',
  createScene,
  (build) => {
    it('steps, poses, moves and culls the horde in one compute pass, then draws', async () => {
      const { log, next } = await build();
      const events = recorded(log, () => {
        next();
        next();
      });
      // The first frame takes no step and spawns the burning fire and the first smoke; the next,
      // 0.05 s later, takes three and spawns nothing until the steps have owed it.
      expect(events).toEqual([...moved(0, 2), ...drawn, ...moved(3, 0), ...drawn]);
    });

    it('creates nothing and writes only the clocks and the fire in twenty seconds of waking, wandering, fighting and falling', async () => {
      const { log, next } = await build();
      next();
      next();

      // The machine starts every action on the GPU. The flicker's time and the two clocks are all
      // a frame writes, beside the particles' bursts and how far ahead they draw.
      const before = log.counts();
      for (let i = 2; i < 400; i++) {
        const from = log.writes.length;
        next();
        const written = log.writes.slice(from).map(({ buffer }) => buffer.label);
        const fire = new Set(['burstBuffer', 'queued', 'ahead']);
        expect(written.filter((label) => !fire.has(label))).toEqual([
          'time',
          'now',
          'clockUniform',
        ]);
      }
      expect(log.counts()).toEqual(before);
    });
  },
  {
    // The meshes' bakes and the poses' kernel; the grid's eight kernels, the machine's and the
    // march's; the flames' and the smoke's four each; the colour pipelines of the yard, the walls
    // and the braziers' iron for each look of the pool, and the horde's; four culling kernels;
    // the shadow pipelines of the horde, the walls and the iron; the sky, the two particle
    // drawings, three bloom passes and the final
    pipelines: 6 + 10 + 8 + (9 + 3) + 4 + 6 + (1 + 2) + 3 + 1,
    options: { count: 8 },
    features: ['indirect-first-instance'],
    gap: 0.05,
  },
);

it('binds at most 8 storage buffers a stage, the machine step 8 of them', async () => {
  const files = (await filesOf('skeletons', fromDisk)) as SceneOptions;
  const { root } = recordingRoot({ features: ['indirect-first-instance'] });
  const counted = census.create(root);
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  const scene = await createScene(root, surface, { ...files, count: 8 });

  expect(counted.overBudget).toEqual([]);
  // Its rows, events and rolls; the members; the plays and clips the verbs write; the grid's cells
  // and copies
  const step = counted.pipelines.find(({ label }) => label === 'Doing_stepping')!;
  const kinds = step.bindings.map((binding) => binding.kind);
  expect([
    kinds.filter((k) => k === 'storage').length,
    kinds.filter((k) => k === 'uniform').length,
  ]).toEqual([8, 4]);

  scene.destroy();
  counted.destroy();
});
