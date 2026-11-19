import { expect, it } from 'vitest';
import { createScene } from '../src/night-market/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/**
 * A draw for each of the 30 meshes the solid things use, buildings, props, the table and potions,
 * then the seller, the customer and the lantern's iron and glass.
 */
const solid = Array<string>(34).fill('draw-indexed');
/** The lamps, their glass, the windows, the string lights, the lantern, its cord and the logs. */
const GLOWING = 13;
/** The poses, the fire's and the rain's spawns, steps and compactions. */
const COMPUTE = 12;
/** The moon's two cascades, then the lantern's and the fire's six faces and a layer per sign lamp. */
const SHADOW_PASSES = 2 + 14;
const frame = (blurred = true) => [
  // The swinging lantern's and the walkers' rows, kept for the motion, then the frame's moves.
  ...Array.from({ length: 2 }, () => ['copy-buffer', 'copy-buffer', 'copy-buffer']).flat(),
  ...compute(COMPUTE),
  ...Array.from({ length: SHADOW_PASSES }, () => pass(...solid)).flat(),
  // The ground, the solid and glowing things, the people, the sky, the fire and the rain.
  ...pass(
    'draw',
    ...solid,
    ...Array<string>(GLOWING).fill('draw-indexed'),
    'draw',
    'draw-indirect',
    'draw-indirect',
  ),
  // TAA; with depth of field, the focus pull reading the drawn depth and copying the focus into
  // the blur's settings, then its prepare, tiles, gather and composite; then the bloom and the
  // tone curve into the surface.
  ...pass('draw'),
  'copy-buffer',
  ...(blurred ? [...compute(), 'copy-buffer'] : []),
  ...Array.from({ length: blurred ? 4 : 0 }, () => pass('draw')).flat(),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'night-market',
  createScene,
  (build) => {
    it('draws the shadows, the scene, TAA, depth of field and the bloom, with one submit', async () => {
      const { log, run } = await build();
      await run(3);
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame());
    });

    it('drops the focus pull and the four depth-of-field passes with it off', async () => {
      const { log, scene, run } = await build();
      await run(3);
      scene.settings.values.dof = false;
      await scene.settings.write!('dof');
      await run(2);
      const counts = log.counts();
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(false));
      expect(log.counts()).toEqual(counts);
    });

    it('waves the person a tap lands on and still pulls the focus, creating nothing', async () => {
      const { log, scene, run } = await build();
      await run(3);
      const counts = log.counts();
      const tap = async (x: number, y: number) => {
        const writes = log.writes.length;
        scene.pointing.press(x, y);
        await run(1);
        return log.writes.length - writes;
      };
      const seller = new Float32Array(3);
      expect(scene.view.project([1.1, 1.5, 2.1], seller)).toBe(true);

      const missed = await tap(0.5, 0.05);
      const waved = await tap((seller[0] + 1) / 2, (1 - seller[1]) / 2);
      expect(waved).toBeGreaterThan(missed);
      expect(log.counts()).toEqual(counts);
    });
  },
  { pipelines: 37 },
);
