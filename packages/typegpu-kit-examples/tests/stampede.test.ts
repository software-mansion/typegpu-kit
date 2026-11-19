import { expect, it } from 'vitest';
import { createScene } from '../src/stampede/scene.ts';
import { BLOOM, describeExample, pass, settle, SORTING } from './utils/examples.ts';

const indexed = (count: number) => Array<string>(count).fill('draw-indexed-indirect');
/**
 * Each fixed step sorts the herd's grid, moves the herd and changes its gaits, then moves the dust
 * and kicks up more.
 */
const STEP = [...SORTING, ...Array<string>(4).fill('dispatch')];
/**
 * The dust's liveLists, the poses, the pick, the hovered name, the cards' frames, and culling the
 * skins, the cards and the scenery for the eye and the sun.
 */
const PER_FRAME = 18;
const frame = (steps: number, { picked = false, bursts = 0 } = {}) => [
  'begin-compute',
  // A thunderclap's dust, brought to life before the steps.
  ...Array<string>(bursts).fill('dispatch'),
  ...Array.from({ length: steps }, () => STEP).flat(),
  ...Array<string>(PER_FRAME).fill('dispatch'),
  'end-compute',
  // A click's pick, then the frame's split, cleared for the next.
  ...(picked ? ['copy-buffer'] : []),
  'copy-buffer',
  'clear-buffer',
  // The scenery's eight models, the horse's eight materials and the card, into each of the sun's
  // three cascades.
  ...Array.from({ length: 3 }, () => pass(...indexed(17))).flat(),
  // The land, the scenery, the skins, the cards, the sky, the dust and the name.
  ...pass('draw-indexed', ...indexed(8 + 8 + 1), 'draw', 'draw-indirect', 'draw'),
  // The bloom, then the tone curve into the surface.
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'stampede',
  createScene,
  (build) => {
    it('records the herd, its dust, the pick and the split before the shadow, the scene and the bloom, with one submit', async () => {
      const { log, scene, run } = await build();
      scene.pointing.move!(0.5, 0.5);
      await run(3);
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(3));
    });

    it("reads a click's pick, and claps where it found no horse", async () => {
      const { log, scene, run } = await build();
      await run(3);
      scene.pointing.move!(0.5, 0.8);
      scene.pointing.press!(0.5, 0.8);
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(3, { picked: true }));
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(3, { bursts: 1 }));
      expect(scene.settings.values.follow).toBe('herd');
    });

    it('claps for each of two quick clicks, each picking in its own frame', async () => {
      const { log, scene, next, run } = await build();
      /** The bytes of the clap bursts the dust uploaded last. */
      const bursts = () => log.writes.findLast((w) => w.buffer.label === 'burstBuffer')!.size;
      await run(3);
      scene.pointing.move!(0.5, 0.8);
      scene.pointing.press!(0.5, 0.8);
      scene.pointing.press!(0.4, 0.8);
      log.events.length = 0;
      // Neither pick reads back before the second click's frame
      next();
      next();
      expect(log.events).toEqual([...frame(3, { picked: true }), ...frame(3, { picked: true })]);
      await settle();
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(3, { bursts: 1 }));
      const both = bursts();

      scene.pointing.press!(0.5, 0.8);
      await run(2);
      expect(both).toBe(2 * bursts());
    });

    it('repopulates within its capacity without creating resources', async () => {
      const { log, scene, run } = await build();
      await run(3);
      const before = log.counts();
      scene.settings.values.count = 8192;
      scene.settings.write!('count');
      await run(3);
      scene.settings.values.count = 256;
      scene.settings.write!('count');
      await run(3);
      expect(log.counts()).toEqual(before);
    });
  },
  {
    gap: 0.05,
    options: { count: 256, random: () => 0.5 },
    features: ['indirect-first-instance'],
    pipelines: 55,
  },
);
