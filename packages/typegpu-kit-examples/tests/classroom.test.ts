import { expect, it } from 'vitest';
import { createScene } from '../src/classroom/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/** The room's nine meshes, the lamps' globes among them. */
const room = Array<string>(9).fill('draw-indexed');
/** The motes' simulation step, as the scene's. */
const STEP = 1 / 30;
const frame = (sunShadow: boolean, dust: boolean) => [
  // Spawning the motes, a fixed step, then listing them to draw.
  ...(dust ? compute(3) : []),
  ...(sunShadow ? pass(...room) : []),
  // The room, the sky through the windows, then the motes.
  ...pass(...room, 'draw', ...(dust ? ['draw-indirect'] : [])),
  // The air's grid lit over last frame's and summed from the eye, then added onto the room.
  ...(dust ? [...compute(2), ...pass('draw')] : []),
  // TAA, then keeping the camera for the next frame's motion, the bloom and the tone curve.
  ...pass('draw'),
  'copy-buffer',
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

describeExample(
  'classroom',
  createScene,
  (build) => {
    it('draws the sun’s shadow once, then the room, the dust in the air, TAA and the bloom', async () => {
      const { log, run } = await build();
      log.events.length = 0;
      await run(1, STEP);
      expect(log.events).toEqual(frame(true, true));
      log.events.length = 0;
      await run(1, STEP);
      expect(log.events).toEqual(frame(false, true));
    });

    it('switches the baked GI, the baked AO and the lamps by a write each, and the dust off', async () => {
      const { log, scene, run } = await build();
      await run(3, STEP);
      const counts = log.counts();
      for (const key of ['gi', 'ao', 'lamps'] as const) {
        const writes = log.writes.length;
        scene.settings.values[key] = false;
        await scene.settings.write!(key);
        expect(log.writes.length - writes).toBe(1);
      }
      await run(1, STEP);
      expect(log.counts()).toEqual(counts);

      scene.settings.values.sunbeams = false;
      log.events.length = 0;
      await run(1, STEP);
      expect(log.events).toEqual(frame(false, false));
    });
  },
  { pipelines: 18 },
);
