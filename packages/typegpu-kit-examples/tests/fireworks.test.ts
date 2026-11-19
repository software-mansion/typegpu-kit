import { cameras } from '@typegpu/kit';
import { expect, it } from 'vitest';
import { heightAt } from '../src/fireworks/land.ts';
import { createScene } from '../src/fireworks/scene.ts';
import { BLOOM, compute, describeExample, pass } from './utils/examples.ts';

/** The castle's and the pines' meshes, one draw each. */
const meshes = Array<string>(16).fill('draw-indexed');
/** Embers, stars, shells and flashes, then the stars' streaks and the shells' tails. */
const glows = [...Array<string>(4).fill('draw-indirect'), 'draw-indexed', 'draw-indexed'];
/** Five systems, each compacted in two kernels. */
const COMPACTION = 10;
/** Each fixed step moves the five systems, then follows the stars and the shells. */
const PER_STEP = 7;
const frame = (steps: number, spawns = 0) => [
  ...compute(spawns + steps * PER_STEP + COMPACTION),
  // The land's and the castle's depth at the smoke's resolution, then the smoke behind them.
  ...pass('draw-indexed', ...meshes),
  ...pass('draw-indirect'),
  // The land, the castle, the sky, the smoke and what glows, reflected in the lake's mirror.
  ...pass('draw-indexed', ...meshes, 'draw', 'draw-indirect', ...glows),
  // Then as the eye sees them, with the water before the sky and the smoke's layer after it.
  ...pass('draw-indexed', ...meshes, 'draw', 'draw', 'draw', ...glows),
  ...BLOOM,
  // The exposure adapting to the picture.
  ...compute(),
  // The tone curve and bloom into the surface.
  ...pass('draw'),
  'submit',
];

describeExample(
  'fireworks',
  createScene,
  (build) => {
    it('steps the show, draws it through the mirror and the eye, and only draws it when paused', async () => {
      const { log, scene, run } = await build();
      await run(3);
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(3));
      const lit = log.shaders.filter((code) => code.includes('fn lightOf'));
      // The land and the castle's two sides, through the mirror and the eye, and the smoke's step.
      expect(lit).toHaveLength(7);
      expect(lit.every((code) => code.includes('struct Flash'))).toBe(true);

      scene.settings.values.paused = true;
      scene.settings.write!('paused');
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(0));
    });

    it('uploads each shell its warm-up launches in a submission of its own', async () => {
      const { log } = await build();
      const launching = compute(1 + PER_STEP + COMPACTION).join();
      const launches = log.events
        .join()
        .split('submit')
        .map((submission) => submission.split(launching).length - 1);
      expect(Math.max(...launches)).toBe(1);
      expect(launches.filter((count) => count === 1).length).toBeGreaterThan(1);
    });

    it('sends a shell up for a click on the sky, and none on its own when the show waits for clicks', async () => {
      const { log, scene, run } = await build();
      scene.settings.values.show = 'on click';
      await run(3);
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(3));

      scene.pointing.press!(0.3, 0.2);
      log.events.length = 0;
      await run(1);
      expect(log.events).toEqual(frame(3, 1));
    });

    it('backs away on a narrow view, keeps its zoom through a resize and keeps the eye over the lake and above the land', async () => {
      const { scene, surface, run } = await build();
      await run(3);
      const distance = scene.view.goal.distance;
      surface.setSize({ width: 36, height: 64 });
      await run(2);
      expect(scene.view.goal.distance).toBeGreaterThan(distance);
      surface.setSize({ width: 64, height: 36 });
      await run(2);
      expect(scene.view.goal.distance).toBeCloseTo(distance);

      // A phone's portrait screen, about 9 by 19.5
      surface.setSize({ width: 36, height: 78 });
      await run(2);
      const eye = cameras.eyeOf(scene.view.goal, [0, 0, 0]);
      expect(Math.hypot(eye[0], eye[2])).toBeLessThan(300);
      expect(heightAt(eye[0], eye[2])).toBeLessThan(0);

      scene.view.goal.pitch = -0.5;
      scene.view.goal.distance = 600;
      await run(1);
      cameras.eyeOf(scene.view.goal, eye);
      expect(eye[1]).toBeGreaterThanOrEqual(Math.max(0, heightAt(eye[0], eye[2])) + 2.5);
    });
  },
  {
    options: { random: () => 0.5 },
    gap: 0.05,
    pipelines: 56,
  },
);
