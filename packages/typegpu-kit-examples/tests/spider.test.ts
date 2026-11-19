import { expect, it } from 'vitest';
import { createScene } from '../src/spider/scene.ts';
import { createWalker, GAIT } from '../src/spider/walker.ts';
import { BLOOM, describeExample, pass } from './utils/examples.ts';

/** A draw for each of the spider's six meshes: the ball, four leg tubes and the ring. */
const meshes = Array<string>(6).fill('draw-indexed');

/**
 * A frame's events: the shadow of the spider, the scene with the spider and the ground, the
 * bloom's eleven passes and the final pass.
 */
const frame = [
  ...pass(...meshes),
  ...pass(...meshes, 'draw-indexed'),
  ...BLOOM,
  ...pass('draw'),
  'submit',
];

const hills = (x: number, z: number) => Math.sin(x * 0.7) * Math.cos(z * 0.5) * 0.8;

describeExample(
  'spider',
  createScene,
  (build) => {
    it('draws the shadow and the scene, blooms and finishes', async () => {
      const { log, next } = await build();
      const loaded = log.events.length;
      next();
      next();
      expect(log.events.slice(loaded)).toEqual([...frame, ...frame]);
    });

    it('walks over hills to a goal, its feet reached and its knees up', () => {
      const walker = createWalker(hills, GAIT);
      const misses: number[] = [];
      let kneesDown = 0;

      for (const goal of [
        [6, -4],
        [-5, 3],
        [2, 7],
      ] as const) {
        walker.goal[0] = goal[0];
        walker.goal[2] = goal[1];
        // Ten seconds of frames.
        for (let step = 0; step < 600; step++) {
          walker.step(1 / 60);
          for (const { chain, foot } of walker.legs) {
            const [coxa, femur, tibia] = chain.bones;
            const hip = coxa!.start;
            const knee = femur!.end;
            const tip = tibia!.end;
            misses.push(Math.hypot(...foot.map((at, k) => at - tip[k])));
            // The knee stands above the line from the hip to the foot.
            const out = Math.hypot(knee[0] - hip[0], knee[2] - hip[2]);
            const reach = Math.hypot(tip[0] - hip[0], tip[2] - hip[2]);
            const below = hip[1] + (tip[1] - hip[1]) * Math.min(1, out / reach);
            if (knee[1] < below) kneesDown++;
          }
        }

        const { body } = walker;
        expect(Math.hypot(body[0] - goal[0], body[2] - goal[1])).toBeLessThan(0.3);
        expect(body[1] - hills(body[0], body[2])).toBeGreaterThan(0.4);
      }

      // A foot may be out of reach for a moment while the body turns.
      misses.sort((a, b) => a - b);
      expect(misses[Math.floor(misses.length * 0.99)]).toBeLessThan(0.02);
      expect(kneesDown / misses.length).toBeLessThan(0.005);
    });
  },
  {
    gap: 0.05,
    // The ground's heights and a bake for each of the seven meshes while loading, a colour and a
    // shadow pipeline for the spider, a colour pipeline for the ground, three bloom passes and
    // the final pass.
    pipelines: 1 + 7 + 3 + 3 + 1,
  },
);
