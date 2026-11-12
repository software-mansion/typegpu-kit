import { expect, it } from 'vitest';
import { VANISH } from '../src/platformer/coins.ts';
import { STEP, type Score } from '../src/platformer/game.ts';
import { createHero, HALF } from '../src/platformer/hero.ts';
import { parseLevel } from '../src/platformer/level.ts';
import { createScene } from '../src/platformer/scene.ts';
import { compute, describeExample, pass } from './utils/examples.ts';

/** A strip of ground with a coin, a far critter and the flag. */
const STRIP = ['', '', '', '@ o          w             F', '###########################'];

/**
 * A frame's events. One compute pass holds the critters', the dust's and the sparkles' steps, any
 * spawns, and two compacting kernels each for the coins, the critters, the dust and the sparkles.
 * A pass draws each backdrop layer and one the level's tiles. The sprites' pass draws the coins,
 * critters, dust and sparkles indirectly, and the hero. One pass finds what casts shadows and two
 * blur them, one prints the layers and sprites on paper and the final pass draws it onto the
 * surface.
 */
const frame = (steps: number, spawns: number, capture: boolean) => [
  ...compute(3 * steps + spawns + 8),
  ...(capture ? ['copy-buffer'] : []),
  ...Array.from({ length: 4 }, () => pass('draw')).flat(),
  ...pass(...Array<string>(4).fill('draw-indirect'), 'draw'),
  ...Array.from({ length: 5 }, () => pass('draw')).flat(),
  'submit',
];

describeExample(
  'platformer',
  createScene,
  (build) => {
    /** The scene, with a copy of each score it hears. */
    const setup = async (map?: readonly string[]) => {
      const scores: Score[] = [];
      const onScore = (score: Score) => scores.push({ ...score });
      const built = await build(map ? { map, onScore } : { onScore });
      return { ...built, scores };
    };

    /** The scene on `map` at 60 Hz, letting readbacks map at every frame. */
    const play = async (map: readonly string[]) => {
      const parts = await setup(map);
      const { bytes, scene, run } = parts;
      const step = () => run(1, 1 / 60);
      const advance = (seconds: number) => run(Math.round(seconds * 60), 1 / 60);

      // The GPU never runs here, so contacts are written into the counters it would add to.
      const counters = bytes(scene.critters.contacts.buffer);
      const contact = (stomps: number, hurts: number) => {
        new Uint32Array(counters.buffer, counters.byteOffset, 2).set([stomps, hurts]);
      };

      return { ...parts, step, advance, contact };
    };

    it('records a frame in one compute pass, the layers, their shadows and their print on paper', async () => {
      const { log, next } = await setup();
      expect(log.shaders.filter((code) => code.includes('@vertex'))).toHaveLength(12);

      // Each frame copies the contacts, which are read at 60 Hz. The second takes four steps, and
      // the hero lands from where it was placed in a puff of dust
      next();
      next();
      expect(log.events).toEqual([...frame(0, 0, true), ...frame(4, 1, true)]);
    });

    it('uploads only the time, and the time not yet stepped, in a frame where nothing changes on the CPU', async () => {
      const { log, advance } = await play(STRIP);

      // The hero lands and stops blinking, and the camera settles on it.
      await advance(3);

      const submitted = log.submissions;
      log.writes.length = 0;
      await advance(0.5);

      // The dust and the sparkles each write how far ahead of the last step they are drawn.
      const labels = log.writes.map((write) => write.buffer.label);
      expect(labels.filter((label) => label === 'time')).toHaveLength(log.submissions - submitted);
      expect(labels.filter((label) => label !== 'time' && label !== 'ahead')).toEqual([]);
    });

    it('kicks up dust from the critters’ kernel when the hero lands on one', async () => {
      const { log } = await setup();

      const step = log.shaders.find((code) => code.includes('fn add('));
      expect(step).toContain('fn emit(');
    });

    it('sparkles where a coin was taken, in the same frame', async () => {
      const { scene, scores, log, step } = await play(STRIP);
      scene.input.right = true;

      while (scores.at(-1)?.coins === 0) {
        log.writes.length = 0;
        await step();
      }

      expect(log.writes.map((write) => write.buffer.label)).toContain('burstBuffer');
    });

    it('finds where each walker turns around', () => {
      const level = parseLevel(['', ' w   #', '###  #', '######']);

      expect(level.critters).toEqual([{ kind: 0, at: [1.5, 2.5], patrol: [0.5, 2.5] }]);
    });

    it('runs into a wall and jumps over three tiles, where the bricks sit', () => {
      const level = parseLevel(['', '', '', '', '@     #', '#######']);
      const hero = createHero(level);
      const floor = 1 + HALF[1];

      for (let t = 0; t < 1; t += STEP) hero.step({ left: false, right: true, jump: false }, STEP);
      expect(hero.position[0]).toBeCloseTo(6 - HALF[0], 3);
      expect(hero.position[1]).toBeCloseTo(floor, 3);
      expect(hero.grounded).toBe(true);

      hero.jump();
      let top = 0;
      for (let t = 0; t < 1; t += STEP) {
        hero.step({ left: false, right: false, jump: true }, STEP);
        top = Math.max(top, hero.position[1] - floor);
      }

      expect(top).toBeGreaterThan(3.2);
      expect(top).toBeLessThan(3.6);
    });

    it('takes a coin it runs through, and the coin goes after it flies off', async () => {
      const { scene, scores, advance } = await play(STRIP);
      expect(scene.coins.table.count).toBe(1);

      scene.input.right = true;
      await advance(0.4);
      expect(scores.at(-1)?.coins).toBe(1);

      await advance(VANISH + 0.1);
      expect(scene.coins.table.count).toBe(0);
    });

    it('bounces off a critter the GPU found it landed on', async () => {
      const { scene, scores, advance, contact } = await play(STRIP);
      await advance(0.1);
      const standing = scene.hero.position[1];

      contact(1, 0);
      await advance(0.1);

      expect(scores.at(-1)?.stomps).toBe(1);
      expect(scene.hero.position[1]).toBeGreaterThan(standing + 0.3);
    });

    it('ignores touches right after a start, then starts over at the next', async () => {
      const { scene, advance, contact } = await play(STRIP);
      scene.input.right = true;

      contact(0, 1);
      await advance(0.5);
      expect(scene.hero.position[0]).toBeGreaterThan(2);

      await advance(1.2);
      contact(0, 2);
      await advance(0.1);
      expect(scene.hero.position[0]).toBeLessThan(1.5);
    });

    it('runs while its buttons are held, jumps from its jump button and shows its score', async () => {
      const { scene, advance } = await play(STRIP);
      const { controls } = scene.ui;
      expect(scene.ui.score()).toBe('coins 0 · stomps 0');

      controls.right.press();
      await advance(0.4);
      expect(scene.ui.score()).toBe('coins 1 · stomps 0');

      controls.right.release();
      await advance(0.3);
      const stopped = scene.hero.position[0];
      await advance(0.2);
      expect(scene.hero.position[0]).toBeCloseTo(stopped, 1);

      const standing = scene.hero.position[1];
      controls.jump.press();
      await advance(0.2);
      controls.jump.release();
      expect(scene.hero.position[1]).toBeGreaterThan(standing + 0.5);
    });

    it('keeps the best time to the flag and starts over', async () => {
      const { scene, scores, advance } = await play(STRIP);
      scene.input.right = true;

      await advance(4.5);

      const best = scores.find((score) => score.best !== undefined)?.best;
      expect(best).toBeGreaterThan(3);
      expect(best).toBeLessThan(4.5);
    });
  },
  {
    // Coins, critters, dust and sparkles each compact with two kernels. The critters' step, a spawn
    // and a step each for dust and sparkles, the backdrop, the tiles, five sprite drawings, the shadows'
    // casters and two blurs, the print on paper and the final pass.
    pipelines: 25,
  },
);
