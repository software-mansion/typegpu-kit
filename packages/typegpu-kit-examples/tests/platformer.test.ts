import { expect, it } from 'vitest';
import { VANISH } from '../src/platformer/coins.ts';
import { HEARTS, STEP, type Score } from '../src/platformer/game.ts';
import { createHero, HALF, POISE, SNEAK, type Input } from '../src/platformer/hero.ts';
import { MAP, parseLevel } from '../src/platformer/level.ts';
import { createScene } from '../src/platformer/scene.ts';
import { compute, describeExample, pass } from './utils/examples.ts';

/** A strip of ground with a coin, a far critter and the flag. */
const STRIP = ['', '', '', '@ o          w             F', '###########################'];

/**
 * A frame's events. One compute pass holds the critters', the dust's, the sparkles' and the
 * bubbles' steps, any spawns, and two compacting kernels each for the coins, the critters, the
 * dust, the sparkles and the bubbles. A pass draws each backdrop layer and one the level's tiles.
 * The sprites' pass draws the coins, critters, bubbles, dust and sparkles indirectly, and the hero. One pass finds what casts shadows and two
 * blur them, one prints the layers and sprites on paper with the health bar over them and the final
 * pass draws it onto the surface.
 */
const frame = (steps: number, spawns: number, capture: boolean) => [
  ...compute(4 * steps + spawns + 10),
  ...(capture ? ['copy-buffer'] : []),
  ...Array.from({ length: 4 }, () => pass('draw')).flat(),
  ...pass(...Array<string>(5).fill('draw-indirect'), 'draw'),
  ...Array.from({ length: 3 }, () => pass('draw')).flat(),
  ...pass('draw', 'draw'),
  ...pass('draw'),
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
      /**
       * Running totals of contacts, of which `rightward` hurts came from a critter on the hero's
       * left. The game takes the increase since the last read.
       */
      const contact = (stomps: number, hurts: number, rightward = 0) => {
        new Uint32Array(counters.buffer, counters.byteOffset, 3).set([stomps, hurts, rightward]);
      };

      return { ...parts, step, advance, contact };
    };

    it('records a frame in one compute pass, the layers, their shadows and their print on paper', async () => {
      const { log, next } = await setup();
      expect(log.shaders.filter((code) => code.includes('@vertex'))).toHaveLength(14);

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

      for (let t = 0; t < 1; t += STEP)
        hero.step({ left: false, right: true, jump: false, down: false }, STEP);
      expect(hero.position[0]).toBeCloseTo(6 - HALF[0], 3);
      expect(hero.position[1]).toBeCloseTo(floor, 3);
      expect(hero.grounded).toBe(true);

      hero.jump();
      let top = 0;
      for (let t = 0; t < 1; t += STEP) {
        hero.step({ left: false, right: false, jump: true, down: false }, STEP);
        top = Math.max(top, hero.position[1] - floor);
      }

      expect(top).toBeGreaterThan(3.2);
      expect(top).toBeLessThan(3.6);
    });

    it('ducks in place, slides at a run down to a sneak, and sneaks from there', () => {
      const level = parseLevel(['', '@                        ', '###########################']);
      const hero = createHero(level);
      const hold = (input: Partial<Input>, seconds: number) => {
        const held = { left: false, right: false, jump: false, down: false, ...input };
        for (let t = 0; t < seconds; t += STEP) hero.step(held, STEP);
      };

      hold({}, 0.5);
      hold({ down: true }, 0.2);
      expect(hero.ducked).toBe(true);
      expect(hero.sliding).toBe(false);
      expect(hero.velocity[0]).toBe(0);

      hold({ right: true }, 0.5);
      expect(hero.velocity[0]).toBeGreaterThan(SNEAK);
      hold({ right: true, down: true }, 0.2);
      expect(hero.sliding).toBe(true);
      expect(hero.velocity[0]).toBeGreaterThan(SNEAK);

      // The slide keeps its way against the stick, and slows down to a sneak
      hold({ left: true, down: true }, 0.1);
      expect(hero.sliding).toBe(true);
      expect(hero.facing).toBe(1);
      hold({ right: true, down: true }, 1);
      expect(hero.sliding).toBe(false);
      expect(hero.velocity[0]).toBeCloseTo(SNEAK, 5);

      hold({ right: true }, 0.5);
      expect(hero.ducked).toBe(false);
      expect(hero.velocity[0]).toBeGreaterThan(SNEAK);
    });

    it('squeezes through a tunnel a tile high only ducked, and stands once out of it', () => {
      const level = parseLevel(['', '    =====    ', '@            ', '#############']);
      const hero = createHero(level);
      const hold = (input: Partial<Input>, seconds: number) => {
        const held = { left: false, right: false, jump: false, down: false, ...input };
        for (let t = 0; t < seconds; t += STEP) hero.step(held, STEP);
      };

      hold({ right: true }, 1);
      expect(hero.position[0]).toBeCloseTo(4 - HALF[0], 3);

      hold({ right: true, down: true }, 1);
      expect(hero.position[0]).toBeGreaterThan(5);

      // Without room to stand, letting go of the duck keeps the hero ducked
      hold({ right: true }, 0.3);
      expect(hero.ducked).toBe(true);
      expect(hero.position[0]).toBeLessThan(9);
      hold({ right: true }, 2);
      expect(hero.ducked).toBe(false);
      expect(hero.position[0]).toBeGreaterThan(9 + HALF[0]);
    });

    it('backflips away from the way it faces when it jumps in a skid', () => {
      const level = parseLevel([
        '',
        '',
        '',
        '',
        '',
        '@                        ',
        '###########################',
      ]);
      const hero = createHero(level);
      const floor = 1 + HALF[1];
      const hold = (input: Partial<Input>, seconds: number) => {
        const held = { left: false, right: false, jump: false, down: false, ...input };
        for (let t = 0; t < seconds; t += STEP) hero.step(held, STEP);
      };

      hold({ right: true }, 1);
      hold({ left: true }, 0.03);
      expect(hero.skidding).toBe(true);
      expect(hero.facing).toBe(-1);
      expect(hero.poised).toBe(false);

      // Letting go of the stick in the skid crouches for the flip
      hold({}, STEP);
      expect(hero.skidding).toBe(true);
      expect(hero.poised).toBe(true);

      hero.jump();
      hold({}, STEP);
      expect(hero.flipping).toBe(true);
      expect(hero.poised).toBe(false);
      expect(hero.velocity[0]).toBeGreaterThan(0);

      // The stick doesn't steer the flip, and a released jump doesn't cut it short
      let top = 0;
      hold({ left: true }, 0.3);
      expect(hero.velocity[0]).toBeGreaterThan(0);
      expect(hero.facing).toBe(-1);
      for (let t = 0; t < 1; t += STEP) {
        hero.step({ left: false, right: false, jump: false, down: false }, STEP);
        top = Math.max(top, hero.position[1] - floor);
      }
      expect(top).toBeGreaterThan(4);
      expect(hero.grounded).toBe(true);
      expect(hero.flipping).toBe(false);
    });

    it('stays crouched for a backflip at least a while after the skid, until it takes a direction', () => {
      const level = parseLevel([
        '',
        '',
        '',
        '',
        '',
        '@                        ',
        '###########################',
      ]);
      const turnThenJump = (wait: number) => {
        const hero = createHero(level);
        const hold = (input: Partial<Input>, seconds: number) => {
          const held = { left: false, right: false, jump: false, down: false, ...input };
          for (let t = 0; t < seconds; t += STEP) hero.step(held, STEP);
        };
        hold({ right: true }, 1);
        hold({ left: true }, 0.03);
        hold({}, wait);
        const skidding = hero.skidding;
        const poised = hero.poised;
        hero.jump();
        hold({ jump: true }, STEP);
        return { skidding, poised, flipping: hero.flipping };
      };

      expect(turnThenJump(POISE - 0.02)).toEqual({ skidding: false, poised: true, flipping: true });
      expect(turnThenJump(POISE + 0.03)).toEqual({
        skidding: false,
        poised: false,
        flipping: false,
      });

      // Taking a direction again ends the crouch at once
      const hero = createHero(level);
      const hold = (input: Partial<Input>, seconds: number) => {
        const held = { left: false, right: false, jump: false, down: false, ...input };
        for (let t = 0; t < seconds; t += STEP) hero.step(held, STEP);
      };
      hold({ right: true }, 1);
      hold({ left: true }, 0.03);
      hold({}, 0.05);
      expect(hero.poised).toBe(true);
      hold({ left: true }, STEP);
      expect(hero.poised).toBe(false);
      // Letting go once the turn is over runs out without a crouch
      hold({ left: true }, POISE);
      hold({}, STEP);
      expect(hero.poised).toBe(false);
    });

    it('jumps as usual from a skid while a direction is held', () => {
      const level = parseLevel([
        '',
        '',
        '',
        '',
        '',
        '@                        ',
        '###########################',
      ]);
      const hero = createHero(level);
      const hold = (input: Partial<Input>, seconds: number) => {
        const held = { left: false, right: false, jump: false, down: false, ...input };
        for (let t = 0; t < seconds; t += STEP) hero.step(held, STEP);
      };

      hold({ right: true }, 1);
      hold({ left: true }, 0.03);
      expect(hero.skidding).toBe(true);
      expect(hero.poised).toBe(false);

      hero.jump();
      hold({ left: true, jump: true }, STEP);
      expect(hero.grounded).toBe(false);
      expect(hero.flipping).toBe(false);
    });

    it('reaches the pillar by the flag only with a backflip', () => {
      const level = parseLevel(MAP);
      const ground = 2;
      /** Whether a run from `x` that way, jumping after `wait` seconds, ends on the pillar. */
      const onPillar = (x: number, way: number, wait: number, turn: boolean) => {
        const hero = createHero(level);
        hero.place(x, ground + 0.5);
        const run = { left: way < 0, right: way > 0, jump: false, down: false };
        for (let t = 0; t < wait; t += STEP) hero.step(run, STEP);
        const back = turn ? { ...run, left: way > 0, right: way < 0 } : run;
        if (turn) for (let t = 0; t < 0.05; t += STEP) hero.step(back, STEP);
        hero.jump();
        // A regular jump steers on toward its target, and a backflip lets go of the stick
        const air = turn ? { ...back, left: false, right: false } : back;
        for (let t = 0; t < 1.5; t += STEP) hero.step({ ...air, jump: true }, STEP);
        return hero.grounded && hero.position[1] - HALF[1] > ground + 3.5;
      };

      for (let x = 100.5; x < 114; x += 0.5) {
        for (const way of [-1, 1]) {
          for (let wait = 0; wait < 1.2; wait += 0.05) {
            expect(onPillar(x, way, wait, false), `jumping from ${x} after ${wait} s`).toBe(false);
          }
        }
      }
      const turns = Array.from({ length: 24 }, (_, k) => k * 0.05);
      expect(turns.some((wait) => onPillar(100.5, 1, wait, true))).toBe(true);
    });

    it('turns walkers where a tile above leaves no room', () => {
      const level = parseLevel(['', '   =', ' w  ', '####']);

      expect(level.critters).toEqual([{ kind: 0, at: [1.5, 1.5], patrol: [0.5, 2.5] }]);
    });

    it('reads forks as walkers of their own', () => {
      const level = parseLevel(['', ' f   #', '###  #', '######']);

      expect(level.critters).toEqual([{ kind: 2, at: [1.5, 2.5], patrol: [0.5, 2.5] }]);
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

    it('ignores touches right after a start, then loses half a raspberry and flies back out of control', async () => {
      const { scene, scores, advance, contact } = await play(STRIP);
      expect(scores.at(-1)?.health).toBe(2 * HEARTS);

      contact(0, 1);
      await advance(0.5);
      expect(scores.at(-1)?.health).toBe(2 * HEARTS);

      // A critter on the hero's left knocks it right and up, and the stick can't turn it
      await advance(1.2);
      scene.input.left = true;
      contact(0, 2, 1);
      await advance(0.1);
      expect(scores.at(-1)?.health).toBe(2 * HEARTS - 1);
      expect(scene.hero.knocked).toBe(true);
      expect(scene.hero.velocity[0]).toBeGreaterThan(0);
      expect(scene.hero.velocity[1]).toBeGreaterThan(0);

      // It can't be hurt again while it blinks, and takes the stick back once it lands
      contact(0, 3, 1);
      await advance(0.5);
      expect(scores.at(-1)?.health).toBe(2 * HEARTS - 1);
      expect(scene.hero.grounded).toBe(true);
      expect(scene.hero.knocked).toBe(false);
      await advance(0.2);
      expect(scene.hero.velocity[0]).toBeLessThan(0);
    });

    it('starts over with every raspberry when the last half goes', async () => {
      const { scene, scores, advance, contact } = await play(STRIP);

      for (let hit = 1; hit < 2 * HEARTS; hit++) {
        await advance(1.6);
        contact(0, hit);
        await advance(0.1);
        expect(scores.at(-1)?.health).toBe(2 * HEARTS - hit);
      }
      await advance(1.6);
      contact(0, 2 * HEARTS);
      await advance(0.1);

      expect(scores.at(-1)?.health).toBe(2 * HEARTS);
      expect(scene.hero.position[0]).toBeLessThan(1.5);
    });

    it('puts the hero back beside a hole it fell in, for half a raspberry, and keeps it safe a while', async () => {
      const map = ['', '', '', '@ o            w  F', '#####   ###########'];
      const { scene, scores, advance } = await play(map);
      scene.input.right = true;

      while ((scores.at(-1)?.health ?? 0) === 2 * HEARTS) await advance(0.05);
      scene.input.right = false;

      expect(scores.at(-1)?.health).toBe(2 * HEARTS - 1);
      expect(scene.hero.position[0]).toBe(4.5);
      expect(scene.hero.position[1]).toBeCloseTo(1 + HALF[1], 3);
      await advance(1.8);
      expect(scene.game.safe).toBe(true);
      await advance(0.4);
      expect(scene.game.safe).toBe(false);
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
    // casters and two blurs, the print on paper, the health bar and the final pass.
    pipelines: 31,
  },
);
