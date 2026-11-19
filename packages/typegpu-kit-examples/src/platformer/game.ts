import type { FixedStep, Frame } from '@typegpu/kit';
import type { TgpuComputePass } from 'typegpu';
import type { Coins } from './coins.ts';
import type { Critters } from './critters.ts';
import type { Effects } from './effects.ts';
import { createHero, HALF, type Input } from './hero.ts';
import type { Level } from './level.ts';

/** Seconds per step. The hero and the critters move at 120 Hz. */
export const STEP = 1 / 120;

/** Invulnerability after restart, in seconds. Exceeds contact-readback latency. */
const SAFE = 1.5;
/** Fall distance below the level before restart, in tiles. */
const FELL = 1;

export interface Score {
  coins: number;
  stomps: number;
  /** Fastest time to the flag in seconds. */
  best: number | undefined;
}

export interface PlatformerParts {
  readonly coins: Coins;
  readonly critters: Critters;
  readonly effects: Effects;
  /** Fixed step shared by the hero, critters and effects, with duration `STEP`. */
  readonly sim: FixedStep;
}

/**
 * Runs toward the flag, collecting coins and stomping critters. Falls and other critter
 * contacts restart the run.
 */
export function createPlatformer(
  level: Level,
  parts: PlatformerParts,
  onScore: (score: Score) => void,
) {
  const { coins, critters, effects, sim } = parts;

  const hero = createHero(level);
  const input: Input = { left: false, right: false, jump: false };
  const score: Score = { coins: 0, stomps: 0, best: undefined };

  let now = 0;
  let startedAt = 0;
  let safeUntil = 0;

  const restart = () => {
    hero.place(...level.start);
    coins.reset(level);
    critters.reset(level);

    score.coins = 0;
    score.stomps = 0;
    startedAt = now;
    safeUntil = now + SAFE;
    onScore(score);
  };

  restart();

  return {
    hero,
    input,

    /** Whether restart invulnerability is active. */
    get safe() {
      return now < safeUntil;
    },

    /**
     * Steps the hero, critters and effects. Record after table flushes and before effect
     * spawning so dust and sparkles appear in the same frame.
     */
    update(frame: Frame, pass: TgpuComputePass) {
      now = frame.time;

      // GPU contacts arrive one or two frames after detection
      const contact = critters.contacts.take();
      if (contact?.stomps) {
        hero.bounce();
        score.stomps += contact.stomps;
        onScore(score);
      }
      if (contact?.hurts && now >= safeUntil) restart();

      let taken = 0;
      hero.start[0] = hero.position[0];
      hero.start[1] = hero.position[1];
      for (let steps = sim.update(frame.dt); steps > 0; steps--) {
        const airborne = !hero.grounded;
        hero.step(input, STEP);
        if (airborne && hero.grounded) effects.puff(hero.position[0], hero.position[1] - HALF[1]);

        taken += coins.collect(hero.position, now, effects.sparkle);
        critters.step(pass);
        effects.dust.step(pass);
        effects.sparks.step(pass);
      }

      if (taken > 0) {
        score.coins += taken;
        onScore(score);
      }

      if (hero.position[0] > level.flag[0]) {
        const run = now - startedAt;
        score.best = Math.min(score.best ?? run, run);
        restart();
      }
      if (hero.position[1] < -FELL) restart();
    },
  };
}
