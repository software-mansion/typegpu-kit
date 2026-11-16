import type { FixedStep, Frame } from '@typegpu/kit';
import type { TgpuComputePass } from 'typegpu';
import type { Coins } from './coins.ts';
import type { Critters } from './critters.ts';
import type { Effects } from './effects.ts';
import { createHero, HALF, type Input } from './hero.ts';
import { solidAt, type Level } from './level.ts';

/** Seconds per step. The hero and the critters move at 120 Hz. */
export const STEP = 1 / 120;

/** Raspberries the hero starts with. A hit or a fall takes half of one. */
export const HEARTS = 3;
/**
 * Seconds the hero can't be hurt after a start or a hit, and after a fall puts it back. Each
 * exceeds contact-readback latency.
 */
const SAFE = { start: 1.5, hit: 1.5, fall: 2 };
/** Fall distance below the level before the hero is put back, in tiles. */
const FELL = 1;
/** Tiles a second a skid kicks its dust, the way the hero slides. */
const KICK = 7;

export interface Score {
  /** Halves of raspberries left, from `2 * HEARTS` down. The run starts over at none. */
  health: number;
  coins: number;
  /** Critters defeated, by stomps and slides. */
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
 * Runs toward the flag, collecting coins and stomping critters. Other critter contacts and falls
 * each take half a raspberry: a hit knocks the hero back, and a fall puts it back on the last
 * ground it stood on. The run starts over when none are left.
 */
export function createPlatformer(
  level: Level,
  parts: PlatformerParts,
  onScore: (score: Score) => void,
) {
  const { coins, critters, effects, sim } = parts;

  const hero = createHero(level);
  const input: Input = { left: false, right: false, jump: false, down: false };
  const score: Score = { health: 2 * HEARTS, coins: 0, stomps: 0, best: undefined };

  let now = 0;
  let startedAt = 0;
  let safeUntil = 0;
  /** The centre of the tile above the last ground the hero stood on, where a fall puts it back. */
  const ground: [number, number] = [...level.start];

  const restart = () => {
    hero.place(...level.start);
    coins.reset(level);
    critters.reset(level);

    score.health = 2 * HEARTS;
    score.coins = 0;
    score.stomps = 0;
    startedAt = now;
    safeUntil = now + SAFE.start;
    ground[0] = level.start[0];
    ground[1] = level.start[1];
    onScore(score);
  };

  /** Takes half a raspberry. Returns false when none are left and the run starts over. */
  const hurt = () => {
    score.health -= 1;
    if (score.health > 0) {
      onScore(score);
      return true;
    }
    restart();
    return false;
  };

  /** Remembers the tile the hero stands over when it is solid, with room to stand above. */
  const remember = () => {
    const x = Math.floor(hero.position[0]);
    const below = Math.round(hero.position[1] - HALF[1]) - 1;
    if (!solidAt(level, x, below) || solidAt(level, x, below + 1) || solidAt(level, x, below + 2)) {
      return;
    }
    ground[0] = x + 0.5;
    ground[1] = below + 1.5;
  };

  restart();

  return {
    hero,
    input,

    /** Whether the hero can't be hurt, after a start, a hit or a fall. */
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
      if (contact?.slides) {
        score.stomps += contact.slides;
        onScore(score);
      }
      // A hit knocks the hero away from the critters that touched it, the way most of them push
      if (contact?.hurts && now >= safeUntil && hurt()) {
        hero.knock(2 * contact.rightward >= contact.hurts ? 1 : -1);
        safeUntil = now + SAFE.hit;
      }

      let taken = 0;
      hero.start[0] = hero.position[0];
      hero.start[1] = hero.position[1];
      for (let steps = sim.update(frame.dt); steps > 0; steps--) {
        const airborne = !hero.grounded;
        const skidding = hero.skidding;
        hero.step(input, STEP);
        if (hero.grounded) remember();
        const feet = hero.position[1] - HALF[1];
        if (airborne && hero.grounded) effects.puff(hero.position[0], feet);
        // A ducked hero turns without kicking up dust
        else if (!skidding && hero.skidding && !hero.ducked)
          effects.puff(hero.position[0], feet, -hero.facing * KICK);
        effects.slide(hero.position[0], feet, hero.sliding ? hero.velocity[0] : 0);

        taken += coins.collect(hero.position, now, effects.sparkle);
        critters.step(pass);
        effects.dust.step(pass);
        effects.sparks.step(pass);
        effects.bubbles.step(pass);
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
      if (hero.position[1] < -FELL && hurt()) {
        hero.place(...ground);
        safeUntil = now + SAFE.fall;
      }
    },
  };
}
