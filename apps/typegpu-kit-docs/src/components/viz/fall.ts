import { CARRY, KICK, leanOn, STEP, swing, TIP } from 'typegpu-kit-examples/dominoes/topple.ts';

export { STEP };

/** A domino the line holds: how far it leans, how fast it turns, and how hard it struck. */
export interface Toppling {
  angle: number;
  spin: number;
  hit: number;
}

export interface Line {
  /** The dominoes as the last step left them, the first pushed first. */
  readonly dominoes: readonly Toppling[];
  /** The seconds the steps have covered. */
  readonly time: number;
  /** Runs one step of `STEP` seconds. */
  step(): void;
}

/**
 * A line of `count` dominoes, the first set spinning as a tap does, each handing on `carry` of its
 * spin to the next when it strikes: the example's `CARRY` unless given.
 */
export function line(count: number, carry = CARRY): Line {
  let dominoes: Toppling[] = Array.from({ length: count }, (_, i) => ({
    angle: 0,
    spin: i === 0 ? KICK : 0,
    hit: 0,
  }));
  let time = 0;

  const next = (i: number): Toppling => {
    const self = dominoes[i];
    let spin = self.spin;
    let pushed = 0;
    const before = dominoes[i - 1];
    if (before) {
      if (self.angle === 0 && spin === 0 && before.hit > 0) spin = before.hit * carry;
      if (before.angle >= leanOn(self.angle) - 0.001) {
        pushed = Math.max(Math.sin(before.angle - TIP), 0);
      }
    }

    const swung = swing(self.angle, spin, pushed);
    let angle = swung.x;
    spin = swung.y;

    const on = dominoes[i + 1];
    const limit = on ? leanOn(on.angle) : Math.PI / 2;
    let hit = 0;
    if (angle >= limit) {
      hit = spin;
      angle = limit;
      spin = on ? Math.min(spin, on.spin) : 0;
    }
    return { angle, spin, hit };
  };

  return {
    get dominoes() {
      return dominoes;
    },
    get time() {
      return time;
    },
    step() {
      dominoes = dominoes.map((_, i) => next(i));
      time += STEP;
    },
  };
}
