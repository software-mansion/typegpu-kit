import { machines } from '@typegpu/kit';

export const HORSES = 40;
export const STEP = 1 / 60;
/** Metres a second: a walking horse gallops above the first, a galloping one walks below the second. */
export const GALLOP_FROM = 5.2;
export const WALK_FROM = 4.2;
/** The fastest pace the figure shows, and how fast a bolting horse runs. */
export const FASTEST = 16;
const BOLT_SPEED = 15.5;
const WALK_SPEED = 1.7;
const GALLOP_SPEED = 11.5;
const RESPONSE_PER_SECOND = 1.6;
/** Urgency below which the herd is calm enough to graze. */
export const CALM_BELOW = 0.3;
/** Seconds an arrow stays lit after a horse took it. */
const LIT_SECONDS = 0.4;

export function createProgram(random: () => number = Math.random) {
  const pace = new Float64Array(HORSES);
  const own = Float64Array.from({ length: HORSES }, () => 0.85 + random() * 0.3);
  let urgency = 0.45;

  const faster = (speed: number) => (i: number) => pace[i] > speed;
  const slower = (speed: number) => (i: number) => pace[i] < speed;
  const calm = () => urgency < CALM_BELOW;
  const startled = { to: 'bolting', on: 'startle' } as const;

  const Gait = machines.define('Gait', {
    events: ['startle'],
    states: {
      walking: {
        next: [
          startled,
          { to: 'galloping', when: faster(GALLOP_FROM) },
          { to: 'grazing', when: calm, rate: 0.03 },
        ],
      },
      galloping: { next: [startled, { to: 'walking', when: slower(WALK_FROM) }] },
      grazing: { next: [startled, { to: 'walking', after: 4, rate: 1 / 3 }] },
      bolting: { next: [{ to: 'galloping', after: 2 }] },
    },
  });

  const horses = machines.run(Gait, { count: HORSES, random });
  const states = Gait.names.length;
  const before = new Uint8Array(HORSES);
  /** Seconds each arrow, `from * states + to`, stays lit. */
  const lit = new Float64Array(states * states);

  const cruise = () => WALK_SPEED + (GALLOP_SPEED - WALK_SPEED) * urgency;
  for (let i = 0; i < HORSES; i++) pace[i] = cruise() * own[i];

  return {
    Gait,
    horses,
    pace: pace as Readonly<Float64Array>,
    lit: lit as Readonly<Float64Array>,

    get urgency() {
      return urgency;
    },

    set urgency(value: number) {
      urgency = value;
    },

    /** Every horse takes `startle` at its next step. */
    startle() {
      for (let i = 0; i < HORSES; i++) horses.send.startle(i);
    },

    /** One step of the machines, then each pace eases towards what its state asks for. */
    step() {
      for (let i = 0; i < HORSES; i++) before[i] = horses.state(i);
      horses.step(STEP);
      for (let k = 0; k < lit.length; k++) lit[k] = Math.max(lit[k] - STEP, 0);

      const ease = 1 - Math.exp(-RESPONSE_PER_SECOND * STEP);
      for (let i = 0; i < HORSES; i++) {
        const state = horses.state(i);
        if (state !== before[i]) lit[before[i] * states + state] = LIT_SECONDS;
        let wanted = cruise() * own[i];
        if (state === Gait.grazing) wanted = 0;
        if (state === Gait.bolting) wanted = BOLT_SPEED;
        pace[i] += (wanted - pace[i]) * ease;
      }
    },

    /** How many horses are in each state, in the definition's order. */
    counts() {
      const counts = Array<number>(states).fill(0);
      for (let i = 0; i < HORSES; i++) counts[horses.state(i)] += 1;
      return counts;
    },
  };
}

export type Program = ReturnType<typeof createProgram>;
