import { clock } from '@typegpu/kit';

/** Seconds below which two times count as the same. */
const SAME = 1e-9;

export interface Options {
  /** The step's length in seconds. */
  readonly step?: number;
  readonly maxSteps?: number;
}

/** What a frame did with its time, laid along the frames' own time. */
export interface Ran {
  /** Where the time each step `update` returned for the frame ends, oldest first. */
  readonly steps: readonly number[];
  /** Where the time those steps took starts. */
  readonly from: number;
  /** From here to `dropped`, the frame's own time that later frames step. */
  readonly carried: number;
  /** From here to the frame's end, the time the clock dropped. */
  readonly dropped: number;
  /** Whether `maxSteps` kept the frame from running a step it had the time for. */
  readonly capped: boolean;
}

/**
 * A `clock.fixedStep` run over frames, with the time it has not stepped kept as spans of the
 * frames' time. The clock gives the step count, its debt and so the time it dropped. Where that
 * time lies is the figure's own reading: steps take the oldest time first, and the clock drops
 * the newest.
 */
export function createProgram({ step = 1 / 60, maxSteps = 4 }: Options = {}) {
  const sim = clock.fixedStep({ step, maxSteps });
  /** The time not stepped yet, oldest first. */
  const owed: [from: number, to: number][] = [];

  return {
    sim,

    /** The time not stepped yet, in seconds, which is the clock's `debt`. */
    get owed() {
      return owed.reduce((sum, [from, to]) => sum + to - from, 0);
    },

    /** Advances the clock by the frame from `start` to `end`. */
    frame(start: number, end: number): Ran {
      const before = sim.debt;
      const count = sim.update(end - start);
      const lost = Math.max(before + end - start - count * sim.step - sim.debt, 0);

      const own: [number, number] = [start, end];
      owed.push(own);
      const from = owed[0][0];

      const steps: number[] = [];
      for (let i = 0; i < count; i++) {
        let left = sim.step;
        let at = from;
        while (left > SAME && owed.length > 0) {
          const span = owed[0];
          const taken = Math.min(left, span[1] - span[0]);
          span[0] += taken;
          left -= taken;
          at = span[0];
          if (span[1] - span[0] < SAME) owed.shift();
        }
        steps.push(at);
      }

      const dropped = end - Math.min(lost, own[1] - own[0]);
      own[1] = dropped;
      if (owed.at(-1) === own && own[1] - own[0] < SAME) owed.pop();

      return {
        steps,
        from,
        carried: Math.min(own[0], dropped),
        dropped,
        capped: count === maxSteps && sim.debt > sim.step - SAME,
      };
    },
  };
}
