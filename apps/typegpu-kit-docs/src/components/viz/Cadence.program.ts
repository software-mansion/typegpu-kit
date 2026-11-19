import { clock } from '@typegpu/kit';

export interface Options {
  /** Frames between runs of the cadence that counts frames. */
  readonly every?: number;
  /** Runs a second of the cadence that keeps time. */
  readonly hz?: number;
}

/** A run of the cadence that keeps time, on the frames' time. */
export interface Tick {
  /** The tick of its schedule the run answers, or its frame's end when the schedule starts there. */
  readonly at: number;
  /** The schedule's ticks it gave up, which no frame ran. */
  readonly missed: readonly number[];
}

export interface Ran {
  /** Whether the cadence that counts frames ran in the frame. */
  readonly counted: boolean;
  /** The run of the cadence that keeps time, if it ran in the frame. */
  readonly timed?: Tick;
}

/**
 * Two `clock.cadence`s asked about the same frames, `{ every }` and `{ hz }`, whose `due` decides
 * every run. The second keeps its schedule to itself, so this keeps a copy, summed from the same
 * frame lengths as the kit's clock.ts sums them: a run within half an interval of its tick moves
 * the tick an interval on, and any other run starts the schedule an interval after itself.
 */
export function createProgram({ every = 4, hz = 25 }: Options = {}) {
  const byFrames = clock.cadence({ every });
  const byRate = clock.cadence({ hz });
  const interval = 1 / hz;
  let elapsed = 0;
  let next = 0;
  let first = true;

  return {
    /** Asks both cadences whether they are due in the frame from `start` to `end`. */
    frame(start: number, end: number): Ran {
      const dt = end - start;
      elapsed += dt;
      const counted = byFrames.due(dt);
      if (!byRate.due(dt)) return { counted };

      // The copy sums from 0 at its first frame; the frames' time is that plus `origin`
      const origin = end - elapsed;
      const onSchedule = !first && Math.abs(elapsed - next) < interval / 2;
      const missed: number[] = [];
      if (!first && !onSchedule) {
        for (let tick = next; tick < elapsed; tick += interval) missed.push(origin + tick);
      }
      const at = origin + (onSchedule ? next : elapsed);
      next = onSchedule ? next + interval : elapsed + interval;
      first = false;
      return { counted, timed: { at, missed } };
    },

    /** Calls `reset()` on both, so each runs in the next frame and the schedule starts there. */
    reset() {
      byFrames.reset();
      byRate.reset();
      first = true;
    },
  };
}
