import { coroutines, frames, type Deferred } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';

export const FRAMES = 36;
export const LANES = ['game', 'pick', 'spawner', 'loader'] as const;
const [GAME, PICK, SPAWNER, LOADER] = [0, 1, 2, 3];

/** How the figure stops the coroutines: cancelling `game`, or destroying the scheduler. */
export type Stop = 'cancel' | 'destroy';

export interface Run {
  /** Chunks of work per frame, by lane; most coroutines run once in a frame they resume in. */
  readonly runs: number[][];
  /** Wait labels by lane and the frame the wait began. */
  readonly waits: { lane: number; frame: number; label: string }[];
  /** Events written beside a lane: a return or a finally. */
  readonly notes: { lane: number; frame: number; label: string; bad: boolean }[];
  /** Last frame each lane was alive in, or FRAMES if it never ended. */
  readonly ends: number[];
}

export interface Options {
  /** The frame to stop in; none by default. */
  readonly stopAt?: number | undefined;
  readonly stop?: Stop;
}

/** Runs the four coroutines over FRAMES frames, stopping them at `stopAt`, and records what happened. */
export function createProgram({ stopAt, stop = 'cancel' }: Options = {}): Run {
  const record: Run = {
    runs: LANES.map(() => Array.from({ length: FRAMES }, () => 0)),
    waits: [],
    notes: [],
    ends: LANES.map(() => FRAMES),
  };

  let current = 0;
  let clock = 0;
  let stopping = false;
  const reads: { request: Deferred<number>; at: number }[] = [];

  const ran = (lane: number) => (record.runs[lane][current] += 1);
  const note = (lane: number, label: string, bad = false) =>
    record.notes.push({ lane, frame: current, label, bad });

  const on = <T>(lane: number, label: string, wait: T) => {
    record.waits.push({ lane, frame: current, label });
    return wait;
  };

  const end = (lane: number) => (record.ends[lane] = current);

  const read = () => {
    const request = coroutines.deferred<number>();
    reads.push({ request, at: current + 3 });
    return request;
  };

  const scheduler = coroutines.create({ now: () => clock });

  function* pick() {
    try {
      ran(PICK);
      const id = yield* on(PICK, 'read()', read());
      ran(PICK);
      note(PICK, `return ${id}`);
      return id;
    } finally {
      if (stopping) note(PICK, 'finally', true);
      end(PICK);
    }
  }

  function* game() {
    try {
      ran(GAME);
      yield* on(GAME, 'seconds(0.1)', coroutines.seconds(0.1));

      ran(GAME);
      const id = yield* on(GAME, 'pick', scheduler.child(pick()));
      note(GAME, `got ${id}`);

      for (;;) {
        ran(GAME);
        yield* on(GAME, 'nextFrame', coroutines.nextFrame);
      }
    } finally {
      if (stopping) note(GAME, 'finally', true);
      end(GAME);
    }
  }

  function* spawner() {
    const pause = coroutines.frames(4);

    try {
      for (;;) {
        ran(SPAWNER);
        yield* on(SPAWNER, 'frames(4)', pause);
      }
    } finally {
      if (stopping) note(SPAWNER, 'finally', true);
      end(SPAWNER);
    }
  }

  function* loader() {
    const slice = coroutines.budget(2);

    try {
      for (let chunk = 0; chunk < 12; chunk++) {
        ran(LOADER);
        clock += 0.7;
        yield* on(LOADER, 'budget(2)', slice);
      }
      note(LOADER, 'loaded');
    } finally {
      if (stopping) note(LOADER, 'finally', true);
      end(LOADER);
    }
  }

  let main: { cancel(): void } | undefined;

  const { root } = recordingRoot();
  const loop = frames.create(root, (frame) => {
    current = frame.index;
    clock = frame.index * (1000 / 60);

    // The map lands between frames; the coroutine resumes at the next run
    for (const { request, at } of reads) if (at === current) request.resolve(7);

    if (current === 0) {
      main = scheduler.start(game());
      scheduler.start(spawner());
    }
    if (current === 14 && !(stop === 'destroy' && stopAt !== undefined && stopAt < 14)) {
      scheduler.start(loader());
    }

    if (current === stopAt) {
      stopping = true;
      if (stop === 'cancel') main!.cancel();
      else scheduler.destroy();
      stopping = false;
    }

    scheduler.run(frame);
  });

  for (let i = 0; i < FRAMES; i++) loop.run(i / 60);

  loop.destroy();
  return record;
}
