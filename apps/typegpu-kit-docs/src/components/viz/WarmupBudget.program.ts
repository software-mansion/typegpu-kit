import { pipelines } from '@typegpu/kit';
import { seeded } from './svg.ts';

export const VSYNC = 1000 / 60;
/** Milliseconds the loading screen itself takes each frame. */
export const COVER = 1.5;
/** Compiles that run at once. */
export const THREADS = 2;

export interface Start {
  readonly at: number;
  readonly end: number;
}

export interface Compile {
  readonly lane: number;
  readonly begin: number;
  readonly end: number;
}

export interface FrameRecord {
  readonly start: number;
  /** When the next frame starts: the next display refresh after this one's work. */
  readonly next: number;
  readonly starts: readonly Start[];
  /** Milliseconds of work in the frame, the loading screen's with the starts'. */
  readonly busy: number;
}

export interface Run {
  readonly frames: readonly FrameRecord[];
  readonly compiles: readonly Compile[];
  readonly progress: readonly { at: number; fraction: number }[];
  /** When the warm-up finished. */
  readonly ready: number;
}

const random = seeded(5);
const costs = Array.from({ length: 10 }, () => ({
  start: 1 + random() * 3,
  compile: 12 + random() * 30,
}));

/** Runs the microtasks a settled promise queues. */
const settle = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};

/**
 * Compiles the items with `budget` milliseconds of starts between yields, a frame drawing at each
 * yield, and records what happened.
 */
export async function createProgram({ budget = 8 } = {}): Promise<Run> {
  let clock = 0;
  const lanes = Array.from({ length: THREADS }, () => 0);
  const compiles: Compile[] = [];
  const progress: { at: number; fraction: number }[] = [];
  const landing: { end: number; resolve: () => void }[] = [];
  let starts: Start[] = [];

  const items: pipelines.Warmable[] = costs.map((cost) => ({
    initAsync() {
      const at = clock;
      clock += cost.start; // resolving WGSL and creating the module, on the main thread
      starts.push({ at, end: clock });

      const lane = lanes.indexOf(Math.min(...lanes));
      const begin = Math.max(clock, lanes[lane]);
      const end = begin + cost.compile;
      lanes[lane] = end;
      compiles.push({ lane, begin, end });

      return new Promise<void>((resolve) => landing.push({ end, resolve }));
    },
  }));

  // `compile` reads the time from `performance.now` and yields through `setTimeout`. Only
  // microtasks run until both are given back, so nothing else sees them
  const { now } = performance;
  const { setTimeout } = globalThis;
  let resume: (() => void) | undefined;
  const state = { done: false };
  performance.now = () => clock;
  globalThis.setTimeout = ((then: () => void) => {
    resume = then;
    return 0;
  }) as typeof globalThis.setTimeout;

  const records: FrameRecord[] = [];
  let start = 0;

  try {
    starts = [];
    const compiled = pipelines.compile(items, {
      budget,
      onProgress: (fraction) => progress.push({ at: clock, fraction }),
    });
    void compiled.then(() => (state.done = true));

    while (!state.done && records.length < 60) {
      // Compiles that finish between frames resolve there, before the next frame runs
      landing.sort((a, b) => a.end - b.end);
      while (landing.length > 0 && landing[0].end <= start) {
        const compile = landing.shift()!;
        clock = compile.end;
        compile.resolve();
        await settle();
      }

      // The first starts run before the first frame; later ones after a yield, in a frame's time
      if (records.length > 0) {
        clock = start;
        starts = [];
        const next = resume;
        resume = undefined;
        next?.();
        await settle();
      }

      const busy = clock - start + COVER;
      const next = start + Math.max(1, Math.ceil(busy / VSYNC)) * VSYNC;
      records.push({ start, next, starts, busy });
      start = next;
    }
  } finally {
    performance.now = now;
    globalThis.setTimeout = setTimeout;
  }

  return { frames: records, compiles, progress, ready: records.at(-1)!.start };
}
