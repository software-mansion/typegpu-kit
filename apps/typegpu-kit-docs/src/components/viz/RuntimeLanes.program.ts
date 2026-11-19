import { frames } from '@typegpu/kit';
import { seeded } from './svg.ts';

/** Milliseconds between two vsyncs at 120 Hz. */
export const VSYNC = 1000 / 120;

/** Where `useScene` runs the scene, as its `on` takes it. */
export type Placement = 'worker' | 'ui' | 'js';
export type Lane = 'js' | 'ui' | 'worker';
export const LANES: readonly Lane[] = ['js', 'ui', 'worker'];

/** The relay's own work on the UI runtime at each vsync, in milliseconds. */
const RELAY = 0.1;
/** A closure sent from a gesture, from the UI runtime to the scene's runtime. */
const HOP = 0.4;
/** A sent closure's own work. */
const CLOSURE = 0.3;
/** History duration in milliseconds. */
const KEEP = 1000;

export interface Block {
  readonly lane: Lane;
  /** The scene's frame, the app's own UI work, the relay's callback or a sent closure. */
  readonly kind: 'scene' | 'app' | 'relay' | 'send';
  readonly start: number;
  readonly end: number;
  /** A scene frame's `dt` from `animate`, in seconds. */
  dt?: number;
  /** A scene frame that is the first to start after a sent closure ran. */
  reads?: boolean;
}

export interface Mark {
  readonly lane: Lane;
  /** The app's UI frame of the vsync at `at` was late, or the relay counted it in `missed()`. */
  readonly kind: 'dropped' | 'missed';
  readonly at: number;
}

export interface Send {
  /** When the gesture sent it, on the UI runtime. */
  readonly at: number;
  readonly arrives: number;
  /** When it ran on the scene's runtime. */
  ran?: number;
  /** When the first frame after it started. */
  read?: number;
}

export interface Stats {
  /** Scene frames a second. */
  readonly hz: number;
  /** Vsyncs, each an app UI frame. */
  readonly shown: number;
  readonly dropped: number;
  /** What the relay's `missed()` counted. */
  readonly missed: number;
}

export interface Options {
  /** Default `'worker'`. */
  readonly on?: Placement;
  /** Simulated scene-frame duration in milliseconds. Default 3.7. */
  readonly frame?: number;
  /** The app's own UI work at each vsync, in milliseconds. Default 3. */
  readonly ui?: number;
}

export interface Program {
  readonly on: Placement;
  /** Milliseconds since it started. */
  readonly now: number;
  readonly blocks: readonly Block[];
  readonly marks: readonly Mark[];
  readonly sends: readonly Send[];
  /** Read at each subsequent frame and vsync. */
  frame: number;
  ui: number;
  /** Queues a gesture closure from the UI runtime. */
  send(): void;
  /** Advances the model by `ms` milliseconds. */
  advance(ms: number): void;
  /** Over the last `ms` milliseconds. */
  stats(ms: number): Stats;
  destroy(): void;
}

export function createProgram(options: Options = {}): Program {
  const on = options.on ?? 'worker';
  const lane: Lane = on;
  const random = seeded(7);

  const blocks: Block[] = [];
  const marks: Mark[] = [];
  const sends: Send[] = [];
  /** When each lane is free again. */
  const free: Record<Lane, number> = { js: 0, ui: 0, worker: 0 };
  let now = 0;
  let vsync = 0;
  /** The frame the relay sent the worker, until it ends. */
  let running = -Infinity;

  let wanted: ((milliseconds: number) => void) | undefined;
  let requests = 0;
  const schedule: frames.Schedule = {
    request(callback) {
      wanted = callback;
      return ++requests;
    },
    cancel(request) {
      if (request === requests) wanted = undefined;
    },
  };

  let current: Block | undefined;
  let seconds: number | undefined;
  const loop = {
    run(time: number) {
      if (current && seconds !== undefined) current.dt = time - seconds;
      seconds = time;
      return true;
    },
  };
  const animation = frames.animate(loop, { schedule });

  const program: Program = {
    on,
    get now() {
      return now;
    },
    blocks,
    marks,
    sends,
    frame: options.frame ?? 3.7,
    ui: options.ui ?? 3,

    send() {
      sends.push({ at: now, arrives: now + HOP });
    },

    advance(ms) {
      const until = now + ms;
      for (;;) {
        const at = vsync * VSYNC;
        const arriving = sends.find((sent) => sent.ran === undefined && sent.arrives <= until);
        if (arriving && arriving.arrives <= at) {
          arrive(arriving);
          continue;
        }
        if (at > until) break;
        tick(at);
        vsync += 1;
      }

      now = until;
      const old = now - KEEP;
      prune(blocks, (block) => block.end < old);
      prune(marks, (mark) => mark.at < old);
      prune(sends, (sent) => (sent.read ?? Infinity) < old);
    },

    stats(ms) {
      const from = now - ms;
      const within = (at: number) => at > from && at <= now;
      const scenes = blocks.filter((block) => block.kind === 'scene' && within(block.start));
      const count = (kind: Mark['kind']) =>
        marks.filter((mark) => mark.kind === kind && within(mark.at)).length;
      const span = Math.min(ms, now);

      return {
        hz: span > 0 ? scenes.length / (span / 1000) : 0,
        shown: Math.floor(now / VSYNC) - Math.floor(Math.max(from, 0) / VSYNC),
        dropped: count('dropped'),
        missed: count('missed'),
      };
    },

    destroy: () => animation.destroy(),
  };

  /** Schedules work no earlier than `at` and after pending work on the lane. */
  function job(where: Lane, kind: Block['kind'], at: number, ms: number) {
    const start = Math.max(at, free[where]);
    const block: Block = { lane: where, kind, start, end: start + ms };
    blocks.push(block);
    free[where] = block.end;
    return block;
  }

  /** Vary simulated frame duration, with occasional longer frames. */
  function cost() {
    const spike = random() < 0.03;
    const tail = lane === 'ui' ? 2 + random() * 0.8 : 1.3 + random() * 0.2;
    return program.frame * (spike ? tail : 0.92 + random() * 0.16);
  }

  /** Runs the frame `animate` asked for, on the scene's lane, from `at`. */
  function fire(at: number, vsyncAt: number) {
    const callback = wanted;
    if (!callback) return;
    wanted = undefined;

    const frame = job(lane, 'scene', at, cost());
    running = frame.end;
    for (const sent of sends) {
      if (sent.ran === undefined || sent.read !== undefined || sent.ran > frame.start) continue;
      sent.read = frame.start;
      frame.reads = true;
    }

    current = frame;
    callback(vsyncAt);
    current = undefined;
  }

  /** Runs UI callbacks each vsync and JS callbacks on alternate vsyncs. */
  function tick(at: number) {
    if (free.ui > at) {
      // Still busy from the last vsync: its callbacks wait, and the app shows no new frame
      marks.push({ lane: 'ui', kind: 'dropped', at });
    } else {
      let from = at;
      if (on === 'worker') {
        from = job('ui', 'relay', at, RELAY).end;
        if (wanted && running > at) marks.push({ lane: 'worker', kind: 'missed', at });
        else if (wanted) fire(from, at);
      }

      const app = job('ui', 'app', from, program.ui);
      if (app.end > at + VSYNC) marks.push({ lane: 'ui', kind: 'dropped', at });
      if (on === 'ui') fire(app.end, at);
    }

    if (on === 'js' && vsync % 2 === 0 && free.js <= at) fire(at, at);
  }

  /** A sent closure runs once the scene's runtime is free, after what was sent before it. */
  function arrive(sent: Send) {
    sent.ran = job(lane, 'send', sent.arrives, CLOSURE).start;
  }

  return program;
}

function prune<T>(list: T[], old: (item: T) => boolean) {
  let first = 0;
  while (first < list.length && old(list[first])) first++;
  list.splice(0, first);
}
