import { counters, frames } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { cadenceOf, delayedMaps, KEPT } from './timeline.ts';

const NAMES = ['stomps'] as const;
type Counts = counters.Counts<(typeof NAMES)[number]>;

export interface Copy {
  readonly slot: number;
  /** The frame its map lands in. */
  readonly until: number;
  /** A `clear` came after it and before it landed, so `take` and `latest` never see it. */
  dropped: boolean;
}

export interface Column {
  readonly index: number;
  /** `clear` ran in this frame, before the kernel. */
  readonly cleared: boolean;
  /** The counters on the GPU after this frame's passes. */
  counters: Counts;
  /** The capture copied the counters. */
  copy: Copy | undefined;
  /** Every staging buffer was mapping. */
  readonly full: boolean;
  /** The rate wanted a copy, which a full frame skips. */
  readonly wanted: boolean;
  /** What `take()` returned at the top of this frame. */
  readonly taken: Counts | undefined;
  /** `latest` after this frame. */
  latest: Counts | undefined;
}

export interface Options {
  /** Frames a staging map takes to land. Default 3. */
  readonly latency?: number;
  /** Reads a second. Default 30. */
  readonly hz?: number;
}

export interface Program {
  /** The last KEPT frames, oldest first. */
  readonly columns: readonly Column[];
  /** Adds `amount` to the counter in the next frame, as a kernel would. */
  add(name: 'stomps', amount: number): void;
  /** Clears the counters in the next frame. */
  clear(): void;
  /** Settles the maps that land now, then runs a frame. */
  step(): Promise<void>;
  destroy(): void;
}

const counts = (stomps = 0): Counts => ({ stomps });

export function createProgram({ latency = 3, hz = 30 }: Options = {}): Program {
  const recording = recordingRoot();
  const { root } = recording;
  const columns: Column[] = [];
  const copies: Copy[] = [];
  const wants = cadenceOf(hz);
  let clearing = false;
  let pending = 0;

  const maps = delayedMaps(root.device, latency, (slot, until) => {
    const copy: Copy = { slot, until, dropped: false };
    copies.push(copy);
    columns.at(-1)!.copy = copy;
  });
  const { busy } = maps;

  const tally = counters.create(root, NAMES, { hz });
  const stored = new Uint32Array(recording.bytes(root.unwrap(tally.buffer)).buffer);

  const loop = frames.create(root, (frame) => {
    const taken = tally.take();

    // A clear lands at the submit, after the stand-in kernel wrote its adds, so a frame that
    // clears leaves its adds to the next one
    const cleared = clearing;
    clearing = false;
    if (cleared) {
      tally.clear(frame);
      for (const copy of copies) if (copy.until > frame.index) copy.dropped = true;
    } else {
      stored[0] = (stored[0] + pending) >>> 0;
    }

    const full = busy.length > 0 && busy.every((until) => until > frame.index);
    const column: Column = {
      index: frame.index,
      cleared,
      counters: counts(),
      copy: undefined,
      full,
      wanted: full ? wants.full(frame.index) : wants.free(frame.index),
      taken: taken && counts(taken.stomps),
      latest: undefined,
    };
    columns.push(column);
    if (columns.length > KEPT) columns.shift();
    if (!cleared) pending = 0;

    tally.capture(frame);
    frame.afterSubmit(() => {
      const { latest } = tally;
      column.counters = counts(stored[0]);
      column.latest = latest && counts(latest.stomps);
    });
  });

  return {
    columns,

    add(_name, amount) {
      pending += amount;
    },

    clear() {
      clearing = true;
    },

    step: () => maps.step(loop),

    destroy() {
      loop.destroy();
      tally.destroy();
    },
  };
}
