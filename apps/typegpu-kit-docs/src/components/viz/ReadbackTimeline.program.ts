import { coroutines, frames, readback } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d } from 'typegpu';
import { cadenceOf, delayedMaps, KEPT } from './timeline.ts';

export interface Copy {
  readonly slot: number;
  /** The frame its map lands in. */
  readonly until: number;
  /** A read was waiting for it, and resolves as it lands. */
  readonly answers: boolean;
}

export interface Column {
  readonly index: number;
  copy: Copy | undefined;
  /** Every staging buffer was mapping. */
  readonly full: boolean;
  /** The rate or a waiting read wanted a copy, which a full frame skips. */
  readonly wanted: boolean;
  /** A read was asked for in this frame. */
  asked: boolean;
  /** Frames whose reads resumed here. */
  readonly resumed: number[];
  /** `latestFrame` after this frame, or -1. */
  latest: number;
  /** `latest` after this frame: the buffer holds the index of the frame that wrote it. */
  value: number | undefined;
}

export interface Options {
  /** Frames a staging map takes to land. Default 3. */
  readonly latency?: number;
  /** Staging buffers. Default 3. */
  readonly slots?: number;
  /** Captures a second without a read waiting, or 0 for none. Default 30. */
  readonly hz?: number;
}

export interface Program {
  /** The last KEPT frames, oldest first. */
  readonly columns: readonly Column[];
  /** Per slot, the frame its map lands in. */
  readonly busy: readonly number[];
  /** Starts a coroutine that reads, in the next frame. */
  ask(): void;
  /** Settles the maps that land now, then runs a frame. */
  step(): Promise<void>;
  destroy(): void;
}

export function createProgram({ latency = 3, slots = 3, hz = 30 }: Options = {}): Program {
  const recording = recordingRoot();
  const { root } = recording;
  const columns: Column[] = [];
  const wants = cadenceOf(hz > 0 ? hz : undefined);
  let asking = false;
  // A read waits from the frame after it was asked for until a capture copies, as `next()` does
  let waiting = false;
  let answering = false;

  const maps = delayedMaps(
    root.device,
    latency,
    (slot, until) => {
      columns.at(-1)!.copy = { slot, until, answers: answering };
      answering = false;
    },
    slots,
  );
  const { busy } = maps;

  const source = root.createBuffer(d.u32).$usage('storage');
  const sourceBytes = new Uint32Array(recording.bytes(root.unwrap(source)).buffer);
  const probe = readback.create(root, source, hz > 0 ? { staging: slots, hz } : { staging: slots });
  const scheduler = coroutines.create();

  const loop = frames.create(root, (frame) => {
    const full = busy.every((until) => until > frame.index);
    const rate = full ? wants.full(frame.index) : wants.free(frame.index);
    const column: Column = {
      index: frame.index,
      copy: undefined,
      full,
      wanted: rate || waiting,
      asked: false,
      resumed: [],
      latest: -1,
      value: undefined,
    };
    columns.push(column);

    sourceBytes[0] = frame.index; // the pass that writes the buffer
    probe.capture(frame);
    if (waiting && !full) {
      answering = true;
      waiting = false;
    }

    // A coroutine that reads, started now. This frame's capture is already recorded, so the next
    // frame's captures for it
    if (asking) {
      asking = false;
      waiting = true;
      column.asked = true;
      const asked = frame.index;

      scheduler.start(
        (function* () {
          yield* coroutines.until(probe.next());
          columns.at(-1)!.resumed.push(asked);
        })(),
      );
    }

    scheduler.run(frame);
    column.latest = probe.latestFrame;
    column.value = probe.latest;
    if (columns.length > KEPT) columns.shift();
  });

  return {
    columns,
    busy,

    ask() {
      asking = true;
    },

    step: () => maps.step(loop),

    destroy() {
      scheduler.destroy();
      loop.destroy();
      probe.destroy();
      source.destroy();
    },
  };
}
