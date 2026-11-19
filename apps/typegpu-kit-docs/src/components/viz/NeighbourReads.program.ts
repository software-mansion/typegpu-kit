import { frames, table } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d } from 'typegpu';
import { floatsOf } from './program.ts';
import { seeded } from './svg.ts';

export const ROWS = 8;
const FRAME_TIME = 1 / 60;

const Cell = table.define('Cell', { device: { value: d.f32 } });

export type Reads = 'in place' | 'copy';

/** One thread of the kernel, which gives its row the value of the row on its left. */
export interface Thread {
  /** The row it writes. */
  readonly row: number;
  /** The row it reads, on the left of its own, the last row for the first. */
  readonly from: number;
  /** The value it read, and wrote. */
  readonly value: number;
  /** Whether an earlier thread had written the row it read. */
  readonly late: boolean;
}

export interface Step {
  /** The rows as the step began, which the table keeps as last frame's. */
  readonly before: readonly number[];
  /** The threads in the order they ran. */
  readonly threads: readonly Thread[];
}

export interface Program {
  /** Where the threads read their neighbour: the rows they write, or last frame's rows. */
  reads: Reads;
  /** The last step. */
  readonly step: Step;
  /** The current rows. */
  readonly rows: readonly number[];
  /** Draws another order for the threads. */
  shuffle(): void;
  /** Runs a step on the rows the last one left. */
  advance(): void;
  /** Puts back the first rows and runs a step on them, as after a change of reads or order. */
  restart(): void;
  destroy(): void;
}

export function createProgram(): Program {
  const random = seeded(2);
  const { root, bytes } = recordingRoot();

  const cells = table.create(root, Cell, { capacity: ROWS, history: true });
  const loop = frames.create(root, (frame) => cells.flush(frame));
  const ids = Array.from({ length: ROWS }, (_, i) => cells.add({ value: i + 1 }));

  // Last frame's rows follow the current ones in the same buffer
  const held = floatsOf(bytes(cells.buffers.device!));
  const current = held.subarray(0, ROWS);
  const previous = held.subarray(ROWS, ROWS * 2);

  // The rows in the order their threads run
  const order = Array.from({ length: ROWS }, (_, i) => i);
  let index = 0;
  let step: Step = { before: [], threads: [] };
  const frame = () => loop.run(index++ * FRAME_TIME);

  const run = () => {
    // The flush copies the rows aside before any thread runs
    frame();
    const before = Array.from(previous);
    const written = new Set<number>();

    const threads = order.map((row): Thread => {
      const from = (row + ROWS - 1) % ROWS;
      const late = program.reads === 'in place' && written.has(from);
      const read = (program.reads === 'copy' ? previous : current)[from];
      // The recording root runs no kernels, so the thread's write goes into the buffer's bytes
      current[row] = read;
      written.add(row);
      return { row, from, value: read, late };
    });
    step = { before, threads };
  };

  const program: Program = {
    reads: 'in place',

    get step() {
      return step;
    },

    get rows() {
      return Array.from(current);
    },

    shuffle() {
      for (let i = ROWS - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    },

    advance: run,

    restart() {
      // A device write lands in the next flush, over what the threads left
      ids.forEach((id, i) => cells.fields.value.set(id, i + 1));
      frame();
      run();
    },

    destroy() {
      loop.destroy();
      cells.destroy();
    },
  };

  program.shuffle();
  frame();
  run();
  return program;
}
