import { frames, history } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, type TgpuBuffer, type UniformFlag } from 'typegpu';
import { floatsOf } from './program.ts';

/** The stage the dot moves on, in the units its position is written in. */
export const STAGE = { width: 200, height: 92 } as const;
/** The loop the dot goes round, on the stage. */
export const LOOP = { x: 100, y: 46, rx: 78, ry: 34 } as const;
/** The positions on the loop. Each write moves the dot to the next. */
export const NOTCHES = 12;
const FRAME_TIME = 1 / 60;

export type Writes = 'each frame' | 'on click';
export type Point = readonly [x: number, y: number];

export interface State {
  readonly current: Point;
  readonly previous: Point;
}

export interface Snapshot {
  readonly index: number;
  /** What the frame wrote before its copies, if it wrote. */
  readonly written: Point | undefined;
  /** `history.create`'s state, as the frame's passes read it. */
  readonly kept: State;
  /** The buffers written straight into `current`, as the frame's passes read them. */
  readonly straight: State;
}

export interface Program {
  /** When a frame writes the dot's next position. Starts each frame. */
  writes: Writes;
  /** Writes the dot's next position in the next frame. */
  write(): void;
  /** Runs one frame and reads what it left in the buffers. */
  step(): Snapshot;
  destroy(): void;
}

/** A position on the loop, which the dot goes round clockwise from the left end. */
export const at = (notch: number): Point => {
  const turn = Math.PI * (1 + (2 * notch) / NOTCHES);
  return [
    Math.round(LOOP.x + LOOP.rx * Math.cos(turn)),
    Math.round(LOOP.y + LOOP.ry * Math.sin(turn)),
  ];
};

/** How far the state says the dot moved this frame. */
export const moved = ({ current, previous }: State) =>
  Math.hypot(current[0] - previous[0], current[1] - previous[1]);

export function createProgram(): Program {
  const { root, bytes } = recordingRoot();
  const read = (buffer: TgpuBuffer<d.Vec2f>): Point => {
    const [x, y] = floatsOf(bytes(buffer));
    return [x, y];
  };

  const kept = history.create(root, d.vec2f, { usage: 'uniform' });
  const make = (): TgpuBuffer<d.Vec2f> & UniformFlag =>
    root.createBuffer(d.vec2f).$usage('uniform');
  const straight = { current: make(), previous: make() };

  let index = 0;
  let notch = 0;
  let asked = false;
  let written: Point | undefined;

  const loop = frames.create(root, (frame) => {
    if (written) {
      const position = d.vec2f(...written);
      kept.write(position);
      straight.current.write(position);
    }
    kept.record(frame);
    straight.previous.copyFrom(straight.current, frame.encoder);
  });

  const program: Program = {
    writes: 'each frame',

    write() {
      asked = true;
    },

    step() {
      written = program.writes === 'each frame' || asked ? at(notch++) : undefined;
      asked = false;
      loop.run(index * FRAME_TIME);

      return {
        index: index++,
        written,
        kept: { current: read(kept.current), previous: read(kept.previous) },
        straight: { current: read(straight.current), previous: read(straight.previous) },
      };
    },

    destroy() {
      loop.destroy();
      kept.destroy();
      straight.current.destroy();
      straight.previous.destroy();
    },
  };

  return program;
}
