import { coroutines, frames } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';

/** The page's openDoor, as the figure shows it; the marks below count its lines from 1. */
export const CODE = [
  'function* openDoor() {',
  '  yield* coroutines.seconds(0.5);',
  '',
  '  while (door.open < 1) {',
  '    const frame = yield* coroutines.nextFrame;',
  '    door.open = Math.min(door.open + frame.dt * 2, 1);',
  '  }',
  '}',
];

export const FPS = 8;
export const FRAMES = 16;

/** What one frame's run did, for the code, the strip and the status line. */
export interface Step {
  readonly frame: number;
  readonly ran: readonly number[];
  readonly paused: number | undefined;
  /** The line it was paused on when this frame's run began. */
  readonly resumed: number | undefined;
  readonly open: number;
  readonly started: boolean;
  readonly done: boolean;
}

export function createProgram() {
  const door = { open: 0 };
  const steps: Step[] = [];
  let ran: number[] = [];
  let paused: number | undefined;
  let pressed = false;
  let started = false;
  let done = false;

  function* openDoor() {
    started = true;
    ran.push(1, 2);
    paused = 2;
    yield* coroutines.seconds(0.5);

    ran.push(4);
    while (door.open < 1) {
      ran.push(5);
      paused = 5;
      const frame = yield* coroutines.nextFrame;

      ran.push(6);
      door.open = Math.min(door.open + frame.dt * 2, 1);
      ran.push(4);
    }

    ran.push(8);
    paused = undefined;
    done = true;
  }

  const scheduler = coroutines.create();
  const loop = frames.create(recordingRoot().root, (frame) => {
    const resumed = paused;
    ran = [];
    scheduler.run(frame);
    steps.push({ frame: frame.index, ran, paused, resumed, open: door.open, started, done });
  });

  return {
    steps: steps as readonly Step[],

    get pressed() {
      return pressed;
    },

    /** Whether it ran every frame it shows. */
    get finished() {
      return steps.length >= FRAMES;
    },

    /** The switch: an input event, so the coroutine starts at the next run. */
    press() {
      if (pressed) return;
      pressed = true;
      scheduler.start(openDoor());
    },

    /** Runs the next frame, unless it ran them all. */
    step() {
      if (steps.length < FRAMES) loop.run(steps.length / FPS);
    },

    destroy() {
      scheduler.destroy();
      loop.destroy();
    },
  };
}
