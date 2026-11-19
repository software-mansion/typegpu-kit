import { afterEach, expect, it, vi } from 'vitest';
import { watchFrames } from '../src/animate.ts';
import { frames } from '../src/index.ts';
import { recordingRoot, steppedSchedule } from '../src/testing.ts';

afterEach(() => {
  vi.unstubAllGlobals();
});

/** The page's animation frames, run by hand, so the default schedule is the one tested. */
function handFrames() {
  const display = steppedSchedule();
  vi.stubGlobal('requestAnimationFrame', display.request);
  vi.stubGlobal('cancelAnimationFrame', display.cancel);
  return display;
}

it('runs the loop at every frame until destroyed, even from inside a frame', () => {
  const display = handFrames();
  const { root } = recordingRoot();

  const times: number[] = [];
  const loop = frames.create(root, ({ time }) => {
    times.push(time);
    if (time >= 0.125) animation.destroy();
  });
  const animation = frames.animate(loop);

  display.step(500, 562.5, 625, 687.5);

  expect(times).toEqual([0, 0.0625, 0.125]);
  expect(display.waiting).toBe(false);
  expect(loop.run(3)).toBe(true);
});

it('adds at most maxDt a frame, a tenth of a second by default', () => {
  const display = handFrames();
  const { root } = recordingRoot();

  const seen: number[] = [];
  const loop = frames.create(root, ({ dt }) => seen.push(dt));
  const animation = frames.animate(loop);
  display.step(0, 62.5, 60_000);
  animation.destroy();

  const capped: number[] = [];
  const other = frames.create(root, ({ dt }) => capped.push(dt));
  frames.animate(other, { maxDt: 0.25 });
  display.step(1000, 1125, 5000);

  expect(seen.slice(0, 2)).toEqual([0, 0.0625]);
  expect(seen[2]).toBeCloseTo(0.1);
  expect(capped).toEqual([0, 0.125, 0.25]);
});

it('runs no frames while the surface is paused, and leaves the pause out of the time', () => {
  const display = handFrames();
  const { root } = recordingRoot();
  const surface = { paused: false };

  const seen: [number, number][] = [];
  const loop = frames.create(root, ({ time, dt }) => seen.push([time, dt]));
  const animation = frames.animate(loop, { surface });

  display.step(1000, 1062.5);
  surface.paused = true;
  display.step(2000, 60_000);
  surface.paused = false;
  display.step(61_000, 61_062.5);

  expect(seen).toEqual([
    [0, 0],
    [0.0625, 0.0625],
    [0.0625, 0],
    [0.125, 0.0625],
  ]);

  animation.destroy();
  expect(display.waiting).toBe(false);
});

it('keeps time nondecreasing for a loop animated again', () => {
  const display = handFrames();
  const { root } = recordingRoot();

  const times: number[] = [];
  const loop = frames.create(root, ({ time }) => times.push(time));
  const first = frames.animate(loop, { maxDt: 0.25 });
  display.step(1000, 1062.5, 9000);
  first.destroy();

  frames.animate(loop);
  display.step(20_000);

  // The second animation starts where the loop stopped, without the gap between them.
  expect(times).toEqual([0, 0.0625, 0.3125, 0.3125]);
});

it('ends the animation at a frame that throws, rethrowing without onError', () => {
  const display = handFrames();
  const { root } = recordingRoot();

  const failure = new Error('lost');
  const loop = frames.create(root, () => {
    throw failure;
  });

  const onError = vi.fn();
  frames.animate(loop, { onError });
  display.step(0);

  expect(onError).toHaveBeenCalledExactlyOnceWith(failure);
  expect(display.waiting).toBe(false);

  frames.animate(loop);
  expect(() => display.step(16)).toThrow(failure);
  expect(display.waiting).toBe(false);
});

it('takes frames from the schedule it is given', () => {
  let requested: ((milliseconds: number) => void) | undefined;
  const schedule = {
    request: vi.fn((callback: (milliseconds: number) => void) => {
      requested = callback;
      return 7;
    }),
    cancel: vi.fn(),
  };
  vi.stubGlobal('requestAnimationFrame', () => {
    throw new Error('The default schedule ran.');
  });

  const { root } = recordingRoot();
  const times: number[] = [];
  const loop = frames.create(root, ({ time }) => times.push(time));
  const animation = frames.animate(loop, { schedule });

  requested?.(1000);
  requested?.(1062.5);
  animation.destroy();

  expect(times).toEqual([0, 0.0625]);
  expect(schedule.cancel).toHaveBeenCalledWith(7);
});

it('runs each unpaused frame through a profiler’s watch, which still lets errors stop it', () => {
  const display = steppedSchedule();
  const { root } = recordingRoot();
  const watched: number[] = [];
  watchFrames((milliseconds, run) => {
    watched.push(milliseconds);
    run(milliseconds);
  });

  try {
    const times: number[] = [];
    const surface = { paused: false };
    const errors: unknown[] = [];
    const loop = frames.create(root, ({ time }) => {
      if (time > 0.1) throw new Error('broken frame');
      times.push(time);
    });
    frames.animate(loop, { schedule: display, surface, onError: (error) => errors.push(error) });

    display.step(100, 150);
    surface.paused = true;
    display.step(200);
    surface.paused = false;
    display.step(300, 400);

    expect(watched).toEqual([100, 150, 300, 400]);
    expect(times.map((time) => Math.round(time * 100) / 100)).toEqual([0, 0.05, 0.05]);
    expect(errors).toEqual([new Error('broken frame')]);
  } finally {
    watchFrames(undefined);
  }
});

it('lets a watch run several frames in one, and adds no time when the display falls behind', () => {
  const display = steppedSchedule();
  const { root } = recordingRoot();
  // As a measurement does, run three frames 10 ms apart in the first display frame
  watchFrames((milliseconds, run) => {
    for (let i = 0; i < 3; i++) run(milliseconds + i * 10);
    watchFrames(undefined);
  });

  try {
    const times: number[] = [];
    const loop = frames.create(root, ({ time }) => times.push(Math.round(time * 1000)));
    frames.animate(loop, { schedule: display });

    // The display's next frame comes before the last frame run ahead, so it adds no time
    display.step(100, 110, 126);
    expect(times).toEqual([0, 10, 20, 20, 36]);
  } finally {
    watchFrames(undefined);
  }
});
