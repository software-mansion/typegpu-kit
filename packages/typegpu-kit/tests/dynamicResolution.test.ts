import { expect, it } from 'vitest';
import { dynamicResolution, type Frame, type Timing } from '../src/index.ts';

/** A timing the test feeds, as a profiler's would fill as samples come back. */
const fake = () => {
  const timing = {
    name: 'frame',
    ms: undefined,
    count: 0,
    last: undefined as number | undefined,
    lastFrame: -1,
  };
  const add = (ms: number, frame: number) => {
    timing.count += 1;
    timing.last = ms;
    timing.lastFrame = frame;
  };
  return { timing: timing as Timing, add };
};

/** Frame `index` of a program whose time runs at 60 frames a second. */
const at = (index: number) => ({ index, time: index / 60 }) as Frame;

/**
 * Runs `frames` frames of a GPU whose frame at a scale costs `cost(scale, i)` milliseconds, each
 * sample coming back `latency` frames after the frame drew it. `arrive` runs as each frame comes,
 * to advance a clock. Returns the scale of every frame.
 */
const run = (
  detail: dynamicResolution.DynamicResolution,
  add: (ms: number, frame: number) => void,
  frames: number,
  cost: (scale: number, i: number) => number,
  { latency = 2, start = 0, arrive = (_i: number): void => {} } = {},
) => {
  const flying: [number, number][] = [];
  const scales: number[] = [];
  for (let i = start; i < start + frames; i++) {
    arrive(i);
    detail.update(at(i));
    scales.push(detail.scale);
    flying.push([cost(detail.scale, i), i]);
    if (flying.length > latency) add(...flying.shift()!);
  }
  return scales;
};

/** The frames at which the scale changed. */
const moves = (scales: number[]) =>
  scales.flatMap((s, i) => (i > 0 && s !== scales[i - 1] ? [i] : []));

it('steps down at once to near the budget when over it, then up a step while under the aim', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });
  // All of this frame's time goes with pixels: 12 ms at full scale
  const scales = run(detail, add, 1200, (s) => 12 * s * s);

  // Aiming for 85% of the budget: 12 s² = 5.1 at s ≈ 0.65, which is 5.07 ms, so a second later it
  // tries 0.7, which is 5.88 ms, under budget
  expect(moves(scales)).toEqual([11, 71]);
  expect(scales.at(-1)).toBe(0.7);
});

it('takes a second step down when time that does not scale makes the first too small', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });
  const scales = run(detail, add, 1200, (s) => 3 + 9 * s * s);

  // 12 ms at 1 guesses 0.65, which costs 6.8 ms. The next median, of 9 samples of frames drawn at
  // 0.65, steps again
  expect(moves(scales)).toEqual([11, 22]);
  expect(3 + 9 * scales.at(-1)! ** 2).toBeLessThan(6);
});

/** A frame with 1 ms that doesn't scale. */
const partlyScaled = (s: number) => 1 + 8 * s * s;

it('steps up slowly from a low scale set after create, never past the budget', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });
  detail.scale = 0.5;
  const cost = partlyScaled;
  const scales = run(detail, add, 2000, cost);

  // Each step up waits a second at the scale
  const steps = moves(scales);
  expect(steps.map((i) => scales[i])).toEqual([0.65, 0.7, 0.75]);
  expect(steps[1] - steps[0]).toBeGreaterThanOrEqual(60);
  expect(scales.at(-1)).toBe(0.75);
  expect(cost(scales.at(-1)!)).toBeLessThan(6);
});

/**
 * 4.5 ms that doesn't scale, as a full-size final pass on a phone, or a GPU that lowers its clock
 * under light work: 4.59 ms at 0.3 guesses no room from pixels alone.
 */
const mostlyFixed = (s: number) => 4.5 + s * s;

it('climbs a step at a time where time that does not scale hides the room', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });
  detail.scale = 0.3;
  const cost = mostlyFixed;
  const scales = run(detail, add, 1000, cost);

  const steps = moves(scales);
  expect(steps.map((i) => scales[i])).toEqual([
    0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
  ]);
  expect(cost(scales.at(-1)!)).toBeLessThan(6);
});

it('rides out noise and spikes, trying the step above less and less often', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });
  const noise = [1, 1.2, 0.85, 1.1, 0.9, 1.25, 0.8];
  // A 100 µs grain, as Chrome rounds timestamps, and a stall of three times the cost every 50
  const scales = run(detail, add, 8000, (s, i) => {
    const ms = 10 * s * s * noise[i % noise.length] * (i % 50 === 0 ? 3 : 1);
    return Math.round(ms * 10) / 10;
  });

  // 0.75 runs over on noise; each try comes after twice the hold of the last, up to 40 seconds
  const tries = moves(scales).filter((i) => scales[i] === 0.75);
  const gaps = tries.slice(1).map((i, k) => i - tries[k]);
  expect(gaps.length).toBe(3);
  expect(gaps[0]).toBeGreaterThan(1200);
  for (const gap of gaps.slice(1)) {
    expect(gap).toBeGreaterThan(2400);
    expect(gap).toBeLessThan(2600);
  }
  expect(scales.at(-1)).toBe(0.7);
});

it('keeps its scale while no timings arrive, as without timestamp-query', () => {
  const { timing } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6 });
  expect(detail.scale).toBe(1);
  detail.scale = 0.6;

  for (let i = 0; i < 1000; i++) detail.update(at(i));
  expect(detail.scale).toBe(0.6);
});

it('ignores samples of frames drawn before a scale is set', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });

  // Eight slow samples at full scale fall short of a median, then the scale is set by hand
  for (let i = 0; i < 8; i++) {
    detail.update(at(i));
    add(20, i);
  }
  detail.update(at(8));
  detail.scale = 0.5;
  expect(detail.scale).toBe(0.5);

  // Slow samples of frames drawn before the change still arrive, then fast ones
  for (let i = 9; i < 12; i++) {
    add(20, i - 3);
    detail.update(at(i));
  }
  for (let i = 12; i < 68; i++) {
    add(2, i);
    detail.update(at(i));
  }
  expect(detail.scale).toBe(0.5);

  // A second after the change, 2 ms at 0.5 guesses 0.8 for 5.1 ms, kept on the step below
  for (let i = 68; i < 70; i++) {
    add(2, i);
    detail.update(at(i));
  }
  expect(detail.scale).toBe(0.75);
});

it('judges by the newest of the samples that arrive between two updates', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });

  for (let i = 0; i < 4; i++) add(i === 3 ? 20 : 1, i);
  detail.update(at(4));
  for (let i = 5; i < 13; i++) {
    add(20, i);
    detail.update(at(i));
  }
  // 20 ms at full scale guesses 0.5 for 5.1 ms
  expect(detail.scale).toBe(0.5);
});

it('keeps away from a scale that ran over budget for a while', () => {
  const { timing, add } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.3, 1] });
  detail.scale = 0.7;
  // A cliff the model can't see: past 0.7, the frame costs more than twice as much
  const scales = run(detail, add, 4000, (s) => (s > 0.7 ? 7 : 3));

  // Up to 0.9, back down twice, up again to the step below the scale that ran over, then held
  // before it tries above it again, twice as long the second time
  const steps = moves(scales);
  const cycle = [0.9, 0.75, 0.6, 0.7];
  expect(steps.map((i) => scales[i])).toEqual([...cycle, ...cycle, ...cycle]);
  expect(steps[4] - steps[0]).toBeGreaterThan(1200);
  expect(steps[8] - steps[4]).toBeGreaterThan(2400);
});

it('computes a budget from the display interval at every judgement', () => {
  const { timing, add } = fake();
  let share = 0.72;
  let clock = 0;
  const detail = dynamicResolution.create(timing, {
    budget: (interval) => share * interval,
    range: [0.3, 1],
    now: () => clock,
  });
  // 10 ms at full scale fits the 12 ms of a 60 Hz display, then one twice as fast leaves 6 ms.
  // The frames' own time runs at 60 a second throughout
  const scales = run(detail, add, 2000, (s) => 10 * s * s, {
    arrive: (i) => void (clock += i < 1000 ? 1000 / 60 : 1000 / 120),
  });

  expect(scales[999]).toBe(1);
  expect(scales.at(-1)).toBeLessThan(0.8);

  share = Number.NaN;
  expect(() => run(detail, add, 20, () => 1, { start: 2000 })).toThrow('budget must be');
});

it('measures the display interval from the shortest recent frames', () => {
  const { timing, add } = fake();
  let clock = 0;
  let measured = 0;
  const detail = dynamicResolution.create(timing, {
    budget: (interval) => (measured = interval),
    now: () => clock,
  });
  let index = 0;
  /** The interval a judgement gets once frames have come `elapsed` milliseconds apart. */
  const after = (...elapsed: number[]) => {
    for (const ms of elapsed) {
      clock += ms;
      add(1, index);
      detail.update(at(index++));
    }
    return measured;
  };
  const [slow, fast] = [1000 / 60, 1000 / 120];

  expect(after(...Array.from({ length: 12 }, () => slow))).toBeCloseTo(slow, 9);
  // One short frame alone does not move it, a second does
  expect(after(fast)).toBeCloseTo(slow, 9);
  expect(after(fast)).toBeCloseTo(fast, 9);
  // One early frame in a burst, then frames that miss every other refresh
  expect(after(fast, 1, fast)).toBeCloseTo(fast, 9);
  const missing = after(slow, slow, slow);
  expect(missing).toBeGreaterThan(fast);
  expect(missing).toBeLessThan(1000 / 119);
});

it('clamps a scale set by hand to its range', () => {
  const { timing } = fake();
  const detail = dynamicResolution.create(timing, { budget: 6, range: [0.4, 0.9] });

  expect(detail.scale).toBe(0.9);
  detail.scale = 2;
  expect(detail.scale).toBe(0.9);
  detail.scale = 0.1;
  expect(detail.scale).toBe(0.4);
  expect(() => (detail.scale = Number.NaN)).toThrow(RangeError);
  expect(() => (detail.scale = Infinity)).toThrow('scale must be finite');
  expect(detail.scale).toBe(0.4);
});

it('rejects a budget or range it cannot follow', () => {
  const { timing } = fake();

  expect(() => dynamicResolution.create(timing, { budget: 0 })).toThrow('budget must be');
  expect(() => dynamicResolution.create(timing, { budget: Infinity })).toThrow('budget must be');
  expect(() => dynamicResolution.create(timing, { budget: 6, range: [0, 1] })).toThrow(
    'range must',
  );
  expect(() => dynamicResolution.create(timing, { budget: 6, range: [0.8, 0.5] })).toThrow(
    'range must',
  );
  // A target set's resolution caps at 1
  expect(() => dynamicResolution.create(timing, { budget: 6, range: [0.5, 1.5] })).toThrow(
    'range must',
  );
});
