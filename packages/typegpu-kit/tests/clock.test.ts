import { describe, expect, it } from 'vitest';
import { clock } from '../src/index.ts';

/** How far a value eased at `hz` frames a second gets towards 1 in one second. */
const easedFor = (hz: number) => {
  let at = 0;
  for (let i = 0; i < hz; i++) at += (1 - at) * clock.ease(1 / hz, 3);
  return at;
};

describe('ease', () => {
  it('closes the same gap in the same time at any frame rate', () => {
    expect(easedFor(30)).toBeCloseTo(easedFor(144), 10);
    expect(easedFor(60)).toBeCloseTo(1 - Math.exp(-3), 10);
    expect(clock.ease(Math.LN2 / 8, 8)).toBeCloseTo(0.5);
    expect(clock.ease(0, 8)).toBe(0);
  });
});

describe('fixed step', () => {
  it('runs whole steps and carries the rest', () => {
    const sim = clock.fixedStep({ step: 0.25 });
    expect([0.1, 0.2, 0.5, 0.1].map((dt) => sim.update(dt))).toEqual([0, 1, 2, 0]);
    expect(sim.alpha).toBeCloseTo(0.6); // 0.15 s of 0.25 carried
    expect(sim.ahead).toBeCloseTo(0.15);

    sim.update(2); // four steps run, four carried
    expect(sim.ahead).toBe(0.25);
    expect(sim.time).toBe(1.75); // seven steps run
  });

  it('runs one step per frame when frames are a step apart, despite rounding', () => {
    const sim = clock.fixedStep({ step: 1 / 60 });
    const steps = Array.from({ length: 600 }, (_, i) => sim.update((i + 1) / 60 - i / 60));

    expect(new Set(steps)).toEqual(new Set([1]));
  });

  it('caps the steps per frame and keeps at most that many in debt', () => {
    const sim = clock.fixedStep({ step: 0.1, maxSteps: 2 });

    expect(sim.update(1)).toBe(2); // 8 steps owed, 2 kept
    expect(sim.update(0)).toBe(0);
    expect(sim.update(0.01)).toBe(2);
    expect(sim.update(0.01)).toBe(0);
  });

  it('runs steps of a new length on the time it carries, keeping the time stepped', () => {
    const sim = clock.fixedStep({ step: 0.25, maxSteps: 4 });
    expect(sim.update(0.6)).toBe(2); // 0.1 s carried

    sim.step = 0.05;
    expect(sim.update(0.1)).toBe(4); // 0.2 s owed, four steps of 0.05
    expect(sim.time).toBeCloseTo(0.7);
    expect(() => (sim.step = 0)).toThrow(RangeError);
  });

  it('counts a missed frame as the usual one when steady, and follows a slower display', () => {
    const sim = clock.fixedStep({ step: 1 / 60, steady: true });
    const steps = [1, 1, 3, 1, 1, 2, 2, 2].map((frames) => sim.update(frames / 60));

    expect(steps).toEqual([1, 1, 1, 1, 1, 1, 2, 2]);
  });

  it('stands still while paused and forgets debt on reset', () => {
    const sim = clock.fixedStep({ step: 0.1 });

    sim.paused = true;
    expect(sim.update(1)).toBe(0);

    sim.paused = false;
    sim.update(0.15);
    sim.reset();
    expect(sim.update(0.06)).toBe(0);
  });

  it('ignores time running backwards or undefined', () => {
    const sim = clock.fixedStep({ step: 0.1 });

    expect(sim.update(-1)).toBe(0);
    expect(sim.update(Number.NaN)).toBe(0);
    expect(() => clock.fixedStep({ step: 0 })).toThrow(RangeError);
    expect(() => clock.fixedStep({ step: 0.1, maxSteps: 0 })).toThrow(RangeError);
  });
});

describe('cadence', () => {
  it('runs every nth frame, first frame included, with the time it skipped', () => {
    const gi = clock.cadence({ every: 3 });
    const runs = [1, 2, 3, 4, 5, 6, 7].map((dt) => (gi.due(dt) ? gi.dt : 0));
    expect(runs).toEqual([1, 0, 0, 9, 0, 0, 18]);
  });

  it('runs hz times a second on average, in the frames closest to its ticks', () => {
    const cadence = clock.cadence({ hz: 25 });
    const runs = Array.from({ length: 120 }, () => cadence.due(1 / 60));
    const gaps = runs.flatMap((run, i) => (run ? [i] : [])).map((i, k, at) => i - (at[k - 1] ?? 0));

    expect(runs.filter(Boolean)).toHaveLength(50);
    expect(new Set(gaps.slice(1))).toEqual(new Set([2, 3]));
  });

  it('hands over the time since its last run', () => {
    const cadence = clock.cadence({ hz: 10 });
    const dts = [0.03, 0.03, 0.03, 0.03].flatMap((dt) => (cadence.due(dt) ? [cadence.dt] : []));
    expect(dts.map((dt) => dt.toFixed(2))).toEqual(['0.03', '0.09']);
  });

  it('starts its schedule again after a stall', () => {
    const cadence = clock.cadence({ hz: 10 });
    const runs = [0.05, 1, 0.05, 0.05].map((dt) => cadence.due(dt));
    expect(runs).toEqual([true, true, false, true]);
  });

  it('keeps its rate when frame times jitter around a divisor of it', () => {
    const readback = clock.cadence({ hz: 30 });
    const runs = Array.from({ length: 60 }, (_, i) => readback.due(i % 2 ? 0.0168 : 0.0165));
    expect(runs.filter(Boolean)).toHaveLength(30);
  });

  it('rejects rates that never run, and every and hz together', () => {
    expect(() => clock.cadence({ hz: 0 })).toThrow(RangeError);
    expect(() => clock.cadence({ hz: Number.NaN })).toThrow(RangeError);
    expect(() => clock.cadence({ every: 0 })).toThrow(RangeError);
    expect(() => clock.cadence({ every: 2.5 })).toThrow(RangeError);
    // @ts-expect-error one of every and hz
    expect(() => clock.cadence({ every: 4, hz: 10 })).toThrow(TypeError);
  });

  it('restarts its phase on reset', () => {
    const gi = clock.cadence({ every: 4 });

    gi.due(1);
    gi.due(1);
    gi.reset();

    expect(gi.due(1)).toBe(true);
    expect(gi.dt).toBe(1);
  });
});

/** Where a spring is after one second, stepped at `hz`, and its speed. */
const springFor = (hz: number, options: clock.SpringOptions) => {
  const pulled = clock.spring(options);
  for (let i = 0; i < hz; i++) pulled.step(1, 1 / hz);
  return [pulled.value, pulled.velocity];
};

describe('spring', () => {
  it('moves the same at any frame rate, bouncing, critical or overdamped', () => {
    for (const damping of [2, 2 * Math.sqrt(40), 30]) {
      const [at30, speed30] = springFor(30, { stiffness: 40, damping });
      const [at144, speed144] = springFor(144, { stiffness: 40, damping });
      expect(at30).toBeCloseTo(at144!, 10);
      expect(speed30).toBeCloseTo(speed144!, 10);
    }
    // Lightly damped, it overshoots; critically damped, it never does
    const bouncing = clock.spring({ stiffness: 40, damping: 2 });
    const peak = Math.max(...Array.from({ length: 60 }, () => bouncing.step(1, 1 / 60)));
    expect(peak).toBeGreaterThan(1);
    const critical = clock.spring({ stiffness: 40, damping: 2 * Math.sqrt(40) });
    const highest = Math.max(...Array.from({ length: 600 }, () => critical.step(1, 1 / 60)));
    expect(highest).toBeLessThanOrEqual(1);
  });

  it('moves a vector in place, keeping a kick to its velocity', () => {
    const value = [0, 0, 0];
    const bend = clock.spring({ stiffness: 55, damping: 4.5, value });
    bend.velocity[1] = 2;
    expect(bend.step([1, 0, -1], 1 / 60)).toBe(value);
    expect(value[0]).toBeGreaterThan(0);
    expect(value[1]).toBeGreaterThan(0);
    expect(value[2]).toBeLessThan(0);
    for (let i = 0; i < 600; i++) bend.step([1, 0, -1], 1 / 60);
    expect(value.map((x) => Math.round(x * 1e6) / 1e6 + 0)).toEqual([1, 0, -1]);
  });

  it('rejects negative or endless stiffness and damping', () => {
    expect(() => clock.spring({ stiffness: -1, damping: 1 })).toThrow('stiffness must be');
    expect(() => clock.spring({ stiffness: 1, damping: Infinity })).toThrow('damping must be');
  });
});
