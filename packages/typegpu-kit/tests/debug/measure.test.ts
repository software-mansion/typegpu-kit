import { expect, it } from 'vitest';
import type { DriveOptions, GpuCommand, Instrumentation } from '../../src/debug/instrument.ts';
import { measure, measureText } from '../../src/debug/measure.ts';

function command(name: string, pass: number, start: number, end: number, split = false) {
  return {
    kind: 'compute',
    name,
    label: undefined,
    group: undefined,
    groups: [],
    pass,
    split,
    pipelines: [name],
    wgsl: [],
    calls: 1,
    uses: [],
    frame: 0,
    timestamps: 'written',
    start: BigInt(Math.round(start * 1e6)),
    end: BigInt(Math.round(end * 1e6)),
  } satisfies GpuCommand;
}

/**
 * Instrumentation whose drives replay ten frames back to back, each a compute pass and the canvas
 * pass, whose start Dawn on Metal writes early. Split per dispatch, the pass takes twice as long.
 * The last whole frame's compute pass takes `hitch` milliseconds longer.
 */
function replaying(count = 10, hitch = 0) {
  const drives: DriveOptions[] = [];
  let clock = 10;
  let passes = 0;
  const instrumentation: Instrumentation = {
    start() {},
    stop() {},
    [Symbol.dispose]() {},
    async drive(hooks, options) {
      drives.push(options);
      for (let i = 0; i < count; i++) {
        hooks.frameStarted(i * options.interval);
        // Every callback of a frame reports its timestamp
        hooks.frameStarted(i * options.interval);
        hooks.callbackTimed('frame', 0.25);
        const at = clock;
        const commands = hooks.splitComputePasses()
          ? [
              command('fill', ++passes, at, at + 1, true),
              command('blur', passes, at + 1, at + 4, true),
              command('canvas', ++passes, at + 4, at + 4.5),
            ]
          : [
              command('fill + blur', ++passes, at, at + 2 + (i === count - 1 ? hitch : 0)),
              command('canvas', ++passes, at - 1, at + 2.5),
              // A pass the program times itself
              { ...command('profiled', ++passes, at, at), timestamps: 'own' as const },
            ];
        clock = Math.max(...commands.map(({ end }) => Number(end) / 1e6)) + 0.25;
        hooks.submitted(commands);
        hooks.timed(commands);
      }
      return 2.75;
    },
  };
  return { instrumentation, drives };
}

const round = (value: number) => Math.round(value * 1000) / 1000;

it('measures whole frames and divides their passes as the split ones do', async () => {
  const { instrumentation, drives } = replaying();
  const measured = await measure(instrumentation, { seconds: 1, dt: 1 / 50, detail: 'dispatch' });

  // Whole frames for a second after warm-up, then a quarter second of split ones
  expect(drives).toEqual([
    { interval: 20, warmup: 250, duration: 1000 },
    { interval: 20, warmup: 0, duration: 250 },
  ]);
  const { parts, ...totals } = measured;
  expect(totals).toEqual({
    frames: 10,
    dt: 0.02,
    // The canvas pass starts with the compute pass, not a millisecond before it
    gpu: { median: 2.5, low: 2.5, high: 2.5, max: 2.5 },
    cpu: 0.25,
    wall: 2.75,
    detail: 'dispatch',
    untimed: ['profiled'],
  });
  expect(parts.map(({ name, ms, calls }) => [name, round(ms), round(calls)])).toEqual([
    ['blur', 1.5, 1],
    ['fill', 0.5, 1],
    ['canvas', 0.5, 1],
  ]);

  expect(measureText(measured)).toBe(
    [
      'measured 10 frames back to back, 20.00ms of scene time apart',
      'gpu 2.50ms (2.50 to 2.50, max 2.50) · a frame every 2.75ms · cpu 0.25ms',
      'gpu   1.50ms  blur',
      'gpu   0.50ms  fill',
      'gpu   0.50ms  canvas',
      'gpu  untimed  profiled',
    ].join('\n'),
  );
});

it('drives only whole frames per pass, and rejects a measurement of no time', async () => {
  const { instrumentation, drives } = replaying();
  const measured = await measure(instrumentation, { seconds: 0.5, dt: 1 / 60, detail: 'pass' });
  expect(drives).toHaveLength(1);
  expect(measured.parts.map(({ name }) => name)).toEqual(['fill + blur', 'canvas']);

  await expect(
    measure(instrumentation, { seconds: 0, dt: 1 / 60, detail: 'pass' }),
  ).rejects.toThrow('seconds and dt above 0');
});

it('reports the slowest frame beside the percentiles', async () => {
  const { instrumentation } = replaying(20, 3);
  const { gpu } = await measure(instrumentation, { seconds: 1, dt: 1 / 60, detail: 'pass' });
  expect(gpu).toEqual({ median: 2.5, low: 2.5, high: 2.5, max: 5 });
});

it('rejects when the GPU reports no frame finished before the measurement ends', async () => {
  const instrumentation: Instrumentation = {
    start() {},
    stop() {},
    [Symbol.dispose]() {},
    // Frames ran, but each burst waited past the end on the one before it
    drive: async () => Number.NaN,
  };
  await expect(
    measure(instrumentation, { seconds: 1, dt: 1 / 60, detail: 'pass' }),
  ).rejects.toThrow('No frame was measured: the GPU did not report the warm-up frames finished');
});
