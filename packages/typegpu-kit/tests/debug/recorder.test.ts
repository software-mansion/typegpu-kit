import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GpuCommand } from '../../src/debug/instrument.ts';
import type { Access, GpuResource } from '../../src/debug/metadata.ts';
import { createRecorder, nameAt, perFrame } from '../../src/debug/recorder.ts';

const resource = (id: number, label: string): GpuResource => ({ id, kind: 'buffer', label });

function command(name: string, uses: [GpuResource, Access][] = [], ms?: number): GpuCommand {
  return {
    kind: 'compute',
    name,
    label: undefined,
    group: undefined,
    groups: [],
    pass: undefined,
    split: false,
    pipelines: [name],
    wgsl: [],
    calls: 1,
    uses: uses.map(([used, access]) => ({ resource: used, access })),
    frame: 0,
    timestamps: 'written',
    ...(ms === undefined ? {} : { start: 1_000_000n, end: 1_000_000n + BigInt(ms * 1e6) }),
  };
}

/** A command the profiler wrote no timestamps for. */
function untimed(submitted: GpuCommand): GpuCommand {
  const copy = { ...submitted };
  delete copy.timestamps;
  return copy;
}

/** Fills in a submitted command's timestamps, in milliseconds, as the instrument does. */
function stamp(submitted: GpuCommand, start: number, end: number): GpuCommand[] {
  submitted.start = BigInt(start * 1e6);
  submitted.end = BigInt(end * 1e6);
  return [submitted];
}

afterEach(() => {
  vi.useRealTimers();
});

describe('recorder', () => {
  it('folds frames into bars and averages them per frame', () => {
    const recorder = createRecorder();

    for (let frame = 0; frame < 4; frame++) {
      recorder.frameStarted(frame * 20);
      const step = command('step');
      recorder.submitted([step]);
      recorder.timed(stamp(step, 0, 2));
      recorder.callbackTimed('tick', 0.5);
    }
    recorder.frameStarted(100);

    // Frames at 0, 20, 40 and 60 ms share the first 100 ms bar.
    const [bar] = recorder.bars();
    expect(bar?.frames).toEqual({ gpu: 4, cpu: 4 });
    expect(perFrame(recorder.bars(), 'gpu')).toEqual([{ name: 'step', ms: 2, calls: 1 }]);
    expect(perFrame(recorder.bars(), 'cpu')).toEqual([{ name: 'tick', ms: 0.5, calls: 1 }]);

    // Five frames started in 100 ms.
    expect(recorder.fps()).toBe(40);
  });

  it('sums the last second up as text, most first, with calls a frame', () => {
    const recorder = createRecorder();

    for (let frame = 0; frame < 4; frame++) {
      recorder.frameStarted(frame * 20);
      const draw = command('draw');
      const blur = command('blur');
      const again = command('blur');
      recorder.submitted([draw, blur, again]);
      recorder.timed([...stamp(draw, 0, 1), ...stamp(blur, 1, 2), ...stamp(again, 2, 4)]);
      recorder.callbackTimed('tick', 0.5);
    }
    recorder.frameStarted(100);

    expect(recorder.summary()).toBe(
      [
        '40 fps · gpu 4.00ms over 4 frames · cpu 0.50ms over 4 frames',
        'gpu   3.00ms  blur ×2',
        'gpu   1.00ms  draw',
        'cpu   0.50ms  tick',
      ].join('\n'),
    );
  });

  it('leaves a frame whose passes went untimed out of the GPU average', () => {
    const recorder = createRecorder();

    recorder.frameStarted(0);
    const timed = command('draw');
    recorder.submitted([timed]);
    recorder.timed(stamp(timed, 0, 2));

    recorder.frameStarted(20);
    const early = command('draw');
    recorder.submitted([early, { ...command('draw'), timestamps: 'dropped' }]);
    recorder.timed(stamp(early, 0, 1));
    recorder.callbackTimed('tick', 0.5);

    recorder.frameStarted(40);
    recorder.frameStarted(100);

    expect(recorder.bars()[0]?.frames).toEqual({ gpu: 2, cpu: 3 });
    expect(perFrame(recorder.bars(), 'gpu')).toEqual([{ name: 'draw', ms: 1, calls: 0.5 }]);
  });

  it('never counts a frame whose timestamps come back after it left the history', () => {
    const recorder = createRecorder();
    recorder.setDetail('pass');
    recorder.frameStarted(0);
    const late = command('draw');
    recorder.submitted([late]);

    // Frames are kept for two seconds; without vsync the GPU can run half a second behind.
    for (let time = 10; time <= 2010; time += 10) recorder.frameStarted(time);
    recorder.timed(stamp(late, 0, 1));

    const counted = (processor: 'gpu' | 'cpu') =>
      recorder.bars().reduce((sum, bar) => sum + bar.frames[processor], 0);
    expect(counted('cpu')).toBe(201);
    expect(counted('gpu')).toBe(200);
    expect(perFrame(recorder.bars(), 'gpu')).toEqual([]);
  });

  it('charges each pass only for the time no earlier pass of the frame covers', async () => {
    const recorder = createRecorder();
    // A capture under a finer detail splits its frame, which then stays out of the graph
    recorder.setDetail('pass');
    const capture = recorder.capture();
    recorder.frameStarted(0);

    // Bloom on a tile-based GPU: each pass begins with its vertex work, then waits for the
    // previous pass's fragments. An independent pass after a gap counts whole.
    const [scene, down, up, ui] = ['scene', 'down', 'up', 'ui'].map((name) => command(name));
    recorder.submitted([scene!, down!, up!, ui!]);
    recorder.timed([
      ...stamp(scene!, 0, 2),
      ...stamp(down!, 0.1, 2.5),
      ...stamp(up!, 0.2, 2.6),
      ...stamp(ui!, 3, 4),
    ]);
    recorder.frameStarted(100);
    recorder.frameStarted(200);

    const own = [
      { name: 'scene', ms: 2, calls: 1 },
      { name: 'down', ms: 0.5, calls: 1 },
      { name: 'up', ms: 0.1, calls: 1 },
      { name: 'ui', ms: 1, calls: 1 },
    ];
    expect(recorder.bars()[0]?.segments.gpu).toEqual(own);
    expect(recorder.bars()[0]?.peak.gpu).toBe(3.6);

    const { commands } = await capture;
    expect(commands.map(({ ms }) => ms)).toEqual(own.map(({ ms }) => ms));
  });

  it('adds timings that arrive after their frame was folded', () => {
    const recorder = createRecorder();
    recorder.frameStarted(0);
    const late = command('late');
    recorder.submitted([late]);
    recorder.frameStarted(200);
    recorder.timed(stamp(late, 0, 3));

    expect(recorder.bars()[0]?.segments.gpu).toEqual([{ name: 'late', ms: 3, calls: 1 }]);
    expect(recorder.bars()[0]?.peak.gpu).toBe(3);
  });

  it('captures the next frame, with the command that last wrote each read', async () => {
    const recorder = createRecorder();
    const [field, sums, input] = [resource(0, 'field'), resource(1, 'sums'), resource(2, 'input')];

    recorder.frameStarted(0);
    expect(recorder.capturing()).toBe(false);

    const capture = recorder.capture();
    recorder.frameStarted(16);
    expect(recorder.capturing()).toBe(true);

    const reduce = command('reduce', [
      [field, 'read'],
      [sums, 'readwrite'],
    ]);
    recorder.submitted([
      untimed(
        command('advect', [
          [input, 'read'],
          [field, 'write'],
        ]),
      ),
      reduce,
      untimed(command('draw', [[sums, 'read']])),
    ]);

    recorder.frameStarted(32);
    expect(recorder.capturing()).toBe(false);

    // The capture waits for the frame's timestamps.
    recorder.timed(stamp(reduce, 2, 2.25));
    const { frame, timings, commands } = await capture;

    expect(frame).toBe(2);
    expect(timings).toBe('complete');
    expect(commands.map(({ displayName, after }) => [displayName, after])).toEqual([
      ['advect', [{ from: undefined, resource: input }]],
      [
        'reduce',
        [
          { from: 0, resource: field },
          { from: undefined, resource: sums },
        ],
      ],
      ['draw', [{ from: 1, resource: sums }]],
    ]);
    expect(commands.map(({ ms }) => ms)).toEqual([undefined, 0.25, undefined]);
    expect(commands[1]).toMatchObject({ start: 0, end: 0.25 });
    expect(JSON.parse(JSON.stringify(commands[1]))).toMatchObject({ start: 0, end: 0.25 });
  });

  it('shares a capture between the calls made before it resolves', async () => {
    vi.useFakeTimers();
    const recorder = createRecorder();
    const first = recorder.capture();
    const second = recorder.capture();
    expect(second).toBe(first);

    recorder.frameStarted(0);
    recorder.submitted([command('draw', [], 1)]);
    recorder.frameStarted(16);
    // The draw's timestamps never come, so the capture resolves when it stops waiting
    await vi.runAllTimersAsync();

    expect((await first).frame).toBe(1);
    expect(recorder.capture()).not.toBe(first);
  });

  it('waits for timestamps that come back late, and says which never did', async () => {
    vi.useFakeTimers();
    const recorder = createRecorder();

    const late = recorder.capture();
    recorder.frameStarted(0);
    const draw = command('draw');
    recorder.submitted([draw]);
    recorder.frameStarted(16);

    // Without vsync the GPU can run half a second behind.
    let resolved = false;
    void late.then(() => (resolved = true));
    await vi.advanceTimersByTimeAsync(350);
    expect(resolved).toBe(false);
    recorder.timed(stamp(draw, 0, 2));
    await expect(late).resolves.toMatchObject({ timings: 'complete', commands: [{ ms: 2 }] });

    const lost = recorder.capture();
    recorder.frameStarted(32);
    recorder.submitted([command('draw')]);
    // Frames go on meanwhile, and timers run late under load, so the frames pass the two seconds
    // the panel keeps before the capture gives up waiting.
    for (let time = 48; time <= 2400; time += 16) {
      recorder.frameStarted(time);
      await vi.advanceTimersByTimeAsync(14);
    }
    await expect(lost).resolves.toMatchObject({ timings: 'late', commands: [{ ms: undefined }] });

    const dropped = recorder.capture();
    recorder.frameStarted(2416);
    recorder.submitted([command('draw'), { ...command('draw'), timestamps: 'dropped' }]);
    recorder.frameStarted(2432);
    await expect(dropped).resolves.toMatchObject({ timings: 'dropped' });
  });

  it('writes timed commands as a Chrome trace, counted from the first timestamp it saw', () => {
    const recorder = createRecorder();
    type Trace = { traceEvents: { name: string; ph: string; ts: number }[] };
    const spans = () =>
      (recorder.trace() as Trace).traceEvents
        .filter(({ ph }) => ph === 'X')
        .map(({ name, ts }) => [name, ts]);

    recorder.frameStarted(0);
    const advect = command('advect');
    const reduce = command('reduce');
    recorder.submitted([advect, command('untimed'), reduce]);
    recorder.timed([...stamp(advect, 10, 11), ...stamp(reduce, 11, 13)]);
    expect(spans()).toEqual([
      ['advect', 0],
      ['reduce', 1000],
    ]);

    // The first frame leaves the history; the next trace still counts from it, so traces line up
    recorder.frameStarted(3000);
    const later = command('later');
    recorder.submitted([later]);
    recorder.timed(stamp(later, 3010, 3011));
    expect(spans()).toEqual([['later', 3_000_000]]);
  });

  it('starts a pass no earlier than the one before it, as Dawn on Metal writes the canvas pass', () => {
    const recorder = createRecorder();
    recorder.frameStarted(0);
    const scene = command('scene');
    const canvas = command('canvas');
    recorder.submitted([scene, canvas]);
    // The canvas pass's start is written before the scene's, while it waits for the scene
    recorder.timed([...stamp(scene, 10, 12), ...stamp(canvas, 1, 13)]);
    recorder.frameStarted(200);

    expect(recorder.bars()[0]?.segments.gpu).toEqual([
      { name: 'scene', ms: 2, calls: 1 },
      { name: 'canvas', ms: 1, calls: 1 },
    ]);
  });

  it('times whole frames, dividing their passes as the frames split per dispatch do', () => {
    const recorder = createRecorder();
    let passes = 0;
    // Each split frame's dispatches take longer than the whole pass: splitting drains the GPU
    // between them
    const frame = (time: number) => {
      recorder.frameStarted(time);
      const pass = ++passes;
      if (recorder.splitComputePasses()) {
        const dispatches = ['fill', 'blur'].map((name) => ({
          ...command(name),
          pass,
          split: true,
        }));
        recorder.submitted(dispatches);
        recorder.timed([...stamp(dispatches[0], 0, 1), ...stamp(dispatches[1], 1, 4)]);
      } else {
        const whole = { ...command('fill + blur'), pipelines: ['fill', 'blur'], pass };
        recorder.submitted([whole]);
        recorder.timed(stamp(whole, 0, 2));
      }
    };

    // Frames 8 and 16 split, and stay out of the GPU's columns. The first seven name the pass
    // whole, as no split frame has come yet; the second bar holds frames 11 to 16
    for (let i = 0; i < 16; i++) frame(i * 10);
    recorder.frameStarted(200);

    const [before, after] = recorder.bars();
    expect(before?.frames).toEqual({ gpu: 9, cpu: 10 });
    expect(before?.segments.gpu[0]).toEqual({ name: 'fill + blur', ms: 14, calls: 7 });
    expect(after?.frames).toEqual({ gpu: 5, cpu: 6 });
    expect(perFrame([after!], 'gpu')).toEqual([
      { name: 'fill', ms: 0.5, calls: 1 },
      { name: 'blur', ms: 1.5, calls: 1 },
    ]);

    recorder.setDetail('pass');
    expect(recorder.detail()).toBe('pass');
    recorder.frameStarted(210);
    expect(recorder.splitComputePasses()).toBe(false);
  });

  it('names dispatches by their debug groups per group, and by their pipeline elsewhere', () => {
    const step = { ...command('step'), group: 'sim', groups: ['water'], split: true };
    const shade = { ...command('shade'), group: 'sim', split: true };

    expect([nameAt(step, 'group'), nameAt(shade, 'group')]).toEqual(['sim › water', 'sim › shade']);
    expect([nameAt(step, 'dispatch'), nameAt(step, 'pass')]).toEqual([
      'sim › water › step',
      'sim › water › step',
    ]);
  });

  it('counts the frame rate again after a gap, as after a measurement', () => {
    const recorder = createRecorder();
    for (let time = 0; time <= 100; time += 10) recorder.frameStarted(time);
    for (let time = 2000; time <= 2050; time += 25) recorder.frameStarted(time);
    expect(recorder.fps()).toBe(40);
  });
});
