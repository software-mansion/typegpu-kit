import { describe, expect, expectTypeOf, it } from 'vitest';
import { type Frame } from '../src/frames.ts';
import { frames, coroutines, type Coroutine, type Wait } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

function setup(
  options?: coroutines.Options,
  record: (frame: Frame, run: () => void) => void = (_frame, run) => run(),
) {
  const { root, log } = recordingRoot();
  const scheduler = coroutines.create(options);
  const timeline: string[] = [];
  let index = -1;

  const loop = frames.create(root, (frame) => {
    index = frame.index;
    record(frame, () => scheduler.run(frame));
  });
  let next = 0;

  return {
    scheduler,
    log,
    timeline,

    note: (text: string) => timeline.push(`${index}: ${text}`),

    step(count = 1) {
      for (let i = 0; i < count; i++) loop.run(next++ / 60);
    },
  };
}

function* broken(atOnce = false) {
  if (!atOnce) yield* coroutines.nextFrame;
  throw new Error('broken');
}

function* bare() {
  yield 1;
}

/** A wait that throws in `begin`, or in `ready` after the first frame, from frozen members. */
const failing = (where: 'begin' | 'ready'): Wait<number> =>
  coroutines.wait(
    Object.freeze({
      begin() {
        if (where === 'begin') throw new Error('begin failed');
      },
      ready(frame: Frame) {
        if (frame.index > 0) throw new Error('ready failed');
        return false;
      },
      value: () => 1,
    }),
  );

describe('coroutines.create', () => {
  it('resumes waits at the right frames with typed values', () => {
    const { scheduler, timeline, note, step } = setup();

    function* timed() {
      const first = yield* coroutines.nextFrame;
      note(`next frame ${first.index}`);

      yield* coroutines.frames(2);
      note('two frames later');

      yield* coroutines.seconds(0.05);
      note('0.05 s later');

      return 'done';
    }

    const coroutine = scheduler.start(timed());
    expectTypeOf(coroutine).toEqualTypeOf<Coroutine<string>>();

    step(8);
    expect(timeline).toEqual(['1: next frame 1', '3: two frames later', '6: 0.05 s later']);
    expect(coroutine.status).toBe('done');
    expect(coroutine.result).toBe('done');
    expect(scheduler.size).toBe(0);
  });

  it('starts child coroutines in the same frame and resumes joiners when they finish', () => {
    const { scheduler, timeline, note, step } = setup();

    function* child() {
      note('child started');
      yield* coroutines.frames(2);
      note('child done');
      return 42;
    }

    function* parent() {
      yield* coroutines.nextFrame;
      const value = yield* scheduler.start(child());
      expectTypeOf(value).toEqualTypeOf<number>();
      note(`parent got ${value}`);
    }

    scheduler.start(parent());

    step(4);
    expect(timeline).toEqual(['1: child started', '3: child done', '3: parent got 42']);
  });

  it('waits for a promise and resumes at the first run after it settles', async () => {
    const { scheduler, timeline, note, step } = setup();
    let resolve!: (value: number) => void;
    const answer = new Promise<number>((settle) => {
      resolve = settle;
    });

    function* asker() {
      note(`got ${yield* coroutines.until(answer)}`);
      try {
        yield* coroutines.until(Promise.reject(new Error('refused')));
      } catch (error) {
        note((error as Error).message);
      }
    }

    scheduler.start(asker());
    step();
    resolve(7);
    step(); // the promise settles after this synchronous frame
    await Promise.resolve();
    step();
    await Promise.resolve();
    step();

    expect(timeline).toEqual(['2: got 7', '3: refused']);
  });

  it('counts seconds on the clock it is given, so timed waits pause with it', () => {
    let simulated = 0;
    const { scheduler, timeline, note, step } = setup({ time: () => simulated });

    function* timer() {
      yield* coroutines.seconds(1);
      note('rang');
    }

    scheduler.start(timer());
    step();
    simulated = 0.5;
    step(100); // frame time runs on while the simulation stays paused
    expect(timeline).toEqual([]);

    simulated = 1;
    step();
    expect(timeline).toEqual(['101: rang']);
  });

  it('fails a throwing coroutine alone and keeps rendering', () => {
    const errors: unknown[] = [];
    const { scheduler, log, timeline, note, step } = setup({ onError: (e) => errors.push(e) });

    function* healthy() {
      for (let i = 0; i < 3; i++) {
        yield* coroutines.nextFrame;
        note('healthy');
      }
    }

    const failed = scheduler.start(broken());
    scheduler.start(healthy());

    step(4);

    expect(failed.status).toBe('failed');
    expect(errors).toEqual([new Error('broken')]);
    expect(timeline).toEqual(['1: healthy', '2: healthy', '3: healthy']);
    expect(log.submissions).toBe(4);
  });

  it('throws a failure at the joiner instead of reporting it', () => {
    const errors: unknown[] = [];
    const { scheduler, timeline, note, step } = setup({ onError: (e) => errors.push(e) });
    const request = coroutines.deferred<number>();

    function* parent() {
      try {
        yield* scheduler.start(
          (function* () {
            yield* request;
          })(),
        );
      } catch (error) {
        note(`caught ${(error as Error).message}`);
      }

      try {
        yield* scheduler.start(broken());
      } catch (error) {
        note(`caught ${(error as Error).message}`);
      }
    }

    function* failsAtOnce() {
      try {
        yield* scheduler.start(broken(true));
      } catch (error) {
        note(`caught ${(error as Error).message} at once`);
      }
    }

    scheduler.start(parent());
    scheduler.start(failsAtOnce());

    step();
    request.reject(new Error('device lost'));
    step(2);

    expect(timeline).toEqual([
      '0: caught broken at once',
      '1: caught device lost',
      '2: caught broken',
    ]);
    expect(errors).toEqual([]);
  });

  it('throws what a wait throws at the yield*, and a TypeError at a yield that is not a wait', () => {
    const errors: unknown[] = [];
    const { scheduler, timeline, note, step } = setup({ onError: (e) => errors.push(e) });

    function* catching(where: 'begin' | 'ready') {
      try {
        yield* failing(where);
      } catch (error) {
        note(`caught ${(error as Error).message}`);
      }
    }

    scheduler.start(catching('begin'));
    scheduler.start(catching('ready'));
    // @ts-expect-error bare yields are not waits
    const bad = scheduler.start(bare());

    expect(() => step(2)).not.toThrow();
    expect(timeline).toEqual(['0: caught begin failed', '1: caught ready failed']);
    expect(bad.status).toBe('failed');
    expect(errors).toEqual([bad.error]);
    expect(bad.error).toBeInstanceOf(TypeError);
  });

  it('cancels with finally blocks, using disposals, leaving its wait and joiner errors', () => {
    const { scheduler, timeline, note, step } = setup();
    const request = coroutines.wait({ ready: () => false, leave: () => note('left') });

    function* scoped() {
      using scope = new DisposableStack();
      scope.defer(() => note('resource freed'));
      try {
        const resumed = yield* request;
        expectTypeOf(resumed).toEqualTypeOf<Frame>();
      } finally {
        note('finally');
      }
    }

    const coroutine = scheduler.start(scoped());

    function* joiner() {
      try {
        yield* coroutine;
      } catch (error) {
        note((error as Error).message);
      }
    }

    scheduler.start(joiner());

    step();
    coroutine.cancel();
    step();

    expect(timeline).toEqual([
      '0: left',
      '0: finally',
      '0: resource freed',
      '1: The coroutine being waited on was cancelled.',
    ]);
  });

  it('cancels a coroutine it started with yield* first, and fails its other waiters', () => {
    const { scheduler, timeline, note, step } = setup();
    const request = coroutines.deferred<number>();
    let shared: Coroutine<number> | undefined;

    function* pick() {
      try {
        return yield* request;
      } finally {
        note('pick finally');
      }
    }

    function* game() {
      try {
        shared = scheduler.start(pick());
        yield* shared;
      } finally {
        note('game finally');
      }
    }

    function* watcher() {
      yield* coroutines.nextFrame;
      try {
        yield* shared!;
      } catch (error) {
        note((error as Error).message);
      }
    }

    const main = scheduler.start(game());
    scheduler.start(watcher());

    step(2);
    main.cancel();
    step();

    expect(timeline).toEqual([
      '1: pick finally',
      '1: game finally',
      '2: The coroutine being waited on was cancelled.',
    ]);
  });

  it('leaves a coroutine it did not start running when a waiter is cancelled', () => {
    const { scheduler, timeline, note, step } = setup();
    const compiled = coroutines.deferred<string>();

    function* warmup() {
      return yield* compiled;
    }

    // Started first, it starts `warm` by yielding it, but `warm` was started outside any coroutine.
    let warm: Coroutine<string> | undefined;
    const reveal = scheduler.start(
      (function* () {
        yield* warm!;
        note('revealed');
      })(),
    );
    warm = scheduler.start(warmup());

    const ready = scheduler.start(
      (function* () {
        note(`ready: ${yield* warm!}`);
      })(),
    );

    step();
    reveal.cancel();
    compiled.resolve('pipelines');
    step();

    expect(warm.status).toBe('done');
    expect(ready.status).toBe('done');
    expect(timeline).toEqual(['1: ready: pipelines']);
  });

  it('reports a yield* in a finally block while cancelling, and runs the cleanup around it', () => {
    const errors: unknown[] = [];
    const { scheduler, timeline, note, step } = setup({ onError: (e) => errors.push(e) });

    function* tidy() {
      using scope = new DisposableStack();
      scope.defer(() => note('resource freed'));

      try {
        try {
          yield* coroutines.nextFrame;
        } finally {
          yield* coroutines.nextFrame;
          note('not reached');
        }
      } finally {
        note('outer finally');
      }
    }

    const coroutine = scheduler.start(tidy());

    step();
    coroutine.cancel();

    expect(timeline).toEqual(['0: outer finally', '0: resource freed']);
    expect(errors).toHaveLength(1);
    expect((errors[0] as Error).message).toContain('yielded in a finally block');
  });

  it('rejects waits that would never resume', () => {
    expect(() => coroutines.seconds(Number.NaN)).toThrow(RangeError);
    expect(() => coroutines.seconds(Infinity)).toThrow(RangeError);
    expect(() => coroutines.frames(-1)).toThrow(RangeError);
    expect(() => coroutines.frames(1.5)).toThrow(RangeError);
    expect(() => coroutines.budget(Number.NaN)).toThrow(RangeError);
  });

  it('lets a coroutine cancel itself', () => {
    const { scheduler, timeline, note, step } = setup();

    function* selfCancelling() {
      try {
        yield* coroutines.nextFrame;
        self.cancel();
        yield* coroutines.nextFrame;
        note('not reached');
      } finally {
        note('finally');
      }
    }

    const self = scheduler.start(selfCancelling());

    step(3);

    expect(self.status).toBe('cancelled');
    expect(timeline).toEqual(['1: finally']);
  });

  it('spreads work over frames with a budget', () => {
    const { scheduler, timeline, note, step } = setup();

    function* work(name: string, ms: number) {
      for (let i = 0; i < 3; i++) {
        note(`${name} ${i}`);
        yield* coroutines.budget(ms);
      }
    }

    scheduler.start(work('sliced', 0));
    scheduler.start(work('whole', Infinity));

    step(3);

    expect(timeline).toEqual([
      '0: sliced 0',
      '0: whole 0',
      '0: whole 1',
      '0: whole 2',
      '1: sliced 1',
      '2: sliced 2',
    ]);
  });

  it('measures a budget and stamps each resume by the clock it is given', () => {
    let clock = 0;
    const { scheduler, timeline, note, step } = setup({ now: () => clock });
    const resumedAt: number[] = [];
    const stamp = coroutines.wait({
      ready(_frame, since) {
        resumedAt.push(since.resumedAt);
        return false;
      },
    });

    function* load() {
      const slice = coroutines.budget(2);
      for (let chunk = 0; chunk < 5; chunk++) {
        note(`chunk ${chunk}`);
        clock += 0.75;
        yield* slice;
      }
      yield* stamp;
    }

    scheduler.start(load());
    step(2);

    expect(timeline).toEqual([
      '0: chunk 0',
      '0: chunk 1',
      '0: chunk 2',
      '1: chunk 3',
      '1: chunk 4',
    ]);
    expect(resumedAt).toEqual([2.25]);
  });

  it('resumes after seconds when the frame time lands exactly on the end', () => {
    const { root } = recordingRoot();
    const scheduler = coroutines.create();
    const resumed: number[] = [];
    const loop = frames.create(root, (frame) => scheduler.run(frame));

    function* wait() {
      yield* coroutines.nextFrame;
      resumed.push((yield* coroutines.seconds(0.2)).time);
    }

    scheduler.start(wait());
    for (const time of [0, 0.1, 0.2, 0.3, 0.4]) loop.run(time);

    expect(resumed).toEqual([0.3]); // 0.3 - 0.1 rounds just below 0.2
  });

  it('cancels coroutines started after destroy without running them', () => {
    const { scheduler, timeline, note, step } = setup();

    scheduler.destroy();
    const coroutine = scheduler.start(
      (function* () {
        note('ran');
        yield* coroutines.nextFrame;
      })(),
    );
    step();

    expect(coroutine.status).toBe('cancelled');
    expect(timeline).toEqual([]);
    expect(scheduler.size).toBe(0);
  });
});
