import type { TgpuRoot } from 'typegpu';
import { afterEach, describe, expect, expectTypeOf, it, vi } from 'vitest';
import { local, stage, watch } from '../src/hosting.ts';
import type { Surface } from '../src/targets.ts';
import { offscreenSurface, recordingRoot, steppedSchedule } from '../src/testing.ts';

afterEach(() => {
  vi.restoreAllMocks();
});

/** Lets the closures queued so far run. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A function that throws `message`. */
const throwing = (message: string) => () => {
  throw new Error(message);
};

describe('local', () => {
  it('keeps the order of own on the calling runtime', async () => {
    const log: string[] = [];
    const owned = local(async () => {
      await Promise.resolve();
      return { total: 0, destroy: () => log.push('destroy') };
    });

    owned.send((value) => {
      value.total += 1;
      log.push(`total ${value.total}`);
    });
    const total = owned.run((value) => value.total);
    owned.destroy();
    owned.send(() => log.push('after destroy'));
    const late = owned.run(() => 0);
    void late.catch(() => {});
    await settle();

    expect(log).toEqual(['total 1', 'destroy']);
    await expect(total).resolves.toBe(1);
    await expect(late).rejects.toThrow('destroyed');
  });

  it('awaits an async closure that run is given, and types its value unwrapped', async () => {
    const owned = local(() => ({ total: 3 }));

    const total = owned.run(async (value) => {
      await Promise.resolve();
      return value.total;
    });
    const failed = owned.run(async () => {
      throw new Error('late');
    });
    void failed.catch(() => {});
    await settle();

    expectTypeOf(total).toEqualTypeOf<Promise<number>>();
    await expect(total).resolves.toBe(3);
    await expect(failed).rejects.toThrow('late');
  });

  it('destroys at once while a run waits for cancellation', async () => {
    const cancelled = Promise.withResolvers<void>();
    const destroy = vi.fn();
    const owned = local(() => ({ capture: () => cancelled.promise, destroy }));
    const capture = owned.run((value) => value.capture());
    await settle();

    owned.destroy();
    await settle();
    expect(destroy).toHaveBeenCalledOnce();
    await expect(owned.run(() => 0)).rejects.toThrow('destroyed');

    cancelled.resolve();
    await capture;
  });

  it('reports what create and sent closures throw', async () => {
    const onError = vi.fn();
    const owned = local(() => ({}), { onError });

    owned.send(() => {
      throw new Error('send');
    });
    local(() => Promise.reject(new Error('create')), { onError });
    await settle();

    const messages = onError.mock.calls.map(([error]) => (error as Error).message);
    expect(messages.toSorted()).toEqual(['create', 'send']);
  });

  it('logs an error whose onError throws, as own does', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});

    local(() => Promise.reject(new Error('create')), {
      onError() {
        throw new Error('onError');
      },
    });
    await settle();

    expect(logged.mock.calls).toEqual([
      [
        expect.objectContaining({ message: 'create' }),
        expect.objectContaining({ message: 'onError' }),
      ],
    ]);
  });
});

/** A scene that is ready when `ready` settles. */
const scene = (ready: Promise<void>) => () => ({ frame() {}, destroy() {}, ready });

/** A root, a surface that pauses, and a scene that writes what happens to it into `order`. */
function staging() {
  const { root, log } = recordingRoot();
  const surface = offscreenSurface(root, { width: 8, height: 8, format: 'bgra8unorm' });
  const schedule = steppedSchedule();
  const step = schedule.step;
  const waiting = () => schedule.waiting;
  const order: string[] = [];

  const createScene = (_root: TgpuRoot, _surface: Surface, options: { name: string }) => ({
    frame: () => {
      order.push(`frame ${options.name}`);
    },
    destroy: () => {
      order.push('destroy');
    },
  });

  return { root, log, surface, order, createScene, schedule, step, waiting };
}

describe('stage', () => {
  it('runs the scene with page code after each frame, and destroys it but not the root or the surface', async () => {
    const { root, log, surface, order, createScene, schedule, step, waiting } = staging();

    const staged = await stage(
      root,
      surface,
      createScene,
      { name: 'reef' },
      {
        schedule,
        onError: (error) => {
          throw error;
        },
        onFrame: (held, frame) => {
          order.push(`page ${held.scene === staged.scene} ${frame.index}`);
        },
      },
    );
    step(0);
    step(16);
    expect(order).toEqual(['frame reef', 'page true 0', 'frame reef', 'page true 1']);

    staged.destroy();
    expect(order.at(-1)).toBe('destroy');
    expect(waiting()).toBe(false);
    expect(staged.loop.run(1)).toBe(false);
    expect(log.alive()).toEqual(['[Texture "offscreen surface"]']);
  });

  it.each(['frame', 'onFrame', 'ready', 'device', 'fail'] as const)(
    'stops its frames at the first error from %s, reports it once, and keeps the scene',
    async (source) => {
      const { root, surface, schedule, step, waiting } = staging();
      const events = new EventTarget();
      const device = Object.assign(Object.create(root.device) as GPUDevice, {
        lost: new Promise(() => {}),
        addEventListener: events.addEventListener.bind(events),
        removeEventListener: events.removeEventListener.bind(events),
      });
      const uncaptured = () =>
        events.dispatchEvent(
          Object.assign(new Event('uncapturederror'), { error: { message: source } }),
        );
      const onError = vi.fn();
      const fails = () => {
        throw new Error(source);
      };
      const destroy = vi.fn();

      const staged = await stage(
        root,
        surface,
        () => ({
          frame: source === 'frame' ? fails : () => {},
          destroy,
          ...(source === 'ready' ? { ready: Promise.reject(new Error(source)) } : {}),
        }),
        undefined,
        {
          schedule,
          watch: device,
          onError,
          ...(source === 'onFrame' ? { onFrame: fails } : {}),
        },
      );
      await settle();
      if (source === 'device') uncaptured();
      if (source === 'fail') staged.fail(new Error(source));
      step(0);
      staged.fail(new Error('later'));
      uncaptured();
      step(16);

      expect(onError.mock.calls).toEqual([[expect.objectContaining({ message: source })]]);
      expect(waiting()).toBe(false);
      expect(destroy).not.toHaveBeenCalled();
    },
  );

  it('reports nothing once destroyed, such as an error scope that resolves late', async () => {
    const { root, surface, createScene, schedule } = staging();
    const onError = vi.fn();
    const staged = await stage(root, surface, createScene, { name: 'reef' }, { schedule, onError });

    staged.destroy();
    staged.fail(new Error('late'));

    expect(onError).not.toHaveBeenCalled();
  });

  it('calls frame on a class scene, so it keeps its this', async () => {
    const { root, surface, schedule, step } = staging();
    class Counter {
      frames = 0;
      frame() {
        this.frames += 1;
      }
      destroy() {}
    }
    const onError = vi.fn();

    const staged = await stage(root, surface, () => new Counter(), undefined, {
      schedule,
      onError,
    });
    step(0, 16);

    expect(onError).not.toHaveBeenCalled();
    expect(staged.scene.frames).toBe(2);
  });

  it('stops at an async frame before submitting it', async () => {
    const { root, log, surface, schedule, step } = staging();
    const onError = vi.fn();
    const onFrame = vi.fn();
    await stage(root, surface, () => ({ async frame() {}, destroy() {} }), undefined, {
      schedule,
      onError,
      onFrame,
    });
    step(0);

    expect(onError.mock.calls).toEqual([
      [expect.objectContaining({ message: expect.stringMatching(/must be synchronous/) })],
    ]);
    expect(onFrame).not.toHaveBeenCalled();
    expect(log.submissions).toBe(0);
  });

  it('runs every cleanup and release when some throw, then rethrows their errors', async () => {
    const { root, surface, schedule, step, waiting } = staging();
    const release = vi.fn();
    const staged = await stage(
      root,
      surface,
      () => ({ frame() {}, destroy: throwing('scene') }),
      undefined,
      {
        schedule,
        onError: () => {},
        release,
        attach: () => ({ destroy: throwing('attach') }),
      },
    );
    step(0);

    let thrown: unknown;
    try {
      staged.destroy();
    } catch (error) {
      thrown = error;
    }
    staged.destroy();

    expect((thrown as AggregateError).errors.map((error: Error) => error.message)).toEqual([
      'attach',
      'scene',
    ]);
    expect(waiting()).toBe(false);
    expect(staged.loop.run(1)).toBe(false);
    expect(release).toHaveBeenCalledOnce();
  });

  it('rethrows what createScene throws, and leaves the root and the surface to the host', async () => {
    const { root, log, surface, schedule } = staging();
    const failing = stage(
      root,
      surface,
      () => Promise.reject(new Error('createScene')),
      {},
      {
        schedule,
        onError: () => {},
      },
    );

    await expect(failing).rejects.toThrow('createScene');
    expect(log.alive()).toEqual(['[Texture "offscreen surface"]']);
  });

  it('animates nothing without a schedule, and leaves the frames to the caller', async () => {
    const { root, surface, order, createScene } = staging();
    const requested = vi.fn();
    vi.stubGlobal('requestAnimationFrame', requested);

    const staged = await stage(
      root,
      surface,
      createScene,
      { name: 'reef' },
      { schedule: false, onError: () => {} },
    );
    staged.loop.run(0);
    vi.unstubAllGlobals();

    expect(requested).not.toHaveBeenCalled();
    expect(order).toEqual(['frame reef']);
  });

  it('frees what attach returned before the scene, then runs release', async () => {
    const { root, surface, order, createScene, schedule } = staging();
    const staged = await stage(
      root,
      surface,
      createScene,
      { name: 'reef' },
      {
        schedule,
        onError: () => {},
        attach: (held) => {
          order.push(`attach ${held.surface === surface}`);
          return { destroy: () => order.push('detach') };
        },
        release: () => order.push('release'),
      },
    );

    staged[Symbol.dispose]();
    staged.destroy();
    expect(order).toEqual(['attach true', 'detach', 'destroy', 'release']);
  });

  it.each([
    { buffer: { destroy() {} } },
    { [Symbol.dispose]() {} },
    { destroy: true },
    null,
    false,
  ])('rejects an invalid attachment at setup: %s', async (attachment) => {
    const { root, surface, order, createScene } = staging();
    await expect(
      stage(
        root,
        surface,
        createScene,
        { name: 'reef' },
        {
          schedule: false,
          onError: () => {},
          // @ts-expect-error an attachment must be a cleanup function or have destroy()
          attach: () => attachment,
        },
      ),
    ).rejects.toThrow('`attach` must return a cleanup function or an object with `destroy()`.');
    expect(order).toEqual(['destroy']);
  });

  it('is ready once the scene says so', async () => {
    const { root, surface, schedule } = staging();
    const { promise, resolve } = Promise.withResolvers<void>();
    const onError = vi.fn();

    const staged = await stage(root, surface, scene(promise), {}, { schedule, onError });
    expect(staged.ready).toBe(false);
    resolve();
    await settle();
    expect(staged.ready).toBe(true);
    expect(onError).not.toHaveBeenCalled();
  });
});

describe('watch', () => {
  it('reports the first uncaptured error or loss once, and nothing once disposed', async () => {
    const { promise: lost, resolve: lose } = Promise.withResolvers<GPUDeviceLostInfo>();
    const events = new EventTarget();
    const device = {
      lost,
      addEventListener: events.addEventListener.bind(events),
      removeEventListener: events.removeEventListener.bind(events),
    } as unknown as GPUDevice;
    const uncaptured = (message: string) =>
      events.dispatchEvent(Object.assign(new Event('uncapturederror'), { error: { message } }));

    const onError = vi.fn();
    const watching = watch(device, onError);
    uncaptured('bad');
    uncaptured('worse');
    lose({ message: 'removed', reason: 'unknown' } as GPUDeviceLostInfo);
    await settle();
    expect(onError.mock.calls).toEqual([[expect.objectContaining({ message: 'bad' })]]);

    const later = vi.fn();
    watch(device, later)[Symbol.dispose]();
    watching[Symbol.dispose]();
    uncaptured('ignored');
    expect(later).not.toHaveBeenCalled();
  });

  it('takes a device destroyed on purpose for no loss', async () => {
    const lost = Promise.resolve({ message: '', reason: 'destroyed' } as GPUDeviceLostInfo);
    const device = {
      lost,
      addEventListener() {},
      removeEventListener() {},
    } as unknown as GPUDevice;
    const onError = vi.fn();

    watch(device, onError);
    await settle();
    expect(onError).not.toHaveBeenCalled();
  });
});
