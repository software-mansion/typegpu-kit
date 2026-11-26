import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fake from '../fake-worklets.ts';

vi.mock('react-native-worklets', () => import('../fake-worklets.ts'));
vi.mock('react-native-webgpu', () => ({ installWebGPU: vi.fn() }));

const { installWebGPU } = await import('react-native-webgpu');
const runtimes = await import('../../src/native/runtimes.ts');

const RN = fake.RuntimeKind.ReactNative;

beforeEach(() => {
  fake.reset();
  vi.stubGlobal('requestAnimationFrame', fake.requestAnimationFrame);
  vi.stubGlobal('cancelAnimationFrame', fake.cancelAnimationFrame);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const callback = () => {};
const plain = () => {};

/** Runs `fn` on `runtime` at once, as code already running there would. */
function onRuntime(runtime: { runtimeId: number }, fn: () => void) {
  fake.scheduleOnRuntimeWithId(runtime.runtimeId, fn);
  return fake.settle();
}

describe('worker', () => {
  it('installs WebGPU and floors zero-delay timers, with their arguments, until destroyed', async () => {
    const worker = runtimes.worker({ timerFloor: 2 });
    const timeout = vi.fn(setTimeout);

    vi.stubGlobal('setTimeout', timeout);
    fake.initialize(worker.runtimeId);
    setTimeout(callback, 0, 'payload');
    setTimeout(callback, 5);
    worker.destroy();
    await fake.settle();

    expect(installWebGPU).toHaveBeenCalledOnce();
    expect(timeout.mock.calls.slice(0, 2)).toEqual([
      [callback, 2, 'payload'],
      [callback, 5],
    ]);
    expect(globalThis.setTimeout).toBe(timeout);
  });
});

describe('vsync', () => {
  it('relays one display frame to the runtime per request', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const display = runtimes.vsync(worker);
    const seen: [number, number][] = [];
    const frame = (milliseconds: number) => seen.push([milliseconds, fake.currentRuntime()]);

    await fake.settle();
    await onRuntime(worker, () => display.request(frame));
    fake.display(16);
    fake.display(24);
    await fake.settle();

    expect(seen).toEqual([[16, worker.runtimeId]]);
    expect(display.missed()).toBe(0);
  });

  it('counts and skips display frames that come while a frame is running', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const display = runtimes.vsync(worker);
    const seen: number[] = [];

    const frame = (milliseconds: number) => {
      seen.push(milliseconds);
      display.request(frame);
      if (milliseconds === 16) fake.display(24); // the display ticks before this frame ends
    };

    await fake.settle();
    await onRuntime(worker, () => display.request(frame));
    fake.display(16);
    await fake.settle();
    fake.display(32);
    await fake.settle();

    expect(seen).toEqual([16, 32]);
    expect(display.missed()).toBe(1);
  });

  it('cancels a request and stops relaying once destroyed', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const display = runtimes.vsync(worker);
    const frame = vi.fn();

    await fake.settle();
    await onRuntime(worker, () => display.cancel(display.request(frame)));
    fake.display(16);
    await fake.settle();

    display.destroy();
    await onRuntime(worker, () => display.request(frame));
    fake.display(32);
    fake.display(48);
    await fake.settle();

    expect(frame).not.toHaveBeenCalled();
  });

  it('drops a display frame already relayed when the request is destroyed or replaced', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const display = runtimes.vsync(worker);
    const first = vi.fn();
    const second = vi.fn();

    await fake.settle();
    let request = 0;
    await onRuntime(worker, () => (request = display.request(first)));
    // Runs on the worker before the frame the display relays for the first request
    fake.scheduleOnRuntimeWithId(worker.runtimeId, () => {
      display.cancel(request);
      display.request(second);
    });
    fake.display(16);
    await fake.settle();
    fake.display(32);
    await fake.settle();
    expect(second.mock.calls).toEqual([[32]]);

    await onRuntime(worker, () => display.request(first));
    fake.display(48);
    display.destroy();
    await fake.settle();
    expect(first).not.toHaveBeenCalled();
  });

  it('asks the display for no frames while nothing requests one, and wakes once at a request', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const display = runtimes.vsync(worker);
    const frame = vi.fn();

    await fake.settle();
    fake.display(16);
    expect(fake.requestedFrames()).toBe(0);

    await onRuntime(worker, () => {
      display.request(frame);
      display.request(frame);
    });
    expect(fake.requestedFrames()).toBe(1);
    fake.display(32);
    await fake.settle();
    expect(frame.mock.calls).toEqual([[32]]);
  });

  it('takes requests only on its runtime', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const display = runtimes.vsync(worker);

    expect(() => display.request(() => {})).toThrow('on worker only');
  });
});

describe('own', () => {
  it('creates the value on its runtime and runs what was sent after it, in order', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const log: string[] = [];

    const owned = runtimes.own(worker, async () => {
      log.push(`create on ${fake.currentRuntime()}`);
      await Promise.resolve();
      return { total: 0, destroy: () => log.push('destroy') };
    });

    owned.send((value, add: number) => {
      value.total += add;
      log.push(`add ${add} on ${fake.currentRuntime()}`);
    }, 1);
    owned.send((value, add: number) => {
      value.total += add;
      log.push(`add ${add}, total ${value.total}`);
    }, 2);
    owned.destroy();
    owned.send(() => log.push('after destroy'));
    await fake.settle();

    expect(log).toEqual([
      `create on ${worker.runtimeId}`,
      `add 1 on ${worker.runtimeId}`,
      'add 2, total 3',
      'destroy',
    ]);
  });

  it('runs a closure from React Native and returns its result', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const owned = runtimes.own(worker, () => ({ total: 5 }));

    const total = owned.run((value, add: number) => value.total + add, 2);
    const thrown = owned.run(() => {
      throw new Error('no');
    });
    void thrown.catch(() => {});
    await fake.settle();

    await expect(total).resolves.toBe(7);
    await expect(thrown).rejects.toThrow('no (on worker)');

    owned.destroy();
    const late = owned.run(() => 1);
    void late.catch(() => {});
    await fake.settle();
    await expect(late).rejects.toThrow('destroyed');
  });

  it('awaits an async closure where the value lives, and returns what it settles with', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const owned = runtimes.own(worker, () => ({ total: 5 }));

    const total = owned.run(async (value) => {
      await Promise.resolve();
      return { total: value.total };
    });
    const failed = owned.run(async () => {
      throw new Error('late');
    });
    void failed.catch(() => {});
    await fake.settle();

    await expect(total).resolves.toEqual({ total: 5 });
    await expect(failed).rejects.toThrow('late (on worker)');
  });

  it('copies what run returns, so a value returned again arrives as it is now', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const owned = runtimes.own(worker, () => ({ hud: { score: 0 } }));

    const first = owned.run(({ hud }) => hud);
    owned.send(({ hud }) => {
      hud.score = 5;
    });
    const second = owned.run(({ hud }) => hud);
    await fake.settle();

    await expect(first).resolves.toEqual({ score: 0 });
    await expect(second).resolves.toEqual({ score: 5 });
  });

  it('copies binary data and dates that run returns', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const bytes = new Uint8Array([1, 2, 3, 4]);
    const owned = runtimes.own(worker, () => ({
      buffer: bytes.buffer,
      view: new DataView(bytes.buffer, 1, 2),
      pixels: bytes.subarray(2),
      at: new Date(1000),
    }));

    const copied = owned.run((value) => value);
    await fake.settle();
    const { buffer, view, pixels, at } = await copied;

    expect(new Uint8Array(buffer)).toEqual(bytes);
    expect(buffer).not.toBe(bytes.buffer);
    expect([view.byteLength, view.getUint8(0)]).toEqual([2, 2]);
    expect(pixels).toEqual(new Uint8Array([3, 4]));
    expect(at.getTime()).toBe(1000);
  });

  it('hands a send that worklets cannot schedule to onError, not to the caller', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const onError = vi.fn();
    const owned = runtimes.own(worker, () => ({}), { onError });

    await onRuntime({ runtimeId: fake.UIRuntimeId }, () =>
      owned.send((_value, _promise: Promise<void>) => {}, Promise.resolve()),
    );
    await fake.settle();

    expect(onError.mock.calls[0]?.[0]).toMatchObject({
      message: '[Worklets] Promises cannot be serialized. (on worker)',
    });
  });

  it('reports what create and sent closures throw, on React Native', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const errors: [string, number][] = [];
    const onError = (error: unknown) =>
      errors.push([(error as Error).message, fake.currentRuntime()]);

    const owned = runtimes.own(worker, () => ({}), { onError });
    owned.send(() => {
      throw new Error('send');
    });
    const broken = runtimes.own(
      worker,
      () => {
        throw new Error('create');
      },
      { onError },
    );
    const never = broken.run(() => 1);
    void never.catch(() => {});
    await fake.settle();

    expect(errors.toSorted()).toEqual([
      ['create (on worker)', RN],
      ['send (on worker)', RN],
    ]);
    await expect(never).rejects.toThrow('destroyed');
  });

  it('calls a worklet onError on the runtime', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const onError = fake.worklet(vi.fn((_error: unknown) => fake.currentRuntime()));

    runtimes.own(
      worker,
      () => {
        throw new Error('create');
      },
      { onError },
    );
    await fake.settle();

    expect(onError.mock.results[0]?.value).toBe(worker.runtimeId);
  });
});

describe('post', () => {
  it('calls on React Native with copies of what it gets from another runtime', async () => {
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const seen = vi.fn((_value: unknown) => fake.currentRuntime());
    const posted = runtimes.post(seen);
    const value = {
      list: [1, { deep: true }],
      bytes: new Float32Array([1, 2]),
      error: new Error('x'),
      byName: new Map([['a', { deep: true }]]),
    };

    await onRuntime(worker, () => posted(value));

    const [received] = seen.mock.calls[0];
    expect(seen.mock.results[0]?.value).toBe(RN);
    expect(received).not.toBe(value);
    expect(received).toMatchObject({
      list: [1, { deep: true }],
      bytes: new Float32Array([1, 2]),
      byName: new Map([['a', { deep: true }]]),
    });
    expect((received as typeof value).bytes).not.toBe(value.bytes);
    expect((received as typeof value).error).toBeInstanceOf(Error);
  });

  it('calls at once, with the same arguments, on React Native', () => {
    const seen = vi.fn();
    const value = { a: 1 };

    runtimes.post(seen)(value);

    expect(seen).toHaveBeenCalledWith(value);
  });

  it('leaves worklets to run where they are called', () => {
    const fn = fake.worklet(() => {});

    expect(runtimes.callable(fn)).toBe(fn);
    expect(runtimes.callable(plain)).not.toBe(plain);
  });
});

describe('cell', () => {
  it('writes some fields and reads all into an object the reader keeps', () => {
    const pointer = runtimes.cell({ x: 0, y: 0, down: false });
    const here = { x: -1, y: -1, down: true };

    pointer.write({ x: 3 });
    pointer.write({ down: true });

    expect(pointer.read(here)).toBe(here);
    expect(here).toEqual({ x: 3, y: 0, down: true });
  });
});

const cameraFrame = (pointer: bigint) => ({
  width: 640,
  height: 480,
  orientation: 'right' as const,
  isMirrored: true,
  timestamp: Number(pointer) / 30,
  released: 0,
  getNativeBuffer() {
    return { pointer, release: () => (this.released += 1) };
  },
});
describe('camera', () => {
  class Pixels {
    released = 0;
    release() {
      this.released += 1;
    }
  }
  class External {
    destroyed = 0;
    readonly source: Pixels;
    constructor(source: Pixels) {
      this.source = source;
    }
    destroy() {
      this.destroyed += 1;
    }
  }

  const device = {
    importExternalTexture: ({ source }: { source: Pixels }) => new External(source),
  } as unknown as GPUDevice;

  // react-native-webgpu 0.23 renamed the function that wraps a buffer
  const wraps = ['createNativeBufferFromPointer', 'createVideoFrameFromNativeBuffer'];

  it.each(wraps)('shows every new frame and releases each once, with %s', async (wrap) => {
    const pixels: Pixels[] = [];
    vi.stubGlobal('RNWebGPU', {
      [wrap]: () => {
        pixels.push(new Pixels());
        return pixels.at(-1);
      },
    });
    const worker = fake.createWorkletRuntime({ name: 'worker' });
    const thread = fake.createWorkletRuntime({ name: 'camera' });
    const camera = runtimes.camera(worker);
    const first = cameraFrame(1n);
    const second = cameraFrame(2n);
    type Read = ReturnType<typeof camera.latest>;
    const read: Read[] = [];
    // Two feeds, each wanting only frames it hasn't seen
    const seen = [0, 0];
    const feeds = () => {
      for (let i = 0; i < seen.length; i++) {
        const picture = camera.latest(device, seen[i]);
        read.push(picture);
        if (picture) seen[i] = picture.serial;
      }
    };

    await onRuntime(thread, () => {
      camera.write(first);
      camera.write(second);
    });
    await onRuntime(worker, () => {
      feeds();
      feeds();
    });
    const [one, two, again, still] = read;

    // The camera's buffers go back at once; the frame replaced before any read goes too
    expect([first.released, second.released]).toEqual([1, 1]);
    expect(pixels.map((each) => each.released)).toEqual([1, 0]);
    expect(one).toMatchObject({ width: 640, height: 480, rotation: 90, mirrored: true });
    expect(one).toMatchObject({ serial: 2, time: 2 / 30 });
    expect(two!.texture).toBe(one!.texture);
    expect([again, still]).toEqual([undefined, undefined]);
    await onRuntime(worker, () => {
      expect(() => camera.latest({ ...device })).toThrow('on one device');
    });

    await onRuntime(thread, () => camera.write(cameraFrame(3n)));
    const importFailure = new Error('Could not import a camera frame.');
    vi.spyOn(device, 'importExternalTexture').mockImplementationOnce(() => {
      throw importFailure;
    });
    await onRuntime(worker, () => {
      expect(() => camera.latest(device)).toThrow(importFailure);
    });
    expect((one!.texture as unknown as External).destroyed).toBe(0);
    expect(pixels.map((each) => each.released)).toEqual([1, 0, 0]);
    await onRuntime(worker, () => {
      feeds();
      read.push(camera.latest(device));
    });
    camera.destroy();
    await fake.settle();
    await onRuntime(thread, () => camera.write(cameraFrame(4n)));

    expect(read.slice(4).map((each) => each?.serial)).toEqual([3, 3, 3]);
    expect((one!.texture as unknown as External).destroyed).toBe(1);
    expect(pixels.map((each) => each.released)).toEqual([1, 1, 1, 1]);
  });
});
