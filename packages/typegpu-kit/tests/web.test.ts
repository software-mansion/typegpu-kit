import type { TgpuRoot } from 'typegpu';
import { afterEach, expect, expectTypeOf, it, vi } from 'vitest';
import { frames, type Frame, type Surface } from '../src/index.ts';
import * as native from '../src/native.ts';
import { recordingRoot, steppedSchedule } from '../src/testing.ts';
import * as web from '../src/web.ts';

type Resize = (entries: unknown[]) => void;

/** The web surface presents by itself, so its views read no frame. */
const stub = {} as Frame;

function fakePage() {
  let resize: Resize | undefined;

  vi.stubGlobal('window', { devicePixelRatio: 2 });
  vi.stubGlobal(
    'ResizeObserver',
    class {
      constructor(callback: Resize) {
        resize = callback;
      }

      observe() {}
      disconnect() {}
    },
  );

  const context = {
    configure: vi.fn(),
    unconfigure: vi.fn(),
    getCurrentTexture: vi.fn(() => ({ createView: () => ({}) })),
  };
  const canvas = {
    width: 300,
    height: 150,
    getContext: () => context,
    getBoundingClientRect: () => ({ width: 400, height: 300 }),
  } as unknown as HTMLCanvasElement;

  const layout = (width: number, height: number) =>
    resize?.([{ devicePixelContentBoxSize: [{ inlineSize: width, blockSize: height }] }]);

  return { canvas, context, layout };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

it('sizes the canvas from its layout when created', () => {
  const { root } = recordingRoot();
  const { canvas } = fakePage();

  const surface = web.surface(root, canvas, { format: 'bgra8unorm' });

  expect([canvas.width, canvas.height]).toEqual([800, 600]);
  expect([surface.width, surface.height]).toEqual([800, 600]);
});

it('resizes the canvas at the next view, not when the layout changes', () => {
  const { root } = recordingRoot();
  const { canvas, layout } = fakePage();
  const surface = web.surface(root, canvas, { format: 'bgra8unorm' });

  const sizes: number[][] = [];
  const loop = frames.create(root, (frame) => {
    sizes.push([surface.width, canvas.width]);
    surface.view(frame);
    sizes.push([surface.width, canvas.width]);
  });

  loop.run(0);
  layout(1000, 700);
  expect(canvas.width).toBe(800);
  loop.run(1);

  expect(sizes).toEqual([
    [800, 800],
    [800, 800],
    [1000, 800],
    [1000, 1000],
  ]);
});

it('caps the pixel ratio when the layout changes, and reports the ratio it chose', () => {
  const { root } = recordingRoot();
  const { canvas, layout } = fakePage();
  vi.stubGlobal('window', { devicePixelRatio: 3 });
  const surface = web.surface(root, canvas, { format: 'bgra8unorm' });

  layout(1170, 2532);
  surface.view(stub);

  expect([canvas.width, canvas.height]).toEqual([780, 1688]);
  expect(surface.pixelRatio).toBe(2);

  vi.stubGlobal('window', { devicePixelRatio: 1.5 });
  layout(585, 1266);
  expect(surface.pixelRatio).toBe(1.5);
});

it('keeps a fixed size from creation, whatever the layout and the screen', () => {
  const { root } = recordingRoot();
  const { canvas, layout } = fakePage();

  const surface = web.surface(root, canvas, {
    format: 'bgra8unorm',
    size: { width: 2560, height: 1440 },
  });
  expect([canvas.width, canvas.height, surface.pixelRatio]).toEqual([2560, 1440, 1]);

  vi.stubGlobal('window', { devicePixelRatio: 3 });
  layout(1170, 2532);
  surface.view(stub);

  expect([canvas.width, canvas.height]).toEqual([2560, 1440]);
  expect([surface.width, surface.height, surface.pixelRatio]).toEqual([2560, 1440, 1]);
});

it('fixes the size at the next view, and follows the layout again without one', () => {
  const { root } = recordingRoot();
  const { canvas, layout } = fakePage();
  const surface = web.surface(root, canvas, { format: 'bgra8unorm' });

  surface.setSize({ width: 1920, height: 1080, pixelRatio: 1.5 });
  expect([surface.width, surface.pixelRatio, canvas.width]).toEqual([1920, 1.5, 800]);
  surface.view(stub);
  expect([canvas.width, canvas.height]).toEqual([1920, 1080]);

  layout(1000, 700);
  surface.setSize(undefined);
  surface.view(stub);
  expect([canvas.width, canvas.height, surface.pixelRatio]).toEqual([1000, 700, 2]);
});

it('rejects a fixed size the device cannot draw, before configuring the canvas', () => {
  const { root } = recordingRoot();
  const { canvas, context } = fakePage();
  const limit = root.device.limits.maxTextureDimension2D;

  for (const size of [
    { width: 0, height: 100 },
    { width: 100.5, height: 100 },
    { width: limit + 1, height: 100 },
  ]) {
    expect(() => web.surface(root, canvas, { format: 'bgra8unorm', size })).toThrow(/whole pixels/);
  }
  expect(context.configure).not.toHaveBeenCalled();

  const surface = web.surface(root, canvas, { format: 'bgra8unorm' });
  expect(() => surface.setSize({ width: 10, height: 10, pixelRatio: 0 })).toThrow(/above 0/);
});

it('takes touch drags from the page while orbit controls are on', () => {
  const element = {
    style: { touchAction: 'auto' },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;

  const controls = web.orbitControls({ goal: { yaw: 0, pitch: 0, distance: 1 } }, element);
  expect(element.style.touchAction).toBe('none');

  controls[Symbol.dispose]();

  expect(element.style.touchAction).toBe('auto');
  expect(element.removeEventListener).toHaveBeenCalledTimes(6);
});

it('zooms with a pinch and orbits with one finger that reports no movement', () => {
  type Listener = (event: Partial<PointerEvent>) => void;
  const listeners = new Map<string, Listener>();
  const element = {
    style: { touchAction: 'auto' },
    setPointerCapture: vi.fn(),
    addEventListener: (type: string, listener: Listener) => listeners.set(type, listener),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const send = (type: string, event: Partial<PointerEvent>) => listeners.get(type)!(event);

  const orbit = { goal: { yaw: 0, pitch: 0, distance: 8 } };
  web.orbitControls(orbit, element);

  send('pointerdown', { pointerId: 1, clientX: 100, clientY: 100 });
  send('pointerdown', { pointerId: 2, clientX: 200, clientY: 100 });
  send('pointermove', { pointerId: 2, clientX: 300, clientY: 100, buttons: 1, movementX: 100 });

  expect(orbit.goal).toEqual({ yaw: 0, pitch: 0, distance: 4 });

  // WebKit reports a touch's movementX as 0, and a lost capture may end a finger without pointerup
  send('lostpointercapture', { pointerId: 2 });
  send('pointermove', { pointerId: 1, clientX: 110, clientY: 100, buttons: 1, movementX: 0 });

  expect(orbit.goal.yaw).toBeCloseTo(-0.08);
  expect(orbit.goal.distance).toBe(4);
});

it('orbits with the secondary button and two fingers, leaving the primary button to others', () => {
  type Listener = (event: Partial<PointerEvent>) => void;
  const listeners = new Map<string, Listener>();
  const element = {
    style: { touchAction: 'auto' },
    setPointerCapture: vi.fn(),
    addEventListener: (type: string, listener: Listener) => listeners.set(type, listener),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const send = (type: string, event: Partial<PointerEvent>) => listeners.get(type)!(event);

  const orbit = { goal: { yaw: 0, pitch: 0, distance: 8 } };
  web.orbitControls(orbit, element, { button: 'secondary' });
  expect(listeners.has('contextmenu')).toBe(true);

  send('pointerdown', { pointerId: 1, clientX: 100, clientY: 100 });
  send('pointermove', { pointerId: 1, clientX: 110, clientY: 100, buttons: 1, movementX: 10 });
  expect(orbit.goal.yaw).toBe(0);
  send('pointermove', { pointerId: 1, clientX: 120, clientY: 100, buttons: 2, movementX: 10 });
  expect(orbit.goal.yaw).toBeCloseTo(-0.08);
  send('pointerup', { pointerId: 1 });

  // Spreading from 100 to 200 pixels halves the distance; the midpoint moves 50 pixels
  send('pointerdown', { pointerId: 2, clientX: 100, clientY: 100 });
  send('pointerdown', { pointerId: 3, clientX: 200, clientY: 100 });
  send('pointermove', { pointerId: 3, clientX: 300, clientY: 100, buttons: 1, movementX: 100 });
  expect(orbit.goal.distance).toBe(4);
  expect(orbit.goal.yaw).toBeCloseTo(-0.08 - 50 * 0.008);
});

it('orbits with the middle button and pans with the secondary, keeping autoscroll and menus shut', () => {
  type Listener = (event: Partial<PointerEvent>) => void;
  const listeners = new Map<string, Listener>();
  const element = {
    clientHeight: 100,
    style: { touchAction: 'auto' },
    setPointerCapture: vi.fn(),
    addEventListener: (type: string, listener: Listener) => listeners.set(type, listener),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const send = (type: string, event: Partial<PointerEvent>) => listeners.get(type)!(event);

  const orbit = {
    goal: { yaw: 0, pitch: 0, distance: 10, target: [0, 0, 0] as [number, number, number] },
  };
  web.orbitControls(orbit, element, { button: 'middle', pan: true });

  send('pointerdown', { pointerId: 1, clientX: 0, clientY: 0 });
  send('pointermove', { pointerId: 1, buttons: 1, movementX: 10, movementY: 10 });
  expect(orbit.goal).toMatchObject({ yaw: 0, target: [0, 0, 0] });
  send('pointermove', { pointerId: 1, buttons: 2, movementX: 10, movementY: 20 });
  expect(orbit.goal.target).toEqual([-1, 2, 0]);
  send('pointermove', { pointerId: 1, buttons: 4, movementX: 10, movementY: 0 });
  expect(orbit.goal.yaw).toBeCloseTo(-0.08);

  const press = { button: 1, preventDefault: vi.fn() };
  send('mousedown', press);
  send('contextmenu', press);
  expect(press.preventDefault).toHaveBeenCalledTimes(2);
});

it('leaves vertical swipes and the plain wheel to a page that scrolls', () => {
  const listeners = new Map<string, (event: Partial<WheelEvent>) => void>();
  const element = {
    style: { touchAction: 'auto' },
    addEventListener: (type: string, listener: (event: Partial<WheelEvent>) => void) =>
      listeners.set(type, listener),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const wheel = (event: Partial<WheelEvent>) =>
    listeners.get('wheel')!({ preventDefault: vi.fn(), ...event });

  const orbit = { goal: { yaw: 0, pitch: 0, distance: 8 } };
  web.orbitControls(orbit, element, { page: true });
  expect(element.style.touchAction).toBe('pan-y');

  wheel({ deltaY: 100 });
  expect(orbit.goal.distance).toBe(8);

  const pinch = { deltaY: 100, ctrlKey: true, preventDefault: vi.fn() };
  wheel(pinch);
  expect(orbit.goal.distance).toBeCloseTo(8 * Math.exp(0.1));
  expect(pinch.preventDefault).toHaveBeenCalled();
});

/** An element that runs every listener of an event's type, as a pointer and controls share it. */
function sharedElement() {
  type Listener = (event: Partial<PointerEvent & WheelEvent>) => void;
  const listeners: [string, Listener][] = [];
  const element = {
    style: { touchAction: 'auto' },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
    setPointerCapture: vi.fn(),
    addEventListener: (type: string, listener: Listener) => listeners.push([type, listener]),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const send = (type: string, event: Partial<PointerEvent & WheelEvent>) => {
    for (const [on, listener] of listeners) if (on === type) listener(event);
  };
  return { element, send };
}

it('restores the original touch-action when controls borrow a pointer destroyed first', () => {
  const { element } = sharedElement();
  const input = web.pointer(element);
  const controls = web.orbitControls({ goal: { yaw: 0, pitch: 0, distance: 8 } }, element, {
    pointer: input,
  });

  input.destroy();
  controls.destroy();
  expect(element.style.touchAction).toBe('auto');
});

it('leaves touch-action and capture to a borrowed pointer, and its own page to the wheel', () => {
  const { element, send } = sharedElement();
  const orbit = { goal: { yaw: 0, pitch: 0, distance: 8 } };
  const input = web.pointer(element, { page: true });
  web.orbitControls(orbit, element, { pointer: input });
  expect(element.style.touchAction).toBe('pan-y');

  send('pointerdown', { pointerId: 1, clientX: 0, clientY: 0, buttons: 1 });
  send('pointermove', { pointerId: 1, clientX: 10, clientY: 0, buttons: 1 });
  expect(element.setPointerCapture).toHaveBeenCalledOnce();
  expect(orbit.goal.yaw).toBeCloseTo(-0.08);

  send('wheel', { deltaY: 100, preventDefault: vi.fn() });
  expect(orbit.goal.distance).toBeCloseTo(8 * Math.exp(0.1));
});

it('shares one Surface type with the native surface', () => {
  expectTypeOf(web.surface).returns.toExtend<Surface>();
  expectTypeOf(native.surface).returns.toExtend<Surface>();
});

it('pans the target with the secondary button, zooms at zoomSpeed and says when it moved', () => {
  type Listener = (event: Partial<PointerEvent & WheelEvent>) => void;
  const listeners = new Map<string, Listener>();
  const element = {
    clientHeight: 100,
    style: { touchAction: 'auto' },
    setPointerCapture: vi.fn(),
    addEventListener: (type: string, listener: Listener) => listeners.set(type, listener),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const send = (type: string, event: Partial<PointerEvent & WheelEvent>) =>
    listeners.get(type)!({ preventDefault: vi.fn(), ...event });

  const orbit = {
    goal: { yaw: 0, pitch: 0, distance: 10, target: [0, 0, 0] as [number, number, number] },
  };
  const onChange = vi.fn();
  const controls = web.orbitControls(orbit, element, { pan: true, zoomSpeed: 0.01, onChange });

  send('pointerdown', { pointerId: 1, clientX: 0, clientY: 0 });
  send('pointermove', { pointerId: 1, buttons: 2, movementX: 10, movementY: 20 });
  expect(orbit.goal.target).toEqual([-1, 2, 0]);
  expect(orbit.goal.yaw).toBe(0);

  send('wheel', { deltaY: 100 });
  expect(orbit.goal.distance).toBeCloseTo(10 * Math.E);
  expect(onChange).toHaveBeenCalledTimes(2);

  controls.destroy();
  expect(element.removeEventListener).toHaveBeenCalledTimes(7);
  expect(() =>
    web.orbitControls({ goal: { yaw: 0, pitch: 0, distance: 1 } }, element, { pan: true }),
  ).toThrow(/goal\.target/);
});

/** A scene as the examples write one: it clears the canvas and counts its frames. */
function counting(_root: TgpuRoot, surface: Surface, options: { label: string }) {
  let count = 0;

  return {
    label: options.label,
    destroy: vi.fn(),

    get frames() {
      return count;
    },

    frame(frame: Frame) {
      count += 1;
      frame.encoder.beginRenderPass({ colorAttachments: [{ view: surface.view(frame) }] }).end();
    },
  };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
const format: GPUTextureFormat = 'bgra8unorm';

const sent = () => {
  throw new Error('sent');
};

const broken = () => {
  throw new Error('create');
};

it('runs a scene on the canvas, with page code after each frame, until destroyed', async () => {
  const { root, log } = recordingRoot();
  const { canvas, context } = fakePage();
  const display = steppedSchedule();
  const seen: string[] = [];

  const hosted = web.host(
    counting,
    { label: 'reef' },
    {
      canvas,
      device: root.device,
      format,
      schedule: display,
      onFrame: ({ scene }, { time, dt }) =>
        seen.push(`${scene.label} ${scene.frames} ${time} ${dt}`),
    },
  );
  hosted.send(({ scene }) => seen.push(`sent ${scene.frames}`));
  await settle();

  display.step(1000, 1062.5);
  hosted.pause();
  expect(await hosted.run(({ surface }) => surface.paused)).toBe(true);
  display.step(2000, 9000);
  hosted.resume();
  await settle();
  display.step(9500, 9562.5);

  expect(seen).toEqual([
    'sent 0',
    'reef 1 0 0',
    'reef 2 0.0625 0.0625',
    'reef 3 0.0625 0',
    'reef 4 0.125 0.0625',
  ]);
  expect(log.submissions).toBe(4);

  const made = await hosted.run(({ scene }) => scene);
  hosted.destroy();
  await settle();

  expect(made.destroy).toHaveBeenCalledOnce();
  expect(context.unconfigure).toHaveBeenCalledOnce();
  expect(display.waiting).toBe(false);
});

it('runs on a root it is given, without frames of its own, and leaves the root alive', async () => {
  const { root } = recordingRoot();
  const { canvas } = fakePage();
  const destroyRoot = vi.spyOn(root, 'destroy');
  const order: string[] = [];

  const hosted = web.host(
    counting,
    { label: 'reef' },
    {
      canvas,
      root,
      format,
      schedule: false,
      attach: ({ scene }) => {
        order.push(`attach ${scene.label}`);
        return { destroy: () => order.push('detach') };
      },
    },
  );
  await hosted.run(({ loop }) => loop.run(0));
  expect(await hosted.run(({ scene }) => scene.frames)).toBe(1);

  hosted.destroy();
  await settle();

  expect(order).toEqual(['attach reef', 'detach']);
  expect(destroyRoot).not.toHaveBeenCalled();
});

/** A scene whose options may be left out. */
const empty = (_root: TgpuRoot, _surface: Surface, options: { label?: string } = {}) => ({
  label: options.label ?? 'none',
  frame() {},
  destroy() {},
});

it('takes no scene options for a scene that needs none', async () => {
  const { root } = recordingRoot();
  const { canvas } = fakePage();

  const hosted = web.host(empty, { canvas, root, format, schedule: false });
  expect(await hosted.run(({ scene }) => scene.label)).toBe('none');
  hosted.destroy();

  const needing = () =>
    // @ts-expect-error a scene that needs options takes them
    web.host(counting, { canvas, root });
  void needing;
  expect(() =>
    web.host(counting, { label: 'reef' }, { canvas, root, device: root.device }),
  ).toThrow(/either device or root/);
});

it('reports each sent closure that throws and keeps drawing, then stops once at the device', async () => {
  const { root } = recordingRoot();
  const { canvas } = fakePage();
  const display = steppedSchedule();
  const events = new EventTarget();
  Object.assign(root.device, { addEventListener: events.addEventListener.bind(events) });
  const onError = vi.fn();

  const hosted = web.host(
    counting,
    { label: 'reef' },
    { canvas, device: root.device, format, schedule: display, onError },
  );
  hosted.send(sent);
  hosted.send(sent);
  await settle();
  display.step(0);

  const uncaptured = () =>
    events.dispatchEvent(
      Object.assign(new Event('uncapturederror'), { error: { message: 'bad' } }),
    );
  uncaptured();
  uncaptured();
  display.step(16);

  expect(onError.mock.calls.map(([error]) => (error as Error).message)).toEqual([
    'sent',
    'sent',
    'bad',
  ]);
  expect(await hosted.run(({ scene }) => scene.frames)).toBe(1);
  expect(display.waiting).toBe(false);
});

it('releases the canvas when createScene throws', async () => {
  const { root } = recordingRoot();
  const { canvas, context } = fakePage();
  const onError = vi.fn();

  web.host(broken, { canvas, device: root.device, format, schedule: false, onError });
  await settle();

  expect(onError.mock.calls[0]?.[0]).toMatchObject({ message: 'create' });
  expect(context.unconfigure).toHaveBeenCalledOnce();
});

it('zooms the pointer by the wheel and by a two-finger pinch', () => {
  type Listener = (event: Partial<PointerEvent & WheelEvent>) => void;
  const listeners = new Map<string, Listener>();
  const element = {
    style: { touchAction: 'auto' },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 400, height: 200 }),
    setPointerCapture: vi.fn(),
    addEventListener: (type: string, listener: Listener) => listeners.set(type, listener),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const send = (type: string, event: Partial<PointerEvent & WheelEvent>) =>
    listeners.get(type)!({ preventDefault: vi.fn(), ...event });

  const zooms: number[] = [];
  web.pointer(element, { zoom: (by) => zooms.push(by) });
  send('wheel', { deltaY: 100 });
  send('pointerdown', { pointerId: 1, clientX: 100, clientY: 0 });
  send('pointerdown', { pointerId: 2, clientX: 200, clientY: 0 });
  send('pointermove', { pointerId: 2, clientX: 300, clientY: 0 });
  expect(zooms).toEqual([Math.exp(0.1), 0.5]);
});

it('reports the pointer from 0 to 1 across the element, +y down, through a drag past its edge', () => {
  type Listener = (event: Partial<PointerEvent>) => void;
  const listeners = new Map<string, Listener>();
  const element = {
    style: { touchAction: 'auto' },
    getBoundingClientRect: () => ({ left: 100, top: 50, width: 400, height: 200 }),
    setPointerCapture: vi.fn(),
    addEventListener: (type: string, listener: Listener) => listeners.set(type, listener),
    removeEventListener: vi.fn(),
  } as unknown as HTMLElement;
  const send = (type: string, event: Partial<PointerEvent>) => listeners.get(type)!(event);

  const seen: [string, number, number][] = [];
  const pointer = web.pointer(element, {
    down: (x, y) => seen.push(['down', x, y]),
    up: (x, y) => seen.push(['up', x, y]),
    leave: () => seen.push(['leave', 0, 0]),
  });
  expect(element.style.touchAction).toBe('none');

  send('pointermove', { clientX: 200, clientY: 100, buttons: 0 });
  expect([pointer.x, pointer.y, pointer.over]).toEqual([0.25, 0.25, true]);

  send('pointerdown', { pointerId: 1, clientX: 300, clientY: 200, buttons: 1 });
  send('pointermove', { clientX: 700, clientY: 250, buttons: 1 });
  send('pointerleave', { buttons: 1 });
  expect([pointer.x, pointer.y, pointer.over, pointer.buttons]).toEqual([1.5, 1, true, 1]);

  send('pointerup', { clientX: 700, clientY: 250, buttons: 0 });
  send('pointerleave', { buttons: 0 });
  expect(pointer.over).toBe(false);
  expect(element.setPointerCapture).toHaveBeenCalledWith(1);
  expect(seen).toEqual([
    ['down', 0.5, 0.75],
    ['up', 1.5, 1],
    ['leave', 0, 0],
  ]);

  pointer.destroy();
  expect(element.style.touchAction).toBe('auto');
});

/** The page's animation frames, run by hand. */
function handFrames() {
  const display = steppedSchedule();
  vi.stubGlobal('requestAnimationFrame', display.request);
  vi.stubGlobal('cancelAnimationFrame', display.cancel);
  return display;
}

/** A queue whose submitted work finishes when the test says. */
function handQueue() {
  const finish: (() => void)[] = [];
  return {
    onSubmittedWorkDone: () =>
      new Promise<undefined>((resolve) => finish.push(() => resolve(undefined))),
    finish: (fence: number) => finish[fence](),
  };
}

const settled = () => new Promise((resolve) => setTimeout(resolve, 0));

it('paces frames so the GPU has at most inFlight frames of work unfinished', async () => {
  const display = handFrames();
  const queue = handQueue();
  const { root } = recordingRoot();
  const times: number[] = [];
  const loop = frames.create(root, ({ time }) => times.push(time));
  const animation = frames.animate(loop, { schedule: web.paced(queue, { inFlight: 2 }) });

  expect(display.waiting).toBe(true);
  display.step(0);
  // The second frame waits for what was submitted before the first.
  expect(display.waiting).toBe(false);
  queue.finish(0);
  await settled();
  expect(display.waiting).toBe(true);

  display.step(10);
  await settled();
  // The third waits for the first frame's work.
  expect(display.waiting).toBe(false);
  queue.finish(2);
  await settled();
  expect(display.waiting).toBe(false);
  queue.finish(1);
  await settled();
  expect(display.waiting).toBe(true);
  display.step(20);

  // Destroyed while a frame waits for the GPU, it asks for no frame once the work is done.
  animation.destroy();
  queue.finish(3);
  await settled();
  expect(display.waiting).toBe(false);
  expect(times).toEqual([0, 0.01, 0.02]);
});

it('paces with a whole number of frames in flight, at least 1', () => {
  const queue = handQueue();
  expect(() => web.paced(queue, { inFlight: 0 })).toThrow('above 0');
  expect(() => web.paced(queue, { inFlight: 1.5 })).toThrow('integer');
});

const videoDevice = () => ({ importExternalTexture: vi.fn(() => ({}) as GPUExternalTexture) });

it('imports a video frame once per device for the readers of one task', async () => {
  const element = {
    readyState: 4,
    HAVE_CURRENT_DATA: 2,
    videoWidth: 4,
    videoHeight: 2,
    currentTime: 1,
  } as HTMLVideoElement;
  const [one, two] = [videoDevice(), videoDevice()];
  const source = web.video(element);
  const read = (device: typeof one, seen?: number) =>
    source.latest(device as unknown as GPUDevice, seen)?.texture;

  const shown = read(one);
  expect(read(one, 0)).toBe(shown);
  expect(read(two)).not.toBe(shown);
  expect(read(one, 1)).toBeUndefined();
  await Promise.resolve();
  expect(read(one)).not.toBe(shown);

  expect(one.importExternalTexture).toHaveBeenCalledTimes(2);
  expect(two.importExternalTexture).toHaveBeenCalledOnce();
});
