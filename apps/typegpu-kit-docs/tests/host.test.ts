import { recordingRoot } from '@typegpu/kit/testing';
import { afterEach, expect, it, vi } from 'vitest';
import type { ExampleScene } from 'typegpu-kit-examples/contract';
import { buttons } from 'typegpu-kit-examples/ui.ts';
import { connectPad, listenKeys, padsOf } from '../src/examples/buttons.ts';
import { film } from '../src/examples/camera.ts';
import { connect, host } from '../src/examples/host.ts';

afterEach(() => {
  vi.unstubAllGlobals();
});

/** A canvas that takes pointer events, 100 by 50 points. */
function fakeCanvas() {
  const captured = new Set<number>();
  const attributes = new Map<string, string>();
  const context = {
    configure: vi.fn(),
    unconfigure: vi.fn(),
    getCurrentTexture: () => ({ createView: () => ({}) }),
  };
  return Object.assign(new EventTarget(), {
    width: 100,
    height: 50,
    style: { touchAction: '', cursor: '' },
    clientWidth: 100,
    clientHeight: 50,
    setPointerCapture: (pointer: number) => void captured.add(pointer),
    hasPointerCapture: (pointer: number) => captured.has(pointer),
    toggleAttribute: () => {},
    getAttribute: (name: string) => attributes.get(name) ?? null,
    setAttribute: (name: string, value: string) => void attributes.set(name, value),
    hasAttribute: (name: string) => attributes.has(name),
    getContext: () => context,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 50 }),
  }) as unknown as HTMLCanvasElement;
}

/** Dispatches a pointer event on `target`, the primary mouse at (50, 25) unless `options` say. */
const point = (target: EventTarget, type: string, options: Record<string, unknown> = {}) =>
  target.dispatchEvent(
    Object.assign(new Event(type, { cancelable: true }), {
      pointerId: 1,
      pointerType: 'mouse',
      isPrimary: true,
      button: 0,
      buttons: 0,
      clientX: 50,
      clientY: 25,
      movementX: 0,
      movementY: 0,
      ...options,
    }),
  );

/** Dispatches a key event on `target`, returning whether its default was prevented. */
const press = (target: EventTarget, type: string, key: string, repeat = false) => {
  const event = Object.assign(new Event(type, { cancelable: true }), { key, repeat });
  target.dispatchEvent(event);
  return event.defaultPrevented;
};

const wheel = (target: EventTarget, deltaY: number, ctrlKey = false) =>
  target.dispatchEvent(
    Object.assign(new Event('wheel', { cancelable: true }), { deltaY, ctrlKey }),
  );

const yawOf = (scene: ExampleScene) => (scene.view!.goal as { yaw: number }).yaw;

const orbitView = () => ({ goal: { yaw: 0, pitch: 0, distance: 4 } });

it('orbits a view with yaw, pitch and distance, with the secondary button when pointing drags', () => {
  const canvas = fakeCanvas();
  const view = orbitView();
  const move = vi.fn();
  const redraw = vi.fn();
  const scene = { view, pointing: { move }, frame() {}, destroy() {} };
  const orbiting = connect(canvas, scene, { redraw });

  point(canvas, 'pointerdown', { buttons: 1 });
  point(canvas, 'pointermove', { buttons: 1, clientX: 60 });
  expect(view.goal.yaw).toBeCloseTo(-0.08);
  expect(move).not.toHaveBeenCalled();
  // A paused figure draws the turn
  expect(redraw).toHaveBeenCalledOnce();
  point(canvas, 'pointerup');
  orbiting[Symbol.dispose]();

  const drag = vi.fn();
  const stroked = orbitView();
  const stroking = connect(canvas, {
    view: stroked,
    pointing: { move, drag },
    frame() {},
    destroy() {},
  });
  point(canvas, 'pointerdown', { buttons: 1 });
  point(canvas, 'pointermove', { buttons: 1, clientX: 60 });
  expect(stroked.goal.yaw).toBe(0);
  expect(drag.mock.calls).toEqual([
    [0.5, 0.5, true],
    [0.6, 0.5, false],
  ]);
  point(canvas, 'pointerup');
  point(canvas, 'pointerdown', { button: 2, buttons: 2 });
  point(canvas, 'pointermove', { buttons: 2, clientX: 60 });
  expect(stroked.goal.yaw).toBeCloseTo(-0.08);
  expect(drag).toHaveBeenCalledTimes(2);
  expect(canvas.style.touchAction).toBe('none');
  expect(canvas.style.cursor).toBe('crosshair');
  stroking[Symbol.dispose]();
  expect(canvas.style.touchAction).toBe('');
  expect(canvas.style.cursor).toBe('');
});

it('sends hover with no button down, and a tap as a move then a press', () => {
  const canvas = fakeCanvas();
  const calls: string[] = [];
  const scene: ExampleScene = {
    pointing: {
      move: (x, y) => void calls.push(`move ${x} ${y}`),
      press: (x, y) => void calls.push(`press ${x} ${y}`),
      leave: () => void calls.push('leave'),
    },
    frame() {},
    destroy() {},
  };
  const connected = connect(canvas, scene);

  point(canvas, 'pointermove', { clientX: 25 });
  point(canvas, 'pointermove', { buttons: 1 });
  point(canvas, 'pointerdown');
  point(canvas, 'pointerup');
  point(canvas, 'pointerleave');
  expect(calls).toEqual(['move 0.25 0.5', 'move 0.5 0.5', 'press 0.5 0.5', 'leave']);
  connected[Symbol.dispose]();
});

it('ends a stroke as the pointer lifts or a second finger lands, and zooms by wheel and pinch', () => {
  const canvas = fakeCanvas();
  const calls: string[] = [];
  const scene: ExampleScene = {
    pointing: {
      move() {},
      drag: () => void calls.push('drag'),
      end: () => void calls.push('end'),
      zoom: (by) => void calls.push(`zoom ${by.toFixed(3)}`),
    },
    frame() {},
    destroy() {},
  };
  const connected = connect(canvas, scene);

  point(canvas, 'pointerdown');
  point(canvas, 'pointerup');
  expect(calls).toEqual(['drag', 'end']);

  calls.length = 0;
  wheel(canvas, 100);
  expect(calls).toEqual([`zoom ${Math.exp(0.1).toFixed(3)}`]);

  calls.length = 0;
  const touch = { pointerType: 'touch', clientY: 0 };
  point(canvas, 'pointerdown', { ...touch, pointerId: 1, clientX: 0 });
  point(canvas, 'pointerdown', { ...touch, pointerId: 2, clientX: 100, isPrimary: false });
  point(canvas, 'pointermove', { ...touch, pointerId: 2, clientX: 200, isPrimary: false });
  expect(calls).toEqual(['drag', 'end', 'zoom 0.500']);
  connected[Symbol.dispose]();
});

it('zooms a figure only on Ctrl-wheel, and an orbit view through its controls', () => {
  const canvas = fakeCanvas();
  const zoom = vi.fn();
  const pointing = { move() {}, drag() {}, zoom };
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      observe() {}
      disconnect() {}
    },
  );
  const figure = connect(canvas, { pointing, frame() {}, destroy() {} }, { page: true });

  wheel(canvas, 100);
  expect(zoom).not.toHaveBeenCalled();
  wheel(canvas, 100, true);
  expect(zoom).toHaveBeenCalledOnce();
  figure[Symbol.dispose]();

  zoom.mockClear();
  const view = orbitView();
  const orbiting = connect(canvas, {
    view,
    pointing: { move() {}, zoom },
    frame() {},
    destroy() {},
  });
  wheel(canvas, 100);
  expect(zoom).not.toHaveBeenCalled();
  expect(view.goal.distance).toBeCloseTo(4 * Math.exp(0.1));
  orbiting[Symbol.dispose]();
});

it('presses buttons by their keys, a held one from its first key to its last, and builds pads', () => {
  const declared = buttons(
    {
      drop: { label: 'drop', keys: ['Space'] },
      jump: { label: '▲', name: 'jump', keys: ['ArrowUp', 'KeyW'], held: true },
    },
    { left: [[null, 'jump']], right: [['drop']] },
  );
  const calls: string[] = [];
  const bound = {
    drop: { press: () => void calls.push('drop') },
    jump: { press: () => void calls.push('jump'), release: () => void calls.push('land') },
  };
  const page = new EventTarget();
  const listening = new AbortController();
  listenKeys(declared, bound, page, listening.signal);

  const key = (type: string, code: string, repeat = false, target: unknown = page) => {
    const event = Object.assign(new Event(type, { cancelable: true }), { code, repeat });
    Object.defineProperty(event, 'target', { value: target });
    page.dispatchEvent(event);
    return event.defaultPrevented;
  };

  expect(key('keydown', 'Space')).toBe(true);
  key('keydown', 'Space', true);
  key('keyup', 'Space');
  key('keydown', 'KeyW');
  key('keydown', 'KeyW', true);
  key('keyup', 'KeyW');
  expect(key('keydown', 'KeyQ')).toBe(false);
  // Space on a focused button presses the button alone, and a key held into it still releases
  const control = { closest: () => ({}) };
  expect(key('keydown', 'Space', false, control)).toBe(false);
  key('keydown', 'KeyW');
  key('keyup', 'KeyW', false, control);
  expect(calls).toEqual(['drop', 'drop', 'jump', 'land', 'jump', 'land']);
  // Two keys hold it until the last comes up, and the page losing focus lets go
  key('keydown', 'ArrowUp');
  key('keydown', 'KeyW');
  key('keyup', 'ArrowUp');
  expect(calls.slice(6)).toEqual(['jump']);
  key('keyup', 'KeyW');
  key('keydown', 'KeyW');
  page.dispatchEvent(new Event('blur'));
  key('keyup', 'KeyW');
  expect(calls.slice(6)).toEqual(['jump', 'land', 'jump', 'land']);
  calls.length = 6;

  listening.abort();
  key('keydown', 'Space');
  expect(calls).toHaveLength(6);

  const { left, right } = padsOf(declared, bound);
  expect(left?.[0]?.[0]).toBeNull();
  expect(left?.[0]?.[1]).toMatchObject({ label: '▲', name: 'jump' });
  expect(right?.[0]?.[0]).toMatchObject({ label: 'drop' });
  expect(right?.[0]?.[0]?.release).toBeUndefined();
  left?.[0]?.[1]?.press();
  left?.[0]?.[1]?.release?.();
  expect(calls.slice(6)).toEqual(['jump', 'land']);
});

it('holds a pad button while a finger, Space or Enter stays down, and taps it on a bare click', () => {
  const classes = new Set<string>();
  const button = Object.assign(new EventTarget(), {
    classList: {
      toggle: (name: string, on: boolean) => void (on ? classes.add(name) : classes.delete(name)),
      contains: (name: string) => classes.has(name),
    },
    setAttribute() {},
    setPointerCapture() {},
  }) as unknown as HTMLButtonElement;
  const calls: string[] = [];
  connectPad(button, {
    label: '▲',
    press: () => void calls.push('down'),
    release: () => void calls.push('up'),
  });

  expect(press(button, 'keydown', ' ')).toBe(true);
  press(button, 'keydown', ' ', true);
  expect(classes.has('held')).toBe(true);
  expect(press(button, 'keyup', ' ')).toBe(true);
  press(button, 'keydown', 'Enter');
  button.dispatchEvent(new Event('blur'));
  expect(calls).toEqual(['down', 'up', 'down', 'up']);

  button.dispatchEvent(Object.assign(new Event('click'), { detail: 0 }));
  button.dispatchEvent(Object.assign(new Event('click'), { detail: 1 }));
  expect(calls.slice(4)).toEqual(['down', 'up']);

  // Two fingers hold it until the last lifts or loses its capture
  const finger = (type: string, pointerId: number) =>
    button.dispatchEvent(Object.assign(new Event(type), { pointerId }));
  finger('pointerdown', 1);
  finger('pointerdown', 2);
  finger('pointerup', 1);
  finger('lostpointercapture', 1);
  expect(calls.slice(6)).toEqual(['down']);
  finger('lostpointercapture', 2);
  expect(calls.slice(6)).toEqual(['down', 'up']);
});

it('turns a figure by arrow keys as far as a drag does, and presses its middle with Space', () => {
  const dragged = orbitView();
  const dragCanvas = fakeCanvas();
  const dragging = connect(dragCanvas, { view: dragged, frame() {}, destroy() {} });
  point(dragCanvas, 'pointerdown', { buttons: 1 });
  point(dragCanvas, 'pointermove', { buttons: 1, clientX: 66 });
  dragging[Symbol.dispose]();

  const canvas = fakeCanvas();
  const view = orbitView();
  const pressed = vi.fn();
  const scene = { view, pointing: { press: pressed }, frame() {}, destroy() {} };
  const figure = connect(canvas, scene, { page: true });
  expect(canvas.getAttribute('aria-label')).toBe('arrow keys turn it, Space presses at its middle');
  expect(canvas.getAttribute('tabindex')).toBe('0');

  expect(press(canvas, 'keydown', 'ArrowRight')).toBe(true);
  expect(view.goal.yaw).toBeCloseTo(dragged.goal.yaw);
  press(canvas, 'keydown', ' ');
  press(canvas, 'keyup', ' ');
  expect(pressed).toHaveBeenCalledWith(0.5, 0.5);

  figure[Symbol.dispose]();
  expect(press(canvas, 'keydown', 'ArrowRight')).toBe(false);
});

it('hosts an example by key with its orbit connected, and a card with no input', async () => {
  vi.stubGlobal('window', { devicePixelRatio: 1 });
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal('navigator', { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
  const { root } = recordingRoot();
  const errors: unknown[] = [];

  const goalOf = async (input: 'page' | false) => {
    const canvas = fakeCanvas();
    const hosted = host('raymarch', canvas, {
      root,
      input,
      schedule: false,
      onError: (error) => errors.push(error),
    });
    const before = await hosted.run(({ scene, loop }) => {
      loop.run(0);
      return yawOf(scene);
    });
    point(canvas, 'pointerdown', { buttons: 1 });
    point(canvas, 'pointermove', { buttons: 1, clientX: 60 });
    const after = await hosted.run(({ scene }) => yawOf(scene));
    hosted.destroy();
    return after - before;
  };

  expect(await goalOf('page')).toBeCloseTo(-0.08);
  expect(await goalOf(false)).toBe(0);
  expect(errors).toEqual([]);
});

/** Resolves once pending promises have settled. */
const settled = () => new Promise((resolve) => setTimeout(resolve));

it('asks for no camera once let go, or without HTTPS', async () => {
  const getUserMedia = vi.fn(() => Promise.reject(new DOMException('', 'NotAllowedError')));
  const part = { kind: 'camera', label: 'camera' } as const;
  const camera = { attach: vi.fn(), frames: 0 };
  const labels: string[] = [];
  const onError = vi.fn();

  vi.stubGlobal('navigator', {});
  const listening = new AbortController();
  film(part, camera, (text) => void labels.push(text), onError, listening.signal).toggle();
  await settled();
  expect(labels).toEqual(['camera needs HTTPS']);

  // Refused at once, then let go
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } });
  const filming = film(part, camera, () => {}, onError, listening.signal);
  await settled();
  listening.abort();
  filming.toggle();
  expect(getUserMedia).toHaveBeenCalledOnce();
  expect(onError).not.toHaveBeenCalled();
});
