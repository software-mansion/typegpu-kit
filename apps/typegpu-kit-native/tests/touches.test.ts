import { describe, expect, it } from 'vitest';
import {
  createTouchRouter,
  TOUCH,
  type PadRect,
  type TouchActions,
  type TouchLayout,
} from '../src/touches.ts';

const left: PadRect = {
  element: 'controls',
  name: 'left',
  held: true,
  latch: false,
  x: 0,
  y: 300,
  width: 50,
  height: 50,
};
const jump: PadRect = {
  element: 'controls',
  name: 'jump',
  held: true,
  latch: false,
  x: 300,
  y: 300,
  width: 50,
  height: 50,
};
const drop: PadRect = {
  element: 'controls',
  name: 'drop',
  held: false,
  latch: false,
  x: 200,
  y: 300,
  width: 50,
  height: 50,
};

/** A router over a 400 × 400 area, logging its actions. */
function route(shape: TouchLayout['shape'], pads: readonly PadRect[] = [left, jump, drop]) {
  const log: string[] = [];
  const actions: TouchActions = {
    press: (pad) => log.push(`press ${pad.name}`),
    release: (pad) => log.push(`release ${pad.name}`),
    orbit: (dx, dy) => log.push(`orbit ${dx} ${dy}`),
    zoom: (by) => log.push(`zoom ${by.toFixed(2)}`),
    tap: (x, y) => log.push(`tap ${x} ${y}`),
    drag: (x, y, start) => log.push(`drag ${x} ${y}${start ? ' start' : ''}`),
    end: () => log.push('end'),
  };
  const router = createTouchRouter(actions);
  router.configure({ width: 400, height: 400, shape, pads });
  return { router, log };
}

const orbiting = { orbit: true, pointing: { press: true, drag: false, zoom: false } };
const painting = { orbit: false, pointing: { press: false, drag: true, zoom: true } };

describe('touch router', () => {
  it('holds two pads together and releases each with its own finger', () => {
    const { router, log } = route(orbiting);
    router.down(1, 10, 310, 0);
    router.down(2, 310, 310, 10);
    router.move(1, 30, 320, 20);
    router.up(1, 30, 320, 30);
    expect(router.active).toBe(true);
    router.up(2, 310, 310, 40);
    expect(log).toEqual(['press left', 'press jump', 'release left', 'release jump']);
    expect(router.active).toBe(false);
  });

  it('presses a held pad on its first finger and releases it on its last', () => {
    const { router, log } = route(orbiting);
    router.down(1, 10, 310, 0);
    router.down(2, 20, 320, 10);
    router.up(1, 10, 310, 20);
    router.down(3, 210, 310, 30);
    router.down(4, 220, 320, 40);
    router.up(3, 210, 310, 50);
    router.up(4, 220, 320, 60);
    router.up(2, 20, 320, 70);
    expect(log).toEqual([
      'press left',
      'press drop',
      'press drop',
      'release drop',
      'release drop',
      'release left',
    ]);
  });

  it('orbits with one finger while another holds a pad', () => {
    const { router, log } = route(orbiting);
    router.down(1, 10, 310, 0);
    router.down(2, 100, 100, 0);
    router.move(2, 120, 100, 16);
    router.move(2, 125, 104, 32);
    router.up(2, 125, 104, 48);
    router.up(1, 10, 310, 60);
    expect(log).toEqual(['press left', 'orbit 20 0', 'orbit 5 4', 'release left']);
  });

  it('taps on a quick still lift and drags past the slop', () => {
    const { router, log } = route(orbiting);
    router.down(1, 100, 200, 0);
    router.move(1, 103, 201, 50);
    router.up(1, 103, 201, 100);
    router.down(2, 100, 100, 200);
    router.move(2, 100 + TOUCH.slop + 1, 100, 250);
    router.up(2, 100 + TOUCH.slop + 1, 100, 300);
    router.down(3, 100, 100, 400);
    router.up(3, 100, 100, 400 + TOUCH.tapMs + 1);
    expect(log).toEqual([`tap ${103 / 400} ${201 / 400}`, `orbit ${TOUCH.slop + 1} 0`]);
  });

  it('strokes at once, and a second finger ends the stroke and pinches', () => {
    const { router, log } = route(painting);
    router.down(1, 100, 200, 0);
    router.move(1, 120, 200, 16);
    router.down(2, 300, 200, 32);
    router.move(2, 220, 200, 48);
    router.move(1, 60, 200, 64);
    router.up(2, 220, 200, 80);
    router.move(1, 10, 10, 96);
    router.up(1, 10, 10, 112);
    expect(log).toEqual(['drag 0.25 0.5 start', 'drag 0.3 0.5', 'end', 'zoom 1.80', 'zoom 0.63']);
  });

  it('releases held pads and ends a stroke when the gesture is cancelled', () => {
    const { router, log } = route(painting);
    router.down(1, 10, 310, 0);
    router.down(2, 100, 100, 0);
    router.cancel(1);
    router.down(3, 210, 310, 10);
    router.reset();
    expect(log).toEqual([
      'press left',
      'drag 0.25 0.25 start',
      'release left',
      'press drop',
      'end',
      'release drop',
    ]);
    expect(router.active).toBe(false);
  });

  it('starts afresh after a reset', () => {
    const { router, log } = route(orbiting);
    router.down(1, 100, 100, 0);
    router.reset();
    router.move(1, 200, 200, 10);
    router.up(1, 200, 200, 20);
    router.down(1, 50, 50, 30);
    router.up(1, 50, 50, 40);
    expect(log).toEqual([`tap 0.125 0.125`]);
  });

  it('leaves the canvas to pads when the scene takes no input', () => {
    const { router, log } = route(undefined);
    router.down(1, 100, 100, 0);
    router.move(1, 200, 100, 10);
    router.up(1, 200, 100, 20);
    router.down(2, 210, 310, 30);
    router.up(2, 210, 310, 40);
    expect(log).toEqual(['press drop', 'release drop']);
  });
});
