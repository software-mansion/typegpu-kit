import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest';
import { GRACE, live } from '../src/components/viz/figure.ts';

/** What the mocked `gpu.ts` gives `live`: a scene's build without a GPU, and the loss listeners. */
const gpu = vi.hoisted(() => ({ lost: [] as (() => void)[] }));

vi.mock('../src/components/viz/gpu.ts', () => ({
  onLost: (listener: () => void) => void gpu.lost.push(listener),
  build: async (
    _figure: unknown,
    _canvas: unknown,
    module: { createScene: (root: unknown, surface: unknown, options: unknown) => unknown },
    options: unknown,
    hosting: { playing?: boolean },
  ) => {
    const scene = (await module.createScene(undefined, undefined, options)) as Fake;
    const running = {
      playing: hosting.playing ?? true,
      play: (on: boolean) => void (running.playing = on),
      redraw: () => {},
      destroy: () => scene.destroy(),
    };
    return { scene, running };
  },
}));

interface Fake {
  value: number;
  readonly view: { goal: { yaw: number; target: number[] }; jump: () => void };
  frame(): void;
  destroy(): void;
}

let observers: { margin: string; change: (entries: { isIntersecting: boolean }[]) => void }[];
const page = Object.assign(new EventTarget(), { hidden: false });

/** Reports the figure inside or outside the observers with `margin`. */
const seen = (margin: string, on: boolean) => {
  for (const each of observers) if (each.margin === margin) each.change([{ isIntersecting: on }]);
};
const near = (on: boolean) => seen('300px', on);
const within = (on: boolean) => seen('200% 0px', on);
const settle = async () => {
  for (let i = 0; i < 4; i++) await vi.advanceTimersByTimeAsync(0);
};

/** A figure with a canvas, holding the status line `note` writes. */
function fakeFigure() {
  let line: { textContent: string; remove(): void } | undefined;
  const canvas = { hidden: false };
  return {
    dataset: { viz: 'test' },
    get status() {
      return line?.textContent;
    },
    querySelector(selector: string) {
      if (selector === 'canvas') return canvas;
      return selector.includes('viz-note') ? (line ?? null) : null;
    },
    append(made: { textContent: string; remove(): void }) {
      line = made;
      made.remove = () => (line = undefined);
    },
  } as unknown as HTMLElement & { readonly status: string | undefined };
}

// Loaded once here, so `live`'s own import of it resolves within a test's frames
beforeAll(() => import('../src/components/viz/gpu.ts'));

beforeEach(() => {
  vi.useFakeTimers();
  observers = [];
  gpu.lost.length = 0;
  page.hidden = false;
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      readonly watching: (typeof observers)[number];
      constructor(
        change: (entries: { isIntersecting: boolean }[]) => void,
        init: IntersectionObserverInit,
      ) {
        this.watching = { margin: init.rootMargin ?? '', change };
        observers.push(this.watching);
      }
      observe() {}
      disconnect() {
        observers = observers.filter((each) => each !== this.watching);
      }
    },
  );
  vi.stubGlobal(
    'document',
    Object.assign(page, {
      createElement: () => ({ textContent: '', className: '', setAttribute() {} }),
    }),
  );
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

it('lets a far figure go, builds it again near with what the reader set, and again after a lost device', async () => {
  const made: Fake[] = [];
  const createScene = (): Fake => {
    const scene: Fake = {
      value: 0,
      view: { goal: { yaw: 0, target: [0, 0, 0] }, jump: vi.fn() },
      frame() {},
      destroy: vi.fn(),
    };
    made.push(scene);
    return scene;
  };
  let set = 1;
  const disposed = vi.fn();
  const figure = fakeFigure();
  const shown = live(figure, async () => ({ createScene }), undefined, {
    animate: true,
    ready(scene) {
      scene.value = set;
      return { [Symbol.dispose]: disposed };
    },
  });

  // Builds once it nears the screen, and never twice
  await settle();
  expect(made).toHaveLength(0);
  near(true);
  within(true);
  near(true);
  await settle();
  expect(made).toHaveLength(1);
  expect(shown.scene?.value).toBe(1);

  // The reader moves a control and the camera, and pauses
  set = 5;
  shown.update((scene) => (scene.value = set));
  shown.scene!.view.goal.yaw = 1.5;
  shown.scene!.view.goal.target[1] = 2;
  shown.play(false);

  // Coming back within the grace keeps the scene; staying away lets it go
  near(false);
  within(false);
  await vi.advanceTimersByTimeAsync(GRACE - 1);
  within(true);
  await vi.advanceTimersByTimeAsync(GRACE);
  expect(made[0].destroy).not.toHaveBeenCalled();
  within(false);
  await vi.advanceTimersByTimeAsync(GRACE);
  expect(made[0].destroy).toHaveBeenCalledOnce();
  expect(disposed).toHaveBeenCalledOnce();
  expect(shown.scene).toBeUndefined();

  // Near again, a new scene starts from what the reader set
  within(true);
  near(true);
  await settle();
  expect(made).toHaveLength(2);
  expect(shown.scene).toBe(made[1]);
  expect(made[1].value).toBe(5);
  expect(made[1].view.goal).toEqual({ yaw: 1.5, target: [0, 2, 0] });
  expect(made[1].view.jump).toHaveBeenCalledOnce();
  expect(shown.playing).toBe(false);

  // A lost device in a hidden page: the figure says so and builds once the page shows again
  page.hidden = true;
  for (const listener of gpu.lost) listener();
  expect(made[1].destroy).toHaveBeenCalledOnce();
  expect(figure.status).toBe('Rebuilding.');
  await settle();
  expect(made).toHaveLength(2);
  page.hidden = false;
  page.dispatchEvent(new Event('visibilitychange'));
  await settle();
  expect(made).toHaveLength(3);
  expect(made[2].value).toBe(5);
  expect(figure.status).toBeUndefined();
});

it('stops a figure whose scene throws, saying why, and builds it no more', async () => {
  const createScene = vi.fn(() => {
    throw new Error('No pipeline.');
  });
  const figure = fakeFigure();
  live(figure, async () => ({ createScene }));

  near(true);
  await settle();
  expect(figure.status).toBe('Figure stopped. No pipeline.');

  near(false);
  near(true);
  await settle();
  expect(createScene).toHaveBeenCalledOnce();
});
