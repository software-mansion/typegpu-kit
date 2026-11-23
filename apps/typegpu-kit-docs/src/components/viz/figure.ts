// Do not access the DOM at module scope; Astro imports these helpers during builds
import type { Frame } from '@typegpu/kit';
import type * as web from '@typegpu/kit/web';
import { clamp, smooth } from '../home/geometry.ts';
import type { SceneModule } from './gpu.ts';

/** Maximum frame duration in seconds, limiting motion after stalls or returning to a tab. */
const MAX_DT = 0.1;

/** Sets up each figure matching `data-viz`. Throws and rejected promises stop only that instance and display its error. */
export function each(name: string, setup: (root: HTMLElement) => unknown) {
  for (const root of document.querySelectorAll<HTMLElement>(`[data-viz='${name}']`)) {
    void Promise.resolve()
      .then(() => setup(root))
      .catch((error: unknown) => fail(root, error));
  }
}

export const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/** `value` with at most `digits` decimals and no trailing zeros. */
export const rounded = (value: number, digits = 2) => String(Number(value.toFixed(digits)));

/** `n` and the word for that many: 1 frame, 2 frames. */
export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

function find<T extends Element>(root: HTMLElement, selector: string): T {
  const found = root.querySelector<T>(selector);
  if (!found) throw new Error(`The ${root.dataset.viz} figure has no ${selector}.`);
  return found;
}

export function control<T extends HTMLInputElement | HTMLSelectElement = HTMLInputElement>(
  root: HTMLElement,
  name: string,
): T {
  return find<T>(root, `[data-control='${name}']`);
}

export function action(root: HTMLElement, name: string, run?: () => void): HTMLButtonElement {
  const button = find<HTMLButtonElement>(root, `[data-action='${name}']`);
  if (run) button.addEventListener('click', run);
  return button;
}

/** Keeps a Slider output formatted and calls `change` when its value changes. */
export function slider(
  root: HTMLElement,
  name: string,
  change?: (value: number) => void,
  format: (value: number) => string = String,
): HTMLInputElement {
  const input = control(root, name);
  const output = input.closest('label')?.querySelector('output');
  const show = () => {
    const value = format(input.valueAsNumber);
    if (output) output.textContent = value;
    input.setAttribute('aria-valuetext', value);
  };

  show();
  input.addEventListener('input', () => {
    show();
    change?.(input.valueAsNumber);
  });

  if (input.closest('.viz-number')) {
    let from: { x: number; value: number; pointer: number } | undefined;

    input.addEventListener('pointerdown', (event) => {
      if (!event.isPrimary || event.button !== 0) return;
      event.preventDefault();
      input.focus({ preventScroll: true });
      from = { x: event.clientX, value: input.valueAsNumber, pointer: event.pointerId };
      input.setPointerCapture(event.pointerId);
      input.classList.add('held');
    });
    input.addEventListener('pointermove', (event) => {
      if (!from || event.pointerId !== from.pointer) return;
      const min = Number(input.min);
      const max = Number(input.max);
      const step = Number(input.step) || 1;
      const perStep = clamp(240 / ((max - min) / step), 2, 14);
      const value = from.value + Math.round((event.clientX - from.x) / perStep) * step;
      const previous = input.value;
      input.valueAsNumber = clamp(value, min, max);
      if (input.value !== previous) input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    input.addEventListener('lostpointercapture', () => {
      from = undefined;
      input.classList.remove('held');
    });
  }
  return input;
}

interface Choice<T extends string> {
  readonly value: T;
  /** Shows `text` as the option's words. */
  relabel(value: T, text: string): void;
}

/** Connects a Choice radio group by `data-choice`, preserving arrow-key navigation and one selected option. Calls `change` on selection. */
export function choice<T extends string>(
  root: HTMLElement,
  name: string,
  change?: (value: T) => void,
): Choice<T> {
  const group = find<HTMLElement>(root, `[data-choice='${name}']`);
  const inputs = [...group.querySelectorAll<HTMLInputElement>('input[type=radio]')];
  group.addEventListener('change', () => change?.(state.value));

  const state: Choice<T> = {
    get value() {
      return (inputs.find((input) => input.checked) ?? inputs[0]).value as T;
    },

    relabel(value, text) {
      const found = inputs.find((input) => input.value === value)?.nextElementSibling;
      if (!found) throw new Error(`The ${name} choice has no ${value}.`);
      found.textContent = text;
    },
  };
  return state;
}

/** A button that turns something on and off through `aria-pressed`, calling `change` on each click. */
export function toggle(button: HTMLButtonElement, change?: (on: boolean) => void) {
  const state = {
    get on() {
      return button.getAttribute('aria-pressed') === 'true';
    },

    set on(on: boolean) {
      button.setAttribute('aria-pressed', String(on));
    },
  };

  button.addEventListener('click', () => {
    state.on = !state.on;
    change?.(state.on);
  });
  return state;
}

/** Replaces the figure status text before its caption. */
export function note(root: HTMLElement, text: string) {
  let line = root.querySelector<HTMLElement>(':scope > .viz-note');
  if (!line) {
    line = document.createElement('p');
    line.className = 'viz-note';
    line.setAttribute('role', 'status');
    const caption = root.querySelector(':scope > figcaption');
    if (caption) caption.before(line);
    else root.append(line);
  }
  line.textContent = text;
}

/** The figure's `.viz-status` line, which states what its picture shows now. */
export const statusLine = (root: HTMLElement) => find<HTMLElement>(root, '.viz-status');

/** Displays the failure in the figure and logs the full error. */
export function fail(root: HTMLElement, error: unknown) {
  console.error(`The ${root.dataset.viz} figure stopped:`, error);
  note(root, `Figure stopped. ${error instanceof Error ? error.message : String(error)}`);
}

/**
 * Calls `run` with the primary pointer's main-button click or tap on `element` that moved at most
 * 4 px between press and release, until `signal` aborts, if one is given.
 */
export function onTap(
  element: HTMLElement,
  run: (event: PointerEvent) => void,
  signal?: AbortSignal,
) {
  let down: { x: number; y: number; pointer: number } | undefined;
  const press = (event: PointerEvent) => {
    if (event.isPrimary && event.button === 0)
      down = { x: event.clientX, y: event.clientY, pointer: event.pointerId };
  };
  const release = (event: PointerEvent) => {
    if (!down || event.pointerId !== down.pointer) return;
    const tapped = Math.hypot(event.clientX - down.x, event.clientY - down.y) <= 4;
    down = undefined;
    if (tapped) run(event);
  };

  const listening = signal ? { signal } : {};
  element.addEventListener('pointerdown', press, listening);
  element.addEventListener('pointerup', release, listening);
  element.addEventListener('pointercancel', () => (down = undefined), listening);
}

interface Keys {
  /** An arrow key went down, stepping 1 on its axis, or 4 with Shift. */
  readonly arrow?: (dx: number, dy: number) => void;
  /** Space or Enter went down, once per hold. */
  readonly down?: () => void;
  /** Space or Enter came up, or focus left while one was down. */
  readonly up?: () => void;
}

const ARROWS: Record<string, readonly [number, number]> = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
};

/**
 * Lets the keyboard do on focused `element` what a pointer does on it, until `signal` aborts. With
 * `say`, what the keys do, it also makes `element` focusable and adds `say` to its name.
 */
export function onKeys(element: Element, keys: Keys, say?: string, signal?: AbortSignal) {
  if (say) {
    const name = element.getAttribute('aria-label');
    if (!name?.endsWith(say)) element.setAttribute('aria-label', name ? `${name}; ${say}` : say);
    if (!element.hasAttribute('tabindex')) element.setAttribute('tabindex', '0');
    // Screen readers pass arrow keys through to an application
    element.setAttribute('role', 'application');
  }

  let held = false;
  const release = () => {
    if (!held) return;
    held = false;
    keys.up?.();
  };
  const listening = signal ? { signal } : {};
  element.addEventListener(
    'keydown',
    (event) => {
      const { key, shiftKey, repeat } = event as KeyboardEvent;
      const arrow = ARROWS[key];
      if (arrow && keys.arrow) {
        event.preventDefault();
        const by = shiftKey ? 4 : 1;
        keys.arrow(arrow[0] * by, arrow[1] * by);
      } else if ((key === ' ' || key === 'Enter') && (keys.down || keys.up)) {
        event.preventDefault();
        if (repeat || held) return;
        held = true;
        keys.down?.();
      }
    },
    listening,
  );
  element.addEventListener(
    'keyup',
    (event) => {
      const { key } = event as KeyboardEvent;
      if ((key !== ' ' && key !== 'Enter') || !(keys.down || keys.up)) return;
      // A button clicks as Space comes up
      event.preventDefault();
      release();
    },
    listening,
  );
  element.addEventListener('blur', release, listening);
}

/** Runs `run` once, when the figure first comes within 300 px of the screen. */
export function whenNear(root: HTMLElement, run: () => unknown) {
  const observer = new IntersectionObserver(
    ([entry]) => {
      if (!entry?.isIntersecting) return;
      observer.disconnect();
      void Promise.resolve()
        .then(run)
        .catch((error: unknown) => fail(root, error));
    },
    { rootMargin: '300px' },
  );
  observer.observe(root);
}

interface AnimationOptions {
  /** Advances by `dt` seconds while playing within 100 px of the screen. Delta is capped at 0.1 and is 0 after a pause. With `every`, advances in fixed intervals. Without this callback, draws only on request. */
  readonly step?: ((dt: number) => void) | undefined;
  /** Seconds between steps, for a figure that moves in whole steps. */
  readonly every?: number;
  /** Draws after stepping, at most once per frame when `redraw` requests it. */
  readonly draw: () => void;
  /** Toggles playback and displays Play or Pause, with aria-pressed set while playing. */
  readonly play?: HTMLButtonElement | undefined;
  /** Keeps the play button's own label and only presses it. */
  readonly keepLabel?: boolean;
  /** Called on a click that starts playing, before the first step, such as to rewind at the end. */
  readonly onPlay?: () => void;
  /** Whether it plays at first. Default true, false under reduced motion. */
  readonly playing?: boolean | undefined;
  /** Frees what the figure holds, when it leaves the page or fails. */
  readonly destroy?: () => void;
  /** Handles errors from `step` or `draw` after stopping animation. Default `fail`. */
  readonly onError?: (error: unknown) => void;
}

export interface Animation {
  readonly playing: boolean;
  /** Plays or pauses, and labels the play button. */
  play(on: boolean): void;
  /** Requests one draw at the next animation frame, coalescing repeated requests. */
  redraw(): void;
  /** Stops it for good and calls the figure's `destroy`. */
  destroy(): void;
}

interface Running {
  readonly root: HTMLElement;
  readonly options: AnimationOptions;
  visible: boolean;
  playing: boolean;
  dirty: boolean;
  /** The animation frame's time at the last step; undefined after a pause. */
  last: number | undefined;
  /** Seconds towards the next step, for `every`. */
  carried: number;
  /** Aborts once it stops, so a figure built again on the same play button reacts once. */
  readonly listening: AbortController;
}

const running = new Set<Running>();
let request = 0;
let watcher: IntersectionObserver | undefined;
let removals: MutationObserver | undefined;

// While a modal dialog covers the page, the figures under it slow to a stop over EASE seconds and
// draw nothing more, then gather speed again once it closes
const EASE = 0.5;
let covered = false;
let pace = 1;
let paced: number | undefined;
const halted = () => covered && pace === 0;

const moving = (entry: Running) =>
  entry.playing && entry.visible && entry.options.step !== undefined && !halted();

/** Asks for an animation frame when some figure has something to do in it, and only then. */
function wake() {
  if (request !== 0) return;
  for (const entry of running) {
    if (entry.dirty || moving(entry)) {
      request = requestAnimationFrame(tick);
      return;
    }
  }
}

function stop(entry: Running) {
  if (!running.delete(entry)) return;
  entry.listening.abort();
  watcher?.unobserve(entry.root);
  entry.options.destroy?.();
}

function advance(entry: Running, now: number) {
  const { step, every, draw } = entry.options;
  let drawing = entry.dirty;
  entry.dirty = false;

  if (moving(entry)) {
    const dt =
      entry.last === undefined ? 0 : Math.min((now - entry.last) / 1000, MAX_DT) * smooth(pace);
    entry.last = now;

    if (every === undefined) {
      step!(dt);
      drawing = true;
    } else {
      entry.carried += dt;
      while (entry.carried >= every && entry.playing) {
        entry.carried -= every;
        step!(every);
        drawing = true;
      }
    }
  } else {
    entry.last = undefined;
  }

  if (drawing) draw();
}

function tick(now: number) {
  request = 0;
  const dt = paced === undefined ? 0 : Math.min((now - paced) / 1000, MAX_DT);
  pace = clamp(pace + (covered ? -dt : dt) / EASE);
  paced = pace === (covered ? 0 : 1) ? undefined : now;

  for (const entry of running) {
    if (!entry.root.isConnected) {
      stop(entry);
      continue;
    }

    try {
      advance(entry, now);
    } catch (error) {
      stop(entry);
      (entry.options.onError ?? ((thrown) => fail(entry.root, thrown)))(error);
    }
  }

  wake();
}

/** Shared observers track visibility within 100 px and stop figures removed from the page, including paused or off-screen figures. */
function watch(entry: Running) {
  watcher ??= new IntersectionObserver(
    (records) => {
      for (const record of records) {
        for (const other of running) {
          if (other.root === record.target) other.visible = record.isIntersecting;
        }
      }
      wake();
    },
    { rootMargin: '100px' },
  );
  watcher.observe(entry.root);

  if (removals) return;
  removals = new MutationObserver(() => {
    for (const other of running) if (!other.root.isConnected) stop(other);
  });
  removals.observe(document.body, { childList: true, subtree: true });
  new MutationObserver(() => {
    covered = document.querySelector('dialog:modal') !== null;
    paced = undefined;
    wake();
  }).observe(document.body, { attributes: true, attributeFilter: ['open'], subtree: true });
}

/** Runs figures on one shared animation frame. Steps run only while playing and on screen. Draws follow steps or explicit requests, including one initial draw. Removing the figure stops it permanently. */
export function animation(root: HTMLElement, options: AnimationOptions): Animation {
  const entry: Running = {
    root,
    options,
    visible: false,
    playing: options.playing ?? !reducedMotion(),
    dirty: true,
    last: undefined,
    carried: 0,
    listening: new AbortController(),
  };

  const { play: button } = options;
  const label = () => {
    if (!button) return;
    if (!options.keepLabel) button.textContent = entry.playing ? 'Pause' : 'Play';
    button.setAttribute('aria-pressed', String(entry.playing));
  };

  const handle: Animation = {
    get playing() {
      return entry.playing;
    },

    play(on) {
      entry.playing = on;
      if (!on) entry.carried = 0;
      label();
      wake();
    },

    redraw() {
      entry.dirty = true;
      wake();
    },

    destroy: () => stop(entry),
  };

  button?.addEventListener(
    'click',
    () => {
      if (!entry.playing) options.onPlay?.();
      handle.play(!entry.playing);
    },
    { signal: entry.listening.signal },
  );

  running.add(entry);
  watch(entry);
  label();
  wake();
  return handle;
}

let names = 0;

/** Gives each figure instance a distinct radio-group name. */
export const uniqueName = (base: string) => `${base}-${(names += 1)}`;

/** Milliseconds a figure stays more than two screens from view before it lets its scene go. */
export const GRACE = 10_000;

interface Residency {
  /** Lets the scene go with the lost device, and builds it again on the next one, saying so. */
  lose(): void;
  /** Lets the scene go for good. */
  stop(): void;
}

/**
 * Builds with `hold` whenever `figure` comes within 300 px of the screen and the page shows, and
 * lets go what it holds once the figure has stayed more than two screens away for `GRACE`. `hold`
 * resolves to what lets its scene go, or undefined to stay without one. A rejection stops it and
 * shows the error.
 */
function resident(figure: HTMLElement, hold: () => Promise<(() => void) | undefined>): Residency {
  let near = false;
  let far = false;
  let building = false;
  let stopped = false;
  let rebuilding = false;
  let held: (() => void) | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  // Counts the scenes let go, so a build that started before one ends unheld
  let era = 0;

  const drop = () => {
    clearTimeout(timer);
    timer = undefined;
    era += 1;
    const release = held;
    held = undefined;
    release?.();
  };

  const linger = () => {
    timer ??= setTimeout(() => {
      timer = undefined;
      if (far) drop();
    }, GRACE);
  };

  const ensure = async (): Promise<void> => {
    if (stopped || building || held || !near) return;
    if (document.hidden) {
      document.addEventListener('visibilitychange', () => void ensure(), { once: true });
      return;
    }
    building = true;
    const at = era;
    try {
      const release = await hold();
      if (!release) stopped = true;
      else if (stopped || at !== era) release();
      else held = release;
    } catch (error) {
      if (at === era) {
        stopped = true;
        fail(figure, error);
      }
    } finally {
      building = false;
    }
    if (at !== era) return ensure();
    if (rebuilding && held) {
      rebuilding = false;
      figure.querySelector(':scope > .viz-note')?.remove();
    }
    if (far) linger();
  };

  const observe = (margin: string, change: (on: boolean) => void) => {
    const observer = new IntersectionObserver(
      (entries) => {
        const last = entries.at(-1);
        if (last) change(last.isIntersecting);
      },
      { rootMargin: margin },
    );
    observer.observe(figure);
    return observer;
  };
  const nearing = observe('300px', (on) => {
    near = on;
    void ensure();
  });
  const staying = observe('200% 0px', (on) => {
    far = !on;
    if (far) linger();
    else {
      clearTimeout(timer);
      timer = undefined;
    }
  });

  return {
    lose() {
      if (stopped || (!held && !building)) return;
      if (near) {
        rebuilding = true;
        note(figure, 'Rebuilding.');
      }
      drop();
      void ensure();
    },

    stop() {
      stopped = true;
      nearing.disconnect();
      staying.disconnect();
      drop();
    },
  };
}

export interface LiveOptions<S extends web.Scene = web.Scene> {
  /** Runs each animation frame while playing and on screen, advancing time only then. Otherwise draws on resize or `redraw` without advancing time. Default false. */
  readonly animate?: boolean;
  /** Initial playback state and button, as accepted by `animation`. */
  readonly playing?: boolean | undefined;
  readonly play?: HTMLButtonElement;
  /** What the figure says while a scene's `load` reads its files the first time. Default `Loading.` */
  readonly loading?: string | undefined;
  /** Default premultiplied alpha, so a figure can leave its ground clear. */
  readonly surface?: web.SurfaceOptions;
  /** Palette cell width in CSS pixels. */
  readonly grade?: number;
  /** Turns the scene's `view` by dragging the canvas, and draws as it turns. Default false. */
  readonly orbit?: boolean;
  /** Page code for every frame, after the scene's `frame`, such as a readout. */
  readonly onFrame?: (frame: Frame, stage: web.Stage<S>) => void;
  /**
   * Gives each scene it builds what the reader set, before its first frame. What it returns is
   * disposed of as the figure lets that scene go, such as pointer input.
   */
  readonly ready?: (scene: S) => Disposable | void;
}

interface Live<S extends web.Scene> {
  /** The scene the figure holds, if it holds one now. */
  readonly scene: S | undefined;
  readonly playing: boolean;
  play(on: boolean): void;
  redraw(): void;
  /** Calls `change` with the scene, if the figure holds one, and draws. */
  update(change: (scene: S) => void): void;
  /** Lets the scene go for good. */
  destroy(): void;
}

/** A scene's camera, if its `view` has an orbit goal. */
const viewOf = (scene: object) =>
  (scene as { view?: { goal?: Record<string, unknown>; jump?: () => void } }).view;

/** A copy of `goal`, its vectors as arrays. */
const copyOf = (goal: Record<string, unknown>) =>
  Object.fromEntries(
    Object.entries(goal).map(([key, value]) => [
      key,
      typeof value === 'object' && value !== null && Symbol.iterator in value
        ? [...(value as Iterable<unknown>)]
        : value,
    ]),
  );

/**
 * Runs the scene module `scene` imports with the shared root once the figure nears the screen,
 * fetching the module only then. Lets the scene go once the figure stays far away, and builds it
 * again as it comes back and after a lost device, keeping the camera's goal and the playback.
 * `options`, or what `options()` returns for each build, is the scene's third argument, unless
 * the module reads its files with `load`. Without WebGPU, or when a scene fails, hides the canvas
 * and says why.
 */
export function live<Options, S extends web.Scene>(
  figure: HTMLElement,
  scene: () => Promise<SceneModule<Options, S>>,
  options?: Options | (() => Options),
  hosting: LiveOptions<S> = {},
): Live<S> {
  const canvas = figure.querySelector('canvas');
  if (!canvas) throw new Error(`The ${figure.dataset.viz} figure has no canvas.`);

  let current: { readonly scene: S; readonly running: Animation } | undefined;
  let playing = hosting.playing;
  let goal: Record<string, unknown> | undefined;
  let first = true;

  const residency = resident(figure, async () => {
    const [gpu, module] = await Promise.all([import('./gpu.ts'), scene()]);
    if (first) gpu.onLost(() => residency.lose());
    const given = typeof options === 'function' ? (options as () => Options)() : options;
    const loading = first ? (hosting.loading ?? 'Loading.') : undefined;
    first = false;
    const built = await gpu.build(
      figure,
      canvas,
      module,
      given as Options,
      { ...hosting, playing, loading },
      (error) => {
        residency.stop();
        canvas.hidden = true;
        fail(figure, error);
      },
    );
    if (!built) return undefined;

    const view = viewOf(built.scene);
    if (view?.goal && goal) {
      for (const [key, value] of Object.entries(goal)) {
        const into = view.goal[key];
        if (Array.isArray(value) && into)
          value.forEach((item, i) => ((into as unknown[])[i] = item));
        else view.goal[key] = value;
      }
      view.jump?.();
    }
    current = built;
    let readied: Disposable | void;
    try {
      readied = hosting.ready?.(built.scene);
    } catch (error) {
      current = undefined;
      built.running.destroy();
      throw error;
    }

    return () => {
      const kept = viewOf(built.scene)?.goal;
      goal = kept && copyOf(kept);
      playing = built.running.playing;
      current = undefined;
      readied?.[Symbol.dispose]();
      built.running.destroy();
    };
  });

  return {
    get scene() {
      return current?.scene;
    },

    get playing() {
      return current?.running.playing ?? playing ?? false;
    },

    play(on) {
      playing = on;
      current?.running.play(on);
    },

    redraw: () => current?.running.redraw(),

    update(change) {
      if (!current) return;
      change(current.scene);
      current.running.redraw();
    },

    destroy: () => residency.stop(),
  };
}
