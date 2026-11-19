// One host for every example on the web: its page, its gallery card and its figures

import { cameras, type Frame } from '@typegpu/kit';
import * as web from '@typegpu/kit/web';
import type { TgpuRoot } from 'typegpu';
import { exampleOf } from 'typegpu-kit-examples';
import type { ExampleScene, Say } from 'typegpu-kit-examples/contract';
import { create, orbits } from 'typegpu-kit-examples/hosts';
import { clamp } from '../components/home/geometry.ts';
import { onKeys } from '../components/viz/figure.ts';
import { TURN } from '../components/viz/orbit.ts';
import { loading } from './files.ts';

export interface HostOptions {
  /** A root shared with other cards and figures. Default a device requesting the example's `features`. */
  readonly root?: TgpuRoot;
  /** Scene options over the scene's own, such as a card's. Default none. */
  readonly options?: object;
  /** Connects input as a full page does, or none, as a gallery card. Default `'page'`. */
  readonly input?: 'page' | false;
  /** With false, the caller runs `stage.loop`. Default `requestAnimationFrame`. */
  readonly schedule?: false;
  /** Caps the canvas's pixel ratio. Default the screen's. */
  readonly maxPixelRatio?: number;
  /** Receives loading status. Default none. */
  readonly say?: Say;
  /** Runs after the scene's `frame` each frame, such as to write a HUD. Default none. */
  readonly onFrame?: (stage: web.Stage<ExampleScene>, frame: Frame) => void;
  /** Connects page code once the scene is up; the host disposes what it returns. Default none. */
  readonly attach?: (stage: web.Stage<ExampleScene>) => Disposable | void;
  /** Receives errors from loading, creating or running the scene. */
  readonly onError: (error: unknown) => void;
}

/** Hosts example `key` on the canvas with `web.host`, loading its files and connecting its input. */
export function host(
  key: string,
  canvas: HTMLCanvasElement,
  options: HostOptions,
): web.Hosted<ExampleScene> {
  const { root, options: given, input = 'page', schedule, maxPixelRatio, say } = options;
  const { onFrame, attach, onError } = options;
  const features = (exampleOf(key).features ?? []) as readonly GPUFeatureName[];
  const files = loading(say);

  return web.host((pageRoot, surface) => create(key, pageRoot, surface, files, given), {
    canvas,
    onError,
    ...(root ? { root } : { optionalFeatures: features }),
    ...(schedule === undefined ? {} : { schedule }),
    ...(maxPixelRatio === undefined ? {} : { maxPixelRatio }),
    ...(onFrame ? { onFrame } : {}),
    attach(stage) {
      const connected = input === 'page' ? connect(canvas, stage.scene) : undefined;
      const own = attach?.(stage);
      return {
        destroy() {
          own?.[Symbol.dispose]();
          connected?.[Symbol.dispose]();
        },
      };
    },
  });
}

export interface ConnectOptions {
  /**
   * Leaves vertical swipes and the wheel to the page, zooms only on pinches and Ctrl-wheel, and
   * arms touch and pen strokes with a tap, as figures do. Default false.
   */
  readonly page?: boolean;
  /** The ring `pointing.reach` sizes, which follows the pointer. Default none. */
  readonly ring?: HTMLElement | null;
  /** Runs after each input, for figures that draw on demand. Default none. */
  readonly redraw?: () => void;
}

/** Sizes `ring` to the pointing's reach, a radius in CSS pixels. */
export const fitRing = (ring: HTMLElement, reach: number) => {
  const size = `${reach * 2}px`;
  if (ring.style.width !== size) ring.style.width = ring.style.height = size;
};

/** Pixels a press may move and still tap. */
const TAP_SLOP = 4;

/** The share of the canvas an arrow key moves the keyboard's point by. */
const AIM = 0.025;

/**
 * Connects the scene's orbit view and pointing to the canvas until disposed. Orbits when
 * `orbits(scene)`, with the secondary button and two fingers when primary strokes go to
 * `pointing.drag`. Hover moves go with no button down, a tap moves then presses, a pointer up or a
 * second finger ends a stroke, and the wheel and two-finger pinches zoom a view the scene turns
 * itself.
 */
export function connect(
  canvas: HTMLCanvasElement,
  scene: ExampleScene,
  options: ConnectOptions = {},
): Disposable {
  const { page = false, ring, redraw = () => {} } = options;
  const listening = new AbortController();
  const { signal } = listening;
  let controls: web.OrbitControls | undefined;
  let pointer: web.Pointer | undefined;
  const disposable = {
    [Symbol.dispose]() {
      listening.abort();
      // Each restores the canvas's touch-action to what the one before it found
      pointer?.destroy();
      controls?.destroy();
    },
  };

  const { pointing } = scene;
  const orbit = orbits(scene);
  if (orbit) {
    controls = web.orbitControls(scene.view, canvas, {
      ...(page && { page: true }),
      // Strokes take the primary button and one finger
      ...(pointing?.drag && { button: 'secondary' as const }),
    });
  }
  const turn = (dx: number, dy: number) => {
    cameras.turn(scene.view!.goal as web.OrbitGoal, dx * TURN, dy * TURN);
    redraw();
  };
  if (!pointing) {
    if (page && orbit) onKeys(canvas, { arrow: turn }, 'arrow keys turn it', signal);
    return disposable;
  }
  const { drag, end, zoom } = pointing;

  // A full page marks a canvas that strokes paint
  if (!page && drag) {
    const cursor = canvas.style.cursor;
    canvas.style.cursor = 'crosshair';
    signal.addEventListener('abort', () => (canvas.style.cursor = cursor));
  }

  const ringed = ring && pointing.reach !== undefined ? ring : undefined;
  const showRing = (x: number, y: number) => {
    if (!ringed) return;
    fitRing(ringed, pointing.reach ?? 0);
    ringed.style.translate = `${x * canvas.clientWidth}px ${y * canvas.clientHeight}px`;
    ringed.hidden = false;
  };
  const hover = (x: number, y: number) => {
    pointing.move?.(x, y);
    showRing(x, y);
    redraw();
  };

  // Figures leave touch and pen strokes to scrolling until a tap arms them
  let armed = !page;
  /** The canvas's touch-action under the pointer, which disarming restores. */
  let before = '';
  const arm = (on: boolean) => {
    armed = on;
    canvas.style.touchAction = on ? 'none' : before;
    canvas.toggleAttribute('data-armed', on);
  };
  const declines = (event: PointerEvent) => event.pointerType !== 'mouse' && !armed;
  if (page && drag) {
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries.at(-1);
        if (entry && !entry.isIntersecting) arm(false);
      },
      { threshold: 0 },
    );
    observer.observe(canvas);
    signal.addEventListener('abort', () => {
      observer.disconnect();
      arm(false);
    });
  }

  /** The pointer that strokes, and where a press that may tap went down. */
  let stroke: number | undefined;
  let pressed: { readonly x: number; readonly y: number; readonly id: number } | undefined;
  const ending = () => {
    if (stroke === undefined) return;
    stroke = undefined;
    end?.();
    redraw();
  };

  pointer = web.pointer(canvas, {
    page,
    move(x, y, event) {
      if (event.pointerId === stroke) {
        drag?.(x, y, false);
        showRing(x, y);
        redraw();
      } else if (event.buttons === 0) {
        hover(x, y);
      }
    },
    down(x, y, event) {
      // A second finger starts a pinch
      if (!event.isPrimary) {
        pressed = undefined;
        ending();
        return;
      }
      if (event.button !== 0) return;
      pressed = { x: event.clientX, y: event.clientY, id: event.pointerId };
      if (!drag || declines(event)) return;
      stroke = event.pointerId;
      drag(x, y, true);
      showRing(x, y);
      redraw();
    },
    up(x, y, event) {
      const tap = pressed?.id === event.pointerId ? pressed : undefined;
      if (tap) pressed = undefined;
      const tapped =
        tap !== undefined &&
        event.type === 'pointerup' &&
        Math.hypot(event.clientX - tap.x, event.clientY - tap.y) <= TAP_SLOP;
      if (tapped) {
        if (page && drag && declines(event)) arm(true);
        hover(x, y);
        pointing.press?.(x, y);
      }
      if (event.pointerId === stroke) ending();
    },
    leave() {
      pointing.leave?.();
      if (ringed) ringed.hidden = true;
      redraw();
    },
    // An orbit view zooms through its controls
    ...(zoom &&
      !orbit && {
        zoom(by: number) {
          zoom(by);
          redraw();
        },
      }),
  });
  before = canvas.style.touchAction;

  // A figure's keyboard turns an orbit view or moves the point it aims at, and presses there,
  // stroking while the arrows move it
  if (page) {
    let aim: [number, number] = [0.5, 0.5];
    let held = false;
    let still = true;
    const presses = pointing.press !== undefined || drag !== undefined;
    const says = orbit
      ? `arrow keys turn it${presses ? ', Space presses at its middle' : ''}`
      : `arrow keys move the point${presses ? ', Space presses there' : ''}`;
    onKeys(
      canvas,
      {
        arrow(dx, dy) {
          if (orbit) return turn(dx, dy);
          aim = [clamp(aim[0] + dx * AIM), clamp(aim[1] + dy * AIM)];
          still = false;
          if (held) drag?.(...aim, false);
          hover(...aim);
        },
        down() {
          held = true;
          still = true;
          drag?.(...aim, true);
          hover(...aim);
        },
        up() {
          held = false;
          if (still) pointing.press?.(...aim);
          if (drag) end?.();
          redraw();
        },
      },
      says,
      signal,
    );
  }
  return disposable;
}
