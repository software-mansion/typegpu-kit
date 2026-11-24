/**
 * Hosts scenes on web canvases and provides surfaces, GPU pacing, pointer input
 * and orbit controls.
 *
 * @module
 * @kind platform
 * @platform web
 * @allocates each frame: a view of the canvas's texture, which is new every frame; under `paced`,
 * a promise of the frame's work; for `video`, an imported texture per read of a frame
 * @owns the canvas's configuration and a resize observer; a hosted scene also owns the surface, its
 * root and the device unless it was given them, the scene's stage and what its `attach` returned
 */

import { tgpu, type TgpuRoot } from 'typegpu';
import type { Schedule } from './animate.ts';
import * as backends from './backends.ts';
import { pan as panBy, turn as turnBy } from './cameras.ts';
import { checkSize, configuration } from './canvases.ts';
import type { Frame } from './frames.ts';
import type * as videos from './video.ts';
import {
  local,
  stage,
  type CanvasOptions,
  type CreateScene,
  type Hosted as AnyHosted,
  type HostOptions as AnyHostOptions,
  type Scene,
  type Stage as AnyStage,
} from './hosting.ts';
import { asError, freeEach } from './owned.ts';
import type { Surface as Base, FixedSize } from './targets.ts';

export type { CreateScene, Scene } from './hosting.ts';
export type { FixedSize } from './targets.ts';

export interface SurfaceOptions extends CanvasOptions {
  /**
   * Fixes the drawing-buffer size before `surface` returns. The canvas CSS size still scales it on screen. Default the layout size multiplied by
   * the capped pixel ratio, following layout changes.
   */
  readonly size?: FixedSize;
  /** Caps drawing-buffer pixels per CSS pixel unless `size` is set. Default 2. */
  readonly maxPixelRatio?: number;
}

export interface Surface extends Base {
  /**
   * Provides the canvas context for a TypeGPU colour attachment `view`. Only `view()` applies
   * pending size changes.
   */
  readonly context: GPUCanvasContext;
  /**
   * Reports the drawing-buffer size in pixels, from the fixed size or the layout size in device
   * pixels. Reports changes before the next `view()` applies them to the canvas.
   */
  readonly width: number;
  /** Reports the drawing-buffer height in pixels, as `width` does. */
  readonly height: number;
  /**
   * Reports the fixed size's pixel ratio, or follows zoom and screen changes when the size follows
   * layout.
   */
  readonly pixelRatio: number;
  /**
   * Reports whether the surface is paused, true between `pause` and `resume`. `animate` skips
   * frames while true.
   */
  readonly paused: boolean;
  /**
   * Fixes the drawing-buffer size from the next `view()`, as `options.size` does. Pass `undefined`
   * to follow layout changes.
   */
  setSize(size: FixedSize | undefined): void;
  /**
   * Pauses the surface for a canvas outside the viewport or a pause menu. Hidden tabs need no
   * call, because browsers stop their animation frames.
   */
  pause(): void;
  /** Resumes the frames after `pause`. */
  resume(): void;
  /**
   * Returns a view of the current canvas texture, created once per texture. Applies pending size
   * changes first, which a pass using `context` would skip.
   */
  view(frame: Frame): GPUTextureView;
}

/**
 * Configures a canvas for the root. Matches its drawing buffer to the layout size or the fixed
 * `options.size`. Throws if the fixed size is invalid.
 *
 * @param canvas the canvas it configures and sizes
 * @returns a surface you draw into and destroy
 */
export function surface(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  options: SurfaceOptions = {},
): Surface {
  const { maxPixelRatio = 2 } = options;
  const backend = backends.of(root);
  const format = options.format ?? backend.preferredCanvasFormat;

  const maxSize = root.device.limits.maxTextureDimension2D;
  const clamp = (size: number) => Math.max(1, Math.min(maxSize, Math.round(size)));

  const layout = {
    width: canvas.width,
    height: canvas.height,
    pixelRatio: Math.min(window.devicePixelRatio || 1, maxPixelRatio),
  };
  let fixed: Required<FixedSize> | undefined;

  const fix = (next: FixedSize | undefined) => {
    fixed = next && checkSize(next, maxSize, 'web.surface');
  };

  const size = () => fixed ?? layout;

  const apply = (nextWidth: number, nextHeight: number, ratio: number) => {
    layout.width = clamp(nextWidth);
    layout.height = clamp(nextHeight);
    layout.pixelRatio = Math.min(ratio, maxPixelRatio);
  };

  // Resizing clears the drawing buffer. The observer runs after drawing, so apply size changes
  // when the next frame acquires its texture
  const commit = () => {
    const { width, height } = size();
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  };

  fix(options.size);
  // A fallback device has no `webgpu` canvas context to configure, so it supplies its own
  const context =
    backend.kind === 'webgpu'
      ? root.configureContext({ canvas, format, ...configuration(options) })
      : backend.configure(canvas, { format, ...configuration(options) });

  const bounds = canvas.getBoundingClientRect();
  apply(bounds.width * layout.pixelRatio, bounds.height * layout.pixelRatio, layout.pixelRatio);
  commit();

  const observer = new ResizeObserver(([entry]) => {
    if (!entry) return;
    const ratio = window.devicePixelRatio || 1;
    const devicePixels = entry.devicePixelContentBoxSize?.[0];

    if (devicePixels) {
      const scale = Math.min(1, maxPixelRatio / ratio);
      apply(devicePixels.inlineSize * scale, devicePixels.blockSize * scale, ratio);
      return;
    }

    const css = entry.contentBoxSize[0];
    if (!css) return;
    const scale = Math.min(ratio, maxPixelRatio);
    apply(css.inlineSize * scale, css.blockSize * scale, ratio);
  });

  // On WebKit, pixel-ratio changes require a layout change to update the drawing-buffer size
  try {
    observer.observe(canvas, { box: 'device-pixel-content-box' });
  } catch {
    observer.observe(canvas);
  }

  let texture: GPUTexture | undefined;
  let current: GPUTextureView | undefined;
  const view = () => {
    commit();

    const next = context.getCurrentTexture();
    if (next !== texture) {
      texture = next;
      current = next.createView();
    }
    return current!;
  };

  let paused = false;
  const destroy = () => {
    observer.disconnect();
    context.unconfigure();
  };

  return {
    context,
    format,

    get width() {
      return size().width;
    },

    get height() {
      return size().height;
    },

    get pixelRatio() {
      return size().pixelRatio;
    },

    get paused() {
      return paused;
    },

    pause() {
      paused = true;
    },

    resume() {
      paused = false;
    },

    setSize: fix,
    view,
    destroy,
    [Symbol.dispose]: destroy,
  };
}

export interface PacedOptions {
  /**
   * Limits the number of frames whose GPU work may be unfinished when the next frame starts. Must
   * be an integer above 0. Default 2.
   */
  readonly inFlight?: number;
}

/**
 * Supplies animation frames that wait while `inFlight` frames remain unfinished on the GPU. Use as
 * `animate`'s `schedule` when frame callbacks can outpace GPU work. Throws unless `inFlight` is an
 * integer above 0.
 *
 * @param queue the device's queue, `root.device.queue`
 * @returns a schedule to give `animate`
 */
export function paced(
  queue: Pick<GPUQueue, 'onSubmittedWorkDone'>,
  options: PacedOptions = {},
): Schedule {
  const { inFlight = 2 } = options;
  if (!Number.isInteger(inFlight) || inFlight < 1) {
    throw new Error(
      `web.paced requires inFlight to be an integer above 0 frames, got ${inFlight}.`,
    );
  }

  // GPU work completes in order, so check frame fences oldest first
  const fences: Promise<unknown>[] = [];
  const requests = new Map<number, number | undefined>();
  let next = 0;

  return {
    request(callback) {
      // Request the fence after submission so it includes this frame's work
      fences.push(queue.onSubmittedWorkDone().catch(() => undefined));
      if (fences.length > inFlight) fences.shift();

      const id = ++next;
      requests.set(id, undefined);
      const start = () => {
        if (!requests.has(id)) return;
        const frame = requestAnimationFrame((milliseconds) => {
          requests.delete(id);
          callback(milliseconds);
        });
        requests.set(id, frame);
      };

      if (fences.length === inFlight) void fences[0].then(start);
      else start();
      return id;
    },

    cancel(id) {
      const frame = requests.get(id);
      if (frame !== undefined) cancelAnimationFrame(frame);
      requests.delete(id);
    },
  };
}

/** Provides the goal written by orbit controls, as in `cameras.orbit`. */
export interface OrbitGoal {
  yaw: number;
  pitch: number;
  distance: number;
  /** Sets the orbit target that `pan` moves. */
  target?: [number, number, number];
}

export interface OrbitControlsOptions {
  /** Sets rotation per dragged pixel, in radians. Default 0.008, as `cameras.turn`. */
  readonly sensitivity?: number;
  /**
   * Sets wheel zoom speed. Each wheel step multiplies distance by e^(deltaY × zoomSpeed). Default
   * 0.001.
   */
  readonly zoomSpeed?: number;
  /**
   * Orbits with this mouse button. `'secondary'` and `'middle'` leave the primary button and one
   * finger to other listeners, such as strokes that paint, and two fingers orbit by the move of
   * their midpoint. Default `'primary'`.
   */
  readonly button?: 'primary' | 'secondary' | 'middle';
  /**
   * Enables panning with the secondary button, or the middle one when the secondary orbits, and
   * with two fingers moving together. Moves `target` as `cameras.pan` does, with the element's
   * height. Requires the goal to have a `target`. Default false.
   */
  readonly pan?: boolean;
  /**
   * Allows page scrolling around an embedded view. Vertical swipes and the wheel scroll the page.
   * Horizontal drags rotate the view. Pinches and the wheel with Ctrl or Command held zoom it. A
   * trackpad pinch sends that wheel modifier. With `pointer`, it decides only the wheel. Default
   * false, so the element handles every drag and wheel input.
   */
  readonly page?: boolean;
  /**
   * Leaves the element's `touch-action` and pointer capture to this pointer on the same element,
   * which sets them from its own `page`. Destroy the controls before the pointer. Default none, so
   * the controls set both.
   */
  readonly pointer?: Pointer;
  /**
   * Runs after each input that changes the goal, for views that draw on demand. Default none.
   * @cpu
   */
  readonly onChange?: () => void;
}

export interface OrbitControls extends Disposable {
  /**
   * Removes the listeners and restores the element's `touch-action`, unless a pointer set it. Also
   * `[Symbol.dispose]`.
   */
  destroy(): void;
}

/** `PointerEvent.buttons` for each button. */
const BUTTONS = { primary: 1, secondary: 2, middle: 4 } as const;

/**
 * Connects pointer input to an orbit camera's goal. Drag to orbit and use the wheel or pinch to
 * zoom. With `pan`, drag the other button to move the target. `update` eases the camera towards
 * the goal. Throws if `pan` is enabled but the goal has no target.
 *
 * @param orbit an orbit camera, whose goal it writes
 * @param element what takes the drag, the wheel and the pinch
 * @returns the controls, which you destroy to remove the listeners
 */
export function orbitControls(
  orbit: { readonly goal: OrbitGoal },
  element: HTMLElement,
  options: OrbitControlsOptions = {},
): OrbitControls {
  const { sensitivity, zoomSpeed = 0.001, button = 'primary', pan = false } = options;
  const { page = false, onChange } = options;
  const borrowed = options.pointer !== undefined;
  const { goal } = orbit;
  if (pan && !goal.target) {
    throw new Error('web.orbitControls requires goal.target when pan is enabled.');
  }
  const turning = BUTTONS[button];
  const panning = button === 'secondary' ? BUTTONS.middle : BUTTONS.secondary;

  const turn = (dx: number, dy: number) => turnBy(goal, dx, dy, { sensitivity });

  const shift = (dx: number, dy: number) =>
    panBy(goal as Required<OrbitGoal>, dx, dy, { height: Math.max(1, element.clientHeight) });

  const fingers = pinching((by, dx, dy) => {
    goal.distance *= by;
    if (pan) shift(dx, dy);
    else if (button !== 'primary') turn(dx, dy);
    onChange?.();
  });

  const press = (event: PointerEvent) => {
    if (!borrowed) element.setPointerCapture(event.pointerId);
    fingers.down(event);
  };

  const drag = (event: PointerEvent) => {
    if (fingers.move(event) || fingers.count > 2) return;

    const { x, y } = fingers.moved;
    if (pan && event.buttons === panning) {
      shift(x, y);
    } else if (event.buttons === turning) {
      turn(x, y);
    } else {
      return;
    }
    onChange?.();
  };

  const zoom = zoomer(page, zoomSpeed, (by) => {
    goal.distance *= by;
    onChange?.();
  });

  // Touch drags otherwise scroll the page and cancel the pointer. In page mode, keep vertical
  // scrolling enabled
  const touchAction = element.style.touchAction;
  if (!borrowed) element.style.touchAction = page ? 'pan-y' : 'none';
  const menus = pan || button === 'secondary';
  const middle = turning === BUTTONS.middle || (pan && panning === BUTTONS.middle);

  element.addEventListener('pointerdown', press);
  element.addEventListener('pointermove', drag);
  element.addEventListener('pointerup', fingers.up);
  element.addEventListener('pointercancel', fingers.up);
  element.addEventListener('lostpointercapture', fingers.up);
  element.addEventListener('wheel', zoom, { passive: false });
  if (menus) element.addEventListener('contextmenu', menu);
  if (middle) element.addEventListener('mousedown', still);

  const destroy = () => {
    if (!borrowed) element.style.touchAction = touchAction;
    element.removeEventListener('pointerdown', press);
    element.removeEventListener('pointermove', drag);
    element.removeEventListener('pointerup', fingers.up);
    element.removeEventListener('pointercancel', fingers.up);
    element.removeEventListener('lostpointercapture', fingers.up);
    element.removeEventListener('wheel', zoom);
    if (menus) element.removeEventListener('contextmenu', menu);
    if (middle) element.removeEventListener('mousedown', still);
  };

  return { destroy, [Symbol.dispose]: destroy };
}

function menu(event: Event) {
  event.preventDefault();
}

// A middle press starts the browser's autoscroll as the default action of `mousedown`
function still(event: MouseEvent) {
  if (event.button === 1) event.preventDefault();
}

/**
 * Follows the pointers down on an element and reports each move of a pair as a pinch: the ratio of
 * their last spread to the new one, above 1 as they close, and the move of their midpoint in CSS
 * pixels. `move` returns whether the pair moved and leaves the event's own move in `moved`.
 */
function pinching(on: (by: number, dx: number, dy: number) => void) {
  const pointers = new Map<number, { x: number; y: number }>();
  const middle = { x: 0, y: 0 };
  const moved = { x: 0, y: 0 };
  let spread = 0;

  const measure = () => {
    const [a, b] = pointers.values();
    middle.x = (a!.x + b!.x) / 2;
    middle.y = (a!.y + b!.y) / 2;
    return Math.hypot(a!.x - b!.x, a!.y - b!.y);
  };

  return {
    get count() {
      return pointers.size;
    },

    moved,

    down(event: PointerEvent) {
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pointers.size === 2) spread = measure();
    },

    move(event: PointerEvent): boolean {
      const held = pointers.get(event.pointerId);
      const { clientX, clientY } = event;
      // WebKit and Firefox on Android report no movementX/Y for touch, so follow the client position
      if (!held || !Number.isFinite(clientX) || !Number.isFinite(clientY)) {
        moved.x = event.movementX;
        moved.y = event.movementY;
        return false;
      }
      moved.x = clientX - held.x;
      moved.y = clientY - held.y;
      held.x = clientX;
      held.y = clientY;
      if (pointers.size !== 2) return false;
      const { x, y } = middle;
      const next = measure();
      on(spread > 0 && next > 0 ? spread / next : 1, middle.x - x, middle.y - y);
      spread = next;
      return true;
    },

    up(event: PointerEvent) {
      pointers.delete(event.pointerId);
      if (pointers.size === 2) spread = measure();
    },
  };
}

/**
 * A wheel listener that zooms by e^(deltaY × `speed`). With `page`, it leaves the wheel to the
 * page unless Ctrl or Command is held, as a trackpad pinch sends it.
 */
function zoomer(page: boolean, speed: number, zoom: (by: number) => void) {
  return (event: WheelEvent) => {
    if (page && !event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    zoom(Math.exp(event.deltaY * speed));
  };
}

/**
 * Runs with a pointer's position on the element: `x` and `y` from 0 to 1 across it with +y down,
 * as `rayAt` takes them and `projectAt` writes them.
 */
export type PointerHandler = (x: number, y: number, event: PointerEvent) => void;

export interface PointerOptions {
  /**
   * Runs as the pointer moves over the element, or anywhere while it holds a button down. Default
   * none.
   * @cpu
   */
  readonly move?: PointerHandler;
  /**
   * Runs as a button goes down on the element. Read the button from `event.button`. Default none.
   * @cpu
   */
  readonly down?: PointerHandler;
  /** Runs as a button comes up, or as the browser cancels the pointer. Default none. @cpu */
  readonly up?: PointerHandler;
  /** Runs as the pointer leaves the element with no button down. Default none. @cpu */
  readonly leave?: () => void;
  /**
   * Runs as the wheel turns or two fingers pinch on the element, with the factor to scale a
   * distance by, above 1 farther: e^(deltaY × 0.001) for the wheel, and the ratio of the fingers'
   * last spread to the new one for a pinch. Default none.
   * @cpu
   */
  readonly zoom?: (by: number) => void;
  /**
   * Allows page scrolling around an embedded element: touch drags scroll the page instead of
   * reaching `move`, and only the wheel with Ctrl or Command held zooms. Default false, so the
   * element takes every touch drag and wheel turn.
   */
  readonly page?: boolean;
  /** Shows the context menu on a secondary click. Default true. Set false to drag with it. */
  readonly menu?: boolean;
}

export interface Pointer extends Disposable {
  /** Holds the last position from 0 to 1 across the element, +y down. Starts at 0.5. */
  readonly x: number;
  /** Holds the last position from 0 to 1 down the element, as `x` does. */
  readonly y: number;
  /**
   * Tells whether the pointer is over the element or holds a button down that began on it. A
   * finger is over only while it touches.
   */
  readonly over: boolean;
  /** Holds the buttons held down, as `PointerEvent.buttons`: 1 primary, 2 secondary, 4 middle. */
  readonly buttons: number;
  /** Removes the listeners and restores the element's `touch-action`. Also `[Symbol.dispose]`. */
  destroy(): void;
}

/**
 * Follows the mouse, touch or pen on an element in the coordinates `rayAt` takes, 0 to 1 across
 * the element with +y down. Read the position each frame or take each move, press, release and
 * zoom. A button pressed on the element keeps the pointer until it comes up, so a drag continues
 * outside and its positions run past 0 and 1.
 *
 * @param element what the pointer moves over, usually the canvas
 * @returns the pointer, which you destroy to remove the listeners
 */
export function pointer(element: HTMLElement, options: PointerOptions = {}): Pointer {
  const { page = false, menu: menus = true, zoom } = options;
  const state = { x: 0.5, y: 0.5, over: false, buttons: 0 };
  const fingers = zoom && pinching(zoom);
  const wheel = zoom && zoomer(page, 0.001, zoom);

  const follow = (event: PointerEvent) => {
    const box = element.getBoundingClientRect();
    state.x = (event.clientX - box.left) / Math.max(1, box.width);
    state.y = (event.clientY - box.top) / Math.max(1, box.height);
    state.buttons = event.buttons;
  };

  const move = (event: PointerEvent) => {
    follow(event);
    state.over = true;
    fingers?.move(event);
    options.move?.(state.x, state.y, event);
  };
  const down = (event: PointerEvent) => {
    element.setPointerCapture(event.pointerId);
    follow(event);
    state.over = true;
    fingers?.down(event);
    options.down?.(state.x, state.y, event);
  };
  const up = (event: PointerEvent) => {
    follow(event);
    fingers?.up(event);
    options.up?.(state.x, state.y, event);
  };
  const leave = (event: PointerEvent) => {
    // A captured pointer leaves only once its button is up
    if (event.buttons !== 0) return;
    state.over = false;
    state.buttons = 0;
    options.leave?.();
  };

  const touchAction = element.style.touchAction;
  element.style.touchAction = page ? 'pan-y' : 'none';

  element.addEventListener('pointermove', move);
  element.addEventListener('pointerdown', down);
  element.addEventListener('pointerup', up);
  element.addEventListener('pointercancel', up);
  element.addEventListener('pointerleave', leave);
  if (fingers) element.addEventListener('lostpointercapture', fingers.up);
  if (wheel) element.addEventListener('wheel', wheel, { passive: false });
  if (!menus) element.addEventListener('contextmenu', menu);

  const destroy = () => {
    element.style.touchAction = touchAction;
    element.removeEventListener('pointermove', move);
    element.removeEventListener('pointerdown', down);
    element.removeEventListener('pointerup', up);
    element.removeEventListener('pointercancel', up);
    element.removeEventListener('pointerleave', leave);
    if (fingers) element.removeEventListener('lostpointercapture', fingers.up);
    if (wheel) element.removeEventListener('wheel', wheel);
    if (!menus) element.removeEventListener('contextmenu', menu);
  };

  return {
    get x() {
      return state.x;
    },
    get y() {
      return state.y;
    },
    get over() {
      return state.over;
    },
    get buttons() {
      return state.buttons;
    },
    destroy,
    [Symbol.dispose]: destroy,
  };
}

/** Provides the stage to sent closures, `onFrame` and `attach`. */
export type Stage<S extends Scene> = AnyStage<S, Surface>;

/** Hosts a scene with `pause` and `resume` for a pause menu or a canvas outside the viewport. */
export type Hosted<S extends Scene> = AnyHosted<S, Surface>;

export interface HostOptions<S extends Scene> extends AnyHostOptions<S, Surface>, SurfaceOptions {
  /** Supplies the canvas for the scene. */
  readonly canvas: HTMLCanvasElement;
  /**
   * Requests these features when the adapter supports them, such as `timestamp-query`. Default
   * none.
   */
  readonly optionalFeatures?: Iterable<GPUFeatureName>;
  /**
   * Creates the root where WebGPU is missing or gives no adapter, such as `webgl.fallback()` from
   * `@typegpu/kit-gl`. Ignored with `device` or `root`. Default none: the host reports the error.
   */
  readonly fallback?: Fallback;
}

/** Creates a root on another graphics API for `web.host`, which owns and destroys it. */
export interface Fallback {
  /** Uses the fallback even where WebGPU works, to see a scene on it. Default false. */
  readonly force?: boolean;
  /**
   * Creates the root. Its `destroy` frees the device with it.
   *
   * @param canvas the canvas the scene is hosted on
   */
  init(canvas: HTMLCanvasElement): TgpuRoot | Promise<TgpuRoot>;
}

async function init(
  canvas: HTMLCanvasElement,
  optionalFeatures: Iterable<GPUFeatureName>,
  fallback: Fallback | undefined,
): Promise<TgpuRoot> {
  if (!fallback) return tgpu.init({ device: { optionalFeatures } });
  if (!fallback.force && navigator.gpu) {
    try {
      return await tgpu.init({ device: { optionalFeatures } });
    } catch {
      // No adapter or no device: the browser has the API but not the hardware path
    }
  }
  return fallback.init(canvas);
}

/**
 * Runs a scene on a canvas with a root, surface, frame loop and animation. `destroy` tears them
 * down in order. Use `send` to control the scene or `onFrame` to read it. If scene options are
 * optional, call `host(createScene, { canvas })`. Throws if both `device` and `root` are supplied.
 *
 * @param createScene the scene file's `createScene`
 * @param options the scene's third argument
 * @param hosting the canvas, the device or root, and what the page runs beside the scene
 * @returns a hosted scene you send closures to and destroy
 */
export function host<S extends Scene>(
  createScene: CreateScene<Record<never, never>, S>,
  hosting: HostOptions<S>,
): Hosted<S>;
export function host<Options, S extends Scene>(
  createScene: CreateScene<Options, S>,
  options: Options,
  hosting: HostOptions<S>,
): Hosted<S>;
export function host<Options, S extends Scene>(
  createScene: CreateScene<Options, S>,
  ...rest: [HostOptions<S>] | [Options, HostOptions<S>]
): Hosted<S> {
  const [options, hosting] = rest.length === 1 ? [{} as Options, rest[0]] : rest;
  const { canvas, device, root: given, optionalFeatures = [], schedule, onFrame, attach } = hosting;
  const report = hosting.onError ?? ((error: unknown) => console.error(error));
  if (device && given) {
    throw new Error('web.host accepts either device or root. A root already provides its device.');
  }

  const create = async () => {
    let root = given;
    let canvasSurface: Surface | undefined;

    const release = () =>
      freeEach([
        () => canvasSurface?.destroy(),
        () => {
          if (!given) root?.destroy();
        },
      ]);

    try {
      root ??= device
        ? tgpu.initFromDevice({ device })
        : await init(canvas, optionalFeatures, hosting.fallback);

      canvasSurface = surface(root, canvas, hosting);

      return await stage(root, canvasSurface, createScene, options, {
        onError: (error) => report(asError(error)),
        watch: root.device,
        release,
        ...(schedule === undefined ? {} : { schedule }),
        ...(onFrame ? { onFrame } : {}),
        ...(attach ? { attach } : {}),
      });
    } catch (error) {
      freeEach([release], [error]);
    }
  };

  const owned = local(create, { onError: report });
  return {
    ...owned,
    pause: () => owned.send((staged) => staged.surface.pause()),
    resume: () => owned.send((staged) => staged.surface.resume()),
  };
}

export interface VideoOptions {
  /** Flips every frame left to right, as a mirror shows a front camera. Default false. */
  readonly mirrored?: boolean;
}

export interface Video extends videos.Source, Disposable {
  /** Stops watching the element for new frames. Leaves the element and its stream running. */
  destroy(): void;
}

/**
 * Reads a playing `<video>`, such as one showing `getUserMedia`'s camera, as a source for
 * `video.create`'s feed or `video.live`. Frames are numbered as the browser presents them. Readers
 * must submit in the task that imported the frame, as a frame loop does, because an imported video
 * frame expires when its task ends. A frame's `time` is its media time.
 *
 * @param element a video that plays, with its stream or file attached
 * @returns a source to attach to feeds and live views and destroy
 */
export function video(element: HTMLVideoElement, options: VideoOptions = {}): Video {
  const { mirrored = false } = options;
  const watched = typeof element.requestVideoFrameCallback === 'function';
  let presented = 0;
  let mediaTime = 0;
  let watching = 0;
  // Without frame callbacks, a frame is new when the playback time moves
  let counted = 0;
  let played = -1;
  // An imported frame expires as its task ends. Readers before the next microtask checkpoint, such
  // as one frame's feed and live view, share its import on each device
  const imports = new Map<GPUDevice, GPUExternalTexture>();

  const count = (_now: number, metadata: VideoFrameCallbackMetadata) => {
    presented += 1;
    mediaTime = metadata.mediaTime;
    watching = element.requestVideoFrameCallback(count);
  };
  if (watched) watching = element.requestVideoFrameCallback(count);

  const destroy = () => {
    if (watched) element.cancelVideoFrameCallback(watching);
  };

  const ready = () =>
    element.readyState >= element.HAVE_CURRENT_DATA &&
    element.videoWidth > 0 &&
    element.videoHeight > 0;
  // The frame showing before the first callback is number 1
  const serial = () => {
    if (watched) return presented + 1;
    if (element.currentTime !== played) {
      played = element.currentTime;
      counted += 1;
    }
    return counted;
  };

  return {
    latest(device, seen) {
      if (!ready()) return undefined;
      const newest = serial();
      if (newest === seen) return undefined;
      let texture = imports.get(device);
      if (!texture) {
        if (imports.size === 0) queueMicrotask(() => imports.clear());
        texture = device.importExternalTexture({ source: element });
        imports.set(device, texture);
      }
      return {
        texture,
        width: element.videoWidth,
        height: element.videoHeight,
        mirrored,
        serial: newest,
        time: watched && presented > 0 ? mediaTime : element.currentTime,
      };
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
