import type { Shape } from 'typegpu-kit-examples/hosts';

/**
 * Routing constants, in points and milliseconds: a canvas finger that moves past `slop` drags, and
 * one that lifts within `slop` and `tapMs` taps.
 */
export const TOUCH = { slop: 8, tapMs: 350 } as const;

/** A pad on screen, in points from the touch area's top left. */
export interface PadRect {
  /** The buttons element that declares it. */
  readonly element: string;
  /** The button's name in that element. */
  readonly name: string;
  /**
   * Whether the button is held: pressed by the first finger on it and released as the last lifts.
   */
  readonly held: boolean;
  /** Whether a tap holds the button down until the next tap. Only for held buttons. */
  readonly latch: boolean;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** The key a pad goes by, such as in the lit pads and the router's holders. */
export const padKey = (element: string, name: string) => `${element}/${name}`;

/** What the touch area holds: its size in points, the scene's shape and the pads over it. */
export interface TouchLayout {
  readonly width: number;
  readonly height: number;
  readonly shape: Pick<Shape, 'orbit' | 'pointing'> | undefined;
  readonly pads: readonly PadRect[];
}

/** What fingers do, in the touch area's points or, for the scene's pointing, from 0 to 1. */
export interface TouchActions {
  /** A finger landed on a pad, or on a held pad no other finger holds. */
  press(pad: PadRect): void;
  /** A finger on a pad lifted or was cancelled, or the last on a held pad. Every press gets one. */
  release(pad: PadRect): void;
  /** Turns an orbiting view by a drag of `dx` and `dy` points. */
  orbit(dx: number, dy: number): void;
  /** Zooms, multiplying the view's distance by `by`, above 1 farther. */
  zoom(by: number): void;
  /** A tap at `x` and `y`, from 0 to 1 across the area. */
  tap(x: number, y: number): void;
  /** A stroke point at `x` and `y`, from 0 to 1 across the area; `start` marks its first. */
  drag(x: number, y: number, start: boolean): void;
  /** Ends the stroke `drag` began. */
  end(): void;
}

type Role = 'pending' | 'stroke' | 'orbit' | 'pinch' | 'ignored';

interface Finger {
  role: Role;
  /** The pad the finger landed on, which keeps it until it lifts. */
  readonly pad: PadRect | undefined;
  readonly x0: number;
  readonly y0: number;
  readonly t0: number;
  x: number;
  y: number;
  /** Whether it moved past the slop. */
  moved: boolean;
}

export interface TouchRouter {
  /** Replaces the layout. Fingers already down keep their routes. */
  configure(layout: TouchLayout): void;
  down(id: number, x: number, y: number, at: number): void;
  move(id: number, x: number, y: number, at: number): void;
  up(id: number, x: number, y: number, at: number): void;
  /** Ends a finger without a tap. */
  cancel(id: number): void;
  /** Cancels every finger down: held pads release and a stroke ends. */
  reset(): void;
  /** Whether any finger is down. */
  readonly active: boolean;
}

/**
 * Routes each finger once, where it lands, and keeps the route until it lifts. A finger on a pad
 * presses it and releases it as it lifts; a held pad stays down until its last finger lifts. On the canvas, the first finger strokes at once when the
 * scene strokes, else waits: past the slop it orbits an orbiting view, and a quick lift within the
 * slop taps. A second canvas finger makes a pinch with the first, which ends a stroke, zooms by the
 * change in their spread and orbits by the move of their midpoint. Further canvas fingers, and the
 * finger left when a pinch loses one, do nothing until they lift. Pads and the canvas never share
 * a finger, so pads, a drag and a pinch run together.
 */
export function createTouchRouter(actions: TouchActions): TouchRouter {
  const fingers = new Map<number, Finger>();
  let layout: TouchLayout = { width: 1, height: 1, shape: undefined, pads: [] };
  /** The fingers on each held pad, by `padKey`. */
  const holders = new Map<string, number>();
  /** The pinch's last spread and midpoint, while two fingers pinch. */
  let pinch: { spread: number; x: number; y: number } | undefined;

  const padAt = (x: number, y: number) => {
    for (const pad of layout.pads) {
      if (x >= pad.x && x <= pad.x + pad.width && y >= pad.y && y <= pad.y + pad.height) {
        return pad;
      }
    }
    return undefined;
  };

  const pinching = () => {
    const pair: Finger[] = [];
    for (const finger of fingers.values()) if (finger.role === 'pinch') pair.push(finger);
    return pair.length === 2 ? (pair as [Finger, Finger]) : undefined;
  };

  const measure = ([a, b]: readonly [Finger, Finger]) => ({
    spread: Math.hypot(a.x - b.x, a.y - b.y),
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
  });

  const canvasFingers = () => {
    let count = 0;
    for (const finger of fingers.values()) if (!finger.pad && finger.role !== 'ignored') count++;
    return count;
  };

  const strokes = () => layout.shape?.pointing?.drag === true;
  const pinches = () => layout.shape?.orbit === true || layout.shape?.pointing?.zoom === true;

  /** Presses a pad, or adds a holder to a held pad and presses it on the first. */
  const press = (pad: PadRect) => {
    if (pad.held) {
      const key = padKey(pad.element, pad.name);
      const count = holders.get(key) ?? 0;
      holders.set(key, count + 1);
      if (count > 0) return;
    }
    actions.press(pad);
  };

  /** Releases a pad, or takes a holder from a held pad and releases it on the last. */
  const release = (pad: PadRect) => {
    if (pad.held) {
      const key = padKey(pad.element, pad.name);
      const count = (holders.get(key) ?? 1) - 1;
      if (count > 0) {
        holders.set(key, count);
        return;
      }
      holders.delete(key);
    }
    actions.release(pad);
  };

  /** Ends a finger's route; `lifted` decides whether a quick lift taps. */
  const finish = (id: number, finger: Finger, at: number, lifted: boolean) => {
    fingers.delete(id);
    if (finger.pad) {
      release(finger.pad);
      return;
    }
    if (finger.role === 'stroke') actions.end();
    if (finger.role === 'pinch') {
      pinch = undefined;
      // The finger left behind waits for its lift
      for (const other of fingers.values()) if (other.role === 'pinch') other.role = 'ignored';
    }
    const quick = lifted && !finger.moved && at - finger.t0 <= TOUCH.tapMs;
    if (
      quick &&
      (finger.role === 'pending' || finger.role === 'stroke') &&
      layout.shape?.pointing
    ) {
      actions.tap(finger.x / layout.width, finger.y / layout.height);
    }
  };

  function move(id: number, x: number, y: number) {
    const finger = fingers.get(id);
    if (!finger || finger.pad || finger.role === 'ignored') return;
    const dx = x - finger.x;
    const dy = y - finger.y;
    finger.x = x;
    finger.y = y;
    if (!finger.moved && Math.hypot(x - finger.x0, y - finger.y0) > TOUCH.slop) {
      finger.moved = true;
    }

    if (finger.role === 'stroke') {
      actions.drag(x / layout.width, y / layout.height, false);
    } else if (finger.role === 'pending') {
      if (finger.moved && layout.shape?.orbit) {
        finger.role = 'orbit';
        actions.orbit(x - finger.x0, y - finger.y0);
      }
    } else if (finger.role === 'orbit') {
      actions.orbit(dx, dy);
    } else if (finger.role === 'pinch' && pinch) {
      const pair = pinching();
      if (!pair) return;
      const next = measure(pair);
      if (next.spread > 0 && pinch.spread > 0) actions.zoom(pinch.spread / next.spread);
      if (layout.shape?.orbit) actions.orbit(next.x - pinch.x, next.y - pinch.y);
      pinch = next;
    }
  }

  return {
    configure(next) {
      layout = next;
    },

    down(id, x, y, at) {
      const pad = padAt(x, y);
      const finger: Finger = { role: 'pending', pad, x0: x, y0: y, t0: at, x, y, moved: false };
      if (pad) {
        fingers.set(id, finger);
        press(pad);
        return;
      }
      const others = canvasFingers();
      fingers.set(id, finger);
      if (others === 0) {
        if (strokes()) {
          finger.role = 'stroke';
          actions.drag(x / layout.width, y / layout.height, true);
        }
        return;
      }
      if (others > 1 || !pinches()) {
        finger.role = 'ignored';
        return;
      }
      for (const other of fingers.values()) {
        if (other === finger || other.pad || other.role === 'ignored') continue;
        if (other.role === 'stroke') actions.end();
        other.role = 'pinch';
        other.moved = true;
      }
      finger.role = 'pinch';
      finger.moved = true;
      const pair = pinching();
      pinch = pair && measure(pair);
    },

    move,

    up(id, x, y, at) {
      const finger = fingers.get(id);
      if (!finger) return;
      if (!finger.pad && finger.role !== 'ignored' && (x !== finger.x || y !== finger.y)) {
        move(id, x, y);
      }
      finish(id, finger, at, true);
    },

    cancel(id) {
      const finger = fingers.get(id);
      if (finger) finish(id, finger, 0, false);
    },

    reset() {
      for (const [id, finger] of fingers) finish(id, finger, 0, false);
      pinch = undefined;
    },

    get active() {
      return fingers.size > 0;
    },
  };
}
