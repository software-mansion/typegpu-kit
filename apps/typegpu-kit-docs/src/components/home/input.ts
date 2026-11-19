import { slider } from '../viz/figure.ts';
import { shown, type Knobs } from './knobs.ts';
import { clamp } from './geometry.ts';

/** Seconds the pointer stays on another sheet before it counts, and off every sheet. */
const DWELL = 0.09;
const LEAVE = 0.15;

/** Switch sheets after `DWELL` seconds over another sheet. Clear the selection after `LEAVE` seconds off every sheet. */
export function hovering() {
  let current = -1;
  let next = -1;
  let waited = 0;
  return {
    get current() {
      return current;
    },

    step(raw: number, dt: number, instant: boolean) {
      if (raw === current) {
        next = current;
        waited = 0;
        return current;
      }
      if (raw !== next) {
        next = raw;
        waited = 0;
      }
      waited += dt;
      if (instant || waited >= (raw < 0 ? LEAVE : DWELL)) {
        current = raw;
        waited = 0;
      }
      return current;
    },
  };
}

export interface PointerOptions {
  /** The sheet a drag sideways pulls, or -1. */
  grab?: (x: number, y: number) => number;
  /** The sheet a click lands on, or -1. */
  pick?: (x: number, y: number) => number;
  /** How far a drag turns the stack, in radians either way, and tilts it down and up. */
  limits?: { yaw: number; down: number; up: number };
  /** What the stage leaves to its own controls. */
  skip?: string;
}

/**
 * What a stack's stage does under the pointer: point to lift, click a sheet, drag to turn, and,
 * where `grab` finds a sheet that slides, drag it sideways to pull it out or push it back.
 */
export function pointerOn(element: HTMLElement, redraw: () => void, options: PointerOptions = {}) {
  const { grab, pick, limits = { yaw: 0.3, down: -0.25, up: 0.1 } } = options;
  const skip = options.skip ?? 'a, button, input, label, li, .side';
  const state = {
    x: -1,
    y: -1,
    over: false,
    yaw: 0,
    pitch: 0,
    dragging: false,
    pulling: -1,
    pulled: 0,
    released: -1,
    clicked: -1,
  };
  let down:
    | { x: number; y: number; yaw: number; pitch: number; id: number; sheet: number; at: number }
    | undefined;
  let mode: 'undecided' | 'turn' | 'pull' | 'scroll' = 'undecided';
  const local = (event: PointerEvent) => {
    const box = element.querySelector('canvas')!.getBoundingClientRect();
    return [event.clientX - box.left, event.clientY - box.top] as const;
  };
  element.addEventListener('pointermove', (event) => {
    [state.x, state.y] = local(event);
    // A touch doesn't hover
    state.over = event.pointerType !== 'touch' && !(event.target as Element).closest(skip);
    if (down && event.pointerId === down.id) {
      const dx = event.clientX - down.x;
      const dy = event.clientY - down.y;
      // A touch moving up or down scrolls the page, which takes it over a few pixels in
      const touch = event.pointerType === 'touch';
      if (mode === 'undecided') {
        if (down.sheet >= 0 && Math.abs(dx) > 6 && Math.abs(dx) > Math.abs(dy) * 1.5) {
          mode = 'pull';
          state.pulling = down.sheet;
        } else if (touch && Math.abs(dy) >= Math.abs(dx) && Math.hypot(dx, dy) > 5) {
          mode = 'scroll';
        } else if (Math.hypot(dx, dy) > 5) {
          mode = 'turn';
          state.dragging = true;
        }
      }
      if (mode === 'pull') state.pulled = dx;
      if (mode === 'turn') {
        state.yaw = clamp(down.yaw - dx * 0.006, -limits.yaw, limits.yaw);
        if (!touch) state.pitch = clamp(down.pitch + dy * 0.004, limits.down, limits.up);
      }
    }
    redraw();
  });
  element.addEventListener('pointerleave', () => {
    state.over = false;
    redraw();
  });
  element.addEventListener('pointerdown', (event) => {
    if ((event.target as Element).closest(skip)) return;
    if (event.button !== 0) return;
    const [x, y] = local(event);
    down = {
      x: event.clientX,
      y: event.clientY,
      yaw: state.yaw,
      pitch: state.pitch,
      id: event.pointerId,
      sheet: grab?.(x, y) ?? -1,
      at: pick?.(x, y) ?? -1,
    };
    mode = 'undecided';
    element.setPointerCapture(event.pointerId);
  });
  const up = (event: PointerEvent) => {
    if (!down || event.pointerId !== down.id) return;
    if (mode === 'pull') state.released = state.pulling;
    if (mode === 'undecided' && event.type === 'pointerup') state.clicked = down.at;
    down = undefined;
    mode = 'undecided';
    state.dragging = false;
    state.pulling = -1;
    state.pulled = 0;
    redraw();
  };
  element.addEventListener('pointerup', up);
  element.addEventListener('pointercancel', up);
  return state;
}

/**
 * Makes `label`, a number in the code, a slider of its knob, as the figures' numbers are. A change
 * shows in every copy of the knob on the page.
 */
export function knob(
  label: HTMLElement,
  change: (key: keyof Knobs, value: number) => void,
  held: (key: keyof Knobs, on: boolean) => void,
) {
  const key = label.dataset.knob as keyof Knobs;
  const format = (value: number) => shown(key, value);
  const input = slider(
    label,
    key,
    (value) => {
      for (const other of document.querySelectorAll<HTMLElement>(`[data-knob='${key}']`)) {
        if (other === label) continue;
        const copy = other.querySelector('input');
        if (copy) {
          copy.valueAsNumber = value;
          copy.setAttribute('aria-valuetext', format(value));
        }
        (other.querySelector('output') ?? other).textContent = format(value);
      }
      change(key, value);
    },
    format,
  );
  input.disabled = false;
  input.addEventListener('pointerdown', () => held(key, true));
  input.addEventListener('lostpointercapture', () => held(key, false));
}
