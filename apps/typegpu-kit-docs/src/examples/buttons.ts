// A `buttons` element on the web: keys press its buttons, and touch pads show them

import type { AnyButtonsKind, Held, PadRows as Rows, Tap } from 'typegpu-kit-examples/ui.ts';
import { onKeys } from '../components/viz/figure.ts';

/** A touch pad's button: its label, and what it does as a finger presses it and lets go. */
export interface Pad {
  readonly label: string;
  /** What a screen reader says, when the label is a symbol. */
  readonly name?: string;
  readonly press: () => void;
  readonly release?: () => void;
  /** A tap presses it and the next tap releases it. */
  readonly latch?: boolean;
}

/** A side of a touch pad: rows of buttons, `null` leaving a gap. */
export type PadRows = readonly (readonly (Pad | null)[])[];

/** The handlers a scene binds to a `buttons` element, by button name. */
export type Pressed = Readonly<Record<string, Tap | Held>>;

/** Page controls that take keys themselves, such as Space on a button. */
const CONTROLS = 'a[href], button, input, select, textarea, [role=button], [role=slider]';

/** Whether a key event goes to a control on the page, such as a panel slider or a link. */
const atControl = (event: KeyboardEvent) => {
  const target = event.target as Partial<HTMLElement> | null;
  return target?.isContentEditable === true || target?.closest?.(CONTROLS) != null;
};

/**
 * Presses a button as one of its keys goes down on `target`, until `signal` aborts, unless a page
 * control has focus. A held button is pressed by the first of its keys to go down and released as
 * the last comes up, or as `target` loses focus; others press on every key repeat.
 */
export function listenKeys(
  declared: AnyButtonsKind,
  bound: Pressed,
  target: EventTarget,
  signal: AbortSignal,
) {
  const named = new Map<string, string>();
  /** The keys holding each held button, which release it even after focus moved to a control. */
  const holders = new Map<string, Set<string>>();
  for (const [name, button] of Object.entries(declared.buttons)) {
    for (const code of button.keys ?? []) named.set(code, name);
    if (button.held) holders.set(name, new Set());
  }

  const key = (event: KeyboardEvent, down: boolean) => {
    const name = named.get(event.code);
    if (name === undefined) return;
    const handler = bound[name];
    const codes = holders.get(name);
    if (!down && codes?.delete(event.code)) {
      event.preventDefault();
      if (codes.size === 0) (handler as Held).release();
      return;
    }
    if (!down || atControl(event)) return;
    event.preventDefault();
    if (!codes) handler.press();
    else if (!event.repeat && !codes.has(event.code)) {
      codes.add(event.code);
      if (codes.size === 1) handler.press();
    }
  };
  // Keys that come up while the page is in the background send no keyup
  const blur = () => {
    for (const [name, codes] of holders) {
      if (codes.size === 0) continue;
      codes.clear();
      (bound[name] as Held).release();
    }
  };
  const listening = { signal };
  target.addEventListener('keydown', (event) => key(event as KeyboardEvent, true), listening);
  target.addEventListener('keyup', (event) => key(event as KeyboardEvent, false), listening);
  target.addEventListener('blur', blur, listening);
}

/**
 * Connects a pad's button: pressed while a finger that landed on it, or Space or Enter while it
 * has focus, stays down, or from tap to tap when it latches. A click with neither, as assistive
 * technology sends, taps it.
 */
export function connectPad(button: HTMLElement, pad: Pad) {
  button.addEventListener('contextmenu', (event) => event.preventDefault());
  /** The pointers and key holding it down; the first presses it and the last releases it. */
  const holders = new Set<number | 'key'>();
  const show = (down: boolean) => {
    button.classList.toggle('held', down);
    if (pad.latch) button.setAttribute('aria-pressed', String(down));
    if (down) pad.press();
    else pad.release?.();
  };
  const press = (holder: number | 'key') => {
    if (pad.latch) show(!button.classList.contains('held'));
    else if (!holders.has(holder)) {
      holders.add(holder);
      if (holders.size === 1) show(true);
    }
  };
  const release = (holder: number | 'key') => {
    if (pad.latch || !holders.delete(holder)) return;
    if (holders.size === 0) show(false);
  };

  if (pad.latch) button.setAttribute('aria-pressed', 'false');
  button.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    if (!pad.latch) button.setPointerCapture(event.pointerId);
    press(event.pointerId);
  });
  const up = (event: PointerEvent) => release(event.pointerId);
  button.addEventListener('pointerup', up);
  button.addEventListener('pointercancel', up);
  button.addEventListener('lostpointercapture', up);
  onKeys(button, { down: () => press('key'), up: () => release('key') });
  button.addEventListener('click', (event) => {
    if (event.detail !== 0 || holders.size > 0) return;
    press('key');
    release('key');
  });
}

/** The pads of a `buttons` element, each pressing its button's handler. */
export function padsOf(
  declared: AnyButtonsKind,
  bound: Pressed,
): { readonly left?: PadRows; readonly right?: PadRows } {
  const side = (rows: Rows<string> | undefined) =>
    rows?.map((row) =>
      row.map((name): Pad | null => {
        if (name === null) return null;
        const { label, name: spoken, held, latch } = declared.buttons[name];
        const handler = bound[name];
        return {
          label,
          ...(spoken ? { name: spoken } : {}),
          press: () => handler.press(),
          ...(held ? { release: () => (handler as Held).release() } : {}),
          ...(held && latch ? { latch: true } : {}),
        };
      }),
    );
  const left = side(declared.pads.left);
  const right = side(declared.pads.right);
  return { ...(left ? { left } : {}), ...(right ? { right } : {}) };
}
