// A pull as pure functions of milliseconds: the vessel's opening, the card leaving it face down, and
// the card's flight into place.

export const clamp = (x: number, low: number, high: number) => Math.min(Math.max(x, low), high);
export const clamp01 = (x: number) => clamp(x, 0, 1);
export const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
const span = (t: number, from: number, to: number) => clamp01((t - from) / (to - from));
const outCubic = (k: number) => 1 - (1 - k) ** 3;
const outQuad = (k: number) => 1 - (1 - k) ** 2;
const smooth = (k: number) => k * k * (3 - 2 * k);
export const inOutQuad = (k: number) => (k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2);
export const inOutCubic = (k: number) => (k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2);
const bump = (k: number) => Math.sin(Math.PI * clamp01(k));

/** When the card starts out of the box, face down, and when it is out and flies. */
export const EXIT = [1460, 1800] as const;

/**
 * When the card's flight starts: for the box in milliseconds since the overlay's intro, for the
 * pack since its tear was finished.
 */
export const LAUNCH = { box: EXIT[1], pack: 1150 } as const;

/** Milliseconds a tear takes when the pack tears itself, at a press. */
export const TEAR = 460;

/** A decaying jolt `since` milliseconds after a knock lands. */
const jolt = (since: number) =>
  since < 0 || since > 450 ? 0 : Math.exp(-since / 80) * Math.sin(since * 0.09);

const KNOCKS = [560, 860, 1100];
const HOLD = 1300;
const POP = 1420;

/** The box `t` milliseconds into the pull, `phase` (0 to 1) shifting its shudder. */
export function boxAt(t: number, phase: number) {
  const before = t < POP;
  const s = t / 1000;

  // A dip under the press, then the lift; after the pop the box settles back
  const dip = t < 0 ? 0 : Math.exp(-t / 90) * clamp01(t / 40) * 0.06;
  const raised = outCubic(span(t, 120, 700)) * 0.34 + (t >= HOLD && before ? 0.03 : 0);
  const recoil = before ? 1 : 1 - outCubic(span(t, POP, POP + 300));

  // The shudder rises toward the hold and stops dead in it; each knock jolts the box
  const rumble = t < HOLD ? span(t, 200, HOLD) ** 2 : 0;
  const knock = KNOCKS.reduce((sum, at, i) => sum + jolt(t - at) * (0.05 + i * 0.022), 0);

  // Light leaks a little more after each knock, floods in the hold and fades as the card rises
  const count = KNOCKS.filter((at) => t >= at).length;
  const flash = KNOCKS.reduce((sum, at) => sum + Math.max(jolt(t - at), 0) * 0.4, 0);
  const leak = before
    ? clamp01(0.1 + 0.18 * count + flash + 0.35 * span(t, HOLD, POP))
    : 1 - span(t, POP, POP + 500);

  const sink = inOutCubic(span(t, 1950, 2550));
  const popped = outCubic(span(t, POP, POP + 520));
  const live = t >= 0;
  return {
    lift: live ? raised * recoil - dip : 0,
    yaw: live ? smooth(span(t, 150, HOLD)) * 0.55 + rumble * 0.01 * Math.sin(s * 37) : 0,
    tiltX: live
      ? rumble * 0.03 * Math.sin(s * 47 + phase * 20) + knock * (phase > 0.5 ? 1 : -1) * 0.7
      : 0,
    tiltZ: live ? rumble * 0.035 * Math.sin(s * 59 + phase * 31) + knock : 0,
    leak: live ? leak : 0.1,
    lidRise: popped * 1.5,
    lidTilt: -popped * 1.25,
    lidSpin: popped * 0.8,
    lidFade: smooth(span(t, POP + 80, POP + 380)),
    open: before ? 0 : clamp01((t - POP) / 80) * (1 - 0.6 * popped) * (1 - sink),
    sink,
  };
}

/**
 * The pack `t` milliseconds after its tear was finished, negative while it waits: the strip flung,
 * the rarity's light in the mouth, the card peeking out, then the pack dropping away from it.
 */
export function packAt(t: number) {
  const drop = inOutCubic(span(t, 620, 1180));
  return {
    /** The pack gives as the tear finishes, 0 → 1 → 0. */
    press: bump(span(t, -80, 200)),
    /** The torn strip flung up and away, then falling into grain. */
    fly: outCubic(span(t, -60, 420)),
    gone: span(t, 160, 520),
    /** The rarity's light in the open mouth: the tell. */
    mouth: outQuad(span(t, 0, 320)) * (1 - drop),
    /** The card rising a little out of the mouth, face down. */
    peek: outCubic(span(t, 220, 640)),
    /** The pack falling away below the card, and into grain. */
    drop,
    sink: inOutQuad(span(t, 760, 1260)),
  };
}

/** 0 to 1: the card back's edge glowing in the rarity's colour, from when the vessel tells. */
export const tellAt = (t: number, vessel: 'box' | 'pack') =>
  vessel === 'box' ? smooth(span(t, 450, 850)) : outQuad(span(t, 0, 420));

/** How far the card is out of the box, face down: 0 inside, 1 clear of the mouth. */
export const outAt = (t: number) => inOutCubic(span(t, EXIT[0], EXIT[1]));

/**
 * A damped spring from 0 to 1 `s` seconds after release: `omega` its stiffness in radians a second,
 * `zeta` its damping, below 1 overshooting.
 */
function spring(s: number, omega: number, zeta: number) {
  if (s <= 0) return 0;
  const decay = Math.exp(-zeta * omega * s);
  if (zeta >= 1) return 1 - decay * (1 + omega * s);
  const damped = omega * Math.sqrt(1 - zeta * zeta);
  return 1 - decay * (Math.cos(damped * s) + ((zeta * omega) / damped) * Math.sin(damped * s));
}

/** Milliseconds from leaving the vessel until the card is still in its place. */
export const FLIGHT = 1500;

/** The card's flight `ms` after it leaves the vessel, every part 0 at the vessel and 1 in place. */
export function flightAt(ms: number) {
  const s = Math.max(ms, 0) / 1000;
  const done = ms >= FLIGHT;
  return {
    /** Along its arc from the vessel to its place, overshooting a little. */
    travel: done ? 1 : spring(s, 8.5, 0.74),
    /** From the vessel's width to its own. */
    grow: done ? 1 : spring(s - 0.04, 8, 0.82),
    /** Toward the viewer in its middle, in its own heights. */
    near: done ? 0 : 0.42 * bump(s / 0.85) * (1 - span(ms, 700, 1100)),
    /** The turn from its back to its face, overshooting and swinging back. */
    turn: done ? 1 : spring(s - 0.14, 10.5, 0.4),
    /** Its top tipping toward the viewer as it is tossed, and settling back. */
    tip: done ? 0 : bump(s / 0.7) - 0.25 * bump((s - 0.55) / 0.5),
    /** A roll that dies away. */
    roll: done ? 0 : Math.sin(s * 9) * Math.exp(-s * 4.2),
    /** The foil's flare sweeping the face once as it comes round, -0.3 to 1.4, and its brightness. */
    sweep: -0.3 + 1.7 * inOutQuad(span(ms, 260, 860)),
    flare: bump(span(ms, 260, 860)),
    /** The quiet row under the card. */
    row: outCubic(span(ms, 900, 1300)),
    done,
  };
}

/** Milliseconds a vessel takes from its resting spot to its place in the overlay. */
export const HAND = 420;

/** Where a vessel stands: its ground and points a unit, and its turn beyond its own. */
export interface Placed {
  readonly anchor: readonly [number, number];
  readonly unit: number;
  readonly yaw: number;
  readonly roll: number;
}

/** A vessel `elapsed` milliseconds after it was handed over `from` its resting spot, on its way to `ground` at `size` points a unit. */
export function handoffAt(
  elapsed: number,
  from: Placed,
  ground: readonly [number, number],
  size: number,
): Placed {
  const k = outCubic(clamp01(elapsed / HAND));
  return {
    anchor: [lerp(from.anchor[0], ground[0], k), lerp(from.anchor[1], ground[1], k)],
    unit: lerp(from.unit, size, k),
    yaw: from.yaw * (1 - k),
    roll: from.roll * (1 - k),
  };
}
