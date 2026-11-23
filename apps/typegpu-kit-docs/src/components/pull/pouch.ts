// The foil pouch, shared by the shader and the page: a flat pillow with crimped seals standing on
// the ground, seen from above at ELEVATION, its pose for a moment of the pull, and the grains the
// tear throws.

import type { Rgb } from '../site/dither.ts';
import { CELL } from './print.ts';
import {
  add,
  apply,
  frameOf,
  normalize,
  place,
  projectFrom,
  rotX,
  rotZ,
  sub,
  times,
  turnOf,
  type Frame,
  type V3,
} from 'typegpu-kit-pull/vessel.ts';
/** The camera's elevation above the ground, in radians. */
export const ELEVATION = 0.2;

/**
 * The pouch in its own units: half a unit wide each way, its crimped seals from `body` to `top`
 * above and below, the tear's seam across the upper seal, the pillow's half thickness at its
 * middle, a seal's, and the radius the torn strip curls up at.
 */
export const POUCH = {
  top: 1.62,
  body: 1.4,
  seam: 1.47,
  depth: 0.24,
  seal: 0.022,
  curl: 0.7,
} as const;

/** One moment of the pouch, in CSS pixels of its canvas, the pouch's units and sRGB from 0 to 1. */
export interface PouchInput {
  /** The ground under the pouch's middle, and CSS pixels a unit. */
  readonly anchor: readonly [number, number];
  readonly unit: number;
  /** Seconds, which only move the shader's noise. */
  readonly time: number;
  readonly body: Frame;
  /** The torn strip; the body's own frame until it tears. */
  readonly strip: Frame;
  /** The seam's light, 0 to 1 across, and the hint's run along it, below 0 for none. */
  readonly tear: number;
  readonly hint: number;
  /** The mouth's light in the tint, and how far it has puffed open. */
  readonly mouth: number;
  /** The body and the strip falling into grain. */
  readonly sink: number;
  readonly gone: number;
  readonly tint: Rgb;
  /** The grains torn off the seam: their middles, half sizes in CSS pixels and brightness. */
  readonly grains: readonly (readonly [number, number, number, number])[];
}

/** Grains the tear throws. */
export const GRAINS = 12;

/**
 * Where the pouch and its strip are: turned, leaning and lifted, and the strip, once torn, flung
 * up and away about the seam's right end.
 */
export function poseOf(pose: {
  yaw: number;
  roll: number;
  lift: number;
  tear: number;
  fly: number;
}): { body: Frame; strip: Frame } {
  const turn = turnOf(pose.yaw, 0, pose.roll);
  const t: V3 = [0, POUCH.top + 0.05 + pose.lift, 0];
  const hinge: V3 = [1, POUCH.seam, 0];
  const bend = times(rotZ(-1.5 * pose.fly), rotX(0.9 * pose.fly));
  const away: V3 = [0.9 * pose.fly, 1.4 * pose.fly, 0.5 * pose.fly];
  const local = sub(add(hinge, away), apply(bend, hinge));
  return { body: frameOf(turn, t), strip: frameOf(times(turn, bend), add(apply(turn, local), t)) };
}

/** Where world point `p` lands on the canvas, in CSS pixels. */
export const project = projectFrom(ELEVATION);

const SIN = Math.sin(ELEVATION);
const COS = Math.cos(ELEVATION);

/** The mouth along the seam on the canvas: its middle, its right and up, and its width. */
export function mouthOf(body: Frame, anchor: readonly [number, number], unit: number) {
  const at = (x: number) => project(place(body, [x, POUCH.seam, 0]), anchor, unit);
  const [left, middle, right] = [at(-1), at(0), at(1)];
  const width = Math.hypot(right[0] - left[0], right[1] - left[1]);
  const across: [number, number] = [(right[0] - left[0]) / width, (right[1] - left[1]) / width];
  return { middle, right: across, up: [across[1], -across[0]] as [number, number], width };
}

/** Each grain: where along the seam it breaks off, how it is thrown, and how big it is. */
const THROWN = Array.from({ length: GRAINS }, (_, k) => ({
  along: (k + Math.random() * 0.8) / GRAINS,
  out: (Math.random() - 0.35) * 0.32,
  up: 0.18 + Math.random() * 0.32,
  half: Math.random() < 0.3 ? CELL : CELL / 2,
}));

/**
 * The grains the tear throws: `ageOf` gives each one's milliseconds since the tear passed `along`
 * the seam, 0 to 1, negative before.
 */
export function grainsAt(
  ageOf: (along: number) => number,
  body: Frame,
  anchor: readonly [number, number],
  unit: number,
): PouchInput['grains'] {
  const life = 520;
  const scale = unit / 90;
  return THROWN.map(({ along, out, up, half }) => {
    const age = ageOf(along);
    if (age < 0 || age > life) return [0, 0, 0, 0] as const;
    const [x, y] = project(place(body, [-1 + 2 * along, POUCH.seam, 0.03]), anchor, unit);
    return [
      x + out * age * scale,
      y - up * age * scale + 0.0011 * age * age * scale,
      half,
      1 - age / life,
    ] as const;
  });
}

export const LIGHT: V3 = normalize([-0.45, 0.75, 0.5]);
export const VIEW: V3 = [0, SIN, COS];
export const HALF: V3 = normalize(add(LIGHT, VIEW));

/** The bounds of the body and the strip, in their own units. */
export const BODY_BOX = {
  middle: [0, 0, 0] as V3,
  half: [1.02, POUCH.top + 0.02, POUCH.depth + 0.1] as V3,
};
export const STRIP_BOX = {
  middle: [0, POUCH.seam + POUCH.curl, 0] as V3,
  half: [1.02, POUCH.curl + 0.08, 0.03] as V3,
};
