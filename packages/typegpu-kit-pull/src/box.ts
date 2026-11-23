// The box's geometry, shared by the shader and its hosts: a cube on the ground seen from above at
// ELEVATION, and its pose at rest and for a moment of the pull.
import type { Rgb01 } from './dither.ts';
import { boxAt, lerp, tellAt } from './timeline.ts';
import {
  frameOf,
  normalize,
  place,
  projectFrom,
  rotX,
  rotY,
  times,
  apply,
  turnOf,
  type Frame,
  type V3,
} from './vessel.ts';

/** The camera's elevation above the ground, in radians. */
export const ELEVATION = 0.42;
/** The box's half-width and its body's height, in units; the lid sits on top. */
export const BODY = { center: [0, 0.8, 0], half: [1, 0.8, 1] } as const;
export const LID = { at: 1.8, half: [1.06, 0.2, 1.06] } as const;
/** The body's top, where the lid's seam runs. */
export const SEAM = 1.6;
/** Units the box sinks to be under the ground, and the band above the ground where it crumbles. */
const SINK = 2.2;
export const BAND = 0.3;

/** The box's three-quarter turn at rest. */
const YAW = 0.6;

/** One moment of the box, in points of its canvas, units of the box and sRGB from 0 to 1. */
export interface BoxInput {
  /** The ground under the box's centre. */
  readonly anchor: readonly [number, number];
  /** CSS pixels a unit: the box is two units wide. */
  readonly unit: number;
  readonly body: Frame;
  readonly lid: Frame;
  readonly leak: number;
  readonly tint: Rgb01;
  readonly open: number;
  /** 0 to 1: the box sunk into the ground, crumbling into grain where it meets it. */
  readonly sink: number;
  readonly lidFade: number;
  /** 0 to 1: the light on the ground around the box. */
  readonly glow: number;
}

/** Where the box and its lid are: turned, tilted, lifted, the lid risen, tipped and spun. */
function poseOf(pose: {
  yaw: number;
  tiltX: number;
  tiltZ: number;
  lift: number;
  lidRise: number;
  lidTilt: number;
  lidSpin: number;
  sink: number;
}): { body: Frame; lid: Frame } {
  const turn = turnOf(pose.yaw, pose.tiltX, pose.tiltZ);
  const lift = pose.lift - pose.sink * SINK;
  const up = apply(turn, [0, LID.at + pose.lidRise, 0]);
  return {
    body: frameOf(turn, [0, lift, 0]),
    lid: frameOf(times(turn, times(rotY(pose.lidSpin), rotX(pose.lidTilt))), [
      up[0],
      up[1] + lift,
      up[2],
    ]),
  };
}

/** Where world point `p` lands on the canvas, in points. */
const project = projectFrom(ELEVATION);

/**
 * The front edges of the body's open top, left to right, in points of the canvas: the lower chain
 * of its four corners on screen.
 */
function rimOf(body: Frame, anchor: readonly [number, number], unit: number) {
  const corners = (
    [
      [-1, SEAM, -1],
      [1, SEAM, -1],
      [1, SEAM, 1],
      [-1, SEAM, 1],
    ] as const
  ).map((c) => project(place(body, c), anchor, unit));
  // Hermes lacks `toSorted`
  corners.sort((a, b) => a[0] - b[0]);
  const left = corners[0];
  const right = corners[3];
  const below = (c: readonly [number, number]) =>
    c[1] > left[1] + ((right[1] - left[1]) * (c[0] - left[0])) / (right[0] - left[0] || 1);
  return [left, ...corners.slice(1, 3).filter(below), right];
}

export const LIGHT: V3 = normalize([-0.5, 0.85, 0.45]);

/** The diamond stamped on each side of the body, 5 cells by 6, as bits from the top left. */
export const GLYPH = [[2], [1, 3], [0, 4], [1, 3], [2], []].reduce(
  (bits, row, j) => row.reduce((acc, i) => acc | (1 << (j * 5 + i)), bits),
  0,
);
/** The glyph's cell in units, and the height of its top above the ground. */
export const GLYPH_CELL = 1 / 6;
export const GLYPH_TOP = 1.3;

/** The box resting in a spot `width` by `height` points: its ground and points a unit. */
export function restOf(width: number, height: number) {
  const unit = Math.min(width / 4.2, height / 3.4);
  return { anchor: [width / 2, height - 0.75 * unit] as const, unit };
}

/**
 * The box at rest on `anchor`, `over` (0 to 1) turning it a little, lifting its lid and growing its
 * light, `press` pushing it down.
 */
export function restingBox(
  anchor: readonly [number, number],
  unit: number,
  over: number,
  press: number,
  accent: Rgb01,
): BoxInput {
  const pose = poseOf({
    yaw: YAW + 0.1 * over,
    tiltX: 0,
    tiltZ: 0,
    lift: 0.05 * over - 0.06 * press,
    lidRise: 0.07 * over,
    lidTilt: -0.05 * over,
    lidSpin: 0,
    sink: 0,
  });
  return {
    anchor,
    unit,
    ...pose,
    tint: accent,
    sink: 0,
    leak: 0.12 + 0.3 * over,
    open: 0,
    lidFade: 0,
    glow: 0.36 + 0.2 * over,
  };
}

/**
 * Where a pull with the box lays out on a stage `width` by `height` points, with the card `cardH`
 * tall and `room` under it: the card's slot, points a unit, and the ground under the box, which
 * stands where the card will be.
 */
export function boxStage(width: number, height: number, cardH: number, room: number) {
  const slot = [width / 2, Math.max(24 + cardH / 2, (height - room) / 2)] as const;
  const size = Math.min(Math.max(Math.min(width, height) * 0.09, 46), 76);
  const ground = [width / 2, Math.min(height - 24, slot[1] + 1.25 * size)] as const;
  return { slot, size, ground };
}

/**
 * The box `t` milliseconds into the pull on `anchor`, turned `yaw` beyond its own, its light going
 * from `accent` to the card's `tint` as it tells: what to draw, where the card leaves it and how
 * wide it is there, and the lip the card is hidden below.
 */
export function pulledBox(
  t: number,
  phase: number,
  anchor: readonly [number, number],
  unit: number,
  yaw: number,
  accent: Rgb01,
  tint: Rgb01,
) {
  const b = boxAt(t, phase);
  const pose = { ...b, yaw: YAW + b.yaw + yaw };
  const drawn = poseOf(pose);
  const standing = poseOf({ ...pose, sink: 0 }).body;
  const tell = tellAt(t, 'box');
  const input: BoxInput = {
    anchor,
    unit,
    body: drawn.body,
    lid: drawn.lid,
    leak: b.leak,
    tint: [
      lerp(accent[0], tint[0], tell),
      lerp(accent[1], tint[1], tell),
      lerp(accent[2], tint[2], tell),
    ],
    open: b.open,
    sink: b.sink,
    lidFade: b.lidFade,
    glow: 0.2 + 0.32 * b.leak,
  };
  return {
    input,
    mouth: project(place(standing, [0, SEAM, 0]), anchor, unit),
    span: 1.7 * unit,
    lip: rimOf(standing, anchor, unit),
  };
}
