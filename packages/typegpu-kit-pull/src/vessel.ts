// What the box and the foil pack share: frames placing their parts on the ground, and the
// orthographic camera looking down at it.

export type V3 = readonly [number, number, number];

/** A rotation by its columns, then a translation: local to world. */
export interface Frame {
  readonly x: V3;
  readonly y: V3;
  readonly z: V3;
  readonly t: V3;
}

type Turn = readonly [V3, V3, V3];

export const rotY = (a: number): Turn => [
  [Math.cos(a), 0, -Math.sin(a)],
  [0, 1, 0],
  [Math.sin(a), 0, Math.cos(a)],
];
export const rotX = (a: number): Turn => [
  [1, 0, 0],
  [0, Math.cos(a), Math.sin(a)],
  [0, -Math.sin(a), Math.cos(a)],
];
export const rotZ = (a: number): Turn => [
  [Math.cos(a), Math.sin(a), 0],
  [-Math.sin(a), Math.cos(a), 0],
  [0, 0, 1],
];
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const normalize = (v: V3): V3 => {
  const n = Math.hypot(...v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};
/** `m` times `v`, with `m` given by its columns. */
export const apply = (m: Turn, v: V3): V3 => [
  m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
  m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
  m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
];
export const times = (a: Turn, b: Turn): Turn => [apply(a, b[0]), apply(a, b[1]), apply(a, b[2])];
/** Turned by `yaw` about the vertical, then `pitch` about x, then `roll` about z. */
export const turnOf = (yaw: number, pitch: number, roll: number) =>
  times(times(rotY(yaw), rotX(pitch)), rotZ(roll));
export const frameOf = (turn: Turn, t: V3): Frame => ({ x: turn[0], y: turn[1], z: turn[2], t });

/** A world point of `frame`'s local `p`. */
export const place = (frame: Frame, p: V3): V3 =>
  add(apply([frame.x, frame.y, frame.z], p), frame.t);

/** Where a world point lands on the canvas, in CSS pixels, seen from `elevation` radians up. */
export const projectFrom =
  (elevation: number) =>
  (p: V3, anchor: readonly [number, number], unit: number): readonly [number, number] => [
    anchor[0] + p[0] * unit,
    anchor[1] - (p[1] * Math.cos(elevation) - p[2] * Math.sin(elevation)) * unit,
  ];
