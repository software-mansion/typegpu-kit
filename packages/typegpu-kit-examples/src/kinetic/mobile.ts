// A hanging mobile of four wire arms, each turning on its wire as a slow torsion pendulum, and the
// CPU placement of its wires, discs and ball

import { materials } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { d, std } from 'typegpu';
import { quat, vec3 } from 'wgpu-matrix';

const paint = (r: number, g: number, b: number) => srgbToLinear(d.vec3f(r, g, b));

/** Calder's colours in lacquer, and the polished, brushed and glazed pieces that mirror the sky. */
export const LOOKS = {
  red: materials.glossy(paint(0.8, 0.09, 0.05), 0.3),
  yellow: materials.glossy(paint(0.97, 0.74, 0.08), 0.3),
  blue: materials.glossy(paint(0.1, 0.3, 0.72), 0.3),
  black: materials.glossy(paint(0.04, 0.04, 0.045), 0.4),
  white: materials.glossy(paint(0.9, 0.89, 0.85), 0.3),
  mirror: materials.metal(d.vec3f(0.95, 0.95, 0.96), 0.04),
  brushed: materials.metal(d.vec3f(0.9, 0.91, 0.92), 0.3),
  brass: materials.metal(d.vec3f(0.95, 0.78, 0.42), 0.1),
  copper: materials.metal(d.vec3f(0.96, 0.64, 0.54), 0.14),
  glaze: materials.glossy(paint(0.93, 0.92, 0.89), 0.15),
  wire: materials.glossy(paint(0.05, 0.05, 0.05), 0.45),
};

export type Look = (typeof LOOKS)[keyof typeof LOOKS];

/** Radius of the wire, in metres. */
export const WIRE = 0.007;
/** How far each face of a disc domes out, as a share of its radius. */
const DOME = 0.14;

/**
 * A disc of radius 1 facing +y, each face a shallow dome, so a mirror disc shows a wide piece of
 * the sky. `v` runs from the front's centre over the rim to the back's.
 */
export const disc = (around: number, rings: number) =>
  meshes.parametric(
    {
      at: (u, v) => {
        'use gpu';
        const r = 1 - std.abs(v * 2 - 1);
        const side = std.sign(0.5 - v);
        const phi = u * Math.PI * 2;
        return d.vec3f(r * std.cos(phi), side * DOME * (1 - r * r), r * std.sin(phi));
      },
      normalAt: (u, v) => {
        'use gpu';
        const r = 1 - std.abs(v * 2 - 1);
        const side = std.sign(0.5 - v);
        const phi = u * Math.PI * 2;
        const slope = 2 * DOME * r;
        return std.normalize(d.vec3f(slope * std.cos(phi), side, slope * std.sin(phi)));
      },
    },
    { cols: around, rows: rings },
  );

/** A disc or a ball hung on a stem from a point along its arm. */
interface Piece {
  /** Where along the arm, from its pivot, positive to the right. */
  readonly at: number;
  readonly stem: number;
  readonly ball?: boolean;
  readonly radius: number;
  readonly look: Look;
  /** How far the disc's face turns from facing across the arm, in radians. */
  readonly turn?: number;
}

/** A wire arm hung by its pivot, turning as `base + swing·sin(pace·t + phase)` and a slower echo. */
interface Arm {
  readonly left: number;
  readonly right: number;
  /** Wire length above the pivot. */
  readonly drop: number;
  readonly base: number;
  readonly swing: number;
  readonly pace: number;
  readonly phase: number;
  readonly pieces: readonly Piece[];
  /** The arm hung from this arm's right end. */
  readonly child?: Arm;
}

// Each arm carries its pieces to the left of its pivot and the next arm from its right end. The
// coloured arms hang each below the pieces of the one above. The polished pieces stay outside the
// circle the lowest arm sweeps, so those two arms turn through each other at one height.

/** The mirror and brushed discs, nearest the eye, with a glazed ball hanging between. */
const LOWEST: Arm = {
  left: 0.6,
  right: 0.6,
  drop: 0.4,
  base: 0.5,
  swing: 0.7,
  pace: -0.31,
  phase: 0.4,
  pieces: [
    { at: -0.6, stem: 0.12, radius: 0.36, look: LOOKS.mirror, turn: 0.15 },
    { at: 0.6, stem: 0.12, radius: 0.34, look: LOOKS.brushed, turn: -0.15 },
    { at: 0, stem: 0.5, radius: 0.09, ball: true, look: LOOKS.glaze },
  ],
};

/** Brass and copper, outside the reach of the arm below. */
const LOW: Arm = {
  left: 1.83,
  right: 0.55,
  drop: 0.75,
  base: -0.31,
  swing: 0.8,
  pace: 0.26,
  phase: 1.1,
  pieces: [
    { at: -0.75, stem: 0.1, radius: 0.28, look: LOOKS.brass, turn: 0.5 },
    { at: -1.37, stem: 0.1, radius: 0.26, look: LOOKS.copper, turn: -0.4 },
    { at: -1.83, stem: 0.1, radius: 0.1, look: LOOKS.yellow, turn: 0.3 },
  ],
  child: LOWEST,
};

const MIDDLE: Arm = {
  left: 1.3,
  right: 0.7,
  drop: 0.8,
  base: -0.16,
  swing: 0.6,
  pace: -0.19,
  phase: 2.3,
  pieces: [
    { at: -1.3, stem: 0.1, radius: 0.24, look: LOOKS.blue, turn: -0.2 },
    { at: -0.8, stem: 0.1, radius: 0.16, look: LOOKS.white, turn: 0.4 },
    { at: -0.4, stem: 0.1, radius: 0.1, look: LOOKS.black, turn: 0.1 },
  ],
  child: LOW,
};

/** The top arm, hung from the ceiling, out of view. */
export const TOP: Arm = {
  left: 1.9,
  right: 0.8,
  drop: 2.2,
  base: 0.35,
  swing: 0.4,
  pace: 0.13,
  phase: 0.2,
  pieces: [
    { at: -1.9, stem: 0.1, radius: 0.3, look: LOOKS.red, turn: 0.2 },
    { at: -1.35, stem: 0.1, radius: 0.12, look: LOOKS.white, turn: -0.3 },
    { at: -0.95, stem: 0.1, radius: 0.18, look: LOOKS.yellow, turn: -0.5 },
    { at: -0.45, stem: 0.15, radius: 0.1, look: LOOKS.black, turn: 0.3 },
  ],
  child: MIDDLE,
};

/** The pivot of the top arm. */
export const HUNG = [0, 3.2, 0] as const;

type Triple = [number, number, number];
type Quat = [number, number, number, number];

/** What draws a piece: a disc or a ball. */
export type Shape = 'disc' | 'ball';

/** Where a piece is, as the instances take it. */
export interface Placed {
  readonly position: Triple;
  readonly rotation: Quat;
  readonly scale: Triple;
}

export interface Part {
  readonly shape: Shape;
  readonly look: Look;
  readonly placed: Placed;
}

/**
 * The pieces, and the two ends of each wire as the lines take them, x, y and z of each: the wires
 * that hang, then the arms' rods, so a rod drawn over the round ends that meet it hides them.
 */
export interface Mobile {
  readonly parts: Part[];
  readonly wires: Float32Array[];
}

const arms = (arm: Arm): Arm[] => [arm, ...(arm.child ? arms(arm.child) : [])];

/** The mobile's pieces and wires, in the order `place` writes them. */
export function mobileOf(top: Arm): Mobile {
  const all = arms(top);
  return {
    parts: all.flatMap((arm) =>
      arm.pieces.map((piece) => ({
        shape: piece.ball ? 'ball' : 'disc',
        look: piece.look,
        placed: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
      })),
    ),
    wires: [...all.flatMap((arm) => [arm, ...arm.pieces]), ...all].map(() => new Float32Array(6)),
  };
}

const UP: Triple = [0, 1, 0];
const FACING = quat.fromAxisAngle([1, 0, 0], Math.PI / 2);

/** Scratch for `place`, which allocates nothing. */
const above: Triple = [0, 0, 0];
const pivot: Triple = [0, 0, 0];
const across: Triple = [0, 0, 0];
const end: Triple = [0, 0, 0];

/** Writes the point `at` along the current arm into `out`. */
const along = (at: number, out: Triple) =>
  vec3.set(pivot[0] + across[0] * at, pivot[1], pivot[2] + across[2] * at, out);

/** Writes a wire from `from` to `to`. */
const wire = (ends: Float32Array, from: Triple, to: Triple) => {
  ends.set(from);
  ends.set(to, 3);
};

/** Writes a wire `length` long that hangs from `from`, and leaves its lower end in `end`. */
const hang = (ends: Float32Array, from: Triple, length: number) => {
  vec3.set(from[0], from[1] - length, from[2], end);
  wire(ends, from, end);
};

/**
 * Writes every piece and wire of the mobile at `time` seconds. Each arm turns on its wire about
 * the vertical, and carries what hangs from it.
 */
export function place(top: Arm, { parts, wires }: Mobile, time: number) {
  let part = 0;
  let strung = 0;
  let rod = wires.length;
  for (let arm: Arm | undefined = top; arm; arm = arm.child) rod--;
  vec3.set(HUNG[0], HUNG[1] + top.drop, HUNG[2], above);

  for (let arm: Arm | undefined = top; arm; arm = arm.child) {
    hang(wires[strung++], above, arm.drop);
    vec3.copy(end, pivot);
    const angle =
      arm.base +
      arm.swing * Math.sin(arm.pace * time + arm.phase) +
      arm.swing * 0.35 * Math.sin(arm.pace * 0.61 * time + arm.phase * 2);
    vec3.set(Math.cos(angle), 0, -Math.sin(angle), across);

    along(-arm.left, above);
    along(arm.right, end);
    wire(wires[rod++], above, end);

    for (let i = 0; i < arm.pieces.length; i++) {
      const piece = arm.pieces[i];
      along(piece.at, above);
      hang(wires[strung++], above, piece.stem);
      const { position, rotation, scale } = parts[part++].placed;
      const { radius } = piece;
      vec3.set(end[0], end[1] - radius, end[2], position);
      vec3.set(radius, radius, radius, scale);
      if (piece.ball) {
        quat.identity(rotation);
        continue;
      }
      quat.fromAxisAngle(UP, angle + (piece.turn ?? 0), rotation);
      quat.multiply(rotation, FACING, rotation);
    }
    along(arm.right, above);
  }
}
