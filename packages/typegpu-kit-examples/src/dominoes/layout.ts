// Lay out the run once on the CPU, with a domino every `SPACING` centimetres along curves. Each
// domino pushes one or two neighbours. One world unit is one centimetre

import { clamp } from 'math';

/** Width, height and thickness in centimetres. */
export const WIDTH = 2.4;
export const HEIGHT = 4.8;
export const THICKNESS = 0.75;
/** Spacing between domino centres along the run, about two thirds of their height. */
export const SPACING = 3.1;

/** Sentinel id for a missing neighbour. */
export const NONE = -1;

export interface Placed {
  /** Foot centre in world coordinates. */
  readonly x: number;
  readonly z: number;
  /** Fall direction in radians around +y, measured from +z. */
  readonly yaw: number;
  /** Distance from the start in dominoes. */
  readonly depth: number;
  /** Id of the preceding domino, or `NONE`. */
  prev: number;
  /** Ids of the next one or two dominoes, or `NONE`. */
  readonly next: [number, number];
  readonly color: number;
  /** Portrait pixel column and row, measured from the top left. */
  readonly cell?: readonly [column: number, row: number];
}

type Point = readonly [x: number, z: number];

/** Maps progress `t` from 0 to 1 to a point on a curve. */
type Curve = (t: number) => Point;

/**
 * Cubic Hermite curve from `from` to `to`, with endpoint tangents `out` and `in`. Tangent
 * lengths control the pull toward each direction.
 */
const hermite =
  (from: Point, out: Point, to: Point, into: Point): Curve =>
  (t) => {
    const t2 = t * t;
    const t3 = t2 * t;
    const a = 2 * t3 - 3 * t2 + 1;
    const b = t3 - 2 * t2 + t;
    const c = -2 * t3 + 3 * t2;
    const d = t3 - t2;
    return [
      a * from[0] + b * out[0] + c * to[0] + d * into[0],
      a * from[1] + b * out[1] + c * to[1] + d * into[1],
    ];
  };

/**
 * Spirals inward around `center` from `radius` to `inner`, decreasing by `pitch` per turn.
 * Starts at `start` radians with direction `turn`, +1 or -1. Angles run from +x toward +z in
 * the xz plane.
 */
const spiral =
  (
    center: Point,
    radius: number,
    inner: number,
    pitch: number,
    start: number,
    turn: number,
  ): Curve =>
  (t) => {
    const angle = (t * (radius - inner) * Math.PI * 2) / pitch;
    const r = radius - (pitch * angle) / (Math.PI * 2);
    const at = start + turn * angle;
    return [center[0] + r * Math.cos(at), center[1] + r * Math.sin(at)];
  };

/** Smoothstep from 0 to 1, clamping `t` to that range. */
const smooth = (t: number) => {
  const c = clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
};

/** Offsets `curve` to its left by `offset(length)` at distance `length`. */
const beside =
  (curve: Curve, length: number, offset: (along: number) => number): Curve =>
  (t) => {
    const e = 1e-4;
    const [x, z] = curve(t);
    const [x1, z1] = curve(Math.min(t + e, 1));
    const [x0, z0] = curve(Math.max(t - e, 0));
    const tx = x1 - x0;
    const tz = z1 - z0;
    const norm = Math.hypot(tx, tz) || 1;
    const side = offset(t * length);
    // Left is relative to the curve direction, viewed from +y
    return [x + (tz / norm) * side, z - (tx / norm) * side];
  };

/** Samples a curve as a polyline of `steps` points and measures its length. */
const sample = (curve: Curve, steps: number) => {
  const points = Array.from({ length: steps + 1 }, (_, i) => curve(i / steps));
  let length = 0;
  for (let i = 1; i < points.length; i++) {
    length += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  }
  return { points, length };
};

/**
 * Spaces dominoes evenly from distance `first` to the curve end, about `SPACING` apart. The
 * last domino stands at the endpoint.
 */
const along = (curve: Curve, first = 0) => {
  const { points, length } = sample(curve, 6000);
  const gaps = Math.max(Math.round((length - first) / SPACING), 1);
  const spacing = (length - first) / gaps;
  const stands: { x: number; z: number; yaw: number }[] = [];
  let walked = 0;
  let next = first;
  for (let i = 1; i < points.length; i++) {
    const [x0, z0] = points[i - 1];
    const [x1, z1] = points[i];
    const step = Math.hypot(x1 - x0, z1 - z0);
    while (walked + step >= next - 1e-3) {
      const t = Math.min((next - walked) / step, 1);
      stands.push({
        x: x0 + (x1 - x0) * t,
        z: z0 + (z1 - z0) * t,
        yaw: Math.atan2(x1 - x0, z1 - z0),
      });
      next += spacing;
    }
    walked += step;
  }
  return stands.slice(0, gaps + 1);
};

/** Palette indices. `PALETTE` in `dominoes.ts` holds the colours. */
const [IVORY, SAFFRON, CORAL, CRIMSON, TEAL, BLUE] = [0, 1, 2, 3, 4, 5];

/** Colour sections repeating along the run's depth: bands, stripes and stepped gradients. */
const SECTIONS: readonly { readonly length: number; readonly at: (i: number) => number }[] = [
  { length: 96, at: (i) => [SAFFRON, CORAL, CRIMSON, CORAL, SAFFRON, IVORY][Math.floor(i / 16)] },
  { length: 80, at: (i) => (Math.floor(i / 10) % 2 === 0 ? BLUE : IVORY) },
  {
    length: 72,
    at: (i) =>
      [TEAL, TEAL, TEAL, BLUE, TEAL, BLUE, BLUE, TEAL, BLUE, BLUE, BLUE, BLUE][Math.floor(i / 6)],
  },
  { length: 60, at: (i) => [IVORY, SAFFRON, IVORY][Math.floor(i / 20)] },
  { length: 120, at: (i) => [CRIMSON, TEAL, CORAL, BLUE, SAFFRON][Math.floor(i / 24)] },
  { length: 80, at: (i) => (Math.floor(i / 10) % 2 === 0 ? CRIMSON : SAFFRON) },
];
const CYCLE = SECTIONS.reduce((sum, section) => sum + section.length, 0);

const colorAt = (depth: number) => {
  let i = depth % CYCLE;
  for (const section of SECTIONS) {
    if (i < section.length) return section.at(i);
    i -= section.length;
  }
  return IVORY;
};

/** Twin spirals, each with two arms branching from one line. */
const SPIRAL = { radius: 84, inner: 22, gap: 9 };
/** How far each of two lines starts from the line they branch from, in centimetres. */
const SPLIT = 1.5;
/** Centimetres over which the spiral arms part from `SPLIT` to half a ring gap each. */
const PARTING = 30;
const CENTERS: readonly Point[] = [
  [-100, -135],
  [100, -135],
];

/**
 * Catmull-Rom spline through `points`. Each interior tangent follows the line between the
 * neighbouring points. Endpoint tangents follow the first and last segments.
 */
const through = (points: readonly Point[]): Curve => {
  const at = (i: number) => points[clamp(i, 0, points.length - 1)];
  const tangent = (i: number): Point => [
    (at(i + 1)[0] - at(i - 1)[0]) / 2,
    (at(i + 1)[1] - at(i - 1)[1]) / 2,
  ];
  const legs = points.length - 1;
  return (t) => {
    const leg = Math.min(Math.floor(t * legs), legs - 1);
    return hermite(at(leg), tangent(leg), at(leg + 1), tangent(leg + 1))(t * legs - leg);
  };
};

/** The lead follows an S-bend across the front before turning toward the first fork. */
const LEAD = through([
  [30, 116],
  [-10, 112],
  [-50, 100],
  [-76, 76],
  [-68, 48],
  [-36, 32],
  [-8, 14],
  [0, -6],
  [0, -10],
]);

/**
 * Orders the lead before two branches to the spirals. Each branch splits into two spiral arms
 * winding inward side by side.
 */
export function layOut(): Placed[] {
  const placed: Placed[] = [];

  /** Adds dominoes after `from`, linking it to the first. Returns the last index. */
  const line = (stands: { x: number; z: number; yaw: number }[], from: number) => {
    let prev = from;
    const depth0 = from === NONE ? 0 : placed[from].depth + 1;
    stands.forEach((stand, k) => {
      const index = placed.length;
      placed.push({
        ...stand,
        depth: depth0 + k,
        prev,
        next: [NONE, NONE],
        color: colorAt(depth0 + k),
      });
      if (prev !== NONE) {
        const pushes = placed[prev].next;
        pushes[pushes[0] === NONE ? 0 : 1] = index;
      }
      prev = index;
    });
    return prev;
  };

  const fork = line(along(LEAD), NONE);
  const { x, z, yaw } = placed[fork];
  const forward: Point = [Math.sin(yaw), Math.cos(yaw)];

  CENTERS.forEach((center, side) => {
    const turn = side === 0 ? 1 : -1;
    const middle = SPIRAL.radius - SPIRAL.gap / 2;
    const entry: Point = [center[0], center[1] + middle];

    const lateral = -turn * SPLIT;
    const start: Point = [
      x + forward[0] * SPACING - forward[1] * lateral,
      z + forward[1] * SPACING + forward[0] * lateral,
    ];
    // Leave the fork straight ahead and arrive heading around the spiral
    const reach = Math.hypot(entry[0] - start[0], entry[1] - start[1]);
    const out: Point = [forward[0] * reach * 0.9, forward[1] * reach * 0.9];
    const into: Point = [-turn * reach * 1.4, 0];
    const split = line(along(hermite(start, out, entry, into)), fork);

    const curve = spiral(center, middle, SPIRAL.inner, SPIRAL.gap * 2, Math.PI / 2, turn);
    const length = sample(curve, 4000).length;
    for (const arm of [1, -1]) {
      const offset = (s: number) => arm * (SPLIT + (SPIRAL.gap / 2 - SPLIT) * smooth(s / PARTING));
      line(along(beside(curve, length, offset), SPACING), split);
    }
  });

  return placed;
}

/**
 * Keeps the first `count` dominoes in wave order, shortening the arms. Removes links to omitted
 * dominoes so each line ends by falling flat.
 */
export function firstOf(run: readonly Placed[], count: number): Placed[] {
  // Sort in place because Hermes has no `toSorted`
  const order = run.map((_, i) => i);
  order.sort((a, b) => run[a].depth - run[b].depth || a - b);
  const kept = order.slice(0, count);
  kept.sort((a, b) => a - b);
  const index = new Map(kept.map((old, i) => [old, i]));
  const at = (old: number) => (old === NONE ? NONE : (index.get(old) ?? NONE));
  return kept.map((old) => {
    const each = run[old];
    const [a, b] = [at(each.next[0]), at(each.next[1])];
    return { ...each, prev: at(each.prev), next: a === NONE ? [b, NONE] : [a, b] };
  });
}
