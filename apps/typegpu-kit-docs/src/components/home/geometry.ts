import type { Kind, SheetState, StackView } from './stack.ts';

export const clamp = (x: number, low = 0, high = 1) => Math.min(Math.max(x, low), high);
export const smooth = (x: number) => {
  const t = clamp(x);
  return t * t * (3 - 2 * t);
};
export const easeOut = (x: number) => 1 - (1 - clamp(x)) ** 3;
/** Past the end a little, then back: for the camera settling. */
export const settle = (x: number) => {
  const t = clamp(x);
  const c = 1.4;
  return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2;
};
/**
 * Moves `from`, going at `pace` a second, towards `to` as a critically damped spring does in `dt`,
 * and returns where it is and how fast it goes. Its pace changes smoothly, so a target that moves
 * in uneven steps doesn't jolt it.
 */
export const spring = (from: number, pace: number, to: number, dt: number, rate = 10) => {
  const away = from - to;
  const push = (pace + rate * away) * dt;
  const decay = Math.exp(-rate * dt);
  return [to + (away + push) * decay, (pace - rate * push) * decay] as const;
};
export const mix = (a: number, b: number, t: number) => a + (b - a) * t;

/** Half a sheet's side, in world units. */
export const HALF = 2;
/** A sheet's diagonal, which spans its width on screen at a yaw of 45°. */
export const DIAG = HALF * 2 * Math.SQRT2;
/** CSS pixels the pointer must be inside a sheet to leave the one it is on for it. */
const MARGIN = 5;
export const DEG = Math.PI / 180;

/** A sheet in a stack as the page moves it, before it is placed. */
export interface Stacked {
  kind: Kind;
  /** How far it has come away from the one above, 0 to 1. */
  open: number;
  /** How much room it takes in the stack: 1 for a sheet, less for an empty slot. */
  weight: number;
  /** 1 while its module is in the frame, 0 while its place is an empty socket; between, it dithers in. */
  presence: number;
  hot: number;
  /** How far it is pushed toward the middle while pointed at, in world units. */
  lift: number;
  /** How far it is dragged toward the middle, in world units. */
  pull: number;
  /** Whether its module is out of the frame, as the page last left it. */
  out: boolean;
  keeps: number;
  /** How bright its seat's dashed outline is while it is not all there, from 0 to 1. */
  rim: number;
}

export const stacked = (kind: Kind): Stacked => ({
  kind,
  open: 1,
  weight: 1,
  presence: 1,
  hot: 0,
  lift: 0,
  pull: 0,
  out: false,
  keeps: 1,
  rim: 1,
});

/** How far a sheet drifts as it dithers in or out, in world units. */
const ARRIVE = HALF * 0.3;

/**
 * How `place` moves a sheet off its seat on the stack's axis: by its pull alone, to hit-test it;
 * by its push as well, to label it; and by its drift as it dithers in or out too, to draw it.
 */
export type Placing = 'resting' | 'labelled' | 'drawn';

/**
 * Place sheets top-first from height `top`, with `gap` between open sheets. Push and pull along
 * `toward`, the way to the canvas's middle in world x and z. Writes into `into`, which a caller
 * that places every frame keeps, and returns it.
 */
export function place(
  list: readonly Stacked[],
  gap: number,
  toward: readonly [number, number] = [1, 0],
  placing: Placing = 'drawn',
  top = 0,
  into: SheetState[] = [],
): SheetState[] {
  let y = top;
  for (let i = 0; i < list.length; i++) {
    const sheet = list[i];
    if (i > 0) {
      const room = (list[i - 1].weight + sheet.weight) / 2;
      y -= gap * room * sheet.open + 0.004;
    }
    const presence = smooth(sheet.presence);
    const pushed = placing === 'resting' ? 0 : sheet.lift;
    const drift = placing === 'drawn' ? (1 - presence) * ARRIVE : 0;
    const side = sheet.pull + pushed + drift;
    const placed = (into[i] ??= {
      kind: sheet.kind,
      x: 0,
      y: 0,
      z: 0,
      half: HALF,
      hot: 0,
      shown: 0,
      keeps: 1,
    });
    placed.kind = sheet.kind;
    placed.x = side * toward[0];
    placed.y = y;
    placed.z = side * toward[1];
    placed.half = HALF;
    placed.hot = sheet.hot;
    placed.shown = sheet.kind === 'socket' ? 0 : presence;
    placed.keeps = sheet.keeps;
    placed.rim = sheet.rim;
  }
  into.length = list.length;
  return into;
}

/** The corners of a sheet's top face, in CSS pixels of the canvas. */
export const cornersOf = (view: StackView, sheet: SheetState) =>
  [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ].map(([x, z]) => view.toCanvas([sheet.x + x! * sheet.half, sheet.y, sheet.z + z! * sheet.half]));

/** How far (x, y) is inside a convex quad, in CSS pixels; negative outside. */
const depthIn = (corners: readonly (readonly [number, number])[], x: number, y: number) => {
  let area = 0;
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = corners[i];
    const [bx, by] = corners[(i + 1) % 4];
    area += ax * by - bx * ay;
  }
  const sign = Math.sign(area) || 1;
  let least = Infinity;
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = corners[i];
    const [bx, by] = corners[(i + 1) % 4];
    const length = Math.hypot(bx - ax, by - ay) || 1;
    least = Math.min(least, (sign * ((bx - ax) * (y - ay) - (by - ay) * (x - ax))) / length);
  }
  return least;
};

/** Where (x, y) falls on a sheet's top face, from 0 to 1 across its texture, or undefined off it. */
export const uvOn = (view: StackView, sheet: SheetState, x: number, y: number) => {
  const [c0, c1, , c3] = cornersOf(view, sheet);
  const ax = c1![0] - c0![0];
  const ay = c1![1] - c0![1];
  const bx = c3![0] - c0![0];
  const by = c3![1] - c0![1];
  const det = ax * by - bx * ay;
  if (Math.abs(det) < 1e-6) return undefined;
  const px = x - c0![0];
  const py = y - c0![1];
  const u = (px * by - bx * py) / det;
  const v = (ax * py - px * ay) / det;
  return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? ([u, v] as const) : undefined;
};

/** Return the topmost sheet under the pointer, or -1. Extend `held` by `MARGIN` and inset other sheets by the same amount to prevent edge flicker. */
export function pointedIn(
  view: StackView,
  sheets: readonly SheetState[],
  x: number,
  y: number,
  held = -1,
) {
  for (let i = 0; i < sheets.length; i++) {
    if (depthIn(cornersOf(view, sheets[i]), x, y) > (i === held ? -MARGIN : MARGIN)) return i;
  }
  return -1;
}

/** The heights of `count` sheets `gap` apart, top first. */
export const evenly = (count: number, gap: number) =>
  Array.from({ length: count }, (_, i) => -i * (gap + 0.004));

/** Fit sheets at `heights` inside canvas box (x0, y0, x1, y1), in CSS pixels. Reserve `lift` above the top sheet. */
export function fitted(
  yaw: number,
  pitch: number,
  heights: readonly number[],
  lift: number,
  box: readonly [number, number, number, number],
  width: number,
  height: number,
) {
  const f = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
  const r = [Math.cos(yaw), 0, -Math.sin(yaw)];
  const u = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]];
  let low = [Infinity, Infinity];
  let high = [-Infinity, -Infinity];
  for (const [i, top] of heights.entries()) {
    for (const y of [top + (i === 0 ? lift : 0), top - 0.1]) {
      for (const [x, z] of [
        [-HALF, -HALF],
        [HALF, -HALF],
        [HALF, HALF],
        [-HALF, HALF],
      ] as const) {
        const pu = x * r[0] + y * r[1] + z * r[2];
        const pv = x * u[0] + y * u[1] + z * u[2];
        low = [Math.min(low[0], pu), Math.min(low[1], pv)];
        high = [Math.max(high[0], pu), Math.max(high[1], pv)];
      }
    }
  }
  const [x0, y0, x1, y1] = box;
  const perPixel = Math.max((high[0] - low[0]) / (x1 - x0), (high[1] - low[1]) / (y1 - y0));
  const cu = (low[0] + high[0]) / 2;
  const cv = (low[1] + high[1]) / 2;
  return {
    scale: (perPixel * height) / 2,
    anchor: [((x0 + x1) / 2 - cu / perPixel) / width, ((y0 + y1) / 2 + cv / perPixel) / height] as [
      number,
      number,
    ],
  };
}
