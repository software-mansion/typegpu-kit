import { std } from 'typegpu';
import { POINTS, SKIN } from './slider.ts';
import type { V3 } from './tray.ts';

/** A point along the spine and its tangent and normal, as `sample` writes them. */
const along = { x: 0, y: 0, tx: 1, ty: 0, nx: 0, ny: 1 };
/** The normals of a segment and of those before and after it. */
const normals = { here: [0, 0], before: [0, 0], after: [0, 0] };

/** Writes the unit normal of segment `i`, clamped to the spine's first and last, into `out`. */
const segmentNormal = (spine: Float32Array, i: number, out: number[]) => {
  const k = std.clamp(i, 0, POINTS - 2);
  const dx = spine[k * 2 + 2] - spine[k * 2];
  const dy = spine[k * 2 + 3] - spine[k * 2 + 1];
  const segment = Math.hypot(dx, dy) || 1e-6;
  out[0] = -dy / segment;
  out[1] = dx / segment;
};

/**
 * Point, tangent and normal at arc length `s` along a spine of xy points, written into `along`. The
 * normal blends between the segments' normals at each point, so it turns smoothly along the spine.
 */
const sample = (spine: Float32Array, s: number) => {
  let left = Math.max(s, 0);
  for (let i = 0; i < POINTS - 1; i++) {
    const ax = spine[i * 2];
    const ay = spine[i * 2 + 1];
    const dx = spine[i * 2 + 2] - ax;
    const dy = spine[i * 2 + 3] - ay;
    const segment = Math.hypot(dx, dy) || 1e-6;
    if (left <= segment || i === POINTS - 2) {
      const t = Math.min(left / segment, 1);
      const { here, before, after } = normals;
      segmentNormal(spine, i, here);
      segmentNormal(spine, i - 1, before);
      segmentNormal(spine, i + 1, after);
      const fromX = (here[0] + before[0]) * (1 - t) + (here[0] + after[0]) * t;
      const fromY = (here[1] + before[1]) * (1 - t) + (here[1] + after[1]) * t;
      const n = Math.hypot(fromX, fromY) || 1e-6;
      along.nx = fromX / n;
      along.ny = fromY / n;
      along.x = ax + dx * t;
      along.y = ay + dy * t;
      along.tx = along.ny;
      along.ty = -along.nx;
      return;
    }
    left -= segment;
  }
  along.x = along.y = along.ty = along.nx = 0;
  along.tx = along.ny = 1;
};

export const spineLength = (spine: Float32Array) => {
  let total = 0;
  for (let i = 0; i < POINTS - 1; i++) {
    total += Math.hypot(spine[i * 2 + 2] - spine[i * 2], spine[i * 2 + 3] - spine[i * 2 + 1]);
  }
  return total;
};

/** The spine point nearest a point, as `nearest` finds it. */
export interface Nearest {
  /** Its arc length along the spine. */
  s: number;
  /** How far above the jelly's top the point is. */
  above: number;
}

/** Finds the spine point nearest `x, y`, into `out`, and returns `out`. */
export const nearest = (spine: Float32Array, x: number, y: number, out: Nearest) => {
  let best = Infinity;
  let at = 0;
  let above = 0;
  let travelled = 0;
  for (let i = 0; i < POINTS - 1; i++) {
    const ax = spine[i * 2];
    const ay = spine[i * 2 + 1];
    const dx = spine[i * 2 + 2] - ax;
    const dy = spine[i * 2 + 3] - ay;
    const segment = Math.hypot(dx, dy) || 1e-6;
    const t = std.clamp(((x - ax) * dx + (y - ay) * dy) / (segment * segment), 0, 1);
    const px = ax + dx * t;
    const py = ay + dy * t;
    const gap = Math.hypot(x - px, y - py);
    if (gap < best) {
      best = gap;
      at = travelled + segment * t;
      above = ((x - px) * -dy + (y - py) * dx) / segment;
    }
    travelled += segment;
  }
  out.s = at;
  out.above = above - SKIN;
  return out;
};

/** Where a critter's body sits on the jelly's top, as `seat` writes it. */
export interface Seat {
  readonly surface: V3;
  readonly normal: V3;
  readonly tangent: V3;
}

export const createSeat = (): Seat => ({
  surface: [0, 0, 0],
  normal: [0, 0, 0],
  tangent: [0, 0, 0],
});

/**
 * Writes where a critter's body sits on the jelly's top, which is the spine's left side, at arc
 * length `s` and `z` across into `out`, and returns `out`.
 */
export const seat = (spine: Float32Array, s: number, z: number, out: Seat) => {
  sample(spine, s);
  const { surface, normal, tangent } = out;
  surface[0] = along.x + along.nx * SKIN;
  surface[1] = along.y + along.ny * SKIN;
  surface[2] = z;
  normal[0] = along.nx;
  normal[1] = along.ny;
  normal[2] = 0;
  tangent[0] = along.tx;
  tangent[1] = along.ty;
  tangent[2] = 0;
  return out;
};
