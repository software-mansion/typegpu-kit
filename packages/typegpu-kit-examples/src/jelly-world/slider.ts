import { sdBezier } from '@typegpu/sdf';
import { baker } from '@typegpu/kit';
import { d, std, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { vec3n } from 'wgpu-matrix';
import type { V3 } from './tray.ts';

const offset: V3 = [0, 0, 0];

/** How far a ray from `origin` along unit `direction` passes from `point`. */
export const missOf = (origin: V3, direction: V3, point: V3) => {
  vec3n.sub(point, origin, offset);
  return vec3n.length(vec3n.addScaled(offset, direction, -vec3n.dot(offset, direction), offset));
};

/** Points along the jelly's spine, from the pinned anchor to the dragged end. */
export const POINTS = 17;
/** The anchor and the rest position of the dragged end, in world units on the slider plane. */
const START = [-1, 0] as const;
const END_X = 0.9;
/** Lifts the spine off the floor. The dragged end rides a little higher than the rest. */
const LIFT = -0.03;
const BASE = START[1] + LIFT;
const END_Y = 0.08 + LIFT;
const LENGTH = END_X - START[0];
const REST = LENGTH / (POINTS - 1);

/** Where the dragged end may go, in world x. */
export const RANGE = [-0.33, END_X] as const;
/** Half the jelly's thickness around its spine. */
export const SKIN = 0.024;

/** Physics step in seconds and the substeps and constraint passes within it. */
export const STEP = 1 / 120;
const SUBSTEPS = 3;
const ITERATIONS = 16;
/** Velocity lost per substep. */
const DAMPING = 0.01;
/** Stiffness of the spine against stretching, per constraint pass. */
const STRETCHING = 0.1;
/** Stiffness of the spine against bending, strongest at its ends. */
const BENDING = { strength: 0.1, exponent: 1.2 };
/** Upward push in the middle of a compressed spine, which makes it arch. */
const ARCH = { strength: 2, deadzone: 0.01 };
/** Downward pull on the spine per rider. */
const LOAD = 0.6;
/** Pulls the points next to each end towards the floor. */
const FLAT_ENDS = 0.05;

/**
 * The box the jelly lies in, in world units, with half its depth across the slider plane in `z`.
 * The baked field covers its x and y, each texel holding the distance to the spine, the progress
 * along it, and the distance gradient.
 */
export const BOX = {
  left: START[0] - LENGTH * 0.02,
  right: END_X + LENGTH * 0.1,
  bottom: -0.3,
  top: 0.65,
  z: 0.25,
};
const BAKE = { width: 256, height: 128 };
/** Finite-difference step for the baked gradient, in world units. */
const GRADIENT_STEP = 0.03;

/** Steps the spine takes before the first frame, so it starts at rest. */
const SETTLE = 120;

/**
 * Lays the spine out at rest, with its dragged end at `end` in world x, as an arc of its length
 * bulging up between its ends, so it starts untangled.
 */
function layArc(position: Float32Array, previous: Float32Array, end: number) {
  const run = [end - START[0], END_Y - BASE] as const;
  const span = Math.hypot(run[0], run[1]);
  // The arc's half-angle, where sin(half) / half = span / LENGTH
  let [low, high] = [0, Math.PI];
  for (let i = 0; i < 40; i++) {
    const half = (low + high) / 2;
    if (Math.sin(half) / half > span / LENGTH) low = half;
    else high = half;
  }
  const half = (low + high) / 2;
  const turn = Math.atan2(run[1], run[0]);
  const radius = LENGTH / (2 * half);
  const centre = [
    START[0] + run[0] / 2 + Math.sin(turn) * radius * Math.cos(half),
    BASE + run[1] / 2 - Math.cos(turn) * radius * Math.cos(half),
  ];
  for (let i = 0; i < POINTS; i++) {
    const angle = turn + Math.PI / 2 + half - 2 * half * (i / (POINTS - 1));
    position[i * 2] = previous[i * 2] = centre[0] + radius * Math.cos(angle);
    position[i * 2 + 1] = previous[i * 2 + 1] = Math.max(
      BASE,
      centre[1] + radius * Math.sin(angle),
    );
  }
}

/**
 * A Verlet spine on the CPU, drawn as quadratic Bézier segments that a kernel bakes into a
 * distance field the march samples. Its dragged end starts at `start` in world x.
 */
export function createSlider(root: TgpuRoot, options: { start?: number } = {}) {
  const { start = END_X } = options;
  const position = new Float32Array(POINTS * 2);
  const previous = new Float32Array(POINTS * 2);
  const normal = new Float32Array(POINTS * 2);
  const control = new Float32Array((POINTS - 1) * 2);
  const cap = new Float32Array(4);
  const loads = new Float32Array(POINTS);
  const point: V3 = [0, 0, 0];
  let target = std.clamp(start, RANGE[0], RANGE[1]);
  layArc(position, previous, target);

  const points = root.createReadonly(d.arrayOf(d.vec2f, POINTS));
  /** Where each point was when `track` was last called, and how far back that is from it now. */ const drawn =
    new Float32Array(POINTS * 2);
  const back = new Float32Array(POINTS * 2);
  const moved = root.createReadonly(d.arrayOf(d.vec2f, POINTS));
  const controls = root.createReadonly(d.arrayOf(d.vec2f, POINTS - 1));
  /** The last segment, as its start in xy and its end in zw, for the rounded cap. */
  const end = root.createUniform(d.vec4f);
  const texture = root
    .createTexture({ size: [BAKE.width, BAKE.height], format: 'rgba16float' })
    .$usage('sampled', 'storage');

  const baking = baker.create(root, texture, {
    texel: (uv) => {
      'use gpu';
      const at = d.vec2f(
        BOX.left + uv.x * (BOX.right - BOX.left),
        BOX.top - uv.y * (BOX.top - BOX.bottom),
      );
      const dx = d.vec2f(GRADIENT_STEP, 0);
      const dy = d.vec2f(0, GRADIENT_STEP);

      let nearest = d.f32(1e10);
      let along = d.f32(0);
      let right = d.f32(1e10);
      let left = d.f32(1e10);
      let up = d.f32(1e10);
      let down = d.f32(1e10);
      for (let i = 0; i < POINTS - 1; i++) {
        const a = points.$[i];
        const b = points.$[i + 1];
        const c = controls.$[i];
        const distance = sdBezier(at, a, c, b);
        if (distance < nearest) {
          nearest = distance;
          // Progress within the segment, projected onto its chord
          const chord = b - a;
          const t = std.saturate(std.dot(at - a, chord) / std.max(std.dot(chord, chord), 1e-12));
          along = (d.f32(i) + t) / (POINTS - 1);
        }
        right = std.min(right, sdBezier(at + dx, a, c, b));
        left = std.min(left, sdBezier(at - dx, a, c, b));
        up = std.min(up, sdBezier(at + dy, a, c, b));
        down = std.min(down, sdBezier(at - dy, a, c, b));
      }

      const gradient = d.vec2f(right - left, up - down) / (2 * GRADIENT_STEP);
      return d.vec4f(nearest, along, gradient);
    },
  });

  const pin = () => {
    position[0] = previous[0] = START[0];
    position[1] = previous[1] = BASE;
    const last = (POINTS - 1) * 2;
    position[last] = previous[last] = target;
    position[last + 1] = previous[last + 1] = END_Y;
  };

  const project = (i: number, j: number, rest: number, stiffness: number) => {
    const dx = position[j * 2] - position[i * 2];
    const dy = position[j * 2 + 1] - position[i * 2 + 1];
    const length = Math.hypot(dx, dy);
    if (length < 1e-8) return;
    // Pinned ends have no inverse mass and take none of the correction
    const wi = i === 0 || i === POINTS - 1 ? 0 : 1;
    const wj = j === 0 || j === POINTS - 1 ? 0 : 1;
    if (wi + wj === 0) return;
    const scale = ((length - rest) / length) * stiffness;
    const ci = (wi / (wi + wj)) * scale;
    const cj = (wj / (wi + wj)) * scale;
    position[i * 2] += dx * ci;
    position[i * 2 + 1] += dy * ci;
    position[j * 2] -= dx * cj;
    position[j * 2 + 1] -= dy * cj;
  };

  const substep = (h: number) => {
    const compression = Math.max(0, 1 - Math.abs(target - START[0]) / LENGTH);
    for (let i = 1; i < POINTS - 1; i++) {
      const t = i / (POINTS - 1);
      const edge = ARCH.deadzone;
      const window = std.smoothstep(edge, 1 - edge, t) * std.smoothstep(edge, 1 - edge, 1 - t);
      const arch = ARCH.strength * Math.sin(Math.PI * t) * window * compression;
      const x = position[i * 2];
      const y = position[i * 2 + 1];
      position[i * 2] = x + (x - previous[i * 2]) * (1 - DAMPING);
      position[i * 2 + 1] = Math.max(
        BASE,
        y + (y - previous[i * 2 + 1]) * (1 - DAMPING) + (arch - loads[i] * LOAD) * h * h,
      );
      previous[i * 2] = x;
      previous[i * 2 + 1] = y;
    }
    pin();

    for (let it = 0; it < ITERATIONS; it++) {
      for (let i = 0; i < POINTS - 1; i++) project(i, i + 1, REST, STRETCHING);
      for (let i = 1; i < POINTS - 1; i++) {
        const fromMiddle = Math.abs(i / (POINTS - 1) - 0.5) * 2;
        const k = BENDING.strength * (0.05 + 0.95 * fromMiddle ** BENDING.exponent);
        project(i - 1, i + 1, 2 * REST, k);
      }
      for (const i of [1, POINTS - 2]) {
        position[i * 2 + 1] += (BASE - position[i * 2 + 1]) * FLAT_ENDS;
      }
      pin();
    }
  };

  /** Sets each segment's control point where the tangents at its ends meet. */
  const shape = () => {
    for (let i = 0; i < POINTS; i++) {
      const a = Math.max(0, i - 1);
      const b = Math.min(POINTS - 1, i + 1);
      const dx = position[b * 2] - position[a * 2];
      const dy = position[b * 2 + 1] - position[a * 2 + 1];
      const length = Math.hypot(dx, dy);
      if (length < 1e-6) {
        normal[i * 2] = i > 0 ? normal[i * 2 - 2] : 0;
        normal[i * 2 + 1] = i > 0 ? normal[i * 2 - 1] : 1;
        continue;
      }
      normal[i * 2] = -dy / length;
      normal[i * 2 + 1] = dx / length;
    }

    for (let i = 0; i < POINTS - 1; i++) {
      const ax = position[i * 2];
      const ay = position[i * 2 + 1];
      const bx = position[i * 2 + 2];
      const by = position[i * 2 + 3];
      control[i * 2] = (ax + bx) * 0.5;
      control[i * 2 + 1] = (ay + by) * 0.5;
      if (i === 0 || i === POINTS - 2) continue;

      const na = [normal[i * 2], normal[i * 2 + 1]] as const;
      const nb = [normal[i * 2 + 2], normal[i * 2 + 3]] as const;
      // Nearly parallel tangents meet far away or nowhere: meet them halfway along the chord
      // instead, which keeps the bends at the points smooth
      const cross = na[1] * -nb[0] - -na[0] * nb[1];
      if (na[0] * nb[0] + na[1] * nb[1] > 0.99 || Math.abs(cross) <= 1e-6) {
        const half = Math.hypot(bx - ax, by - ay) * 0.5;
        control[i * 2] = (ax + na[1] * half + bx - nb[1] * half) * 0.5;
        control[i * 2 + 1] = (ay - na[0] * half + by + nb[0] * half) * 0.5;
        continue;
      }
      const t = ((bx - ax) * -nb[0] - (by - ay) * nb[1]) / cross;
      control[i * 2] = ax + t * na[1];
      control[i * 2 + 1] = ay - t * na[0];
    }

    const last = (POINTS - 2) * 2;
    cap.set(position.subarray(last, last + 4));
  };

  const step = () => {
    for (let s = 0; s < SUBSTEPS; s++) substep(STEP / SUBSTEPS);
    shape();
  };
  for (let i = 0; i < SETTLE; i++) step();
  drawn.set(position);

  return {
    /** The baked field, sampled by the march. */
    field: texture.createView(),
    end,
    /** How far back each point of the spine was when the last frame was drawn, in x and y. */
    moved,
    /** Writes `moved` for the frame about to be drawn. */
    track() {
      for (let i = 0; i < POINTS * 2; i++) back[i] = drawn[i] - position[i];
      drawn.set(position);
      moved.write(back);
    },
    /** Sets where the dragged end goes, clamped to `RANGE`. */
    drag(x: number) {
      target = std.clamp(x, RANGE[0], RANGE[1]);
    },
    get target() {
      return target;
    },
    /** Whether a ray passes within `reach` of a point of the spine. */
    passes(origin: V3, direction: V3, reach: number) {
      for (let i = 0; i < POINTS; i++) {
        point[0] = position[i * 2];
        point[1] = position[i * 2 + 1];
        if (missOf(origin, direction, point) < reach) return true;
      }
      return false;
    },
    /** The spine's points in x and y. */
    spine: position,
    /** Downward pull on each spine point, such as riders' weight. */
    loads,
    step,
    /** Uploads the spine and bakes its field. */
    bake(pass: TgpuComputePass) {
      points.write(position);
      controls.write(control);
      end.write(cap);
      baking.record(pass);
    },
    initAsync: () => baking.initAsync(),
    destroy() {
      texture.destroy();
      points.buffer.destroy();
      moved.buffer.destroy();
      controls.buffer.destroy();
    },
  };
}
