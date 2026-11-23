import { cameras } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';

export const KINDS = ['standard', 'reversed'] as const;
export type Kind = (typeof KINDS)[number];

/** One 32-bit float and its bits, as a `depth32float` target stores a depth. */
const float = new Float32Array(1);
const bits = new Uint32Array(float.buffer);
const bitsOf = (value: number) => ((float[0] = value), bits[0]);
const floatOf = (index: number) => ((bits[0] = index), float[0]);

/** How many values a 32-bit float can hold from 0 to 1, less the 1. */
const VALUES = bitsOf(1);

/** The float next to `value` on the side away from the eye, or towards it at the far plane. */
const beside = (kind: Kind, value: number) => {
  const index = bitsOf(value);
  const away = kind === 'standard' ? 1 : -1;
  return floatOf(index + (index + away < 0 || index + away > VALUES ? -away : away));
};

/** Halvings that take two doubles a far plane apart down to neighbours. */
const HALVINGS = 80;

/** Where the eye sits on the z axis, looking down it, so a point's distance is its depth in view. */
const EYE = 1;

/**
 * Two orbit cameras over the same planes, one with `reversed: true`, and the depth each projects a
 * distance to, as a `depth32float` target rounds it.
 */
export function createProgram({ near = 0.1, far = 1000 } = {}) {
  const { root } = recordingRoot();
  const camera = (reversed: boolean) =>
    cameras.orbit(root, {
      size: { width: 16, height: 9 },
      target: [0, 0, 0],
      distance: EYE,
      distanceRange: [EYE, EYE],
      yaw: 0,
      pitch: 0,
      near,
      far,
      reversed,
    });
  const views = { standard: camera(false), reversed: camera(true) };
  const projected = [0, 0, 0];

  /** The depth the camera projects a point `distance` ahead of the eye to, before rounding. */
  const depth = (kind: Kind, distance: number) => {
    if (!views[kind].project([0, 0, EYE - distance], projected)) {
      throw new Error(`Nothing projects from ${distance} ahead of the eye.`);
    }
    return projected[2];
  };

  /** The distance between the planes that projects to `value`, or the plane past which it lies. */
  const distance = (kind: Kind, value: number) => {
    const { near: from, far: to } = views[kind].clip;
    const rising = depth(kind, to) > depth(kind, from);
    let low = from;
    let high = to;
    for (let i = 0; i < HALVINGS; i++) {
      const middle = (low + high) / 2;
      if (middle === low || middle === high) break;
      if (depth(kind, middle) < value === rising) low = middle;
      else high = middle;
    }
    return (low + high) / 2;
  };

  /** The depth the target holds for a surface at `distance`. */
  const stored = (kind: Kind, at: number) => Math.fround(depth(kind, at));

  return {
    get near() {
      return views.standard.clip.near;
    },

    get far() {
      return views.standard.clip.far;
    },

    /** Moves both cameras' planes. */
    clip(nearer: number, farther: number) {
      for (const kind of KINDS) {
        views[kind].clip.near = nearer;
        views[kind].clip.far = farther;
        views[kind].jump();
      }
    },

    depth,
    stored,
    distance,

    /** Where every `every`th value a float can hold from 0 to 1 falls, nearest the eye first. */
    ticks(kind: Kind, every: number) {
      const falls: number[] = [];
      for (let index = 0; index <= VALUES; index += every) {
        falls.push(distance(kind, floatOf(index)));
      }
      return kind === 'standard' ? falls : falls.toReversed();
    },

    /** How far apart in the world the value stored at `at` and the next one away from the eye lie. */
    step(kind: Kind, at: number) {
      const value = stored(kind, at);
      return Math.abs(distance(kind, beside(kind, value)) - distance(kind, value));
    },

    /** The values a float can hold whose distances lie from `from` to `to`, at most `most`. */
    values(kind: Kind, from: number, to: number, most: number) {
      const falls: number[] = [];
      const first = bitsOf(stored(kind, from));
      const away = kind === 'standard' ? 1 : -1;
      for (let i = 0; i <= most; i++) {
        const index = first + i * away;
        if (index < 0 || index > VALUES) break;
        const at = distance(kind, floatOf(index));
        if (at > to) break;
        if (at >= from) falls.push(at);
      }
      return falls;
    },

    /** Whether surfaces at `at` and `gap` behind it store different depths. */
    apart: (kind: Kind, at: number, gap: number) => stored(kind, at) !== stored(kind, at + gap),
  };
}

export type Program = ReturnType<typeof createProgram>;
