import { ao } from '@typegpu/kit';
import { std, type d } from 'typegpu';
import { clamp } from '../home/geometry.ts';
import { seeded } from './svg.ts';

export type Settings = d.Infer<typeof ao.Settings>;

/** A point of the cut: across the screen and up from the ground, in world units. */
export type Point = readonly [x: number, height: number];

/** One depth read of the walk. */
export interface Step {
  /** The pixel it read, which can lie off the screen, where the read counts for nothing. */
  readonly pixel: number;
  /** Where the depth there puts the surface. */
  readonly at: Point;
  /** How much the read counts, from 1 near the pixel to 0 at the end of the reach. */
  readonly weight: number;
  /** The cosine of the horizon it gives, eased towards the lowest by `weight`. */
  readonly horizon: number;
}

/** What the walk from one pixel finds on one side. */
export interface Side {
  readonly steps: readonly Step[];
  /** The step that set the horizon, if one rose above the pixel's own surface. */
  readonly highest: Step | undefined;
  /** The horizon's angle from the direction to the eye, in radians, positive to the right. */
  readonly angle: number;
}

export interface Trace {
  readonly pixel: number;
  readonly at: Point;
  readonly left: Side;
  readonly right: Side;
  /** The share of the sky this slice leaves open, before the denoise averages it and raises it to `power`. */
  readonly visibility: number;
}

// The trace's constants in ao.ts, which it keeps to itself
const STEPS = 3;
const REACH = 1.457;
const FALLOFF = 0.615;
const CLOSEST = 1.3;
const HALF_PI = Math.PI / 2;
const GOLDEN = 0.6180339887498949;
/** The least light the denoise leaves. */
const LEAST = 0.03;
/** Eye distances within this share of the distance count as the same surface to the denoise. */
const SAME = 0.03;

/**
 * The eye's height over the ground. The orthographic trace sees the same from any height, and the
 * denoise's test for the same surface scales with it.
 */
const EYE = 2;

/** The width of a pixel on the surface, in world units. */
const TEXEL = 0.025;

/**
 * Ground with a crease, a block and a thinner one a narrow gap away. Walls stand on pixel edges.
 */
const PROFILE: readonly Point[] = [
  [0, 0],
  [0.6, 0],
  [0.8, -0.3],
  [1, 0],
  [1.55, 0],
  [1.55, 0.25],
  [2, 0.25],
  [2, 0],
  [2.25, 0],
  [2.25, 0.25],
  [2.35, 0.25],
  [2.35, 0],
  [3, 0],
];

/** As the picture draws a view-space point `[x, z]`, z away from the eye. */
const drawn = ([x, z]: readonly [number, number]): Point => [x, EYE - z];

/** The step whose horizon rose highest above `low`, if one did. */
const highest = (steps: readonly Step[], low: number) =>
  steps.reduce<Step | undefined>(
    (best, step) => (step.horizon > (best?.horizon ?? low + 1e-6) ? step : best),
    undefined,
  );

/** A side the walk never left, whose horizon is the surface itself. */
const open = (angle: number): Side => ({ steps: [], highest: undefined, angle });

export interface Options {
  readonly profile?: readonly Point[];
  readonly settings?: Partial<Settings>;
}

/**
 * One slice of `ao.create`'s trace on the CPU, over the depth an orthographic eye straight above a
 * cut surface draws: a row of pixels, and the one direction that lies in the cut.
 */
export function createProgram({ profile = PROFILE, settings: given }: Options = {}) {
  /** Patched as `occlusion.settings` is. */
  const settings: Settings = { ...ao.defaults, ...given };
  const pixels = Math.round(profile.at(-1)![0] / TEXEL);

  /** The height of the surface the eye sees at `x`. */
  const heightAt = (x: number) => {
    for (let i = 1; i < profile.length; i++) {
      const [x0, y0] = profile[i - 1];
      const [x1, y1] = profile[i];
      if (x >= x0 && x < x1) return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
    }
    return profile.at(x < profile[0][0] ? 0 : -1)![1];
  };

  /** Each pixel's random offset for its steps, in its row and the rows above and below it. */
  const random = seeded(2016);
  const jitters = Array.from({ length: 3 }, () => Float64Array.from({ length: pixels }, random));

  /** The depth prepass: the distance from the eye at each pixel centre. */
  const depth = Float64Array.from({ length: pixels }, (_, i) => EYE - heightAt((i + 0.5) * TEXEL));

  // In view space a point is [x, z]: x across the screen, z away from the eye. The eye is
  // orthographic, so the direction to it is [0, -1] from every pixel
  const placed = (sample: number): [number, number] => [
    sample * TEXEL,
    depth[clamp(Math.floor(sample), 0, pixels - 1)],
  ];

  /**
   * `fromDepth` on one axis: the normal from the neighbour nearer in depth, as its angle from the
   * direction to the eye, `n` in the trace.
   */
  const normalAt = (pixel: number, center: readonly [number, number]) => {
    const left = placed(pixel - 0.5);
    const right = placed(pixel + 1.5);
    const rightward =
      pixel === 0 ||
      (pixel < pixels - 1 && Math.abs(right[1] - center[1]) < Math.abs(center[1] - left[1]));
    const across = rightward
      ? [right[0] - center[0], right[1] - center[1]]
      : [center[0] - left[0], center[1] - left[1]];
    const length = Math.hypot(across[0], across[1]);
    // The cross product with the screen's other axis, which the cut leaves flat
    const facing = [across[1] / length, -across[0] / length];
    const cosNormal = std.saturate(-facing[1]);
    return { cosNormal, n: Math.sign(facing[0]) * Math.acos(cosNormal) };
  };

  /** `horizonOf`: the cosine of the horizon a sample gives, eased to `low` by XeGTAO's falloff. */
  const horizonOf = (sample: number, center: readonly [number, number], low: number): Step => {
    const at = placed(sample);
    const delta = [at[0] - center[0], at[1] - center[1]];
    const length = Math.hypot(delta[0], delta[1]);
    const reach = settings.radius * REACH;
    const range = FALLOFF * reach;
    const inside = sample >= 0 && sample < pixels;
    const weight = inside ? std.saturate((reach - length) / range) : 0;
    return {
      pixel: Math.floor(sample),
      at: drawn(at),
      weight,
      horizon: std.mix(low, -delta[1] / Math.max(length, 1e-6), weight),
    };
  };

  /**
   * The trace's fragment for `pixel`, with one slice turned into the cut. `jitter` is the random
   * offset of its steps, from 0 to 1, which the trace draws for every pixel.
   */
  const trace = (pixel: number, jitter = 0.5): Trace => {
    const position = pixel + 0.5;
    const center = placed(position);
    const { cosNormal, n } = normalAt(pixel, center);
    const low0 = Math.cos(n + HALF_PI);
    const low1 = Math.cos(n - HALF_PI);

    const reach = settings.radius * REACH;
    const spread = reach / TEXEL;
    const base = { pixel, at: drawn(center) };
    if (spread < CLOSEST) {
      return { ...base, left: open(n - HALF_PI), right: open(n + HALF_PI), visibility: 1 };
    }
    const closest = CLOSEST / spread;

    const right: Step[] = [];
    const left: Step[] = [];
    for (let step = 0; step < STEPS; step++) {
      const shift = std.fract(jitter + step * STEPS * GOLDEN);
      const s = ((step + shift) / STEPS) * ((step + shift) / STEPS) + closest;
      const offset = std.round(spread * s);
      right.push(horizonOf(position + offset, center, low0));
      left.push(horizonOf(position - offset, center, low1));
    }
    const horizon0 = Math.max(low0, ...right.map((step) => step.horizon));
    const horizon1 = Math.max(low1, ...left.map((step) => step.horizon));

    const h0 = n + Math.max(-Math.acos(horizon1) - n, -HALF_PI);
    const h1 = n + Math.min(Math.acos(horizon0) - n, HALF_PI);
    const arc0 = (cosNormal + 2 * h0 * Math.sin(n) - Math.cos(2 * h0 - n)) / 4;
    const arc1 = (cosNormal + 2 * h1 * Math.sin(n) - Math.cos(2 * h1 - n)) / 4;
    // The fade for small screen radii, then the slice. The normal lies in the cut, so its
    // projected length is 1
    const visibility = std.saturate((10 - spread) / 100) * 0.5 + arc0 + arc1;

    return {
      ...base,
      left: { steps: left, highest: highest(left, low1), angle: h0 },
      right: { steps: right, highest: highest(right, low0), angle: h1 },
      visibility,
    };
  };

  return {
    settings,
    profile,
    texel: TEXEL,
    pixels,
    heightAt,
    trace,

    /** How far from the pixel a depth read still counts, in world units: `radius` × 1.457. */
    get reach() {
      return settings.radius * REACH;
    },

    /**
     * The denoise at `pixel`: the 3 × 3 traces around it that lie on its surface, averaged, raised
     * to `power` and never below 0.03. The cut repeats the row above and below, which differ only
     * in the random offset of their steps.
     */
    kept(pixel: number) {
      const middle = depth[pixel];
      let sum = 0;
      let total = 0;
      for (const row of jitters) {
        for (let x = -1; x <= 1; x++) {
          const near = clamp(pixel + x, 0, pixels - 1);
          const weight = std.saturate(2 - Math.abs(depth[near] - middle) / (middle * SAME));
          sum += trace(near, row[near]).visibility * weight;
          total += weight;
        }
      }
      return Math.max(std.pow(std.saturate(sum / total), settings.power), LEAST);
    },
  };
}

export type Program = ReturnType<typeof createProgram>;
