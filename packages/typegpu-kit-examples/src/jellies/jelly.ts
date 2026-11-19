// A jelly's parts: its state, the strands under its bell and the bell's shape

import { d, std } from 'typegpu';

/** A moon jelly, a shallow saucer with a fine fringe and four gonads, or a taller sea nettle. */
export const MOON = 0;
export const NETTLE = 1;

/** The most jellies the tank holds. */
export const MAX = 32;
/** Tentacles round each margin, and oral arms under each bell. */
export const TENTACLES = 40;
export const ARMS = 4;
export const STRANDS = TENTACLES + ARMS;
/** Points along each strand, its root on the bell. */
export const POINTS = 24;

export const Jelly = d.struct({
  /** The bell's apex, in world space. */
  position: d.vec3f,
  /** Pulses so far; the fraction is where the bell is in its beat. */
  phase: d.f32,
  velocity: d.vec3f,
  /** The bell's radius, relaxed. */
  size: d.f32,
  /** The way the bell faces, a unit vector out of its apex. */
  axis: d.vec3f,
  /** How far the bell has contracted, from 0 to 1. */
  squeeze: d.f32,
  tint: d.vec3f,
  kind: d.u32,
  /** The colour of the bell's own glow. */
  glow: d.vec3f,
  /** A number from 0 to 1 that sets the jelly's own rhythm. */
  seed: d.f32,
  /** Where the jelly keeps returning to, so the tank keeps its near, middle and far. */
  home: d.vec3f,
  /** Seconds between pulses. */
  period: d.f32,
  /** When the jelly was last disturbed, in scene seconds. It flashes and darts away. */
  startled: d.f32,
});
export type Jelly = d.Infer<typeof Jelly>;

/** A tentacle or an oral arm. Jelly `j` owns strands `j * STRANDS` on, its tentacles first. */
export const Strand = d.struct({
  /** Where round the bell it hangs from, in radians. */
  around: d.f32,
  length: d.f32,
  /** Width at the root, in world units. */
  width: d.f32,
  /** 1 for an oral arm, 0 for a tentacle. */
  arm: d.f32,
  /** How far, and how quickly, waves curl down it. */
  curl: d.f32,
  pace: d.f32,
});
export type Strand = d.Infer<typeof Strand>;

/** How far round each bell reaches from its apex, in radians, and its depth per unit radius. */
const REACH = [1.05, 1.3] as const;
const DEPTH = [0.3, 0.72] as const;
/** Lobes round each margin. */
const LOBES = 16;
/** The share of a bell's width a full contraction takes from its margin, and the height it adds. */
const PINCH = 0.28;
const LIFT = 0.4;

/** Depth of a relaxed bell per unit radius. */
export const depthOf = (kind: number) => {
  'use gpu';
  return std.select(d.f32(DEPTH[0]), d.f32(DEPTH[1]), kind === NETTLE);
};

/**
 * A point of a jelly's unit bell around +y, its apex at the origin, at `time`: `v` runs from the
 * apex to the margin and `around` round the axis, in radians. A squeezed bell draws its margin in
 * and grows taller. No two bells, and no two lappets, are quite alike.
 */
export const bellPoint = (
  jelly: d.InferGPU<typeof Jelly>,
  time: number,
  v: number,
  around: number,
) => {
  'use gpu';
  const nettle = jelly.kind === NETTLE;
  const squeeze = jelly.squeeze;
  const seed = jelly.seed * 40;
  const reach = std.select(d.f32(REACH[0]), d.f32(REACH[1]), nettle);
  const theta = v * reach;
  const v2 = v * v;
  const v4 = v2 * v2;
  // Lappets of uneven spacing and depth
  const uneven = around * LOBES * 0.5 + std.sin(around * 5 + seed) * 0.35;
  const deep = 0.6 + 0.4 * std.sin(around * 7 + seed * 1.3);
  const lappets = std.abs(std.cos(uneven)) * deep;
  const scallop = 1 + (lappets - 0.5) * v4 * v2 * 0.05;
  // A slightly lopsided bell
  const lopsided = 1 + std.sin(around + seed) * 0.035 + std.sin(around * 2 + seed * 0.7) * 0.02;
  // The margin snaps in on a contraction and ripples as it does
  const ripple = std.sin(around * 6) * v4 * squeeze * (1 - squeeze) * 0.12;
  const pinch = 1 - squeeze * PINCH * v2 * std.sqrt(v) + ripple;
  // A sea nettle's margin flares out a little when relaxed
  const margin = std.smoothstep(0.72, 1, v);
  const flare = std.select(0, margin * margin * 0.04 * (1 - squeeze), nettle);
  const r = (std.sin(theta) / std.sin(reach)) * pinch * scallop * lopsided + flare;
  // The margin undulates even at rest
  const swell =
    margin *
    margin *
    (std.sin(around * 3 + time * 1.1 + seed) * 0.035 + std.sin(around * 5 - time * 0.8) * 0.015);
  // A moon jelly's margin turns down a little at its very edge
  const lip = std.select(std.smoothstep(0.82, 1, v) * 0.07, -flare * 0.9, nettle);
  const cupped = (1 - std.cos(theta)) / (1 - std.cos(reach));
  const y = -depthOf(jelly.kind) * (1 + squeeze * LIFT) * cupped - lip + swell;
  return d.vec3f(r * std.cos(around), y, r * std.sin(around));
};

/** The jelly's own frame: `right` and `forward` across its axis. */
export const frameOf = (axis: d.v3f) => {
  'use gpu';
  const ref = std.select(d.vec3f(0, 0, 1), d.vec3f(1, 0, 0), std.abs(axis.z) > 0.9);
  const right = std.normalize(std.cross(axis, ref));
  return d.mat3x3f(right, axis, std.cross(right, axis));
};

/** Where a point of a jelly's unit bell is in the world. */
export const placed = (jelly: d.InferGPU<typeof Jelly>, local: d.v3f) => {
  'use gpu';
  return jelly.position + frameOf(jelly.axis) * local * jelly.size;
};

/**
 * How far the bell contracts at `beat`, the fraction of a pulse: in over `closing` of the beat,
 * out over `opening`, then a glide.
 */
export const contraction = (beat: number, closing: number, opening: number) => {
  'use gpu';
  const shut = std.smoothstep(0, closing, beat);
  const open = 1 - std.smoothstep(closing, closing + opening, beat);
  return std.select(open, shut, beat < closing);
};
