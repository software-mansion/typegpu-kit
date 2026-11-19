import { hexToRgb, srgbToLinear } from '@typegpu/color';
import { perlin2d } from '@typegpu/noise';
import { d, std, tgpu } from 'typegpu';

/** Contour levels, the deepest first: ten soundings, the shoreline, then the land. */
export const LEVELS = 41;
/** The shoreline's level index; those below it are soundings. */
export const SHORE = 10;
/** Every `INDEX`th level from the shoreline is an index contour, drawn bold. */
const INDEX = 5;
/** Height between levels, in the terrain's units and in world units once lifted. */
const STEP = 0.04;
/** World units the lowest ring sits over the plinth's top. */
const BASE = 0.05;
/** World units from the peak to the farthest a ring reaches. */
export const REACH = 4.3;

/** The terrain height at level `ring`. */
export const levelOf = (ring: number) => {
  'use gpu';
  return (d.f32(ring) - SHORE) * STEP;
};

/** World height of ring `ring` lifted by `lift`, from resting on the plinth at 0 to its level at 1. */
export const liftedOf = (ring: number, lift: number) => {
  'use gpu';
  return BASE + d.f32(ring) * STEP * lift;
};

/**
 * The island's height at `p` on the plinth: one peak, its slopes roughened by noise that fades
 * near the summit so every level closes round it, falling away before `REACH` so every sounding
 * closes inside it.
 */
export const heightAt = (p: d.v2f) => {
  'use gpu';
  const r = std.length(p);
  const peak = 2.1 * std.exp(-(r * r) / 5.2) - 0.55;
  const rough = perlin2d.sample(p * 0.42 + 3.1) * 0.75 + perlin2d.sample(p * 1.05 - 7.4) * 0.3;
  return peak + rough * std.smoothstep(0.4, 2.2, r) - std.smoothstep(2.4, REACH, r) * 0.75;
};

/** Hypsometric stops in sRGB: the deep soundings, the shallow ones, then the land up to the peak. */
const DEEP = [0x1d4e89, 0x4f92c4].map((hex) => hexToRgb(hex));
const LAND = [
  0x3f8f5a, 0x67a65b, 0x93b95b, 0xbcc566, 0xd9c36c, 0xd5a159, 0xbf7d4d, 0xa4664a, 0x8c5446,
].map((hex) => hexToRgb(hex));
/** The shoreline's pale sea glass. */
const GLASS = hexToRgb(0xd6ece0);
/** The slate that intermediate contours lean towards. */
const MUTE = hexToRgb(0x3a4048);

/** The colour at `t` from 0 to 1 along `stops`. */
const along = (stops: d.v3f[], t: number) => {
  const at = t * (stops.length - 1);
  const k = Math.min(Math.floor(at), stops.length - 2);
  return std.mix(stops[k], stops[k + 1], at - k);
};

/** Whether level `ring` is an index contour. */
export const isIndex = (ring: number) => {
  'use gpu';
  return (ring + LEVELS * INDEX - SHORE) % INDEX === 0;
};

/** Level `ring`'s tint in sRGB. */
const tintOf = (ring: number) => {
  if (ring === SHORE) return GLASS;
  const srgb =
    ring < SHORE
      ? along(DEEP, ring / (SHORE - 1))
      : along(LAND, (ring - SHORE - 1) / (LEVELS - SHORE - 2));
  return isIndex(ring) ? srgb : std.mix(srgb, MUTE, 0.45);
};

/**
 * Hypsometric tints, the deepest sounding first. The shoreline is pale sea glass, and the
 * intermediate contours between index contours lean towards slate.
 */
export const tints = tgpu.const(
  d.arrayOf(d.vec3f, LEVELS),
  Array.from({ length: LEVELS }, (_, ring) => srgbToLinear(tintOf(ring))),
);
