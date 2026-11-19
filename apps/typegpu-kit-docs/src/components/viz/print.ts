import { d, std } from 'typegpu';
import { bayerCell } from '../site/dither.ts';

/** Prints display colour through an 8×8 Bayer screen, preserving its hue. */
export const printColor = (color: d.v3f, cell: d.v2u, ground: d.v3f) => {
  'use gpu';
  const levels = std.saturate(color) * 6;
  const ink = (std.floor(levels) + std.step(d.vec3f(bayerCell(cell)), std.fract(levels))) / 6;
  return std.max(ground, ink);
};
