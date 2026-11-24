import { d, std } from 'typegpu';
import { bayerCell, plain } from '../site/dither.ts';

const printDithered = (color: d.v3f, cell: d.v2u, ground: d.v3f) => {
  'use gpu';
  const levels = std.saturate(color) * 6;
  const ink = (std.floor(levels) + std.step(d.vec3f(bayerCell(cell)), std.fract(levels))) / 6;
  return std.max(ground, ink);
};

const printSmooth = (color: d.v3f, _cell: d.v2u, ground: d.v3f) => {
  'use gpu';
  return std.max(ground, std.saturate(color));
};

/** Prints display colour through an 8×8 Bayer screen, preserving its hue, or smooth with the dither off. */
export const printColor = plain ? printSmooth : printDithered;
