// The site's dither, which every canvas of a pull prints through: 8×8 Bayer cells and the greys
// above the ground.

import { d, std } from 'typegpu';

/** Points, or CSS pixels, a cell. */
export const CELL = 3;

/** The threshold of `cell` in the 8×8 Bayer matrix, in (0, 1), from its coordinates' bits. */
export const bayerCell = (cell: d.v2u) => {
  'use gpu';
  const x = cell.x & 7;
  const y = cell.y & 7;
  const v = x ^ y;
  const index =
    ((v & 1) << 5) |
    ((y & 1) << 4) |
    ((v & 2) << 2) |
    ((y & 2) << 1) |
    ((v & 4) >>> 1) |
    ((y & 4) >>> 2);
  return (d.f32(index) + 0.5) / 64;
};

/**
 * `value`, 0 to 1, as one of five tones, 0 to 4, dithered through `threshold` and pulled toward
 * the tones themselves so a flat face prints mostly solid.
 */
export const toneOf = (value: number, threshold: number) => {
  'use gpu';
  const raw = std.saturate(value) * 4;
  const level = std.round(raw) + (raw - std.round(raw)) * 0.4;
  const low = std.min(std.floor(level), 3);
  return d.u32(low + std.step(threshold, level - low));
};

/** An sRGB colour from 0 to 1. */
export type Rgb01 = readonly [number, number, number];

/** The canvases' colours. The greys run from the page's ground up. */
export interface Colors {
  readonly grey: readonly [Rgb01, Rgb01, Rgb01, Rgb01, Rgb01, Rgb01, Rgb01];
  readonly accent: readonly [Rgb01, Rgb01];
}

/** `hex`, `#rrggbb`, from 0 to 1. */
export const rgbOf = (hex: string): Rgb01 => [
  Number.parseInt(hex.slice(1, 3), 16) / 255,
  Number.parseInt(hex.slice(3, 5), 16) / 255,
  Number.parseInt(hex.slice(5, 7), 16) / 255,
];

/** The site's tokens as they stand, for a host or a test that reads none. */
export const COLORS: Colors = {
  grey: [
    rgbOf('#0a0d11'),
    rgbOf('#141a20'),
    rgbOf('#323b45'),
    rgbOf('#5d6772'),
    rgbOf('#929ca8'),
    rgbOf('#c3cad2'),
    rgbOf('#eaeef2'),
  ],
  accent: [rgbOf('#4d79ff'), rgbOf('#dbf0ff')],
};
