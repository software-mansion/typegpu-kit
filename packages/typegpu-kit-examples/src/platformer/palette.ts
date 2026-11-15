import { linearRgbToOklab, srgbToLinear } from '@typegpu/color';
import { d, std, tgpu } from 'typegpu';

/**
 * The 32 colours the print may show, dark to light. Fitted with k-means in OKLab to the game's
 * inks, their full shadows and lit cut edges, and the sky's gradient, weighting the ground's greens
 * so each keeps an entry of its own and sprites' colours so small ones keep their hue. The ten
 * colours of the hero's sheet then replaced entries near them, so the hero prints as drawn.
 */
const PALETTE = [
  '#140f1a',
  '#222034',
  '#15322d',
  '#50182c',
  '#382952',
  '#244b3f',
  '#743f57',
  '#873f24',
  '#396157',
  '#6b4099',
  '#306b3a',
  '#3f5f9b',
  '#af2a5a',
  '#a05024',
  '#4a8652',
  '#7581a2',
  '#b97034',
  '#6a9c7e',
  '#f24d59',
  '#d77b83',
  '#6aa2e2',
  '#61c24a',
  '#9badb7',
  '#88bb89',
  '#e19d57',
  '#ea9a75',
  '#93bad9',
  '#accbe3',
  '#a2df74',
  '#ffcc38',
  '#cbddea',
  '#ffffff',
];

const decode = (hex: string) =>
  [1, 3, 5].map((at) => Number.parseInt(hex.slice(at, at + 2), 16) / 255) as [
    number,
    number,
    number,
  ];

/** OKLab of an sRGB colour, as `linearRgbToOklab(srgbToLinear(…))` computes it on the GPU. */
const labOf = ([r, g, b]: readonly number[]) => {
  const [lr, lg, lb] = [r!, g!, b!].map((c) =>
    c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4,
  ) as [number, number, number];
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return d.vec3f(
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  );
};

/** Each palette entry's colour and its OKLab coordinates, which matching measures in. */
const ENTRIES = PALETTE.map((hex) => {
  const rgb = decode(hex);
  return { rgb: d.vec3f(...rgb), lab: labOf(rgb) };
});

/** A 4×4 Bayer matrix, row by row, as thresholds between 0 and 1. */
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((k) => (k + 0.5) / 16);
const Thresholds = d.arrayOf(d.f32, 16);
const thresholds = tgpu.const(Thresholds, BAYER);

const Entry = d.struct({ rgb: d.vec3f, lab: d.vec3f });

/** The display colour `color` in OKLab. */
const labAt = (color: d.v3f) => {
  'use gpu';
  return linearRgbToOklab(srgbToLinear(std.saturate(color)));
};

/** The palette entry nearest `lab`. */
const nearestTo = (lab: d.v3f) => {
  'use gpu';
  let best = Entry({ rgb: ENTRIES[0].rgb, lab: ENTRIES[0].lab });
  let gap = d.f32(1e9);
  for (const entry of tgpu.unroll(ENTRIES)) {
    const offset = entry.lab - lab;
    const distance = std.dot(offset, offset);
    if (distance < gap) {
      gap = distance;
      best = Entry({ rgb: entry.rgb, lab: entry.lab });
    }
  }
  return best;
};

/** The palette colour nearest the display colour `color`. */
export const nearest = (color: d.v3f) => {
  'use gpu';
  return nearestTo(labAt(color)).rgb;
};

/** The ordered-dithering threshold of cell `cell`, between 0 and 1, repeating every 4 cells. */
export const thresholdAt = (cell: d.v2i) => {
  'use gpu';
  const x = ((cell.x % 4) + 4) % 4;
  const y = ((cell.y % 4) + 4) % 4;
  return thresholds.$[y * 4 + x];
};

/**
 * Shows the display colour `color` as one of the two palette colours around it: the nearest, or
 * the next one past it where `color` lies further toward it than `threshold`. Thresholds that
 * vary from cell to cell mix the two in proportion, as an ordered dither.
 */
export const dithered = (color: d.v3f, threshold: number) => {
  'use gpu';
  const lab = labAt(color);
  const near = nearestTo(lab);
  // The entry on the far side of `color` from the nearest one, the other end of its blend
  const far = nearestTo(lab * 2 - near.lab);
  const span = far.lab - near.lab;
  const reach = std.dot(span, span);
  if (reach < 1e-8) return near.rgb;
  const along = std.saturate(std.dot(lab - near.lab, span) / reach);
  return std.select(near.rgb, far.rgb, along > threshold);
};
