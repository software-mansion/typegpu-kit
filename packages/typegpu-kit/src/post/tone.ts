/**
 * These `'use gpu'` functions apply tone curves and dither.
 *
 * @module
 */

import { aces as acesCurve, neutral as neutralCurve } from '@typegpu/color';
import { randf } from '@typegpu/noise';
import { d, std } from 'typegpu';

/** This `'use gpu'` function maps linear colour to display range. */
export type Tone = (color: d.v3f) => d.v3f;

const LUMA = d.vec3f(0.2126, 0.7152, 0.0722);

/** Returns the relative luminance of a linear colour. */
export const luminance = (color: d.v3f) => {
  'use gpu';
  return std.dot(color, LUMA);
};

/**
 * Applies the Narkowicz approximation of the ACES filmic curve, `aces` from `@typegpu/color`. It
 * increases contrast and shifts bright colours towards white.
 */
export const aces = acesCurve;

/**
 * Applies Khronos PBR Neutral, `neutral` from `@typegpu/color`. It preserves hue and saturation
 * below the highlights, then reduces them.
 */
export const neutral = neutralCurve;

/**
 * Applies Reinhard to luminance to preserve hues. The curve approaches white without reaching
 * it.
 */
export const reinhard = (color: d.v3f) => {
  'use gpu';
  return color / (1 + luminance(color));
};

// AgX base inset and outset matrices and log range, from Blender 4.0's AgX configuration
const AGX_INSET = d.mat3x3f(
  0.842479062253094,
  0.0423282422610123,
  0.0423756549057051,
  0.0784335999999992,
  0.878468636469772,
  0.0784336,
  0.0792237451477643,
  0.0791661274605434,
  0.879142973793104,
);
const AGX_OUTSET = d.mat3x3f(
  1.19687900512017,
  -0.0528968517574562,
  -0.0529716355144438,
  -0.0980208811401368,
  1.15190312990417,
  -0.0980434501171241,
  -0.0990297440797205,
  -0.0989611768448433,
  1.15107367264116,
);
const AGX_LOW = -12.47393;
const AGX_HIGH = 4.026069;

/**
 * Applies AgX, Blender's default view transform. Bright colours fade towards white with little hue
 * shift, and the curve approaches white without reaching it.
 */
export const agx = (color: d.v3f) => {
  'use gpu';
  const inset = AGX_INSET * std.max(color, d.vec3f(1e-10));
  const stops = std.clamp(std.log2(inset), d.vec3f(AGX_LOW), d.vec3f(AGX_HIGH));
  const x = (stops - AGX_LOW) / (AGX_HIGH - AGX_LOW);

  // Sigmoid contrast in log space, the polynomial fit by Benjamin Wrensch
  const x2 = x * x;
  const x4 = x2 * x2;
  const curve =
    x4 * x2 * 15.5 -
    x4 * x * 40.14 +
    x4 * 31.96 -
    x2 * x * 6.868 +
    x2 * 0.4298 +
    x * 0.1191 -
    0.00232;

  // The curve gives display values with a 2.2 power; return them linear as the other curves do
  const display = std.saturate(AGX_OUTSET * curve);
  return std.pow(display, d.vec3f(2.2));
};

/**
 * Adds triangular noise to encoded colour, seeded by `pixel`. `steps` sets the amplitude in
 * 8-bit steps. This reduces banding in dark 8-bit gradients. The final pass applies it before
 * storing output.
 */
export const dither = (encoded: d.v3f, pixel: d.v2f, steps: number) => {
  'use gpu';
  randf.seed2(pixel);

  return encoded + ((randf.sample() - randf.sample()) * steps) / 255;
};
