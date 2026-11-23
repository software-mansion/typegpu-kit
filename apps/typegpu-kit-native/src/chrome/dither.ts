// The app's dithered look: the docs site's colours printed through the pull's Bayer cells, for the
// shaders that draw the gallery's banner and the veil over a loading example

import { d, std } from 'typegpu';
import { rgbOf } from 'typegpu-kit-pull/dither.ts';
import { colors } from '../theme.ts';

const rgb = (hex: string) => d.vec3f(...rgbOf(hex));

/** The theme's colours in sRGB from 0 to 1, as the canvas takes them. */
export const INK = {
  ground: rgb(colors.ground),
  ground3: rgb(colors.ground3),
  rule: rgb(colors.ruleStrong),
  muted: rgb(colors.muted),
  accent: rgb(colors.accent),
  accentHigh: rgb(colors.accentHigh),
} as const;

/** `value` from 0 to 1 printed in `levels` steps, the cell's threshold choosing between two. */
export const printed = (value: number, levels: number, threshold: number) => {
  'use gpu';
  const steps = d.f32(levels);
  const scaled = std.saturate(value) * steps;
  return (std.floor(scaled) + std.step(threshold, std.fract(scaled))) / steps;
};

/**
 * The site's ramp from the ground through the rules to the accent, printed: `value` from 0 to 1
 * picks a step, and the cell's threshold rounds it up or down.
 */
export const ramp = (value: number, threshold: number) => {
  'use gpu';
  const step = printed(value, 5, threshold) * 5;
  let color = INK.ground;
  if (step >= 1) color = INK.ground3;
  if (step >= 2) color = INK.rule;
  if (step >= 3) color = std.mix(INK.rule, INK.accent, 0.5);
  if (step >= 4) color = INK.accent;
  if (step >= 5) color = INK.accentHigh;
  return color;
};
