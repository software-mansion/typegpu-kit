import { srgbToLinear } from '@typegpu/color';
import { perlin2d } from '@typegpu/noise';
import { baker } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';

/** Texture dimensions. Each domino in the 100 × 60 field gets 4 × 4 texels. */
export const SIZE = [400, 240] as const;

/** Width over height, so distances measured in picture heights stay round. */
const ASPECT = SIZE[0] / SIZE[1];

/** sRGB palette ordered from sky to foreground hills. */
const NIGHT = d.vec3f(0.12, 0.1, 0.36);
const DUSK = d.vec3f(0.55, 0.2, 0.55);
const ROSE = d.vec3f(0.95, 0.4, 0.5);
const EMBER = d.vec3f(0.98, 0.55, 0.15);
const GLOW = d.vec3f(0.99, 0.64, 0.2);
const SUN = d.vec3f(1, 0.92, 0.45);
const SUN_LOW = d.vec3f(1, 0.8, 0.22);
const FAR = d.vec3f(0.5, 0.22, 0.55);
const NEAR = d.vec3f(0.16, 0.14, 0.42);
const NEAREST = d.vec3f(0.06, 0.05, 0.12);

/** Sun centre measured from the picture top left, and radius, in picture heights. */
const SUN_AT = d.vec2f(0.5, 0.56);
const SUN_RADIUS = 0.27;

/** Maps height from the top at 0 to the horizon at 1 through colour bands. */
const skyAt = (y: number) => {
  'use gpu';
  let color = std.mix(NIGHT, DUSK, std.smoothstep(0, 0.22, y));
  color = std.mix(color, ROSE, std.smoothstep(0.18, 0.38, y));
  color = std.mix(color, EMBER, std.smoothstep(0.36, 0.52, y));
  return std.mix(color, GLOW, std.smoothstep(0.5, 0.62, y));
};

/** Hill height at `x`, measured from the picture top. Ridged noise adds peaks to `base`. */
const ridgeAt = (x: number, base: number, height: number, scale: number, seed: number) => {
  'use gpu';
  const ridge = 1 - std.abs(perlin2d.sample(d.vec2f(x * scale, seed)));
  const roll = perlin2d.sample(d.vec2f(x * scale * 0.35, seed + 7));
  return base - (ridge * ridge * 0.7 + roll * 0.5) * height;
};

/**
 * Draws a banded sky, a striped sun and three hill ranges. Large shapes remain visible at 100 ×
 * 60.
 */
export function createSunset(root: TgpuRoot) {
  const texture = root
    .createTexture({ size: SIZE, format: 'rgba16float' })
    .$usage('storage', 'sampled');

  const drawing = baker.create(root, texture, {
    texel: (uv) => {
      'use gpu';
      // Use picture heights on both axes to keep the sun round
      const x = uv.x * ASPECT;
      let color = skyAt(uv.y);

      const fromSun = d.vec2f(x - SUN_AT.x * ASPECT, uv.y - SUN_AT.y);
      const below = std.max(uv.y - SUN_AT.y + 0.04, 0);
      const barred = std.fract(below * 28) < below * 4.5;
      if (std.length(fromSun) < SUN_RADIUS && !barred) {
        color = std.mix(SUN, SUN_LOW, std.smoothstep(-SUN_RADIUS, SUN_RADIUS, fromSun.y));
      }

      if (uv.y > ridgeAt(x, 0.72, 0.22, 3.2, 1)) color = FAR;
      if (uv.y > ridgeAt(x, 0.83, 0.2, 4.5, 2)) color = NEAR;
      if (uv.y > ridgeAt(x, 0.94, 0.16, 2.4, 3)) color = NEAREST;

      return d.vec4f(srgbToLinear(color), 1);
    },
  });

  return { drawing, texture };
}
