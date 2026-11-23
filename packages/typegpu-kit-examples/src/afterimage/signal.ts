import { linearToSrgb, srgbToLinear } from '@typegpu/color';
import type { SizeSource } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';

/** The 4×4 ordered dither's threshold at a pixel, from 1/32 to 31/32. */
export const ordered = (pixel: d.v2u) => {
  'use gpu';
  const x = pixel.x & 3;
  const y = pixel.y & 3;
  const low = ((x & 1) ^ (y & 1)) * 2 + (y & 1);
  const high = (((x >>> 1) & 1) ^ ((y >>> 1) & 1)) * 2 + ((y >>> 1) & 1);
  return (d.f32(low * 4 + high) + 0.5) / 16;
};

export const PRECISION = 31;

export function createSignal(root: TgpuRoot, size: SizeSource) {
  const precision = root.createUniform(d.f32, PRECISION);
  const dimensions = [size.width, size.height] as [number, number];
  const grid = root.createUniform(d.vec2f, dimensions);
  return {
    precision,
    fit() {
      if (dimensions[0] === size.width && dimensions[1] === size.height) return;
      dimensions[0] = size.width;
      dimensions[1] = size.height;
      grid.write(dimensions);
    },
    grade: (color: d.v3f, uv: d.v2f) => {
      'use gpu';
      if (precision.$ === 0) return d.vec3f(color);
      const threshold = ordered(d.vec2u(std.floor(uv * grid.$)));
      const encoded = linearToSrgb(std.max(color, d.vec3f(0)));
      return srgbToLinear(std.floor(encoded * precision.$ + threshold) / precision.$);
    },
  };
}
