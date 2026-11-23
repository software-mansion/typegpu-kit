import { hexToRgb, srgbToLinear } from '@typegpu/color';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuRenderCommands,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';

/** Picture size in pixels and elapsed growth time in seconds. */
export const World = d.struct({ size: d.vec2f, time: d.f32 });

/** Converts an sRGB hex colour, such as 0xc0301c, to linear light. */
export const hex = (rgb: number) => srgbToLinear(hexToRgb(rgb));

const AGAR = hex(0x080b10);
const DEEP = hex(0x030406);

/**
 * Sheet coordinates have (0, 0) at the picture centre, with y pointing down. One unit is half
 * the picture's longer side. The mould grid covers -1 to 1 on both axes, so on a wide picture its
 * top and bottom run past the edges.
 */
export const sheetAt = (pixel: d.v2f, size: d.v2f) => {
  'use gpu';
  return ((pixel - size * 0.5) * 2) / std.max(size.x, size.y);
};

/** Draws the agar dish over the whole picture, darkening towards its edges. */
export function createAgar(
  root: TgpuRoot,
  world: TgpuUniform<typeof World>,
  options: { format: GPUTextureFormat },
) {
  const { format } = options;
  const ground = tgpu.fragmentFn({ in: { pixel: d.builtin.position }, out: d.vec4f })(({
    pixel,
  }) => {
    'use gpu';
    const at = sheetAt(pixel.xy, world.$.size);
    return d.vec4f(std.mix(AGAR, DEEP, std.smoothstep(0.2, 1.1, std.length(at))), 1);
  });

  const drawing = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: ground,
    targets: { format },
  });

  return {
    initAsync: () => drawing.initAsync(),
    draw(pass: TgpuRenderCommands) {
      pass.setPipeline(drawing);
      pass.draw(3);
    },
  };
}
