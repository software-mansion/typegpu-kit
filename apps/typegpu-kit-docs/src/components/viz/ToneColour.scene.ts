import type { Frame, Surface } from '@typegpu/kit';
import { linearToSrgb } from '@typegpu/color';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { FULL, GAIN, ramp } from 'typegpu-kit-examples/slime-mould/veins.ts';
import { COLORS } from '../site/dither.ts';
import { printColor } from './print.ts';

export { FULL, GAIN };

const GROUND_SRGB = d.vec3f(...COLORS.grey[0]);

/** Rows top to bottom: the ramp, the cross-section in flat amber, and through the tone and ramp. */
export const ROWS = 3;

/** Of each row, the part above its band that the label takes. */
export const LABEL = 0.4;

/** The cross-section's bumps of trail: where each lies, from 0 to 1, its density and its width. */
const SECTION = [
  [0.05, 0.5, 0.02],
  [0.11, 0.8, 0.015],
  [0.18, 1, 0.02],
  [0.27, 1.5, 0.015],
  [0.36, 3, 0.012],
  [0.47, 6, 0.012],
  [0.58, 12, 0.014],
  [0.7, 30, 0.02],
  [0.82, 4, 0.012],
  [0.9, 0.8, 0.02],
  [0.96, 0.5, 0.02],
] as const;

const bumps = tgpu.const(
  d.arrayOf(d.vec3f, SECTION.length),
  SECTION.map(([at, density, width]) => d.vec3f(at, density, width)),
);

const Look = d.struct({ gain: d.f32, picked: d.f32 });
const look = tgpu.accessor(Look);

const densityOn = (x: number) => {
  'use gpu';
  let sum = d.f32(0);
  for (let i = 0; i < SECTION.length; i++) {
    const bump = bumps.$[i];
    const off = (x - bump.x) / bump.z;
    sum += bump.y * std.exp(-off * off);
  }
  return sum;
};

const toneOf = (density: number) => {
  'use gpu';
  return std.log(1 + density * look.$.gain) / std.log(1 + FULL * look.$.gain);
};

const strips = tgpu.fragmentFn({ in: { uv: d.vec2f, position: d.builtin.position }, out: d.vec4f })(
  ({ uv, position }) => {
    'use gpu';
    const pixel = std.fwidth(uv.x);
    const row = uv.y * ROWS;
    if (std.fract(row) < LABEL) return d.vec4f();

    const band = d.u32(row);
    let color = ramp(toneOf(densityOn(uv.x)));
    if (band === 0) {
      color = ramp(uv.x);
      if (std.abs(uv.x - look.$.picked) < pixel * 1.5) color = d.vec3f(1);
    }
    if (band === 1) color = ramp(0.7) * std.saturate(densityOn(uv.x));

    const shown = linearToSrgb(std.saturate(color));
    return d.vec4f(printColor(shown, d.vec2u(position.xy), GROUND_SRGB), 1);
  },
);

/** The slime mould's colour ramp, and a cross-section of trail drawn flat and through its tone. */
export function createScene(root: TgpuRoot, surface: Surface) {
  const values = root.createUniform(Look, { gain: GAIN, picked: 0 });
  const pipeline = root.with(look, values).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: strips,
    targets: { format: surface.format },
  });

  return {
    /** The tone curve's gain, and the tone the ramp marks. */
    set(gain: number, picked: number) {
      values.write({ gain, picked });
    },

    frame(frame: Frame) {
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), clearValue: [0, 0, 0, 0] },
      });
      pass.setPipeline(pipeline);
      pass.draw(3);
      pass.end();
    },

    destroy() {
      values.buffer.destroy();
    },
  };
}
