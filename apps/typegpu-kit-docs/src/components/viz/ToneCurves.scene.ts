import { type Frame, type Surface, tone, type Tone } from '@typegpu/kit';
import { linearToSrgb, srgbToLinear } from '@typegpu/color';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { COLORS } from '../site/dither.ts';
import { printColor } from './print.ts';

const GROUND_SRGB = d.vec3f(...COLORS.grey[0]);

/** Scene light at the strips' ends, in stops, as on the plot's axis. */
export const LOW = -7;
export const HIGH = 6;

/** Strips, top to bottom, in the order `toned` picks them. */
export const STRIPS = ['aces', 'neutral', 'reinhard', 'agx', 'none'] as const;

export type Strip = (typeof STRIPS)[number];

const none = (color: d.v3f) => {
  'use gpu';
  return d.vec3f(color);
};

export const CURVES: Record<Strip, Tone> = {
  aces: tone.aces,
  neutral: tone.neutral,
  reinhard: tone.reinhard,
  agx: tone.agx,
  none,
};

/** Of each row, the part the strip fills; the rest is the gap below it. */
export const FILL = 0.75;

const accent = srgbToLinear(d.vec3f(...COLORS.accent[0]));

/** The colours the figure offers, linear, each with a brightest channel of 1; the first at first. */
export const HUES = {
  blue: [accent.x / accent.z, accent.y / accent.z, 1],
  violet: [0.4, 0.04, 1],
  moss: [0.25, 1, 0.06],
} as const satisfies Record<string, readonly [number, number, number]>;

export type Hue = keyof typeof HUES;

const Strip = d.struct({ color: d.vec3f, exposure: d.f32 });
const strip = tgpu.accessor(Strip);

const toned = (color: d.v3f, row: number) => {
  'use gpu';
  if (row === 0) return tone.aces(color);
  if (row === 1) return tone.neutral(color);
  if (row === 2) return tone.reinhard(color);
  if (row === 3) return tone.agx(color);
  return none(color);
};

const strips = tgpu.fragmentFn({ in: { uv: d.vec2f, position: d.builtin.position }, out: d.vec4f })(
  ({ uv, position }) => {
    'use gpu';
    const band = uv.y * (STRIPS.length - 1 + FILL);
    if (std.fract(band) > FILL) return d.vec4f();

    const light = std.exp2(LOW + (HIGH - LOW) * uv.x) * strip.$.exposure;
    const shown = linearToSrgb(toned(strip.$.color * light, d.u32(band)));

    return d.vec4f(printColor(shown, d.vec2u(position.xy), GROUND_SRGB), 1);
  },
);

export function createScene(root: TgpuRoot, surface: Surface) {
  const values = root.createUniform(Strip, { color: d.vec3f(...HUES.blue), exposure: 1 });
  const pipeline = root.with(strip, values).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: strips,
    targets: { format: surface.format },
  });

  return {
    /** A linear colour whose brightest channel is 1, and the exposure, as a factor. */
    set(color: readonly [number, number, number], exposure: number) {
      values.write({ color: d.vec3f(...color), exposure });
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
