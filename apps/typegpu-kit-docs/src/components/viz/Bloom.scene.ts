import { targets, type Frame, type Surface, bloom, finish } from '@typegpu/kit';
import { linearToSrgb, srgbToLinear } from '@typegpu/color';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { COLORS } from '../site/dither.ts';
import { printColor } from './print.ts';

const GROUND_SRGB = d.vec3f(...COLORS.grey[0]);

/** Discs, each twice as bright as the one before, from half of white. */
export const DISCS = 6;

/** The discs' end colours and the ground's, in linear light. */
const WHITE = srgbToLinear(d.vec3f(...COLORS.grey[6]));
const AMBER = srgbToLinear(d.vec3f(0.97, 0.71, 0.31));
const GROUND = srgbToLinear(d.vec3f(...COLORS.grey[0]));

const pixels = tgpu.accessor(d.vec2f);

const discs = tgpu.fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })(({
  position,
}) => {
  'use gpu';
  const size = pixels.$;
  const radius = std.min(size.y * 0.12, (size.x / DISCS) * 0.3);
  let color = d.vec3f();

  for (const i of tgpu.unroll(std.range(DISCS))) {
    const center = d.vec2f(((i + 0.5) / DISCS) * size.x, size.y * 0.5);
    const distance = std.length(position.xy - center);
    const cover = std.saturate(radius - distance + 0.5);
    const hue = std.mix(WHITE, AMBER, d.f32(i) / (DISCS - 1));
    const peak = std.max(std.max(hue.x, hue.y), hue.z);
    color = std.mix(color, (hue / peak) * std.exp2(d.f32(i) - 1), cover);
  }

  return d.vec4f(color, 1);
});

export interface Look {
  threshold: number;
  glowIntensity: number;
}

export interface Options {
  /** The bloom's levels at first. Default 6. */
  readonly levels?: number;
}

export function createScene(root: TgpuRoot, surface: Surface, options: Options = {}) {
  const screen = targets.create(root, surface, { hdr: { format: 'rgba16float', sampled: true } });
  const source = () => screen.textures.hdr;

  const extent = root.createUniform(d.vec2f);
  const ground = root.with(pixels, extent).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: discs,
    targets: { format: 'rgba16float' },
  });

  // The changes so far, so a remade bloom and final pass start with them
  const look: Partial<Look> = {};
  const make = (levels: number) => {
    const glow = bloom.create(root, { source, levels, settings: look });
    const final = finish.create(root, {
      format: surface.format,
      source,
      settings: { ...look, vignette: 0, dither: 0 },
      glow,
      grade: (color, uv) => {
        'use gpu';
        return srgbToLinear(
          printColor(linearToSrgb(std.max(color, GROUND)), d.vec2u(uv * extent.$), GROUND_SRGB),
        );
      },
    });
    return { glow, final };
  };

  let { glow, final } = make(options.levels ?? 6);
  const written = [0, 0];

  return {
    /** Writes the threshold and the glow intensity into the bloom's settings and the final pass's. */
    set(change: Partial<Look>) {
      Object.assign(look, change);
      final.settings.patch(change);
      glow.settings.patch(change);
    },

    /** Replaces the bloom with one of `levels` levels, and the final pass that adds its glow. */
    remake(levels: number) {
      glow.destroy();
      ({ glow, final } = make(levels));
    },

    /** Draws the discs, their glow and the final pass. */
    frame(frame: Frame) {
      screen.fit();
      const [width, height] = screen.textures.hdr.props.size;
      if (written[0] !== width || written[1] !== height) {
        written[0] = width!;
        written[1] = height!;
        extent.write(d.vec2f(width!, height!));
      }

      const scene = frame.encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      scene.setPipeline(ground);
      scene.draw(3);
      scene.end();

      glow.record(frame);

      const last = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame) },
      });
      final.draw(last);
      last.end();
    },

    destroy() {
      glow.destroy();
      screen.destroy();
      extent.buffer.destroy();
    },
  };
}
