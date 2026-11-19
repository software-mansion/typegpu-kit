import { taa, type Frame } from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { common, d, std, tgpu, type TgpuRoot, type TgpuTexture } from 'typegpu';

const layout = tgpu.bindGroupLayout({ motion: { texture: d.texture2d(d.f32) } });

/** Motion scale in pixels per frame. The lattice moves much less than one pixel. */
const SCALE = 1;

/**
 * Displays motion as colour. Grey means still, red and cyan mean horizontal motion, and green
 * and magenta mean vertical motion. Black marks unwritten pixels.
 */
export function createMotionView(
  root: TgpuRoot,
  options: { format: GPUTextureFormat; motion: () => TgpuTexture & { usableAsSampled: true } },
) {
  const { format, motion } = options;
  const fragment = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })(({ position }) => {
    'use gpu';
    const moved = std.textureLoad(layout.$.motion, d.vec2i(position.xy), 0).xy;
    if (!taa.written(moved)) return d.vec4f(0, 0, 0, 1);
    const pixels = moved * d.vec2f(std.textureDimensions(layout.$.motion));
    const shade = std.saturate(pixels / (SCALE * 2) + 0.5);
    return d.vec4f(shade.x, shade.y, 1 - shade.x, 1);
  });

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format },
  });
  // Refresh the binding when the motion target is rebuilt
  const bound = memo((texture: TgpuTexture & { usableAsSampled: true }) =>
    pipeline.with(root.createBindGroup(layout, { motion: texture })),
  );

  return {
    initAsync: () => pipeline.initAsync(),
    /** Draws into a pass of its own on `view`, such as `surface.view(frame)`. */
    record(frame: Frame, view: GPUTextureView) {
      const pass = frame.encoder.beginRenderPass({ colorAttachments: { view } });
      pass.setPipeline(bound(motion()));
      pass.draw(3);
      pass.end();
    },
  };
}
