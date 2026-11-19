import type { Frame, Surface } from '@typegpu/kit';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';

export async function createScene(root: TgpuRoot, surface: Surface) {
  const time = root.createUniform(d.f32);

  const rings = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    const wave = std.sin(std.length(uv - 0.5) * 40 - time.$ * 3) * 0.5 + 0.5;
    return d.vec4f(0.2, 0.6, 1, 1) * wave;
  });
  const drawing = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: rings,
    targets: { format: surface.format },
  });

  return {
    frame(frame: Frame) {
      time.write(frame.time);
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame) },
      });
      pass.setPipeline(drawing);
      pass.draw(3);
      pass.end();
    },
  };
}
