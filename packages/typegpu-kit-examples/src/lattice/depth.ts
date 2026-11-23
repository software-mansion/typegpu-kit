import { cameras, type CameraBase, type Frame, type targets } from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';

const layout = tgpu.bindGroupLayout({ depth: { texture: d.textureDepth2d() } });

/** Eye distance shown as white. Twice this distance is half as bright. */
const BRIGHT = 4;

/** Displays depth as greyscale eye distance from `view`, with black for unwritten pixels. */
export function createDepthView(
  root: TgpuRoot,
  view: CameraBase,
  options: { format: GPUTextureFormat; depth: () => targets.Depth },
) {
  const { format, depth } = options;
  const fragment = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })(({ position }) => {
    'use gpu';
    const z = std.textureLoad(layout.$.depth, d.vec2i(position.xy), 0);
    if (z >= 1) return d.vec4f(0, 0, 0, 1);
    const distance = cameras.distanceOf(cameras.current.$, z);
    return d.vec4f(d.vec3f(std.saturate(BRIGHT / distance)), 1);
  });

  const pipeline = root.pipe(view.inject()).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format },
  });
  // Refresh the binding when the depth target is rebuilt
  const bound = memo((texture: targets.Depth) =>
    pipeline.with(root.createBindGroup(layout, { depth: texture.createView(d.textureDepth2d()) })),
  );

  return {
    initAsync: () => pipeline.initAsync(),
    /** Draws into a pass of its own on `view`, such as `surface.view(frame)`. */
    record(frame: Frame, into: GPUTextureView) {
      const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: into } });
      pass.setPipeline(bound(depth()));
      pass.draw(3);
      pass.end();
    },
  };
}
