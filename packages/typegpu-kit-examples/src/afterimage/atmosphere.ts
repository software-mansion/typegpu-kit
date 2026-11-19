import {
  cameras,
  shading,
  targets,
  type CameraBase,
  type Frame,
  type Lighting,
  type SizeSource,
  type post,
} from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { ordered } from './signal.ts';

const layout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  depth: { texture: d.textureDepth2d() },
  sampler: { sampler: 'filtering' },
});

export function createAtmosphere(
  root: TgpuRoot,
  view: CameraBase,
  light: Lighting,
  options: { size: SizeSource; source: () => post.FloatTexture; depth: () => targets.Depth },
) {
  const { size, source, depth } = options;
  const screen = targets.create(root, size, { hdr: { format: 'rgba16float', sampled: true } });
  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const fragment = tgpu.fragmentFn({
    in: { uv: d.vec2f, position: d.builtin.position },
    out: d.vec4f,
  })(({ uv, position }) => {
    'use gpu';
    const dimensions = std.textureDimensions(layout.$.depth);
    const pixel = std.min(d.vec2i(uv * d.vec2f(dimensions)), d.vec2i(dimensions) - 1);
    const z = std.textureLoad(layout.$.depth, pixel, 0);
    const camera = cameras.current.$;
    const ray = cameras.rayOf(camera, uv * d.vec2f(2, -2) + d.vec2f(-1, 1));
    const distance = std.min(
      38,
      cameras.distanceOf(camera, z) /
        std.max(0.1, std.dot(ray.direction, cameras.forwardOf(camera))),
    );
    const stride = distance / 24;
    // March from the dither's threshold, so the steps' grain repeats its pattern
    const offset = ordered(d.vec2u(position.xy));
    let beam = d.f32(0);
    for (let step = d.u32(0); step < 24; step++) {
      const world = ray.origin + ray.direction * ((d.f32(step) + offset) * stride);
      const inside =
        std.smoothstep(-6.6, -5.8, world.x) *
        (1 - std.smoothstep(4.6, 5, world.x)) *
        (1 - std.smoothstep(9.5, 10.1, world.y)) *
        std.smoothstep(-20, -18.5, world.z) *
        (1 - std.smoothstep(13.5, 15, world.z));
      const density = 0.018 * std.exp(-std.max(0, world.y) * 0.065) * inside;
      const lit = shading.sunShare(shading.pointAt(world, d.vec3f(0, 1, 0)));
      beam += lit * density * stride;
    }
    const facing = std.pow(std.saturate(std.dot(ray.direction, shading.towardsSun())), 8);
    const shafts = d.vec3f(1, 0.69, 0.36) * (beam * (1 + facing * 2.5));
    const picture = std.textureSampleLevel(layout.$.source, layout.$.sampler, uv, 0).xyz;
    return d.vec4f(picture + shafts, 1);
  });
  const pipeline = root
    .pipe(view.inject())
    .pipe(light.inject())
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: { format: 'rgba16float' },
    });
  const bound = memo((color: post.FloatTexture, z: targets.Depth) =>
    pipeline.with(root.createBindGroup(layout, { source: color, depth: z, sampler })),
  );
  const output = { colorAttachments: { view: screen.views.hdr } };
  bound(source(), depth());

  return {
    screen,
    initAsync: () => pipeline.initAsync(),
    record(frame: Frame) {
      screen.fit();
      output.colorAttachments.view = screen.views.hdr;
      const pass = frame.encoder.beginRenderPass(output);
      pass.setPipeline(bound(source(), depth()));
      pass.draw(3);
      pass.end();
    },

    destroy() {
      screen.destroy();
    },
  };
}
