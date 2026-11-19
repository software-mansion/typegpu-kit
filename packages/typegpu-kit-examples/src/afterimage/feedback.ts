import { targets, type Frame, type SizeSource, type post } from '@typegpu/kit';
import { memo } from '@typegpu/kit/memo';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';

const State = d.struct({ retained: d.f32, glow: d.f32, vertical: d.f32 });
const layout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  previous: { texture: d.texture2d(d.f32) },
  glow: { texture: d.texture2d(d.f32) },
  sampler: { sampler: 'filtering' },
});

export function createFeedback(
  root: TgpuRoot,
  size: SizeSource,
  glow: post.Rendered,
  options: { source: () => post.FloatTexture },
) {
  const { source } = options;
  const screen = targets.create(root, size, {
    a: { format: 'rgba16float', sampled: true },
    b: { format: 'rgba16float', sampled: true },
  });
  const state = { retained: 0, glow: 0, vertical: 0 };
  const uniform = root.createUniform(State, state).$name('afterimage feedback');
  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    const current = std.textureSampleLevel(layout.$.source, layout.$.sampler, uv, 0).xyz;
    const light = std.textureSampleLevel(layout.$.glow, layout.$.sampler, uv, 0).xyz;
    let color = current + light * uniform.$.glow;
    if (uniform.$.retained > 0) {
      const texel = d.vec2f(0, 1) / d.vec2f(std.textureDimensions(layout.$.previous));
      const above = std.textureSampleLevel(layout.$.previous, layout.$.sampler, uv - texel, 0).xyz;
      const below = std.textureSampleLevel(layout.$.previous, layout.$.sampler, uv + texel, 0).xyz;
      const previous = std.textureSampleLevel(layout.$.previous, layout.$.sampler, uv, 0).xyz;
      const filtered = std.mix(
        previous,
        previous * 0.5 + (above + below) * 0.25,
        uniform.$.vertical,
      );
      color = std.max(color, filtered * uniform.$.retained);
    }
    return d.vec4f(color, 1);
  });
  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment,
    targets: { format: 'rgba16float' },
  });
  const bound = memo(
    (
      color: post.FloatTexture,
      light: post.FloatTexture,
      a: post.FloatTexture,
      b: post.FloatTexture,
    ) => [
      pipeline.with(
        root.createBindGroup(layout, { source: color, previous: b, glow: light, sampler }),
      ),
      pipeline.with(
        root.createBindGroup(layout, { source: color, previous: a, glow: light, sampler }),
      ),
    ],
  );
  const output = { colorAttachments: { view: screen.views.a } };
  bound(source(), glow.output, screen.textures.a, screen.textures.b);

  let index = 0;
  let valid = false;

  return {
    initAsync: () => pipeline.initAsync(),
    get output() {
      return index === 0 ? screen.textures.a : screen.textures.b;
    },
    reset() {
      valid = false;
    },
    record(frame: Frame, values: { halfLife: number; bloom: number; vertical: number }) {
      if (screen.fit()) valid = false;
      state.retained =
        valid && values.halfLife > 0 ? Math.exp((-Math.LN2 * frame.dt) / values.halfLife) : 0;
      state.glow = values.bloom;
      state.vertical = values.vertical;
      uniform.write(state);
      index = 1 - index;
      output.colorAttachments.view = index === 0 ? screen.views.a : screen.views.b;
      const pass = frame.encoder.beginRenderPass(output);
      pass.setPipeline(bound(source(), glow.output, screen.textures.a, screen.textures.b)[index]);
      pass.draw(3);
      pass.end();
      valid = true;
    },

    destroy() {
      screen.destroy();
    },
  };
}
