import { bloom, finish, targets, type Frame, type Surface } from '@typegpu/kit';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';

/** Bright enough at the crests that bloom has something to spread. */
const TINT = d.vec4f(0.6, 1.8, 3, 1);

/** The ripple How the kit works builds, drawn with or without the glow it adds. */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const screen = targets.create(root, surface, { hdr: { format: 'rgba16float', sampled: true } });
  const hdr = () => screen.textures.hdr;
  const aspect = root.createUniform(d.f32);
  const measure = () => {
    const [width, height] = screen.textures.hdr.props.size;
    aspect.write(width! / height!);
  };
  measure();
  const phase = root.createMutable(d.f32);

  const advance = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    phase.$ += 0.05;
  });
  const rings = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    const at = (uv - 0.5) * d.vec2f(aspect.$, 1);
    const wave = std.sin(std.length(at) * 40 - phase.$) * 0.5 + 0.5;
    return TINT * (wave * wave * wave);
  });

  const advancing = root.createComputePipeline({ compute: advance });
  const drawing = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: rings,
    targets: { format: 'rgba16float' },
  });
  const glow = bloom.create(root, { source: hdr });
  const final = finish.create(root, {
    format: surface.format,
    source: hdr,
    glow,
    settings: { vignette: 0 },
  });
  await Promise.all([
    advancing.initAsync(),
    drawing.initAsync(),
    glow.initAsync(),
    final.initAsync(),
  ]);
  const { glowIntensity } = finish.defaults;
  let glowing = true;

  return {
    /** Adds the glow, or draws the rings as the plain frame does. */
    glow(on: boolean) {
      glowing = on;
      final.settings.patch({ glowIntensity: on ? glowIntensity : 0 });
    },

    frame(frame: Frame) {
      if (screen.fit()) measure();

      // A redraw without time passing leaves the ripple where it was
      if (frame.dt > 0) {
        const compute = frame.encoder.beginComputePass();
        compute.setPipeline(advancing);
        compute.dispatchWorkgroups(1);
        compute.end();
      }

      const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      pass.setPipeline(drawing);
      pass.draw(3);
      pass.end();

      if (glowing) glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      glow.destroy();
      screen.destroy();
    },
  };
}
