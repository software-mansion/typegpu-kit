import { images, type Frame, type Surface } from '@typegpu/kit';
import { linearToSrgb } from '@typegpu/color';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Loading } from 'typegpu-kit-examples/contract';

export const LEVELS = ['mips', 'none'] as const;
export type Levels = (typeof LEVELS)[number];
export const FILTERS = ['16×', '1×'] as const;
export type Filter = (typeof FILTERS)[number];
export const COLORS = ['srgb', 'linear'] as const;
export type Color = (typeof COLORS)[number];

/** The eye's height over the ground, in metres, and the side of one tile of sand. */
const EYE = 0.6;
const TILE = 2;

/** Poly Haven's gravelly sand, which the quiver trees' ground also reads, 0.9 MB. */
export async function load({ read, decode }: Pick<Loading, 'read' | 'decode'>) {
  const bytes = new Uint8Array(await read('quiver-trees/textures/gravelly-sand.webp'));
  return { sand: await images.decode({ bytes, mimeType: 'image/webp' }, decode) };
}

export type TexturesOptions = Awaited<ReturnType<typeof load>>;

/** The width of the surface over its height. */
const wide = tgpu.accessor(d.f32);

const layout = tgpu.bindGroupLayout({
  picture: { texture: d.texture2d(d.f32) },
  smooth: { sampler: 'filtering' },
});

/** Ground seen from `EYE` metres up, under a horizon a seventh of the way down. */
const ground = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
  'use gpu';
  const ray = std.normalize(d.vec3f((uv.x * 2 - 1) * 0.55 * wide.$, 0.22 - uv.y * 1.1, -1));
  // Every pixel samples, as derivatives need; the sky's are faded out
  const hit = ray * (EYE / std.max(-ray.y, 0.002));
  const color = std.textureSample(layout.$.picture, layout.$.smooth, hit.xz / TILE).rgb;
  // Fades the ground into the page's dark ground at the horizon
  const fade = std.saturate(-ray.y * 12);
  return d.vec4f(std.saturate(linearToSrgb(color)) * fade, fade);
});

/** A plane of sand to the horizon, through a texture with or without mips, as colour or data, and a sampler. */
export function createScene(root: TgpuRoot, surface: Surface, options: TexturesOptions) {
  const made = {
    srgb: {
      mips: images.create(root, options.sand),
      none: images.create(root, options.sand, { mips: false }),
    },
    linear: {
      mips: images.create(root, options.sand, { color: 'linear' }),
      none: images.create(root, options.sand, { color: 'linear', mips: false }),
    },
  };
  const samplers = {
    '16×': images.sampler(root, { anisotropy: 16 }),
    '1×': images.sampler(root),
  };
  // A group for each texture and sampler, made once
  const filtered = (texture: images.Texture) => {
    const picture = texture.createView(d.texture2d(d.f32));
    return {
      '16×': root.createBindGroup(layout, { picture, smooth: samplers['16×'] }),
      '1×': root.createBindGroup(layout, { picture, smooth: samplers['1×'] }),
    };
  };
  const groups = {
    srgb: { mips: filtered(made.srgb.mips), none: filtered(made.srgb.none) },
    linear: { mips: filtered(made.linear.mips), none: filtered(made.linear.none) },
  };
  let written = surface.width / surface.height;
  const aspect = root.createUniform(d.f32, written);
  const pipeline = root.with(wide, aspect).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: ground,
    targets: { format: surface.format },
  });

  const chosen = { levels: 'mips' as Levels, filter: '16×' as Filter, color: 'srgb' as Color };

  return {
    /** Chooses what the next frame draws. */
    show(next: Partial<typeof chosen>) {
      Object.assign(chosen, next);
    },

    frame(frame: Frame) {
      if (surface.width / surface.height !== written) {
        written = surface.width / surface.height;
        aspect.write(written);
      }
      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), clearValue: [0, 0, 0, 0] },
      });
      const { color, levels, filter } = chosen;
      pass.setPipeline(pipeline);
      pass.setBindGroup(groups[color][levels][filter]);
      pass.draw(3);
      pass.end();
    },

    destroy() {
      for (const texture of [...Object.values(made.srgb), ...Object.values(made.linear)]) {
        texture.destroy();
      }
      aspect.buffer.destroy();
    },
  };
}
