import { baker, cameras, pipelines, tone, video, type Feed } from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { perlin3d } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type TgpuCommandEncoder,
  type TgpuRenderCommands,
  type TgpuRoot,
} from 'typegpu';

/** The cold glass the reflection sits behind. */
const GLASS = d.vec3f(0.45, 0.62, 1);
/** How much of the camera's own colour shows through the glass's tint. */
const COLOR = 0.3;

export const Framing = d.struct({
  /** Linear brightness of the reflection. */
  brightness: d.f32,
  /** 1 while a camera streams, which the reflection then samples at full resolution. */
  live: d.u32,
});

/**
 * Draws a dim reflection behind the embers over the feed's rectangle of the world, from the origin
 * to its size in texels: the camera's newest frame in place while one streams, otherwise the feed,
 * into which it paints drifting smoke.
 */
export function createMirror(
  root: TgpuRoot,
  options: {
    readonly feed: Feed;
    readonly live: video.LiveView;
    readonly view: cameras.CameraBase;
    readonly format: GPUTextureFormat;
  },
) {
  const { feed, live, view, format } = options;
  const [width, height] = feed.texture.props.size;
  const framing = root.createUniform(Framing);
  /** Seconds the smoke has drifted. */
  const drift = root.createUniform(d.f32);
  const reading = feed.texture.createView(d.texture2d(d.f32));
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  /** The feed's corners in a strip, with feed coordinates from its top-left corner. */
  const corner = tgpu.vertexFn({
    in: { i: d.builtin.vertexIndex },
    out: { position: d.builtin.position, uv: d.vec2f },
  })(({ i }) => {
    'use gpu';
    const uv = d.vec2f(d.f32(i & 1), d.f32(i >>> 1));
    const at = d.vec2f(uv.x * width, (1 - uv.y) * height);
    return { position: cameras.toClip(d.vec3f(at, 0)), uv };
  });

  const reflect = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    let encoded = std.textureSampleLevel(reading.$, sampler.$, uv, 0).rgb;
    if (framing.$.live === 1) {
      encoded = video.sample(uv).rgb;
    }
    const seen = srgbToLinear(encoded);
    const grey = tone.luminance(seen);
    const glass = std.mix(GLASS * grey, seen, COLOR);
    return d.vec4f(glass * framing.$.brightness, 1);
  });

  const reflecting = root
    .pipe(live.inject())
    .pipe(view.inject())
    .createRenderPipeline({
      vertex: corner,
      fragment: reflect,
      targets: { format },
      primitive: { topology: 'triangle-strip' },
    });

  /** Smoke shapes slow enough to read as figures, with edges that move. */
  const smoke = baker.create(root, feed.texture, {
    texel: (uv) => {
      'use gpu';
      const at = d.vec3f(uv.x * (width / height) * 2.2, uv.y * 2.2, drift.$ * 0.22);
      const body = perlin3d.sample(at) + perlin3d.sample(at * 2.1 + 7) * 0.45;
      const shape = std.smoothstep(0.02, 0.1, body);
      const light = 0.06 + shape * (0.55 + 0.3 * (1 - uv.y));
      return d.vec4f(d.vec3f(light), 1);
    },
  });

  return {
    framing,
    initAsync: () => pipelines.compile([reflecting, smoke]),

    /** Draws the reflection into the caller's pass. */
    draw(pass: TgpuRenderCommands) {
      pass.setPipeline(reflecting);
      pass.draw(4);
    },

    /** Paints smoke at `seconds` into the feed's texture, in place of a camera frame. */
    paint(encoder: TgpuCommandEncoder, seconds: number) {
      drift.write(seconds);
      smoke.record(encoder);
    },
  };
}
