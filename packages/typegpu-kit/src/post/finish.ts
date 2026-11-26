import { memo, passInto } from '../memo.ts';
import { adopt, type AnyRenderPass } from '../passes.ts';
import { type SizeSource } from '../targets.ts';
import { linearToSrgb, srgbToLinear } from '@typegpu/color';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuCommandEncoder,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';
import { encoderOf, type Frame } from '../frames.ts';
import type * as profiler from '../profiler.ts';
import { current, type Exposure } from './exposure.ts';
import { fitRegion, within, type FloatTexture, type Rendered, type Source } from './post.ts';
import { sourceOf } from './source.ts';
import { dither, neutral, type Tone } from './tone.ts';

export const Settings = d.struct({
  /** Multiplies the source before tone mapping. */
  exposure: d.f32,
  /**
   * Sets corner darkening from 0 for none to 1 for full darkening. With premultiplied alpha,
   * corners fade out.
   */
  vignette: d.f32,
  /** Sets dither amplitude in 8-bit steps of the stored output. A value of 0 disables it. */
  dither: d.f32,
  /** Scales the glow from `options.glow` added to the source. */
  glowIntensity: d.f32,
});

export const defaults: d.Infer<typeof Settings> = {
  exposure: 1,
  vignette: 0.35,
  dither: 1,
  glowIntensity: 0.12,
};

/**
 * This type selects source filtering. `linear` interpolates texels. `sharp` keeps texels square
 * and blends one-pixel seams, so fractional pixel-art scaling avoids uneven texel widths and
 * scrolling shimmer.
 */
export type Filter = 'linear' | 'sharp';

export interface Options {
  /** Sets the output format, such as the canvas format. */
  readonly format: GPUTextureFormat;
  /**
   * Provides the picture at any size: a rendered picture, such as `aa` or `blur`, or a function
   * returning a texture, such as `() => screen.textures.hdr`. `draw` reads it on every call, so
   * the texture can be rebuilt.
   * @cpu
   */
  readonly source: Source;
  /**
   * Selects the source region from its top-left corner, such as `regions.hdr` when drawing below
   * full resolution. `draw` reads it on every call. Default the region of a rendered `source`,
   * else the whole texture.
   */
  readonly region?: SizeSource;
  /**
   * Adds a bloom, or a glow of your own such as an emission target, before tone mapping. `draw`
   * reads its `output` and `region` on every call. Default none.
   */
  readonly glow?: Rendered;
  /**
   * Multiplies `settings.exposure` by an exposure adapted to the picture on the GPU, such as one
   * from `exposure.create` recorded before this pass. Default none.
   */
  readonly exposure?: Pick<Exposure, 'inject'>;
  /**
   * Maps linear source colours to display range with `aces`, `neutral`, `reinhard` or a custom
   * curve. Use `null` for sources already in display range. Default `neutral`.
   * @gpu
   */
  readonly tone?: Tone | null;
  /**
   * Selects `linear` filtering or `sharp` filtering for fractionally scaled pixel art. `sharp`
   * keeps texels square and blends seams. Default `linear`.
   */
  readonly filter?: Filter;
  /**
   * Applies the sRGB transfer function in the shader. Use false for encoded colours, such as
   * pixel art painted in screen colours. Formats ending in `-srgb` encode on storage. Default
   * true unless `format` ends in `-srgb`.
   */
  readonly encode?: boolean;
  /**
   * Grades or tints the colour after tone mapping and before the vignette, encoding and dither.
   * It receives the colour and output UV coordinates from 0 to 1. Read other pixels with
   * `finish.picture(uv)` and glow with `finish.glow(uv)`. Default none.
   * @gpu
   */
  readonly grade?: (color: d.v3f, uv: d.v2f) => d.v3f;
  /**
   * Selects output alpha handling. `premultiplied` requires premultiplied source colour,
   * preserves alpha and raises it where glow adds light. The output remains premultiplied so a
   * surface can show its background. Default `opaque`, which writes alpha 1.
   */
  readonly alpha?: 'opaque' | 'premultiplied';
  /**
   * Sets initial values over `defaults`. Default `defaults`, with exposure 1, vignette 0.35,
   * dither 1 and glowIntensity 0.12.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Finish {
  /**
   * Holds `defaults` with overrides from `settings`. Use `patch` to upload only the fields you
   * supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /** Compiles the pipeline. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Draws into the caller's pass, TypeGPU's or a raw one, which usually targets the canvas. */
  draw(pass: AnyRenderPass): void;
  /**
   * Opens a pass on `view`, such as `surface.view(frame)`, draws into it and ends it. A
   * `viewport` draws the picture into that rectangle and clears the rest, transparent with
   * `alpha: 'premultiplied'` and black otherwise. Optional timer writes measure the pass.
   */
  record(target: Frame | TgpuCommandEncoder, view: GPUTextureView, options?: RecordOptions): void;
}

/** A rectangle of the output in pixels, from its top-left corner. */
export interface Viewport {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface RecordOptions extends profiler.Timed {
  /** Draws into this rectangle of the view, such as a letterbox. Default the whole view. */
  readonly viewport?: Viewport;
}

const sourceLayout = tgpu.bindGroupLayout({ source: { texture: d.texture2d(d.f32) } });
const glowLayout = tgpu.bindGroupLayout({ glow: { texture: d.texture2d(d.f32) } });

const Regions = d.struct({ source: d.vec2f, glow: d.vec2f });

const outside = (): never => {
  throw new Error(
    'Call `finish.picture` and `finish.glow` only from the `grade` callback of `finish.create`.',
  );
};
const settingsOf = tgpu.accessor(Settings, outside);
const regionsOf = tgpu.accessor(Regions, outside);
const samplerOf = tgpu.accessor(d.sampler(), outside);
const sharp = tgpu.slot(false);
const glowing = tgpu.slot(false);
const adapting = tgpu.slot(false);
const premultiplied = tgpu.slot(false);
const toneOf = tgpu.slot<Tone | null>(neutral);

const sourceAt = (uv: d.v2f) => {
  'use gpu';
  const part = regionsOf.$.source;
  const size = d.vec2f(std.textureDimensions(sourceLayout.$.source));

  let at = (uv * part) / size;
  if (sharp.$) {
    const texel = uv * part;
    const seam = std.floor(texel + 0.5);
    const blend = std.clamp((texel - seam) / std.fwidth(texel), d.vec2f(-0.5), d.vec2f(0.5));
    at = (seam + blend) / size;
  }

  return std.textureSampleLevel(sourceLayout.$.source, samplerOf.$, within(at, part, size), 0);
};

/**
 * Samples glow at output UV coordinates from 0 to 1. It returns linear colour before applying
 * `glowIntensity`, or zero without glow. Call this only from the final pass `grade` function.
 */
export const glow = (uv: d.v2f) => {
  'use gpu';
  const part = regionsOf.$.glow;
  if (!glowing.$) return d.vec3f();

  const size = d.vec2f(std.textureDimensions(glowLayout.$.glow));
  const at = within((uv * part) / size, part, size);
  return std.textureSampleLevel(glowLayout.$.glow, samplerOf.$, at, 0).xyz;
};

const shade = (uv: d.v2f) => {
  'use gpu';
  const read = sourceAt(uv);
  let color = read.xyz;
  let coverage = d.f32(1);
  if (premultiplied.$) coverage = read.w;

  if (glowing.$) {
    const light = glow(uv) * settingsOf.$.glowIntensity;
    color += light;
    if (premultiplied.$) {
      coverage = std.saturate(coverage + std.max(light.x, std.max(light.y, light.z)));
    }
  }

  // Unpremultiply colour before tone mapping and grading
  if (premultiplied.$) color = color / std.max(coverage, 0.0001);
  let exposure = settingsOf.$.exposure;
  if (adapting.$) exposure = exposure * current.$;
  color = color * exposure;
  if (toneOf.$ !== null) color = toneOf.$(color);
  return d.vec4f(color, coverage);
};

/**
 * Samples the source with glow, exposure and tone mapping at output UV coordinates from 0 to 1.
 * This is the colour passed to `grade`. Call it only from the final pass `grade` function.
 */
export const picture = (uv: d.v2f) => {
  'use gpu';
  return shade(uv).xyz;
};

/**
 * Creates the final pass of a frame. It reads the source, adds glow, tone maps, grades, applies
 * the vignette, encodes and dithers. It draws one triangle into the caller pass.
 *
 * @allocates on resize: a bind group for a new source or glow
 * @owns 2 buffers, a sampler, a pipeline, 2 bind groups: the settings, the regions it reads, a
 * group for the glow and one for each texture the source returns
 * @records `record` into a pass of its own on the view you give, usually the canvas's, or `draw`
 * into the caller's pass, after the bloom's `record`.
 * @param options.source the picture, rendered or a texture, read at every `draw`
 * @param options.region the part of the source that holds the picture, read at every `draw`
 * @param options.glow a bloom, or a glow of your own, added before tone mapping
 * @param options.exposure an adapted exposure, read on the GPU
 * @returns a final pass you draw every frame
 */
export function create(root: TgpuRoot, options: Options): Finish {
  const { format, tone = neutral, filter = 'linear', grade, glow: added } = options;
  const source = sourceOf(options);
  const adapted = options.exposure;
  const srgbOutput = format.endsWith('-srgb');
  const encode = options.encode ?? !srgbOutput;
  const covered = options.alpha === 'premultiplied';

  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });
  const regions = root.createUniform(Regions);
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const fragment = tgpu.fragmentFn({
    in: { position: d.builtin.position, uv: d.vec2f },
    out: d.vec4f,
  })(({ position, uv }) => {
    'use gpu';
    const shaded = shade(uv);
    let coverage = shaded.w;
    let color = shaded.xyz;
    if (grade !== undefined) color = grade(color, uv);

    const edge = uv - 0.5;
    const fade = std.saturate(1 - std.dot(edge, edge) * 2 * settings.$.vignette);
    if (covered) coverage = coverage * fade;
    else color = color * fade;
    if (encode) color = linearToSrgb(color);

    // Apply noise in encoded colour space. An -srgb output encodes during storage, so decode
    // before writing. Scale noise by coverage to keep transparent pixels clear
    const steps = settings.$.dither;
    if (srgbOutput)
      color = srgbToLinear(dither(linearToSrgb(color), position.xy, steps) * coverage);
    else color = dither(color, position.xy, steps) * coverage;

    return d.vec4f(color, coverage);
  });

  const pipeline = (adapted ? root.pipe(adapted.inject()) : root)
    .with(settingsOf, settings)
    .with(regionsOf, regions)
    .with(samplerOf, () => sampler.$)
    .with(sharp, filter === 'sharp')
    .with(glowing, added !== undefined)
    .with(adapting, adapted !== undefined)
    .with(premultiplied, covered)
    .with(toneOf, tone)
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: { format },
    })
    .$name('finish');

  const glowGroup = memo((texture: FloatTexture) =>
    root.createBindGroup(glowLayout, { glow: texture }),
  );

  // Keep a group for every source texture, so any number of sources taking turns binds nothing
  // new once each has drawn
  const bound = new WeakMap<FloatTexture, (light?: FloatTexture) => typeof pipeline>();
  const boundTo = (texture: FloatTexture) => {
    let bind = bound.get(texture);
    if (!bind) {
      const read = root.createBindGroup(sourceLayout, { source: texture });
      bind = memo((light?: FloatTexture) =>
        light ? pipeline.with(read).with(glowGroup(light)) : pipeline.with(read),
      );
      bound.set(texture, bind);
    }
    return bind;
  };

  const written = { source: [0, 0] as [number, number], glow: [0, 0] as [number, number] };
  const measure = (texture: FloatTexture, light: FloatTexture | undefined) => {
    const moved = fitRegion(written.source, texture, source.region);
    const glowMoved = light !== undefined && fitRegion(written.glow, light, added?.region);
    if (moved || glowMoved) regions.write(written);
  };

  const draw = (pass: AnyRenderPass) => {
    const target = adopt(pass);
    const texture = source.output;
    const light = added?.output;
    measure(texture, light);

    target.setPipeline(boundTo(texture)(light));
    target.draw(3);
  };

  const into = passInto(covered ? [0, 0, 0, 0] : [0, 0, 0, 1]);

  return {
    settings,

    initAsync: () => pipeline.initAsync(),

    draw,

    record(target, view, recording) {
      const pass = encoderOf(target).beginRenderPass(into(view, recording?.timestampWrites));
      const viewport = recording?.viewport;
      if (viewport) {
        pass.setViewport(viewport.x, viewport.y, viewport.width, viewport.height, 0, 1);
      }
      draw(pass);
      pass.end();
    },
  };
}
