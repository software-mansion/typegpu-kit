import { memo, passInto } from '../memo.ts';
import { type SizeSource } from '../targets.ts';
import { srgbToLinear } from '@typegpu/color';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuCommandEncoder,
  type TgpuRenderCommands,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';
import { encoderOf, type Frame } from '../frames.ts';
import type * as profiler from '../profiler.ts';
import { fitRegion, within, type FloatTexture, type Source } from './post.ts';
import { sourceOf } from './source.ts';

export const Settings = d.struct({
  /** Sets the minimum contrast to smooth, as a fraction of the brightest local luma. */
  edge: d.f32,
  /** Sets the minimum contrast to smooth in dark areas where the `edge` fraction is too small. */
  floor: d.f32,
  /** Sets how much to soften detail smaller than a pixel, from 0 to 1. */
  subpixel: d.f32,
});

/** These settings match FXAA 3.11 at its default quality. */
export const defaults: d.Infer<typeof Settings> = {
  edge: 0.166,
  floor: 0.0833,
  subpixel: 0.75,
};

export interface Options {
  /** Sets the output format, such as the canvas format. */
  readonly format: GPUTextureFormat;
  /**
   * Provides an encoded picture in display range, such as an `rgba8unorm` target drawn by the
   * final pass, as a function returning the texture or a rendered picture. `draw` reads it on
   * every call and throws for `-srgb` formats, which decode on sampling.
   * @cpu
   */
  readonly source: Source;
  /**
   * Selects the source region from its top-left corner, such as `regions.ldr` when drawing below
   * full resolution. FXAA stretches it over the output. `draw` reads it on every call. Default the
   * region of a rendered `source`, else the whole texture.
   */
  readonly region?: SizeSource;
  /**
   * Sets initial values over `defaults`. Default `defaults`, with edge 0.166, floor 0.0833 and
   * subpixel 0.75.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Fxaa {
  /**
   * Holds `defaults` with overrides from `settings`. Use `patch` to upload only the fields you
   * supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /** Compiles the pipeline. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Draws into the caller's pass, which usually targets the canvas. */
  draw(pass: TgpuRenderCommands): void;
  /**
   * Opens a pass on `view`, such as `surface.view(frame)`, draws into it and ends it. Optional
   * timer writes measure the pass.
   */
  record(target: Frame | TgpuCommandEncoder, view: GPUTextureView, options?: profiler.Timed): void;
}

/**
 * These search steps use texels in each direction along an edge. The last step estimates the
 * remaining length when the edge has not ended.
 */
const STEPS = [1, 1.5, 2, 4];
const GUESS = 12;
const LUMA = d.vec3f(0.299, 0.587, 0.114);

const sourceLayout = tgpu.bindGroupLayout({ source: { texture: d.texture2d(d.f32) } });

/**
 * Creates an FXAA 3.11 pass at the default quality. It smooths edges after the source is in
 * display range. Each pixel follows its edge in both directions to estimate its length, then
 * samples across the edge. It draws one triangle into the caller pass.
 *
 * @allocates on resize: a bind group for a new source
 * @owns 2 buffers, a sampler, a pipeline, a bind group: the settings, the region it reads and the
 * source's group
 * @records `record` into a pass of its own on the view you give, usually the canvas's, or `draw`
 * into the caller's pass, after the pass that drew the source.
 * @param options.source the picture as the screen shows it, read at every `draw`
 * @param options.region the part of the source that holds the picture, read at every `draw`
 * @returns a pass you draw every frame
 */
export function create(root: TgpuRoot, options: Options): Fxaa {
  const { format } = options;
  const source = sourceOf(options);
  const srgbOutput = format.endsWith('-srgb');
  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });
  const part = root.createUniform(d.vec2f);
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const read = (at: d.v2f) => {
    'use gpu';
    const size = d.vec2f(std.textureDimensions(sourceLayout.$.source));
    return std.textureSampleLevel(sourceLayout.$.source, sampler.$, within(at, part.$, size), 0);
  };
  const lumaAt = (at: d.v2f) => {
    'use gpu';
    return std.dot(read(at).xyz, LUMA);
  };

  const smoothed = (output: d.v2f) => {
    'use gpu';
    const size = d.vec2f(std.textureDimensions(sourceLayout.$.source));
    const texel = 1 / size;
    const uv = (output * part.$) / size;
    const centre = read(uv);
    const middle = std.dot(centre.xyz, LUMA);
    // Texture y increases downwards
    const up = lumaAt(uv - d.vec2f(0, texel.y));
    const down = lumaAt(uv + d.vec2f(0, texel.y));
    const left = lumaAt(uv - d.vec2f(texel.x, 0));
    const right = lumaAt(uv + d.vec2f(texel.x, 0));

    const brightest = std.max(std.max(std.max(up, down), std.max(left, right)), middle);
    const darkest = std.min(std.min(std.min(up, down), std.min(left, right)), middle);
    const contrast = brightest - darkest;
    // At zero contrast, zero thresholds would divide 0 by 0 below
    if (contrast <= 0 || contrast < std.max(settings.$.floor, brightest * settings.$.edge)) {
      return centre;
    }

    const upLeft = lumaAt(uv - texel);
    const downRight = lumaAt(uv + texel);
    const upRight = lumaAt(uv + d.vec2f(texel.x, -texel.y));
    const downLeft = lumaAt(uv + d.vec2f(-texel.x, texel.y));

    const around = (2 * (up + down + left + right) + upLeft + upRight + downLeft + downRight) / 12;
    const apart = std.smoothstep(0, 1, std.abs(around - middle) / contrast);
    const subpixel = apart * apart * settings.$.subpixel;

    // Detect horizontal edges through vertical luma differences
    const acrossY =
      std.abs(up + down - 2 * middle) * 2 +
      std.abs(upLeft + downLeft - 2 * left) +
      std.abs(upRight + downRight - 2 * right);
    const acrossX =
      std.abs(left + right - 2 * middle) * 2 +
      std.abs(upLeft + upRight - 2 * up) +
      std.abs(downLeft + downRight - 2 * down);
    const alongX = acrossY >= acrossX;

    let across = std.select(d.vec2f(texel.x, 0), d.vec2f(0, texel.y), alongX);
    let opposite = std.select(right, down, alongX);
    const before = std.select(left, up, alongX);
    if (std.abs(before - middle) >= std.abs(opposite - middle)) {
      across *= -1;
      opposite = before;
    }
    const gradient = std.abs(opposite - middle);

    const along = std.select(d.vec2f(0, texel.y), d.vec2f(texel.x, 0), alongX);
    const edge = uv + across * 0.5;
    const average = (middle + opposite) * 0.5;
    const enough = gradient * 0.25;
    let ahead = d.vec2f(edge);
    let behind = d.vec2f(edge);
    let aheadEnd = d.f32(0);
    let behindEnd = d.f32(0);
    let aheadDone = false;
    let behindDone = false;
    for (const step of tgpu.unroll(STEPS)) {
      if (!aheadDone) {
        ahead += along * step;
        aheadEnd = lumaAt(ahead) - average;
        aheadDone = std.abs(aheadEnd) >= enough;
      }
      if (!behindDone) {
        behind -= along * step;
        behindEnd = lumaAt(behind) - average;
        behindDone = std.abs(behindEnd) >= enough;
      }
    }
    if (!aheadDone) ahead = ahead + along * GUESS;
    if (!behindDone) behind = behind - along * GUESS;

    const toAhead = std.dot(ahead - uv, along) / std.dot(along, along);
    const toBehind = std.dot(uv - behind, along) / std.dot(along, along);
    const nearer = std.min(toAhead, toBehind);
    const end = std.select(behindEnd, aheadEnd, toAhead <= toBehind);
    let shift = 0.5 - nearer / (toAhead + toBehind);
    if (end < 0 === middle < average) shift = 0;
    return read(uv + across * std.max(shift, subpixel));
  };

  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    const color = smoothed(uv);
    // Decode before writing an -srgb target to prevent encoding the source twice
    if (srgbOutput) return d.vec4f(srgbToLinear(color.xyz), color.w);
    return color;
  });

  const pipeline = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: { format },
    })
    .$name('fxaa');

  const bound = memo((texture: FloatTexture) => {
    const { format: stored } = texture.props;
    if (stored.endsWith('-srgb')) {
      throw new Error(
        `FXAA uses encoded colour to detect edges. Sampling ${stored} decodes that colour. Draw the picture into ${stored.replace('-srgb', '')} instead.`,
      );
    }
    return pipeline.with(root.createBindGroup(sourceLayout, { source: texture }));
  });

  const written: [number, number] = [0, 0];
  const draw = (pass: TgpuRenderCommands) => {
    const texture = source.output;
    if (fitRegion(written, texture, source.region)) part.write(written);
    pass.setPipeline(bound(texture));
    pass.draw(3);
  };

  const into = passInto();

  return {
    settings,

    initAsync: () => pipeline.initAsync(),

    draw,

    record(target, view, timing) {
      const pass = encoderOf(target).beginRenderPass(into(view, timing?.timestampWrites));
      draw(pass);
      pass.end();
    },
  };
}
