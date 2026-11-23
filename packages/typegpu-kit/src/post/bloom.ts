import * as pipelines from '../pipelines.ts';
import * as targets from '../targets.ts';
import * as profiler from '../profiler.ts';
import { memo, type PassDescriptor } from '../memo.ts';
import { type SizeSource } from '../targets.ts';
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
import { fitRegion, within, type FloatTexture, type Rendered, type Source } from './post.ts';
import { sourceOf } from './source.ts';
import { luminance } from './tone.ts';

/** WebGPU guarantees render, filter and blend support for this format. */
const FORMAT = 'rgba16float';

export const Settings = d.struct({
  /** Sets the brightness threshold for glow. At 0, every colour contributes some glow. */
  threshold: d.f32,
  /** Sets how far below the threshold glow begins, as a fraction of the threshold. */
  knee: d.f32,
  /** Sets the distance between upsample taps, in texels of the smaller level. */
  spread: d.f32,
});

export const defaults: d.Infer<typeof Settings> = {
  threshold: 1,
  knee: 0.5,
  spread: 1,
};

export interface Options {
  /**
   * Provides the HDR picture: a rendered picture, such as `aa`, or a function returning a
   * texture, such as `() => screen.textures.hdr`. `record` reads it on every call, so the texture
   * can be rebuilt.
   * @cpu
   */
  readonly source: Source;
  /**
   * Selects the source region from its top-left corner, such as `regions.hdr` when drawing below
   * full resolution. `record` reads it on every call. Default the region of a rendered `source`,
   * else the whole texture.
   */
  readonly region?: SizeSource;
  /**
   * Sets the number of levels, each half the size of the preceding level. Small sources stop at
   * one texel. More levels spread glow farther. Default 6.
   */
  readonly levels?: number;
  /**
   * Selects the linear source colour that contributes glow. This `'use gpu'` function runs once
   * per texel of the first level. It can call `threshold` and carry fills with `pipe`. Default
   * `threshold`.
   * @gpu
   */
  readonly bright?: (color: d.v3f) => d.v3f;
  /**
   * Sets initial values over `defaults`. Default `defaults`, with threshold 1, knee 0.5 and
   * spread 1.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Bloom extends Rendered, Disposable {
  /**
   * Holds `defaults` with overrides from `settings`. Use `patch` to upload only the fields you
   * supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /**
   * Provides the glow at half the source size in mip 0 of an `rgba16float` texture. Use it as
   * the final pass `glow` or another pass `source`. Rebuilding the chain replaces the texture.
   */
  readonly output: FloatTexture;
  /**
   * Gives the glow region from the top-left corner of `output`, set by the last `record` call.
   */
  readonly region: SizeSource;
  /** Compiles the bloom pipelines. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Records bloom passes after the source is written. Optional timer writes measure the passes
   * together.
   */
  record(target: Frame | TgpuCommandEncoder, options?: profiler.Timed): void;
  /** Destroys the glow texture. */
  destroy(): void;
}

const layout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  sampler: { sampler: 'filtering' },
  /** Measures the source region in texels; `uv` spans that region. */
  region: { uniform: d.vec2f },
});

const settings = tgpu.accessor(Settings, (): never => {
  throw new Error(
    'Call `bloom.threshold` only from the `bright` callback of `bloom.create` so it can read bloom settings.',
  );
});
const brightOf = tgpu.slot<(color: d.v3f) => d.v3f>();

const tap = (uv: d.v2f, offset: d.v2f) => {
  'use gpu';
  const size = d.vec2f(std.textureDimensions(layout.$.source));
  const region = layout.$.region;
  const at = within((uv * region + offset) / size, region, size);

  return std.textureSampleLevel(layout.$.source, layout.$.sampler, at, 0).xyz;
};

const Boxes = d.struct({
  inner: d.vec3f,
  topLeft: d.vec3f,
  topRight: d.vec3f,
  bottomLeft: d.vec3f,
  bottomRight: d.vec3f,
});

const boxes = (uv: d.v2f) => {
  'use gpu';
  const a = tap(uv, d.vec2f(-2, -2));
  const b = tap(uv, d.vec2f(0, -2));
  const c = tap(uv, d.vec2f(2, -2));
  const e = tap(uv, d.vec2f(-2, 0));
  const f = tap(uv, d.vec2f());
  const g = tap(uv, d.vec2f(2, 0));
  const h = tap(uv, d.vec2f(-2, 2));
  const i = tap(uv, d.vec2f(0, 2));
  const j = tap(uv, d.vec2f(2, 2));

  const inner =
    tap(uv, d.vec2f(-1, -1)) +
    tap(uv, d.vec2f(1, -1)) +
    tap(uv, d.vec2f(-1, 1)) +
    tap(uv, d.vec2f(1, 1));

  return Boxes({
    inner: inner * 0.25,
    topLeft: (a + b + e + f) * 0.25,
    topRight: (b + c + f + g) * 0.25,
    bottomLeft: (e + f + h + i) * 0.25,
    bottomRight: (f + g + i + j) * 0.25,
  });
};

/**
 * Weight by 1 / (1 + luma) to prevent a single bright texel from causing flicker. Negative
 * channels, which a source may hold, are cut first, so the weight stays positive.
 */
const karis = (color: d.v3f, weight: number) => {
  'use gpu';
  const kept = std.max(color, d.vec3f());
  return d.vec4f(kept, 1) * (weight / (1 + luminance(kept)));
};

/**
 * Extracts colour above the brightness threshold for the default `bright` function. It uses the
 * brightest channel and eases the threshold over the fraction set by `knee`. The retained
 * fraction scales the colour. It reads bloom settings, so call it only from `bright`.
 */
export const threshold = (color: d.v3f) => {
  'use gpu';
  const brightness = std.max(color.x, std.max(color.y, color.z));
  const knee = settings.$.threshold * settings.$.knee + 0.0001;
  const soft = std.clamp(brightness - settings.$.threshold + knee, 0, 2 * knee);
  const kept = std.max((soft * soft) / (4 * knee), brightness - settings.$.threshold);

  return color * (kept / std.max(brightness, 0.0001));
};

const prefilter = /* @__PURE__ */ tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({
  uv,
}) => {
  'use gpu';
  const box = boxes(uv);
  const sum =
    karis(box.inner, 0.5) +
    karis(box.topLeft, 0.125) +
    karis(box.topRight, 0.125) +
    karis(box.bottomLeft, 0.125) +
    karis(box.bottomRight, 0.125);

  return d.vec4f(brightOf.$(sum.xyz / sum.w), 1);
});

const downsample = /* @__PURE__ */ tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({
  uv,
}) => {
  'use gpu';
  const box = boxes(uv);
  const corners = box.topLeft + box.topRight + box.bottomLeft + box.bottomRight;

  return d.vec4f(box.inner * 0.5 + corners * 0.125, 1);
});

const upsample = /* @__PURE__ */ tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({
  uv,
}) => {
  'use gpu';
  const r = settings.$.spread;
  const edges =
    tap(uv, d.vec2f(-r, 0)) +
    tap(uv, d.vec2f(r, 0)) +
    tap(uv, d.vec2f(0, -r)) +
    tap(uv, d.vec2f(0, r));
  const corners =
    tap(uv, d.vec2f(-r, -r)) +
    tap(uv, d.vec2f(r, -r)) +
    tap(uv, d.vec2f(-r, r)) +
    tap(uv, d.vec2f(r, r));

  return d.vec4f((tap(uv, d.vec2f()) * 4 + edges * 2 + corners) / 16, 1);
});

/**
 * Creates bloom from a chain of half-size textures. Each level downsamples with 13 taps, then a
 * tent filter adds it to the next larger level. The result is mip 0 of `output`, at half the
 * source size. With `region`, passes draw and sample only that region; changing the region
 * allocates nothing.
 *
 * @allocates on resize: a new chain and the bind groups of its passes; so does a new source
 * @owns targets, a sampler, 3 pipelines, 2 bind groups and a buffer per level: the chain of glow
 * levels, the settings, a group for each pass, and the region each pass reads
 * @records `record` after whatever wrote the source; then `output` holds the glow.
 * @param options.source the HDR picture, read at every `record`
 * @param options.region the part of the source that holds the picture, read at every `record`
 * @returns a bloom you record every frame and destroy
 */
export function create(root: TgpuRoot, options: Options): Bloom {
  const { levels = 6 } = options;
  const source = sourceOf(options);
  // The source texture as `record` or `output` last took it, which the chain follows
  let picture = source.output;

  const half: SizeSource = {
    get width() {
      return picture.props.size[0] / 2;
    },
    get height() {
      return (picture.props.size[1] ?? 1) / 2;
    },
  };

  const chain = targets.create(root, half, {
    glow: { format: FORMAT, mips: levels, sampled: true },
  });
  const uniform = root.createUniform(Settings, { ...defaults, ...options.settings });
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const configured = root.with(settings, uniform).with(brightOf, options.bright ?? threshold);
  const pipeline = (fragment: typeof downsample, blend?: GPUBlendState) =>
    configured.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: blend ? { format: FORMAT, blend } : { format: FORMAT },
    });

  const first = pipeline(prefilter);
  const down = pipeline(downsample);
  const up = pipeline(upsample, pipelines.blend.add);

  // Keep region uniforms allocated and write only changed values, in texels
  const reads = Array.from({ length: levels + 1 }, () =>
    root.createBuffer(d.vec2f).$usage('uniform'),
  );
  const written = Array.from({ length: levels + 1 }, () => [0, 0]);
  const drawn = Array.from({ length: levels + 1 }, () => [0, 0]);

  const group = (input: FloatTexture | GPUTextureView, read: number) =>
    root.createBindGroup(layout, { source: input, sampler, region: reads[read] });

  // Keep each bind group on its pass pipeline so a frame only sets the pipeline on the pass
  const passesOf = memo((texture: FloatTexture, views: readonly GPUTextureView[]) => ({
    downPipelines: views.map((_, level) =>
      (level === 0 ? first : down).with(group(level === 0 ? texture : views[level - 1], level)),
    ),
    upPipelines: views.map((view, level) => up.with(group(view, level + 1))),
    downPasses: views.map((view): PassDescriptor => ({
      colorAttachments: { view, loadOp: 'clear', storeOp: 'store' },
    })),
    upPasses: views.map((_, level): PassDescriptor => ({
      colorAttachments: { view: views[Math.max(0, level - 1)], loadOp: 'load', storeOp: 'store' },
    })),
  }));

  const measure = (count: number) => {
    fitRegion(drawn[0], picture, source.region);

    for (let level = 0; level < count; level++) {
      const parent = drawn[level];
      const child = drawn[level + 1];
      child[0] = Math.min(Math.max(1, chain.width >> level), Math.ceil(parent[0] / 2));
      child[1] = Math.min(Math.max(1, chain.height >> level), Math.ceil(parent[1] / 2));
    }

    for (let read = 0; read <= count; read++) {
      const width = drawn[read][0];
      const height = drawn[read][1];
      const last = written[read];
      if (last[0] === width && last[1] === height) continue;

      reads[read].write(d.vec2f(width, height));
      last[0] = width;
      last[1] = height;
    }
  };

  const fitted = () => {
    picture = source.output;
    chain.fit();
    return chain.levels.glow;
  };

  const follow = () => {
    const views = fitted();
    measure(views.length);
    return passesOf(picture, views);
  };
  follow();

  const destroy = () => chain.destroy();

  const glowRegion: SizeSource = {
    get width() {
      return drawn[1][0];
    },

    get height() {
      return drawn[1][1];
    },
  };

  return {
    settings: uniform,

    get output() {
      fitted();
      return chain.textures.glow;
    },

    region: glowRegion,

    initAsync: () => pipelines.compile([first, down, up]),

    record(target, timing) {
      const encoder = encoderOf(target);
      const timestampWrites = timing?.timestampWrites;
      const { downPipelines, upPipelines, downPasses, upPasses } = follow();
      const { opening, closing } = profiler.split(timestampWrites);

      // Place timer writes only on the first and last passes. A single-level chain needs both
      // writes on one pass
      const count = downPasses.length;
      downPasses[0].timestampWrites = count > 1 ? opening : timestampWrites;
      if (count > 1) upPasses[1].timestampWrites = closing;

      for (let level = 0; level < count; level++) {
        const pass = encoder.beginRenderPass(downPasses[level]);
        const size = drawn[level + 1];
        pass.setViewport(0, 0, size[0], size[1], 0, 1);
        pass.setPipeline(downPipelines[level]);
        pass.draw(3);
        pass.end();
      }

      for (let level = count - 1; level > 0; level--) {
        const pass = encoder.beginRenderPass(upPasses[level]);
        const size = drawn[level];
        pass.setViewport(0, 0, size[0], size[1], 0, 1);
        pass.setPipeline(upPipelines[level]);
        pass.draw(3);
        pass.end();
      }
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
