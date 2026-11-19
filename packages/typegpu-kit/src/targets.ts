/**
 * @module
 * @allocates on resize: the whole set, new textures and their views; a new resolution allocates
 * nothing
 * @owns a texture per target: replaced as a set when the size changes
 * @records `fit()` before the passes that use the targets; it returns true after a rebuild.
 */

import type { d, RenderFlag, SampledFlag, StorageFlag, TgpuRoot, TgpuTexture } from 'typegpu';
import type { Checked } from './inject.ts';
import type { Frame } from './frames.ts';

interface Common {
  readonly format: GPUTextureFormat;
  /**
   * Scales each side relative to the set size. Rounds up to whole pixels, with a minimum of 1.
   * Default 1.
   */
  readonly scale?: number;
  /**
   * Sets the sample count per pixel. Multisampled targets cannot be storage textures or have mip
   * levels. They are transient unless sampled or given `transient: false`. Default 1.
   */
  readonly samples?: number;
}

interface Kept extends Common {
  /**
   * Sets the requested mip level count, with one attachment view per level in `levels`. The size
   * limits the count, so `levels` may be shorter. A 16×9 target has at most 5 levels. Default 1.
   */
  readonly mips?: number;
  /** Allows binding the target as a sampled texture. Default false. */
  readonly sampled?: boolean;
  /**
   * Allows binding the target as a storage texture if TypeGPU supports its format for storage.
   * Default false.
   */
  readonly storage?: boolean;
  /**
   * Keeps multisampled target contents for a later pass to load. Set to `false` to preserve them.
   * Default `true` for a multisampled target that is neither sampled nor stored, otherwise
   * `false`.
   */
  readonly transient?: false;
  /**
   * Makes the target a 2D array of this many layers, with an attachment view per layer in
   * `layers`. A description given `layers` draws into one layer per pass, and shaders sample the
   * whole array as `texture_2d_array` or `texture_depth_2d_array`. A layered target has one
   * sample and one mip level. Default none, a 2D texture.
   */
  readonly layers?: number;
}

interface Stored extends Kept {
  readonly format: d.WgslStorageTextureProps['format'];
  readonly storage: true;
}

interface Unstored extends Kept {
  readonly storage?: false;
}

interface Transient extends Common {
  /**
   * Discards target contents at the end of each pass and clears at the start. Uses memoryless
   * attachments where `GPUTextureUsage.TRANSIENT_ATTACHMENT` exists, or plain attachments
   * elsewhere. Cannot be sampled. Use for MSAA colour and depth. Default `true` for a multisampled
   * target that is neither sampled nor stored, otherwise `false`.
   */
  readonly transient: true;
  readonly mips?: 1;
  readonly sampled?: never;
  readonly storage?: never;
  readonly layers?: never;
}

export type Spec = Unstored | Stored | Transient;

/** Supplies a size in pixels, such as a canvas surface. Read at each `fit`. */
export interface SizeSource {
  /** Reports the width in pixels. */
  readonly width: number;
  /** Reports the height in pixels. */
  readonly height: number;
  /** Reports pixels per point, as a surface's `pixelRatio` does. Default 1. */
  readonly pixelRatio?: number;
}

/**
 * Provides a destination for frames, such as a canvas from `web.surface` or `native.surface`, or a
 * texture from `offscreenSurface` in a test. Pass the frame to `view` for code that works with
 * every destination.
 */
export interface Surface extends SizeSource, Disposable {
  /** Reports the format of its textures, which pipelines that draw into it target. */
  readonly format: GPUTextureFormat;
  /** Reports the drawing-buffer size in pixels. */
  readonly width: number;
  /** Reports the drawing-buffer height in pixels. */
  readonly height: number;
  /**
   * Reports drawing-buffer pixels per CSS pixel or point after applying the cap. A one-point line
   * uses this many drawing-buffer pixels.
   */
  readonly pixelRatio: number;
  /**
   * Returns a view of the current texture, created once per texture. React Native requires the
   * frame to present after submission.
   */
  view(frame: Frame): GPUTextureView;
  /**
   * Releases the drawing destination. Canvas surfaces unconfigure their contexts. Web surfaces
   * also stop observing the canvas. Also `[Symbol.dispose]`.
   */
  destroy(): void;
}

/**
 * Fixes the drawing-buffer size in pixels regardless of layout or screen changes. Accepted by each
 * surface's `setSize`.
 */
export interface FixedSize {
  readonly width: number;
  readonly height: number;
  /** Sets the surface's reported pixels per scene point. Default 1. */
  readonly pixelRatio?: number;
}

/**
 * Scales the source dimensions and pixels per point by `scale`. A function scale is read at every access, so targets
 * and cameras following this size also follow its changes.
 *
 * @param source the size it scales, such as a surface
 * @param scale a factor, or a function that returns it
 * @returns a size that targets and cameras can follow
 */
export function scaled(source: SizeSource, scale: number | (() => number)): SizeSource {
  const factor = typeof scale === 'number' ? () => scale : scale;

  return {
    get width() {
      return source.width * factor();
    },

    get height() {
      return source.height * factor();
    },

    get pixelRatio() {
      return (source.pixelRatio ?? 1) * factor();
    },
  };
}

// Round up for pixel coverage while allowing floating-point error. For example, 1.1 * 100 is
// 110.00000000000001
const pixels = (size: number) => Math.max(1, Math.ceil(size - 1e-6));

type FloatFormat = Exclude<GPUTextureFormat, `${string}int` | `depth${string}` | 'stencil8'>;

/**
 * This type accepts a single-sampled 2D texture read as floats, such as a sampled target, bloom
 * output or TAA output. Post-processing passes read mip 0. Filtering 32-bit float formats
 * requires `float32-filterable`.
 */
export type FloatTexture = TgpuTexture<{
  size: readonly number[];
  format: FloatFormat;
  sampleCount?: 1;
}> &
  SampledFlag;

type DepthFormat = 'depth16unorm' | 'depth24plus' | 'depth32float';

/**
 * This type accepts a sampled depth target without stencil, using `depth16unorm`, `depth24plus`
 * or `depth32float`, as the modules that read depth take it.
 */
export type Depth = TgpuTexture<{ size: readonly number[]; format: DepthFormat; sampleCount?: 1 }> &
  SampledFlag;

export type Target<S extends Spec> = TgpuTexture<
  {
    size: S extends { layers: number } ? [number, number, number] : [number, number];
    format: S['format'];
  } & (S extends { samples: infer N extends number } ? { sampleCount: N } : {}) &
    (S extends { mips: number } ? { mipLevelCount: number } : {})
> &
  RenderFlag &
  (S extends { sampled: true } ? SampledFlag : {}) &
  (S extends { storage: true } ? StorageFlag : {});

export interface Options {
  /**
   * Sets the fraction of each side to draw, capped at 1 and read at each `fit()`. Textures keep
   * the source size, so changing resolution allocates nothing. Passes use viewports for the drawn
   * regions, and sampling reads only those regions. Default 1.
   * @cpu
   */
  readonly resolution?: number | (() => number);
}

/** Keeps every key of a specs literal, as the general `Spec` where the literal is not one. */
type Specs<S> = { readonly [K in keyof S]: S[K] extends Spec ? S[K] : Spec };

export interface Targets<S extends Record<string, Spec> = Record<string, Spec>> extends Disposable {
  /** Provides the creation specifications for each target. */
  readonly specs: S;
  /** Reports the source size rounded up to whole pixels, before applying each target's scale. */
  readonly width: number;
  /** Reports the source height rounded up to whole pixels, before applying each target's scale. */
  readonly height: number;
  /** Reports the source's pixels per point, such as a surface's `pixelRatio`, or 1 without one. */
  readonly pixelRatio: number;
  /**
   * Reports which targets are transient. Passes clear them at the start and discard their contents
   * at the end with `storeOp: 'discard'`.
   */
  readonly transient: { readonly [K in keyof S]: boolean };
  /** Reports whether transient attachments can be memoryless on this device. */
  readonly memoryless: boolean;
  /** Provides the current textures. A size change replaces them. */
  readonly textures: { readonly [K in keyof S]: Target<S[K]> };
  /** Provides attachment views of mip 0, created once per rebuild. */
  readonly views: { readonly [K in keyof S]: GPUTextureView };
  /** Provides attachment views of every mip level. */
  readonly levels: { readonly [K in keyof S]: readonly GPUTextureView[] };
  /** Provides an attachment view of each layer, one for a target without `layers`. */
  readonly layers: { readonly [K in keyof S]: readonly GPUTextureView[] };
  /**
   * Reports each drawn region from the top-left corner of its target. Multiplies target size by
   * resolution and rounds up. `fit()` updates the same objects for the lifetime of the set.
   */
  readonly regions: { readonly [K in keyof S]: SizeSource };

  /**
   * Rebuilds all textures if the source size changed, destroying the old textures. Updates
   * `regions` for the current resolution. Returns true after a rebuild so callers can update bind
   * groups that hold old textures. Call before recording passes that use the targets.
   */
  fit(): boolean;
  /**
   * Reports whether the textures and regions match the source's size and the resolution, as they
   * do after `fit()`. A pass description throws when it begins a pass into a set that does not.
   */
  readonly fitted: boolean;
  /** Destroys the current textures. */
  destroy(): void;
}

/**
 * Creates render targets that follow a size. Replaces the whole set when the size changes. Put
 * targets needed only sometimes in their own set and create that set when needed. Throws for
 * transient targets with sampling, storage or mip levels, for multisampled targets with storage or
 * mip levels, and for layered targets that are multisampled, mipped or have more layers than the
 * device allows.
 *
 * @param source the size it follows, such as a surface
 * @param specs a format and options per target, by name
 * @param options.resolution the fraction of each side drawn, which changes without allocating
 * @returns a set of targets you fit every frame and destroy
 */
export function create<const S extends Record<string, object>>(
  root: TgpuRoot,
  source: SizeSource,
  specs: Checked<S, { readonly [K in keyof S]: Spec }>,
  options?: Options,
): Targets<Specs<S>>;
export function create(
  root: TgpuRoot,
  source: SizeSource,
  specs: Record<string, Spec>,
  options: Options = {},
): Targets {
  // List entries once so fitting an unchanged size creates no arrays
  const entries = Object.entries(specs);

  const transient: Record<string, boolean> = {};

  // Check requested mip counts before a small texture size can hide an invalid request
  for (const [name, spec] of entries) {
    const mipped = (spec.mips ?? 1) > 1;

    if (spec.transient && (spec.sampled || spec.storage || mipped)) {
      throw new Error(`Target '${name}' is transient, so it can't be sampled, stored or mipped.`);
    }

    const multisampled = (spec.samples ?? 1) > 1;
    if (multisampled && (spec.storage || mipped)) {
      throw new Error(`Target '${name}' is multisampled, so it can't be stored or mipped.`);
    }

    const { layers } = spec as { layers?: number };
    if (layers !== undefined) {
      const most = root.device.limits.maxTextureArrayLayers;
      if (!Number.isInteger(layers) || layers < 1 || layers > most) {
        throw new RangeError(`Target '${name}' needs 1 to ${most} layers, got ${layers}.`);
      }
      if (multisampled || mipped) {
        throw new Error(`Target '${name}' has layers, so it can't be multisampled or mipped.`);
      }
    }

    transient[name] = spec.transient ?? (multisampled && !spec.sampled);
  }

  const { resolution = 1 } = options;
  const fraction = typeof resolution === 'number' ? () => resolution : resolution;

  // TRANSIENT_ATTACHMENT is a browser texture usage, not a device feature
  const memoryless = 'TRANSIENT_ATTACHMENT' in GPUTextureUsage;
  const maxSize = root.device.limits.maxTextureDimension2D;
  const side = (size: number, scale = 1) => Math.min(maxSize, pixels(size * scale));

  const textures: Record<string, TgpuTexture> = {};
  const views: Record<string, GPUTextureView> = {};
  const levels: Record<string, GPUTextureView[]> = {};
  const layered: Record<string, GPUTextureView[]> = {};
  const regions: Record<string, { width: number; height: number }> = {};

  for (const name of Object.keys(specs)) regions[name] = { width: 0, height: 0 };

  let width = 0;
  let height = 0;
  let measured = 0;

  const release = () => {
    for (const texture of Object.values(textures)) texture.destroy();
  };

  const build = () => {
    release();
    for (const [name, spec] of entries) {
      const size = [side(width, spec.scale), side(height, spec.scale)] as [number, number];
      const mips = Math.min(spec.mips ?? 1, 1 + Math.floor(Math.log2(Math.max(...size))));
      const { layers } = spec as { layers?: number };
      const texture = root.createTexture({
        size: layers === undefined ? size : [...size, layers],
        format: spec.format,
        sampleCount: spec.samples ?? 1,
        mipLevelCount: mips,
      });

      // `Spec` has checked storage usage, but TypeGPU cannot check it against the general
      // GPUTextureFormat type here
      const usage = texture as unknown as { $usage(...usages: string[]): void };
      if (transient[name]) usage.$usage(memoryless ? 'transient' : 'render');
      else {
        usage.$usage('render');
        if (spec.sampled) usage.$usage('sampled');
        if (spec.storage) usage.$usage('storage');
      }

      // Attachment views must cover one mip level, while TypeGPU's default view covers all levels
      // Reuse each view until resize so consumers such as bloom can memoise it
      const raw = root.unwrap(texture);
      const one = layers === undefined ? {} : ({ dimension: '2d', arrayLayerCount: 1 } as const);
      levels[name] = Array.from({ length: mips }, (_, level) =>
        raw.createView({ ...one, baseMipLevel: level, mipLevelCount: 1 }),
      );
      layered[name] =
        layers === undefined
          ? [levels[name][0]]
          : Array.from({ length: layers }, (_, layer) =>
              layer === 0 ? levels[name][0] : raw.createView({ ...one, baseArrayLayer: layer }),
            );

      views[name] = levels[name][0];
      textures[name] = texture;
    }
  };

  const measure = (scale: number) => {
    for (const [name, spec] of entries) {
      const [fullWidth, fullHeight] = textures[name].props.size as [number, number];
      const region = regions[name];
      region.width = Math.min(fullWidth, pixels(width * scale * (spec.scale ?? 1)));
      region.height = Math.min(fullHeight, pixels(height * scale * (spec.scale ?? 1)));
    }

    measured = scale;
  };

  const fit = () => {
    const nextWidth = pixels(source.width);
    const nextHeight = pixels(source.height);
    const resized = nextWidth !== width || nextHeight !== height;
    const scale = Math.min(1, fraction());

    if (resized) {
      width = nextWidth;
      height = nextHeight;
      build();
    }

    if (resized || scale !== measured) measure(scale);
    return resized;
  };
  fit();

  return {
    get width() {
      return width;
    },

    get height() {
      return height;
    },

    get pixelRatio() {
      return source.pixelRatio ?? 1;
    },

    specs,
    transient: transient as Targets['transient'],
    memoryless,

    textures: textures as Targets['textures'],
    views: views as Targets['views'],
    levels: levels as Targets['levels'],
    layers: layered as Targets['layers'],

    regions: regions as Targets['regions'],
    fit,

    get fitted() {
      return (
        pixels(source.width) === width &&
        pixels(source.height) === height &&
        Math.min(1, fraction()) === measured
      );
    },

    destroy: release,
    [Symbol.dispose]: release,
  };
}
