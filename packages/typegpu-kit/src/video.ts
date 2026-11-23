/**
 * Fits camera frames into a texture, or samples them in place.
 * A feed fits a frame of any size, rotation or mirroring into a fixed texture, which keeps the last
 * frame until the next arrives. A live view binds the newest frame itself for `video.sample`.
 * Sources supply frames, such as `web.video` reading a `<video>`.
 *
 * @module
 * @unstable
 */

import { common, d, std, tgpu, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';
import * as bindings from './bindings.ts';
import { encoderOf, type Frame } from './frames.ts';
import type { Inject } from './inject.ts';
import * as pipelines from './pipelines.ts';

/** How a frame of another shape fills the texture. */
export type Fit = 'cover' | 'contain' | 'stretch';

/** Quarter turns, clockwise, that show a frame upright. */
export type Rotation = 0 | 90 | 180 | 270;

/** One imported frame and how to show it. */
export interface Imported {
  /** The frame on the feed's device. Valid until the frame that read it submits. */
  readonly texture: GPUExternalTexture;
  /** The frame's width in texels, before rotation. */
  readonly width: number;
  /** The frame's height in texels, before rotation. */
  readonly height: number;
  /** Turns the frame clockwise before fitting. Default 0. */
  readonly rotation?: Rotation;
  /** Flips the upright frame left to right, as a mirror shows a front camera. Default false. */
  readonly mirrored?: boolean;
  /**
   * When the frame was captured, or where it sits in a video, in seconds on its source's own clock.
   * Compare only times from one source.
   */
  readonly time?: number;
}

/** A source's newest frame. */
export interface Latest extends Imported {
  /** Numbers the source's frames from 1, in the order they arrive. */
  readonly serial: number;
}

/** Hands feeds and live views frames as they arrive. Any number of them can read one source. */
export interface Source {
  /**
   * Imports the newest frame on `device`. Returns undefined before the first frame, and when the
   * newest is numbered `seen`, so a reader that wants only new frames imports nothing more. The
   * source keeps each frame until a newer one replaces it, and the picture stays valid until the
   * frame that read it submits.
   */
  latest(device: GPUDevice, seen?: number): Latest | undefined;
}

/** Where a feed's texture lies on the upright frame, from 0 to 1 across the frame. */
export interface Crop {
  /** The texture's top-left corner. */
  readonly origin: readonly [number, number];
  /** The texture's width and height. `'cover'` stays within 1; `'contain'` reaches past it. */
  readonly extent: readonly [number, number];
}

export interface Options {
  /** Sets the texture's width and height in texels. */
  readonly size: readonly [number, number];
  /** Selects how a frame of another aspect fills the texture. Default `'cover'`. */
  readonly fit?: Fit;
}

export interface Feed extends Disposable {
  /**
   * Holds the last frame as sRGB-encoded `rgba8unorm`, so a model reads byte values and a lit
   * shader decodes with `srgbToLinear`. Uncovered texels of `'contain'` are transparent black.
   */
  readonly texture: ReturnType<typeof createTexture>;
  /** Counts frames written. Compare it between frames to tell a new one apart. */
  readonly frames: number;
  /** The last frame's width and height after rotation, or zero before the first. */
  readonly frameSize: readonly [number, number];
  /**
   * Where the texture lies on the last frame, upright. A live view created with this feed's `size`
   * and `fit` as its `crop` samples the same part of the frame.
   */
  readonly crop: Crop;
  /**
   * Reports when the last frame was captured, or where it sits in a video, in seconds on its
   * source's clock. Reports 0 when its source gives none.
   */
  readonly time: number;
  /** Sets where `update` takes frames from, or `undefined` to stop and keep the last frame. */
  attach(source: Source | undefined): void;
  /** Records the attached source's frame fitted into `frame.encoder` if it is new. */
  update(frame: Frame): void;
  /**
   * Records `picture` fitted into the texture, in a render pass of its own in `target`, a frame or
   * an encoder.
   */
  record(target: Frame | TgpuCommandEncoder, picture: Imported): void;
  /** Compiles the fitting pipeline. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Destroys the texture. */
  destroy(): void;
}

const FORMAT = 'rgba8unorm';

/** How a stored frame turns upright, and the part of it the output shows. */
const Orientation = d.struct({
  /** Upright frame texels per output texel along each axis, as a fraction of the frame. */
  scale: d.vec2f,
  /** Quarter turns, clockwise. */
  turns: d.u32,
  mirrored: d.u32,
});

const frameLayout = tgpu.bindGroupLayout({
  frame: { externalTexture: d.textureExternal() },
});

function createTexture(root: TgpuRoot, size: readonly [number, number]) {
  return root
    .createTexture({ size: [size[0], size[1]], format: FORMAT })
    .$usage('sampled', 'render');
}

/**
 * Scales a frame of `width` by `height` texels, upright, into `size` as `fit` asks, writing how
 * many upright frame texels span a texel of `size` along each axis, as a fraction of the frame.
 */
function fitInto(
  into: d.v2f,
  width: number,
  height: number,
  size: readonly [number, number],
  fit: Fit,
) {
  const across = size[0] / width;
  const down = size[1] / height;
  // Stretching scales each axis on its own; the others keep the frame's aspect
  const kept = fit === 'cover' ? Math.max(across, down) : Math.min(across, down);
  into.x = fit === 'stretch' ? 1 : across / kept;
  into.y = fit === 'stretch' ? 1 : down / kept;
}

/**
 * Follows the shape of a source's frames: their upright size, and the orientation that fits them
 * into `crop`, or shows the whole frame without one.
 */
function orientation(crop: Options | undefined) {
  const frameSize: [number, number] = [0, 0];
  const value = { scale: d.vec2f(1), turns: 0, mirrored: 0 };
  const last = { width: 0, height: 0, turns: -1, mirrored: false };

  return {
    frameSize,
    value,

    /** Takes `picture`'s shape, and returns whether it changed. */
    follow(picture: Imported): boolean {
      const turns = (picture.rotation ?? 0) / 90;
      const mirrored = picture.mirrored ?? false;
      const { width, height } = picture;
      if (
        width === last.width &&
        height === last.height &&
        turns === last.turns &&
        mirrored === last.mirrored
      ) {
        return false;
      }
      Object.assign(last, { width, height, turns, mirrored });

      const sideways = turns % 2 === 1;
      frameSize[0] = sideways ? height : width;
      frameSize[1] = sideways ? width : height;
      if (crop) fitInto(value.scale, frameSize[0], frameSize[1], crop.size, crop.fit ?? 'cover');
      value.turns = turns;
      value.mirrored = mirrored ? 1 : 0;
      return true;
    },
  };
}

/**
 * Maps a point of the fitted picture, in [0, 1], to the stored frame it was cropped, mirrored and
 * turned from. Returns a point outside [0, 1] where `'contain'` leaves the picture uncovered.
 */
const stored = (uv: d.v2f, scale: d.v2f, turns: number, mirrored: number) => {
  'use gpu';
  let at = (uv - 0.5) * scale + 0.5;
  if (std.any(std.lt(at, d.vec2f())) || std.any(std.gt(at, d.vec2f(1)))) return d.vec2f(-1);
  if (mirrored === 1) at = d.vec2f(1 - at.x, at.y);
  if (turns === 1) return d.vec2f(at.y, 1 - at.x);
  if (turns === 2) return 1 - at;
  if (turns === 3) return d.vec2f(1 - at.y, at.x);
  return at;
};

/** Holds what `video.sample` reads: the newest frame, its crop and how it turns upright. */
const showing = bindings.layout({
  frame: { externalTexture: d.textureExternal() },
  sampler: { sampler: 'filtering' },
  showing: { uniform: Orientation },
});

/**
 * Samples a live view's newest frame at `uv`, from 0 to 1 across its `crop` of the frame turned
 * upright and mirrored as its source asks, the whole frame by default. Transparent black before
 * the first frame and where `'contain'` leaves the crop uncovered. Fill it with `live.inject()`.
 */
export const sample = (uv: d.v2f) => {
  'use gpu';
  const shown = showing.$.showing;
  const at = stored(uv, shown.scale, shown.turns, shown.mirrored);
  if (at.x < 0) return d.vec4f();
  return std.textureSampleBaseClampToEdge(showing.$.frame, showing.$.sampler, at);
};

/**
 * Creates a texture of `size` and the pass that fits frames into it.
 *
 * @allocates every `record`: a bind group for the imported frame. For a live view, each new frame: an
 * external texture from the source and a bind group from the share that holds its part; a web
 * video imports again every frame
 * @owns a texture, a buffer, a sampler, a pipeline, bind groups: a feed's picture, fitting, fit and
 * each recorded frame's group; a live view's blank frame, orientation and sampler
 * @injects hook the frame, sampler and orientation `video.sample` reads, from a live view's
 * part
 * @binds 0 storage, 1 uniform: `video.sample`, the orientation, beside the frame and its sampler
 * @records `update` and `record` into a render pass of their own, before the passes that read
 * the texture.
 * @returns a feed you update every frame and destroy
 */
export function create(root: TgpuRoot, options: Options): Feed {
  const { size } = options;
  const texture = createTexture(root, size);
  const target = root.unwrap(texture).createView();
  const oriented = orientation(options);
  const fitting = root.createUniform(Orientation);
  const destroy = () => texture.destroy();
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    const at = stored(uv, fitting.$.scale, fitting.$.turns, fitting.$.mirrored);
    if (at.x < 0) return d.vec4f();
    return std.textureSampleBaseClampToEdge(frameLayout.$.frame, sampler.$, at);
  });

  const pipeline = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: { format: FORMAT },
    })
    .$name('video');

  let attached: Source | undefined;
  let seen = 0;
  let frames = 0;
  let time = 0;
  const crop = { origin: [0, 0] as [number, number], extent: [1, 1] as [number, number] };
  const pass = {
    colorAttachments: { view: target, loadOp: 'clear' as const, storeOp: 'store' as const },
  };

  const record = (into: Frame | TgpuCommandEncoder, picture: Imported) => {
    // Write the fitting only when the frame's shape changes
    if (oriented.follow(picture)) {
      const { scale } = oriented.value;
      crop.origin[0] = 0.5 - scale.x / 2;
      crop.origin[1] = 0.5 - scale.y / 2;
      crop.extent[0] = scale.x;
      crop.extent[1] = scale.y;
      fitting.write(oriented.value);
    }

    const group = root.createBindGroup(frameLayout, { frame: picture.texture });
    const recording = encoderOf(into).beginRenderPass(pass);
    recording.setBindGroup(group);
    recording.setPipeline(pipeline);
    recording.draw(3);
    recording.end();
    frames += 1;
    time = picture.time ?? 0;
  };

  return {
    texture,

    get frames() {
      return frames;
    },

    get time() {
      return time;
    },

    frameSize: oriented.frameSize,
    crop,

    attach(next) {
      if (next !== attached) seen = 0;
      attached = next;
    },

    update(frame) {
      if (!attached) return;
      const picture = attached.latest(root.device, seen);
      if (!picture || picture.serial === seen) return;
      seen = picture.serial;
      record(frame, picture);
    },

    record,

    initAsync: () => pipelines.compile([pipeline]),
    destroy,
    [Symbol.dispose]: destroy,
  };
}

export interface LiveView extends Disposable {
  /**
   * Provides the newest frame, its sampler and orientation for `video.sample`. Share it before
   * creating the pipelines that sample, because each new frame is a new external texture: give
   * `passes.create` `{ shared: { camera: live.part } }`, or call `bindings.share(root, { camera:
   * live.part })` and set its group on each pass of your own that samples.
   */
  readonly part: bindings.Part;
  /** Counts new frames shown. Compare it between frames to tell a new one apart. */
  readonly frames: number;
  /** The newest frame's width and height after rotation, or zero before the first. */
  readonly frameSize: readonly [number, number];
  /**
   * Reports when the newest frame was captured, or where it sits in a video, in seconds on its
   * source's clock. Reports 0 when its source gives none.
   */
  readonly time: number;
  /**
   * Sets the source, or `undefined` to stop and show transparent black, because a live view holds
   * no frame of its own.
   */
  attach(source: Source | undefined): void;
  /**
   * Binds the source's newest frame. Call it every frame before the passes that sample, because an
   * imported frame is valid only until its frame submits.
   *
   * @returns whether the frame is new
   */
  update(frame: Frame): boolean;
  /** Fills `video.sample` for pipelines created from the configuration it is piped into. */
  inject(): Inject;
  /** Destroys the placeholder texture. */
  destroy(): void;
}

export interface LiveOptions {
  /**
   * Crops each frame as a feed with this `size` and `fit` does, so `video.sample(uv)` reads the
   * part of the frame the feed's texture holds at `uv`. Default the whole frame.
   */
  readonly crop?: Options;
}

/**
 * Creates a live view that samples each source's newest frame where it lies, without a copy.
 * Shaders call `video.sample(uv)` and see the frame upright and cropped. They sample an external
 * texture, so they read it with filtering and without mips or `textureLoad`. Use `video.create` to
 * keep frames, compare them or read them as an ordinary texture.
 *
 * @returns a live view you update every frame and destroy
 */
export function live(root: TgpuRoot, options: LiveOptions = {}): LiveView {
  const { crop } = options;
  const blank = root.createTexture({ size: [1, 1], format: FORMAT }).$usage('sampled');
  const destroy = () => blank.destroy();
  // WebGPU binds a 2D view where an external texture goes. Its texels start transparent black
  const nothing: GPUExternalTexture | GPUTextureView = root.unwrap(blank).createView();
  const oriented = orientation(crop);
  const state = root.createBuffer(Orientation, oriented.value).$usage('uniform');
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const part = showing.bind(
    { frame: nothing, sampler, showing: state },
    {
      unshared:
        'Share the live view before creating pipelines that read `video.sample`. The live view binds a new external texture for each frame. Give `passes.create` the option `{ shared: { camera: live.part } }`, or call `bindings.share(root, { camera: live.part })` for your own passes.',
    },
  );

  let attached: Source | undefined;
  let bound: GPUExternalTexture | GPUTextureView = nothing;
  let seen = 0;
  let frames = 0;
  let time = 0;

  const show = (texture: GPUExternalTexture | GPUTextureView) => {
    if (texture === bound) return;
    bound = texture;
    part.replace({ frame: texture });
  };

  return {
    part,

    get frames() {
      return frames;
    },

    get time() {
      return time;
    },

    frameSize: oriented.frameSize,

    attach(next) {
      if (next !== attached) seen = 0;
      attached = next;
      if (!next) show(nothing);
    },

    update() {
      if (!attached) return false;
      const picture = attached.latest(root.device);
      if (!picture) {
        show(nothing);
        return false;
      }
      show(picture.texture);
      if (oriented.follow(picture)) state.write(oriented.value);

      time = picture.time ?? 0;
      if (picture.serial === seen) return false;
      seen = picture.serial;
      frames += 1;
      return true;
    },

    inject: () => part.inject(),
    destroy,
    [Symbol.dispose]: destroy,
  };
}
