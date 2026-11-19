import * as cameras from '../cameras.ts';
import * as history from '../history.ts';
import * as passes from '../passes.ts';
import * as targets from '../targets.ts';
import type * as profiler from '../profiler.ts';
import { memo, passOf } from '../memo.ts';
import { type CameraBase } from '../cameras.ts';
import { type Depth, type SizeSource } from '../targets.ts';
import {
  common,
  d,
  std,
  tgpu,
  type TgpuBindGroup,
  type TgpuCommandEncoder,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';
import { encoderOf, type Frame } from '../frames.ts';
import {
  regionOf,
  textureOf,
  within,
  type FloatTexture,
  type Rendered,
  type Source,
} from './post.ts';

export const Settings = d.struct({
  /**
   * Sets the fraction of current-frame colour to blend, from 0 to 1. Lower values smooth more
   * and respond more slowly to change.
   */
  blend: d.f32,
  /**
   * Limits history colour to this many standard deviations from current colours around the
   * pixel. Lower values reduce ghosting behind moving objects but increase flicker.
   */
  spread: d.f32,
});

export const defaults: d.Infer<typeof Settings> = {
  blend: 0.1,
  spread: 1,
};

export interface Options {
  /**
   * Provides the current linear-colour picture, drawn through `jitter`: a function returning a
   * texture, such as `() => screen.textures.hdr`, or a rendered picture. `record` reads it on
   * every call.
   * @cpu
   */
  readonly source: Source;
  /**
   * Returns the motion texture written by `motion`. Each vector is the preceding UV position
   * minus the current UV position. `record` reads it on every call. Default none, which moves
   * every pixel as `camera` moves `depth`, or keeps it in place without both.
   * @cpu
   */
  readonly motion?: () => FloatTexture;
  /**
   * Returns sampled scene depth without stencil, or `null` for a picture drawn without depth.
   * Motion then follows the nearest surrounding surface to avoid smearing moving edges over the
   * background. `record` reads it on every call. Unused without `motion` or `camera`. With `null`,
   * motion is read at each pixel.
   * @cpu
   */
  readonly depth: (() => Depth) | null;
  /**
   * Supplies the scene camera with its preceding view, such as `cameras.orbit`. With `depth`,
   * pixels without renderer-written motion reconstruct motion from depth and camera movement.
   * Without both, these pixels stay in place. Default none.
   */
  readonly camera?: CameraBase;
  /**
   * Reads `depth` as reversed, 1 at the near plane and 0 at the far plane, without a `camera` to
   * say so. Default the `camera`'s `cameras.reversed`, or false without one.
   */
  readonly reversed?: boolean;
  /**
   * Selects the source region from its top-left corner, such as `regions.hdr` when drawing below
   * full resolution. `record` reads it on every call. Default the region of a rendered `source`,
   * else the whole texture.
   */
  readonly region?: SizeSource;
  /**
   * Resolves to this size, such as the surface, from a source drawn smaller, up to 1.5 times
   * larger per side. History then keeps this size, each output pixel rebuilds the current frame
   * from the source pixels around it, and jitter takes more phases. Scene shaders sample textures
   * with a mip bias of log2(render / output) - 1. Default the source region's size.
   */
  readonly size?: SizeSource;
  /**
   * Returns a texture the size of the source whose red channel, from 0 to 1, takes away history
   * at each pixel, for translucent, dithered or particle drawings that write no motion of their
   * own. 0 accumulates as usual; values above 0.9 count as 0.9, so history never drops out.
   * `record` reads it on every call. Default none.
   * @cpu
   */
  readonly reactive?: () => FloatTexture;
  /** Sets initial values over `defaults`. Default `defaults`, with blend 0.1 and spread 1. */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Taa extends Rendered, Disposable {
  /**
   * Holds `defaults` with overrides from `settings`. Use `patch` to upload only the fields you
   * supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /**
   * Offsets vertices by a different fraction of a source pixel each frame so frames sample the
   * whole pixel together: 8 offsets in turn, or at `size` 8 for each output pixel a source pixel
   * covers, 18 at 1.5 times per side. Supply this as `jitter` in `passes.create`.
   */
  readonly jitter: (position: d.v4f) => d.v4f;
  /** Holds the current clip-space offset used by `jitter`, for custom jitter shaders. */
  readonly offset: TgpuUniform<d.Vec2f>;
  /**
   * Provides the antialiased image in `rgba16float` for the final pass `source`. It is the
   * history texture written by the last `record`, alternating between two textures, or the source
   * while disabled. Its size is `size`, else the source's, and changes to it replace both
   * textures. Consumer `memo` instances keep a bind group per texture.
   */
  readonly output: FloatTexture;
  /**
   * Gives the part of `output` that holds the picture: all of it at `size`, else the source
   * region, which is also the region while disabled.
   */
  readonly region: SizeSource;
  /** Reports whether TAA is on, as the last `enable` set it. */
  readonly enabled: boolean;
  /**
   * Enables or disables TAA. When disabled, jitter is zero from the next frame, `record` records
   * nothing and `output` returns the source. Enabling it again resets history.
   */
  enable(on: boolean): void;
  /** Resets history to the current frame, for example after a camera cut. */
  reset(): void;
  /** Compiles the pipeline and binds both history textures. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Records the resolve after source and motion passes, then sets jitter for the next frame.
   * Records nothing while disabled. Optional timer writes measure the resolve.
   */
  record(target: Frame | TgpuCommandEncoder, options?: profiler.Timed): void;
  /** Destroys the history textures and the jitter sequence. */
  destroy(): void;
}

const Clips = d.struct({ current: d.vec4f, previous: d.vec4f });

/**
 * The unwritten-motion marker must exceed on-screen motion and be exact in `rg16float`. Such
 * pixels reconstruct motion from depth and camera movement.
 */
const UNWRITTEN = 16384;

/**
 * Tells whether a renderer wrote this `taa.motion` value. Pixels no renderer drew keep the
 * output's clear value.
 */
export const written = (motion: d.v2f) => {
  'use gpu';
  return motion.x <= UNWRITTEN / 2;
};

/** UV y increases downwards. The constant half-offset cancels when subtracting positions. */
const uvOf = (clip: d.v4f) => {
  'use gpu';
  return (clip.xy / clip.w) * d.vec2f(0.5, -0.5);
};

/**
 * This output writes surface motion into an `rg16float` target. It stores preceding UV position
 * minus current UV position. It computes the preceding position by compiling renderer vertex
 * code with `history.previously`, which reads the preceding camera and values of other
 * `temporal` fills.
 */
export const motion = passes.output(d.vec2f, {
  clear: d.vec2f(UNWRITTEN),
  varyings: Clips,
  previous: true,
  vary: (current: { position: d.v4f }, previous: { position: d.v4f }) => {
    'use gpu';
    return Clips({ current: current.position, previous: previous.position });
  },
  write: (_pixel: unknown, clips) => {
    'use gpu';
    return uvOf(clips.previous) - uvOf(clips.current);
  },
});

const clipBefore = history.previously(cameras.toClip);

/**
 * Halton offsets use bases 2 and 3 and span -0.5 to 0.5 pixels, starting at the second sample.
 */
const halton = (index: number, base: number) => {
  let fraction = 1;
  let value = 0;
  for (let i = index; i > 0; i = Math.floor(i / base)) {
    fraction /= base;
    value += fraction * (i % base);
  }
  return value - 0.5;
};
/** Phases at the source's size, and per output pixel's worth of source area at `size`. */
const COUNT = 8;
/** Phases at most, which a source half the output's size per side reaches. */
const MOST = 32;
const offsetsOf = (count: number) =>
  Array.from({ length: count }, (_, i) => [halton(i + 1, 2), halton(i + 1, 3)]);

/** The largest value an `rgba16float` picture holds. */
const LARGEST = 65504;

/** The most `reactive` takes away from history, so some always remains. */
const RESPONSIVE = 0.9;

const layout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  history: { texture: d.texture2d(d.f32) },
});
const movingLayout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  motion: { texture: d.texture2d(d.f32) },
  history: { texture: d.texture2d(d.f32) },
});
const depthLayout = tgpu.bindGroupLayout({ depth: { texture: d.textureDepth2d() } });
const reactiveLayout = tgpu.bindGroupLayout({ reactive: { texture: d.texture2d(d.f32) } });

const State = d.struct({
  /** Measures the source region in texels. */
  region: d.vec2f,
  /** A value of 1 permits blending with history. */
  valid: d.u32,
});

/** What a resolve at `size` gathers from the source pixels around an output pixel. */
const Gathered = d.struct({
  /** Output pixels per source pixel. */
  ratio: d.vec2f,
  /** The output pixel's centre, in source pixels. */
  here: d.vec2f,
  /** How far this frame's jitter moved the picture, in source pixels. */
  shift: d.vec2f,
  /** The sum of colours weighted by Lanczos-2, and of the weights. */
  color: d.vec3f,
  weight: d.f32,
  /** The least and greatest colours around it, to bound the filter's ringing. */
  low: d.vec3f,
  high: d.vec3f,
});

const toYCoCg = (color: d.v3f) => {
  'use gpu';
  return d.vec3f(
    std.dot(color, d.vec3f(0.25, 0.5, 0.25)),
    std.dot(color, d.vec3f(0.5, 0, -0.5)),
    std.dot(color, d.vec3f(-0.25, 0.5, -0.25)),
  );
};

/** The clamp boxes reach outside the RGB cube, so this cuts the negative channels it would give. */
const fromYCoCg = (color: d.v3f) => {
  'use gpu';
  const rgb = d.vec3f(color.x + color.y - color.z, color.x + color.z, color.x - color.y - color.z);
  return std.max(rgb, d.vec3f());
};

const NEAR = [-1, 0, 1];

/** Lanczos-2 at `x` output pixels, from -2 to 2, as a polynomial. */
const lanczos = (x: number) => {
  'use gpu';
  const v = x * x;
  return (-((v - 4) * (v - 4)) * (v - 1) * (1 - 0.354377 * v)) / 16;
};

/**
 * Creates temporal antialiasing from subpixel-jittered frames. The resolve blends each frame
 * with history sampled where `motion`, or depth and camera movement, place the preceding pixel.
 * Without either, it samples history at the same pixel. It clamps history colour to
 * the local colour spread to reduce trails and weights by brightness to reduce bright-pixel
 * flicker. With `size`, it resolves a smaller source up to that size: each output pixel rebuilds
 * the current frame from the nine source pixels around it with Lanczos-2 weights placed by this
 * frame's jitter, and blends less of it where those weights add up to less. Record its single
 * pass after scene passes and before the final pass reads `output`.
 *
 * @allocates on resize: targets for the new size and the bind groups of its passes; so does a new
 * source, motion, depth or reactive texture
 * @owns targets, 4 buffers, a sampler, a pipeline, 2 to 4 bind groups: two histories, each frame
 * written into one from the other, the settings, the region and whether to blend, the jitter and
 * its sequence, a group each for the frames that read either history with the source and
 * `motion`, one for `depth` and one for `reactive`
 * @binds 1 uniform: `jitter`, this frame's offset
 * @records `record` after the passes that draw the source and the motion, before what reads
 * `output`.
 * @param options.source this frame's picture, drawn through `jitter`, read at every `record`
 * @param options.motion how far each pixel's surface moved since last frame, read at every
 * `record`; without it, pixels move only with the camera
 * @param options.depth the scene's depth, for the motion of the nearest surface around each pixel
 * and for motion from the camera, or `null`
 * @returns a resolve you record every frame and destroy
 */
export function create(root: TgpuRoot, options: Options): Taa {
  const { motion: moved, depth, camera: view } = options;
  const source = textureOf(options.source);
  const region = regionOf(options.source, options.region);
  const moving = moved !== undefined;
  const reprojected = depth !== null && view !== undefined;
  const dilated = depth !== null && (moving || reprojected);
  const pictures = moving ? movingLayout : layout;
  const upscaled = options.size !== undefined;
  const responsive = options.reactive !== undefined;

  const full: SizeSource = {
    get width() {
      return source().props.size[0];
    },
    get height() {
      return source().props.size[1] ?? 1;
    },
  };
  const drawn = region ?? full;

  const screen = targets.create(root, options.size ?? full, {
    even: { format: 'rgba16float', sampled: true },
    odd: { format: 'rgba16float', sampled: true },
  });
  const offsets = offsetsOf(upscaled ? MOST : COUNT);
  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });
  const state = root.createUniform(State);
  const offset = root.createUniform(d.vec2f);

  const sequence = root.createBuffer(d.arrayOf(d.vec2f, offsets.length));
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const jitter = (position: d.v4f) => {
    'use gpu';
    return position + d.vec4f(offset.$ * position.w, 0, 0);
  };

  const tap = (x: number, y: number) => {
    'use gpu';
    return std.textureSampleLevel(pictures.$.history, sampler.$, d.vec2f(x, y), 0).xyz;
  };

  const historyAt = (at: d.v2f) => {
    'use gpu';
    const size = d.vec2f(std.textureDimensions(pictures.$.history));
    const part = state.$.region;
    const middle = std.floor(at - 0.5) + 0.5;
    const f = at - middle;
    const w0 = f * (-0.5 + f * (1 - f * 0.5));
    const w1 = 1 + f * f * (-2.5 + f * 1.5);
    const w2 = f * (0.5 + f * (2 - f * 1.5));
    const w3 = f * f * (f * 0.5 - 0.5);
    const w12 = w1 + w2;
    // History at `size` fills its texture
    const low = within((middle - 1) / size, upscaled ? size : part, size);
    const mid = within((middle + w2 / w12) / size, upscaled ? size : part, size);
    const high = within((middle + 2) / size, upscaled ? size : part, size);

    // Omit low-weight corner taps and renormalize the remaining weights
    const sum =
      tap(mid.x, low.y) * (w12.x * w0.y) +
      tap(low.x, mid.y) * (w0.x * w12.y) +
      tap(mid.x, mid.y) * (w12.x * w12.y) +
      tap(high.x, mid.y) * (w3.x * w12.y) +
      tap(mid.x, high.y) * (w12.x * w3.y);
    const weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return std.max(sum / weight, d.vec3f());
  };

  const cameraMotion = (pixel: d.v2f, z: number) => {
    'use gpu';
    const uv = pixel / state.$.region;
    const ndc = d.vec2f(uv.x * 2 - 1, 1 - uv.y * 2) - offset.$;
    const ray = cameras.rayOf(cameras.current.$, ndc);
    // Use clip z and w, linear along the ray, to reconstruct the surface distance from depth
    const start = cameras.current.$.viewProjection * d.vec4f(ray.origin, 1);
    const along = cameras.current.$.viewProjection * d.vec4f(ray.direction, 0);
    const t = (start.z - z * start.w) / (z * along.w - along.z);
    return uvOf(clipBefore(ray.origin + ray.direction * t)) - ndc * d.vec2f(0.5, -0.5);
  };

  // Private, so a resolve without `size` declares none of it
  const gathered = tgpu.privateVar(Gathered);

  /** Starts gathering for the output pixel at `position` and returns the source pixel under it. */
  const gatherAt = (position: d.v2f) => {
    'use gpu';
    const part = state.$.region;
    const ratio = d.vec2f(std.textureDimensions(pictures.$.history)) / part;
    const here = position / ratio;
    gathered.$ = Gathered({
      ratio,
      here,
      shift: offset.$ * d.vec2f(0.5, -0.5) * part,
      color: d.vec3f(),
      weight: 0,
      low: d.vec3f(LARGEST),
      high: d.vec3f(-LARGEST),
    });
    return d.vec2i(here);
  };

  /** Adds a source pixel, weighted by how far its jittered sample lies, in output pixels. */
  const gather = (at: d.v2i, color: d.v3f) => {
    'use gpu';
    const sample = d.vec2f(at) + 0.5 - gathered.$.shift;
    const apart = std.clamp((sample - gathered.$.here) * gathered.$.ratio, d.vec2f(-2), d.vec2f(2));
    const w = lanczos(apart.x) * lanczos(apart.y);
    gathered.$.color += color * w;
    gathered.$.weight += w;
    gathered.$.low = std.min(gathered.$.low, color);
    gathered.$.high = std.max(gathered.$.high, color);
  };

  const fragment = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })(({ position }) => {
    'use gpu';
    const part = state.$.region;
    let pixel = d.vec2i(position.xy);
    if (upscaled) pixel = gatherAt(position.xy);
    const last = d.vec2i(part) - 1;

    let sum = d.vec3f();
    let squares = d.vec3f();
    let nearest = d.f32(cameras.farDepth());
    let nearestAt = d.vec2i(pixel);
    for (const y of tgpu.unroll(NEAR)) {
      for (const x of tgpu.unroll(NEAR)) {
        const at = std.clamp(pixel + d.vec2i(x, y), d.vec2i(), last);
        const color = toYCoCg(std.textureLoad(pictures.$.source, at, 0).xyz);
        sum += color;
        squares += color * color;
        if (upscaled) gather(at, color);
        if (dilated) {
          const z = std.textureLoad(depthLayout.$.depth, at, 0);
          if (cameras.reversed.$ ? z > nearest : z < nearest) {
            nearest = z;
            nearestAt = d.vec2i(at);
          }
        }
      }
    }

    let current = toYCoCg(std.textureLoad(pictures.$.source, pixel, 0).xyz);
    if (upscaled) {
      const filtered = gathered.$.color / std.max(gathered.$.weight, 0.0001);
      current = std.clamp(filtered, gathered.$.low, gathered.$.high);
    }
    let travelled = d.vec2f();
    if (moving) {
      travelled = std.textureLoad(movingLayout.$.motion, nearestAt, 0).xy;
      if (!written(travelled)) {
        travelled = d.vec2f();
        if (reprojected) travelled = cameraMotion(d.vec2f(nearestAt) + 0.5, nearest);
      }
    } else if (reprojected) {
      travelled = cameraMotion(d.vec2f(nearestAt) + 0.5, nearest);
    }
    let before = position.xy + travelled * part;
    let outside = before.x < 0 || before.y < 0 || before.x > part.x || before.y > part.y;
    if (upscaled) {
      const size = part * gathered.$.ratio;
      before = position.xy + travelled * size;
      outside = before.x < 0 || before.y < 0 || before.x > size.x || before.y > size.y;
    }
    if (state.$.valid === 0 || outside) {
      return d.vec4f(fromYCoCg(current), 1);
    }

    const mean = sum / 9;
    let extent = std.max(
      std.sqrt(std.max(squares / 9 - mean * mean, d.vec3f())) * settings.$.spread,
      d.vec3f(0.0001),
    );
    // History between source pixels holds detail that no one frame shows, so let it stray further
    if (upscaled) extent *= std.max(gathered.$.ratio.x, gathered.$.ratio.y);
    const away = toYCoCg(historyAt(before)) - mean;
    const reach = std.abs(away) / extent;
    const furthest = std.max(reach.x, std.max(reach.y, reach.z));
    const previous = mean + away / std.max(furthest, 1);

    // Weight by brightness to reduce flicker from bright pixels, the current frame by how much
    // of the filter its samples cover, and history by what `reactive` leaves of it
    let now = settings.$.blend / (1 + std.max(current.x, 0));
    let then = (1 - settings.$.blend) / (1 + std.max(previous.x, 0));
    if (upscaled) now *= std.saturate(gathered.$.weight);
    if (responsive) {
      then *= 1 - std.clamp(std.textureLoad(reactiveLayout.$.reactive, pixel, 0).x, 0, RESPONSIVE);
    }
    return d.vec4f(fromYCoCg((current * now + previous * then) / (now + then)), 1);
  });

  const viewed = view ? root.pipe(view.inject()) : root;
  const pipeline = (
    options.reversed === undefined ? viewed : viewed.with(cameras.reversed, options.reversed)
  )
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: { format: 'rgba16float' },
    })
    .$name('taa');

  // Keep one memo entry per history texture and share the depth group. Store groups on the
  // pipeline so recording only sets the pipeline on the pass
  const depthGroup = memo((deep: Depth) =>
    root.createBindGroup(depthLayout, { depth: deep.createView(d.textureDepth2d()) }),
  );
  const reactiveGroup = memo((quick: FloatTexture) =>
    root.createBindGroup(reactiveLayout, { reactive: quick }),
  );
  const pictured = memo(
    (picture: FloatTexture, motions: FloatTexture | undefined, read: FloatTexture) => {
      const group = motions
        ? root.createBindGroup(movingLayout, { source: picture, motion: motions, history: read })
        : root.createBindGroup(layout, { source: picture, history: read });
      // Create the groups on the device now, so `initAsync` can bind both histories ahead
      root.unwrap(group);
      return group;
    },
  );
  const bound = memo((group: TgpuBindGroup, deep?: Depth, quick?: FloatTexture) => {
    let held = pipeline.with(group);
    if (deep) {
      const deepGroup = depthGroup(deep);
      root.unwrap(deepGroup);
      held = held.with(deepGroup);
    }
    if (quick) {
      const quickGroup = reactiveGroup(quick);
      root.unwrap(quickGroup);
      held = held.with(quickGroup);
    }
    return held;
  });
  const boundTo = (read: FloatTexture) =>
    bound(
      pictured(source(), moved?.(), read),
      dilated ? depth?.() : undefined,
      options.reactive?.(),
    );
  const passInto = passOf();

  let enabled = true;
  let fresh = true;
  let parity = 0;
  let index = 0;
  const measured = { width: 0, height: 0, valid: -1, phases: COUNT };

  const measure = () => {
    const width = Math.min(full.width, drawn.width);
    const height = Math.min(full.height, drawn.height);
    if (width !== measured.width || height !== measured.height) {
      sequence.write(offsets.map(([x, y]) => d.vec2f((x! * 2) / width, (y! * 2) / height)));
      // History at `size` stays put while the source region changes
      if (!upscaled) fresh = true;
    }
    if (upscaled) {
      const area = (screen.width / width) * (screen.height / height);
      measured.phases = Math.min(MOST, Math.ceil(COUNT * area - 1e-6));
    }

    const valid = enabled && !fresh ? 1 : 0;
    if (width !== measured.width || height !== measured.height || valid !== measured.valid) {
      state.write({ region: d.vec2f(width, height), valid });
      measured.width = width;
      measured.height = height;
      measured.valid = valid;
    }
    fresh = false;
  };

  const destroy = () => {
    screen.destroy();
    sequence.destroy();
  };

  return {
    settings,
    jitter,
    offset,

    get output() {
      if (!enabled) return source();
      return parity === 1 ? screen.textures.even : screen.textures.odd;
    },

    get region() {
      return upscaled && enabled ? screen : drawn;
    },

    get enabled() {
      return enabled;
    },

    enable(on) {
      if (on === enabled) return;
      enabled = on;
      fresh = true;
      // Once per toggle: the next frames draw without jitter until TAA records again
      if (!on) offset.write(d.vec2f());
    },

    reset() {
      fresh = true;
    },

    initAsync() {
      // Bound before the wait: a destroy during compilation frees the histories
      boundTo(screen.textures.odd);
      boundTo(screen.textures.even);
      return pipeline.initAsync();
    },

    record(target, timing) {
      if (!enabled) return;
      const encoder = encoderOf(target);
      const timestampWrites = timing?.timestampWrites;
      if (screen.fit()) fresh = true;
      measure();

      const reading = parity === 0 ? screen.textures.odd : screen.textures.even;
      const writing = parity === 0 ? screen.views.even : screen.views.odd;
      const descriptor = passInto(writing);
      descriptor.timestampWrites = timestampWrites;
      const pass = encoder.beginRenderPass(descriptor);
      if (!upscaled && (measured.width < full.width || measured.height < full.height)) {
        pass.setViewport(0, 0, measured.width, measured.height, 0, 1);
      }
      pass.setPipeline(boundTo(reading));
      pass.draw(3);
      pass.end();
      parity = 1 - parity;

      // Copy the next jitter offset through the encoder to avoid a CPU write each frame
      index = (index + 1) % measured.phases;
      root
        .unwrap(encoder)
        .copyBufferToBuffer(
          root.unwrap(sequence),
          index * d.sizeOf(d.vec2f),
          root.unwrap(offset.buffer),
          0,
          d.sizeOf(d.vec2f),
        );
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
