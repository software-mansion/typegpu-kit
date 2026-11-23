import * as bindings from '../bindings.ts';
import * as cameras from '../cameras.ts';
import * as passes from '../passes.ts';
import * as pipelines from '../pipelines.ts';
import * as targets from '../targets.ts';
import * as profiler from '../profiler.ts';
import { type CameraBase } from '../cameras.ts';
import { type Inject } from '../inject.ts';
import { memo, passInto } from '../memo.ts';
import { type Depth, type SizeSource } from '../targets.ts';
import { encoderOf, type Frame } from '../frames.ts';
import { decode, encode } from './octahedral.ts';
import { randf } from '@typegpu/noise';
import {
  common,
  d,
  std,
  tgpu,
  type SampledFlag,
  type TgpuCommandEncoder,
  type TgpuRoot,
  type TgpuTexture,
  type TgpuUniform,
} from 'typegpu';

export const Settings = d.struct({
  /** Sets how far occluders count, in world units. */
  radius: d.f32,
  /** Raises visibility to this power. Higher values darken creases more. */
  power: d.f32,
});

export const defaults: d.Infer<typeof Settings> = {
  radius: 0.5,
  power: 2.2,
};

/** This type accepts the sampled `rg16float` target that `ao.normal` writes. */
export type Normal = TgpuTexture<{
  size: readonly number[];
  format: 'rg16float';
  sampleCount?: 1;
}> &
  SampledFlag;

/** This type is the `rg16float` texture holding visibility in red and eye distance in green. */
export type Occlusion = TgpuTexture<{ size: [number, number]; format: 'rg16float' }> & SampledFlag;

export interface Options {
  /**
   * Returns a sampled depth texture drawn before shading, such as a half-resolution prepass.
   * `record` reads it on every call. Its size sets the size of the occlusion.
   * @cpu
   */
  readonly depth: () => Depth;
  /**
   * Returns the texture written by the `ao.normal` output in the same pass as `depth`.
   * `record` reads it on every call. Default none, which takes normals from depth.
   * @cpu
   */
  readonly normal?: () => Normal;
  /**
   * Turns the sample directions each frame, for TAA to average. Default false, the same
   * directions every frame.
   */
  readonly temporal?: boolean;
  /** Sets the directions per pixel, 1 to 3, with 3 steps each way. Default 2. */
  readonly slices?: 1 | 2 | 3;
  /**
   * Selects the depth region from its top-left corner, such as `regions.depth` when drawing below
   * full resolution. `record` reads it on every call. Default the whole depth.
   */
  readonly region?: SizeSource;
  /** Sets initial values over `defaults`. Default `defaults`, with radius 0.5 and power 2.2. */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
  /**
   * Controls whether `record` traces in the current frame. When false, the first `record` clears
   * `output` to full visibility and later ones record nothing. Skip the prepass yourself. The
   * textures and pipelines stay allocated. Default always.
   * @cpu
   */
  readonly enabled?: () => boolean;
}

export interface Ao extends Disposable {
  /**
   * Holds `defaults` with overrides from `settings`. Use `patch` to upload only the fields you
   * supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /**
   * Provides the occlusion at the depth's size, visibility in red and eye distance along the view
   * direction in green. A new depth size replaces the texture.
   */
  readonly output: Occlusion;
  /** Gives the region of `output` that `record` writes, matching the depth region. */
  readonly region: SizeSource;
  /**
   * Provides the texture and its region for `ao.at`. Share it before creating the pipelines that
   * shade, so a new depth size replaces the texture without replacing pipelines: give
   * `passes.create` `{ shared: { ao: occlusion.part } }`, or call `bindings.share(root, { ao:
   * occlusion.part })` for passes of your own. A pipeline that reads `ao.at` without the share
   * throws as it builds.
   */
  readonly part: bindings.Part;
  /** Fills `ao.at` for pipelines created from the configuration it is piped into. */
  inject(): Inject;
  /** Compiles the trace and denoise pipelines. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Records the trace and the denoise after the pass that draws `depth`, before the passes that
   * shade, into a frame's encoder, read at the call, or into an encoder. Optional timer writes
   * measure both passes. When disabled, it writes no timestamps, so use a profiler timer around
   * it only while `enabled` is true.
   */
  record(target: Frame | TgpuCommandEncoder, options?: profiler.Timed): void;
  /** Destroys the textures and the frame sequence. */
  destroy(): void;
}

const State = d.struct({
  /** Measures the drawn region in texels. */
  region: d.vec2f,
  /** Counts frames for `temporal`, cycling through `FRAMES`. */
  frame: d.u32,
});

const FRAME_OFFSET = d.memoryLayoutOf(State, (s) => s.frame).offset;
/** Frames before `temporal` repeats its directions. */
const FRAMES = 16;
/** Steps on each side of a slice. */
const STEPS = 3;
/** XeGTAO's defaults: the radius multiplier, the falloff range and the closest sample in texels. */
const REACH = 1.457;
const FALLOFF = 0.615;
const CLOSEST = 1.3;
/** Eye distances within this share of the distance count as the same surface. */
const SAME = 0.03;
/** The `ao.normal` clear value, outside the octahedral square, marks a missing normal. */
const UNWRITTEN = 4;
const HALF_PI = Math.PI / 2;
/** Full visibility at no distance, which `ao.at` matches nowhere. */
const OPEN: GPUColor = [1, 0, 0, 0];

/** Weighs a stored eye distance against another: 1 within `SAME`, 0 beyond twice that. */
const sameSurface = (stored: number, distance: number) => {
  'use gpu';
  return std.saturate(2 - std.abs(stored - distance) / std.max(std.abs(distance) * SAME, 1e-4));
};

/** Holds what `ao.at` reads, shared with the pipelines that shade. */
const shown = /* @__PURE__ */ bindings.layout({
  occlusion: { texture: d.texture2d(d.f32) },
  state: { uniform: State },
});

const Facing = d.struct({ normal: d.vec3f });

/**
 * This output writes the vertex `normal`, as `instances.Corner` carries it, into an `rg16float`
 * target for `ao.create`'s `normal`. It stores the world normal folded onto an octahedron.
 * Where nothing drew, occlusion takes normals from depth.
 */
export const normal = /* @__PURE__ */ passes.output(d.vec2f, {
  clear: d.vec2f(UNWRITTEN),
  varyings: Facing,
  vary: (corner: { readonly normal: d.v3f }) => {
    'use gpu';
    return Facing({ normal: corner.normal });
  },
  write: (_pixel: unknown, facing) => {
    'use gpu';
    return encode(std.normalize(facing.normal));
  },
});

/**
 * Returns the visibility of sky light at a world position, from 0 in a closed crease to 1 in the
 * open. It projects the position through `cameras.current` and blends the 4 nearest texels whose eye
 * distance matches. Where none matches, such as on a translucent surface in front of the opaque
 * one, it returns 1. Call it from shading, such as an `ambient` hook, in a description that shares
 * the occlusion's part.
 */
export const at = (position: d.v3f) => {
  'use gpu';
  const view = cameras.current.$;
  const clip = cameras.toClip(position);
  if (clip.w <= 0) return d.f32(1);

  const region = shown.$.state.region;
  const uv = (clip.xy / clip.w) * d.vec2f(0.5, -0.5) + 0.5;
  const texel = uv * region - 0.5;
  const corner = std.floor(texel);
  const f = texel - corner;
  const distance = std.dot(position - view.position, cameras.forwardOf(view));
  const last = d.vec2i(region) - 1;

  let sum = d.f32(0);
  let total = d.f32(0);
  for (const y of tgpu.unroll([0, 1])) {
    for (const x of tgpu.unroll([0, 1])) {
      const near = std.clamp(d.vec2i(corner) + d.vec2i(x, y), d.vec2i(), last);
      const stored = std.textureLoad(shown.$.occlusion, near, 0).xy;
      const across = (x === 1 ? f.x : 1 - f.x) * (y === 1 ? f.y : 1 - f.y);
      const weight = across * sameSurface(stored.y, distance);
      sum += stored.x * weight;
      total += weight;
    }
  }

  if (total < 1e-4) return d.f32(1);
  return sum / total;
};

/**
 * A point in view space, x right, y up and z away from the eye, at a pixel centre. `project`
 * maps a pixel to x and y at a clip w of 1, scales in xy and offsets in zw, and `w` holds the
 * clip w at the eye and its change with distance, so x and y grow with distance under
 * perspective and stay fixed under orthographic projection.
 */
const placed = (pixel: d.v2f, distance: number, project: d.v4f, w: d.v2f) => {
  'use gpu';
  return d.vec3f((pixel * project.xy + project.zw) * (w.x + distance * w.y), distance);
};

const traceLayout = tgpu.bindGroupLayout({ depth: { texture: d.textureDepth2d() } });
const facingLayout = tgpu.bindGroupLayout({
  depth: { texture: d.textureDepth2d() },
  normal: { texture: d.texture2d(d.f32) },
});
const denoiseLayout = tgpu.bindGroupLayout({ raw: { texture: d.texture2d(d.f32) } });

/**
 * Creates ground-truth ambient occlusion (GTAO) from a depth texture drawn before shading. As
 * Jimenez and others describe it in 2016, a trace walks `slices` directions per pixel with 3 steps
 * each way and XeGTAO's falloff, and a denoise blurs 3 × 3 pixels of the same surface. Shading reads the result through
 * `ao.at`, usually in an `ambient` hook: `(material, point) => shading.hemisphere(material,
 * point) * ao.at(point.position)`, in a description that shares the occlusion's part.
 *
 * @injects hook the texture and region `ao.at` reads, from the part
 * @binds 0 storage, 1 uniform: `ao.at`, the region and frame, beside the texture
 * @allocates on resize: two textures for a new depth size and the bind groups of its passes; so
 * does a new depth or normal texture
 * @owns targets, 2 or 3 buffers, 2 pipelines, 2 bind groups: the raw and the denoised occlusion;
 * the settings, the region and frame, and with `temporal` the frame indices; the trace and the
 * denoise, with a group each
 * @records `record` after the pass that draws `depth`, before the passes that shade; with
 * `temporal`, a copy advances the frame.
 * @param view the camera that drew `depth`, which shading's `cameras.current` must match
 * @param options.depth the depth drawn before shading, read at every `record`
 * @param options.normal normals from `ao.normal` drawn with `depth`, read at every `record`
 * @param options.region the part of the depth that holds the picture, read at every `record`
 * @returns occlusion you record every frame and destroy
 */
export function create(root: TgpuRoot, view: CameraBase, options: Options): Ao {
  const { depth, normal: normals, temporal = false, slices = 2, region } = options;
  const { enabled = () => true } = options;
  if (![1, 2, 3].includes(slices)) {
    throw new RangeError(`The occlusion takes 1 to 3 slices, got ${slices}.`);
  }
  const facing = normals !== undefined;
  const pictures = facing ? facingLayout : traceLayout;

  // The depth as `record` or `output` last took it, which the occlusion follows
  let current = depth();
  const full: SizeSource = {
    get width() {
      return current.props.size[0];
    },
    get height() {
      return current.props.size[1] ?? 1;
    },
  };

  const maps = targets.create(root, full, {
    raw: { format: 'rg16float', sampled: true },
    occlusion: { format: 'rg16float', sampled: true },
  });
  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });
  const state = root.createBuffer(State).$usage('uniform');
  const stateOf = state.as('uniform');
  const sequence = temporal
    ? root.createBuffer(
        d.arrayOf(d.u32, FRAMES),
        Array.from({ length: FRAMES }, (_, i) => i),
      )
    : undefined;
  const part = shown.bind(
    { occlusion: maps.textures.occlusion, state },
    {
      unshared:
        'Share the occlusion texture before creating pipelines that read `ao.at`. Occlusion replaces this texture when its depth texture changes size. Give `passes.create` the option `{ shared: { ao: occlusion.part } }`, or call `bindings.share(root, { ao: occlusion.part })` for your own passes.',
    },
  );

  const depthAt = (pixel: d.v2i) => {
    'use gpu';
    const last = d.vec2i(stateOf.$.region) - 1;
    return std.textureLoad(pictures.$.depth, std.clamp(pixel, d.vec2i(), last), 0);
  };

  const pointAt = (pixel: d.v2i, project: d.v4f, w: d.v2f) => {
    'use gpu';
    const distance = cameras.distanceOf(cameras.current.$, depthAt(pixel));
    return placed(d.vec2f(pixel) + 0.5, distance, project, w);
  };

  /** The normal from the neighbour on each axis whose depth is nearer the centre's. */
  const fromDepth = (pixel: d.v2i, center: d.v3f, project: d.v4f, w: d.v2f) => {
    'use gpu';
    const last = d.vec2i(stateOf.$.region) - 1;
    const left = pointAt(pixel - d.vec2i(1, 0), project, w);
    const right = pointAt(pixel + d.vec2i(1, 0), project, w);
    const above = pointAt(pixel - d.vec2i(0, 1), project, w);
    const below = pointAt(pixel + d.vec2i(0, 1), project, w);

    const rightward =
      pixel.x === 0 ||
      (pixel.x < last.x && std.abs(right.z - center.z) < std.abs(center.z - left.z));
    const downward =
      pixel.y === 0 ||
      (pixel.y < last.y && std.abs(below.z - center.z) < std.abs(center.z - above.z));
    const across = std.select(center - left, right - center, rightward);
    const down = std.select(center - above, below - center, downward);
    return std.normalize(std.cross(across, down));
  };

  const normalAt = (pixel: d.v2i, center: d.v3f, project: d.v4f, w: d.v2f) => {
    'use gpu';
    if (facing) {
      const folded = std.textureLoad(facingLayout.$.normal, pixel, 0).xy;
      if (folded.x < UNWRITTEN / 2) {
        const world = decode(folded);
        const eye = cameras.current.$;
        return d.vec3f(
          std.dot(world, eye.right),
          std.dot(world, eye.up),
          std.dot(world, cameras.forwardOf(eye)),
        );
      }
    }
    return fromDepth(pixel, center, project, w);
  };

  /**
   * The cosine of the horizon a sample gives, eased to the lowest one by XeGTAO's falloff, the
   * reach and one over its falloff range in `falloff`.
   */
  const horizonOf = (
    sample: d.v2f,
    center: d.v3f,
    toEye: d.v3f,
    low: number,
    project: d.v4f,
    w: d.v2f,
    falloff: d.v2f,
  ) => {
    'use gpu';
    const pixel = d.vec2i(std.floor(sample));
    const distance = cameras.distanceOf(cameras.current.$, depthAt(pixel));
    const delta = placed(sample, distance, project, w) - center;
    const length = std.length(delta);
    // A sample off the region sees nothing, so it leaves the horizon low
    const bounds = stateOf.$.region;
    const inside = sample.x >= 0 && sample.y >= 0 && sample.x < bounds.x && sample.y < bounds.y;
    const weight = std.select(0, std.saturate((falloff.x - length) * falloff.y), inside);
    return std.mix(low, std.dot(delta / std.max(length, 1e-6), toEye), weight);
  };

  const trace = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })(({ position }) => {
    'use gpu';
    const pixel = d.vec2i(position.xy);
    const z = depthAt(pixel);
    const eye = cameras.current.$;
    const lens = cameras.lensOf(eye);
    const w = d.vec2f(
      (eye.viewProjection * d.vec4f(eye.position, 1)).w,
      (eye.viewProjection * d.vec4f(cameras.forwardOf(eye), 0)).w,
    );
    const distance = cameras.distanceOf(eye, z);
    if (cameras.reversed.$ ? z <= 0 : z >= 1) return d.vec4f(1, distance, 0, 0);

    const scale = d.vec2f(2, -2) / (lens * stateOf.$.region);
    const project = d.vec4f(scale, d.vec2f(-1, 1) / lens);
    const center = placed(position.xy, distance, project, w);
    // Orthographic views look along z from every pixel
    const toEye = std.select(d.vec3f(0, 0, -1), std.normalize(center * -1), w.y > 0);
    const facingAt = normalAt(pixel, center, project, w);

    // The radius in texels at this distance
    const reach = settings.$.radius * REACH;
    const falloff = d.vec2f(reach, 1 / (FALLOFF * reach));
    const spread = reach / ((w.x + distance * w.y) * scale.x);
    if (spread < CLOSEST) return d.vec4f(1, distance, 0, 0);
    const closest = CLOSEST / spread;

    // Each 3 × 3 block turns its slices through 9 even steps, which the denoise averages. With
    // `temporal`, every frame turns them all further
    let spin = d.f32(0.5);
    if (temporal) {
      randf.seed(d.f32(stateOf.$.frame));
      spin = randf.sample();
      randf.seed3(d.vec3f(position.xy, d.f32(stateOf.$.frame)));
    } else randf.seed2(position.xy);
    const block = d.vec2u(position.xy) % 3;
    const turn = (d.f32(block.x + block.y * 3) + spin) / 9;
    const jitter = randf.sample();

    // Fade out for small screen radii, as XeGTAO does
    let visibility = std.saturate((10 - spread) / 100) * 0.5;
    for (const slice of tgpu.unroll(std.range(slices))) {
      const phi = ((slice + turn) / slices) * Math.PI;
      const direction = d.vec3f(std.cos(phi), std.sin(phi), 0);
      const omega = d.vec2f(direction.x, -direction.y) * spread;

      const ortho = direction - toEye * std.dot(direction, toEye);
      const axis = std.normalize(std.cross(ortho, toEye));
      const projected = facingAt - axis * std.dot(facingAt, axis);
      const projectedLength = std.length(projected);
      const cosNormal = std.saturate(std.dot(projected, toEye) / std.max(projectedLength, 1e-6));
      const n = std.sign(std.dot(ortho, projected)) * std.acos(cosNormal);

      const low0 = std.cos(n + HALF_PI);
      const low1 = std.cos(n - HALF_PI);
      let horizon0 = d.f32(low0);
      let horizon1 = d.f32(low1);
      for (const step of tgpu.unroll(std.range(STEPS))) {
        // Offset each step along the golden-ratio sequence, as XeGTAO does
        const shift = std.fract(jitter + (slice + step * STEPS) * 0.6180339887498949);
        const s = ((step + shift) / STEPS) * ((step + shift) / STEPS) + closest;
        const offset = std.round(omega * s);
        const ahead = position.xy + offset;
        const behind = position.xy - offset;
        horizon0 = std.max(horizon0, horizonOf(ahead, center, toEye, low0, project, w, falloff));
        horizon1 = std.max(horizon1, horizonOf(behind, center, toEye, low1, project, w, falloff));
      }

      const h0 = n + std.max(-std.acos(horizon1) - n, -HALF_PI);
      const h1 = n + std.min(std.acos(horizon0) - n, HALF_PI);
      const arc0 = (cosNormal + 2 * h0 * std.sin(n) - std.cos(2 * h0 - n)) / 4;
      const arc1 = (cosNormal + 2 * h1 * std.sin(n) - std.cos(2 * h1 - n)) / 4;
      // XeGTAO eases the projected length towards 1 against darkening on steep slopes
      visibility += std.mix(projectedLength, 1, 0.05) * (arc0 + arc1);
    }

    return d.vec4f(visibility / slices, distance, 0, 0);
  });

  const denoise = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })(({ position }) => {
    'use gpu';
    const pixel = d.vec2i(position.xy);
    const last = d.vec2i(stateOf.$.region) - 1;
    const middle = std.textureLoad(denoiseLayout.$.raw, pixel, 0).xy;

    let sum = d.f32(0);
    let total = d.f32(0);
    for (const y of tgpu.unroll([-1, 0, 1])) {
      for (const x of tgpu.unroll([-1, 0, 1])) {
        const near = std.clamp(pixel + d.vec2i(x, y), d.vec2i(), last);
        const stored =
          x === 0 && y === 0 ? d.vec2f(middle) : std.textureLoad(denoiseLayout.$.raw, near, 0).xy;
        const weight = sameSurface(stored.y, middle.y);
        sum += stored.x * weight;
        total += weight;
      }
    }

    const open = std.pow(std.saturate(sum / total), settings.$.power);
    return d.vec4f(std.max(open, 0.03), middle.y, 0, 0);
  });

  const traced = root.pipe(view.inject()).createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: trace,
    targets: { format: 'rg16float' },
  });
  const denoised = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: denoise,
    targets: { format: 'rg16float' },
  });

  // Keep each bind group on its pipeline so recording only sets the pipeline on the pass
  const tracing = memo((deep: Depth, faced: Normal | undefined) => {
    const sampled = deep.createView(d.textureDepth2d());
    const group = faced
      ? root.createBindGroup(facingLayout, { depth: sampled, normal: faced })
      : root.createBindGroup(traceLayout, { depth: sampled });
    return traced.with(group);
  });
  const denoising = memo((raw: TgpuTexture & SampledFlag) =>
    denoised.with(root.createBindGroup(denoiseLayout, { raw })),
  );
  const into = passInto(OPEN);

  const written = { width: 0, height: 0 };
  const measure = () => {
    const width = Math.min(full.width, region?.width ?? full.width);
    const height = Math.min(full.height, region?.height ?? full.height);
    if (width === written.width && height === written.height) return;
    state.patch({ region: d.vec2f(width, height) });
    written.width = width;
    written.height = height;
  };

  // Whether a disabled occlusion cleared its output, which a resize replaces
  let cleared = false;
  const follow = () => {
    current = depth();
    if (!maps.fit()) return;
    cleared = false;
    part.replace({ occlusion: maps.textures.occlusion });
  };
  measure();

  let index = 0;

  const destroy = () => {
    maps.destroy();
    sequence?.destroy();
  };

  const occluded: Ao = {
    settings,

    get output() {
      follow();
      return maps.textures.occlusion;
    },

    region: {
      get width() {
        return written.width;
      },

      get height() {
        return written.height;
      },
    },

    part,

    inject: () => part.inject(),

    initAsync: () => pipelines.compile([traced, denoised]),

    record(target, timing) {
      const encoder = encoderOf(target);
      follow();
      if (!enabled()) {
        if (!cleared) encoder.beginRenderPass(into(maps.views.occlusion)).end();
        cleared = true;
        if (timing?.timestampWrites) profiler.untimedFrame(encoder, timing.timestampWrites);
        return;
      }
      cleared = false;
      measure();
      const { opening, closing } = profiler.split(timing?.timestampWrites);
      const below = written.width < full.width || written.height < full.height;

      const tracePass = encoder.beginRenderPass(into(maps.views.raw, opening));
      if (below) tracePass.setViewport(0, 0, written.width, written.height, 0, 1);
      tracePass.setPipeline(tracing(current, normals?.()));
      tracePass.draw(3);
      tracePass.end();

      const denoisePass = encoder.beginRenderPass(into(maps.views.occlusion, closing));
      if (below) denoisePass.setViewport(0, 0, written.width, written.height, 0, 1);
      denoisePass.setPipeline(denoising(maps.textures.raw));
      denoisePass.draw(3);
      denoisePass.end();

      if (!sequence) return;
      // Copy the next frame index through the encoder to avoid a CPU write each frame
      index = (index + 1) % FRAMES;
      root
        .unwrap(encoder)
        .copyBufferToBuffer(
          root.unwrap(sequence),
          index * d.sizeOf(d.u32),
          root.unwrap(state),
          FRAME_OFFSET,
          d.sizeOf(d.u32),
        );
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
  return occluded;
}
