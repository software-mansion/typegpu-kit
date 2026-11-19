import * as bindings from '../bindings.ts';
import * as cameras from '../cameras.ts';
import * as pipelines from '../pipelines.ts';
import * as targets from '../targets.ts';
import * as profiler from '../profiler.ts';
import { type CameraBase } from '../cameras.ts';
import { type Inject } from '../inject.ts';
import { memo, passOf } from '../memo.ts';
import { type SizeSource, type Depth } from '../targets.ts';
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
import { dtOf, encoderOf, type Frame } from '../frames.ts';
import {
  regionOf,
  textureOf,
  within,
  type FloatTexture,
  type Rendered,
  type Source,
} from './post.ts';

/** WebGPU guarantees render, filter and blend support for this format. */
const FORMAT = 'rgba16float';
/** Pixels on each side of a tile, whose largest radius scales the gather around it. */
const TILE = 16;
const HALF_TILE = TILE / 2;

export const Settings = d.struct({
  /** Sets the distance in focus, in world units along the view direction. */
  focus: d.f32,
  /**
   * Sets the diameter of the lens opening, in world units. A 50 mm lens at f/2 in a scene in
   * metres opens 0.025.
   */
  aperture: d.f32,
  /** Sets the largest blur radius, as a fraction of the picture height. */
  limit: d.f32,
});

export const defaults: d.Infer<typeof Settings> = {
  focus: 10,
  aperture: 0.05,
  limit: 0.02,
};

/** Returns the signed blur radius, as a fraction of the picture height, at an eye distance. */
export type Radius = (distance: number, uv: d.v2f) => number;

export interface Options {
  /**
   * Provides the linear picture: a rendered picture, such as `aa`, or a function returning a
   * texture. `record` reads it on every call, so the texture can be rebuilt.
   * @cpu
   */
  readonly source: Source;
  /**
   * Returns a sampled depth texture of the same picture, of any size, such as a half-resolution
   * prepass. `record` reads it on every call. Its drawn part is the source's share of its texture.
   * @cpu
   */
  readonly depth: () => Depth;
  /**
   * Returns the signed blur radius at an eye distance and a point of the picture, from 0 to 1 over
   * its region, as a fraction of the picture height: negative in front of the focus, positive
   * behind it. This `'use gpu'` function runs once per half-resolution texel and once per pixel.
   * It can call `thinLens`, `distanceAt` and `focus` and carry fills with `pipe`. Default
   * `thinLens` at `focus()`.
   * @gpu
   */
  readonly radius?: Radius;
  /**
   * Sets how fast the focus follows `aim`, per second: each second closes `1 - e^-pull` of the
   * way, evenly in blur rather than in distance. With it, each `record` runs the kernel that moves
   * the focus. Default `Infinity`, which focuses at once from the `record` after `aim`.
   */
  readonly pull?: number;
  /** Sets the rings of taps in the gather, 3 for 49 taps or 2 for 25. Default 3. */
  readonly rings?: 2 | 3;
  /**
   * Selects the source region from its top-left corner, such as `regions.hdr` when drawing below
   * full resolution. `record` reads it on every call. Default the region of a rendered `source`,
   * else the whole texture.
   */
  readonly region?: SizeSource;
  /**
   * Sets initial values over `defaults`. Default `defaults`, with focus 10, aperture 0.05 and
   * limit 0.02.
   */
  readonly settings?: Partial<d.InferInput<typeof Settings>>;
}

export interface Dof extends Rendered, Disposable {
  /**
   * Holds `defaults` with overrides from `settings`. Use `patch` to upload only the fields you
   * supply.
   */
  readonly settings: TgpuUniform<typeof Settings>;
  /**
   * Provides the full-size blurred `rgba16float` picture. Use it as the `source` of bloom and the
   * final pass. A new source size replaces the texture.
   */
  readonly output: FloatTexture;
  /** Gives the region of `output` that `record` writes, matching the source region. */
  readonly region: SizeSource;
  /**
   * Provides the half-resolution picture and its regions for `dof.circle`. Share it before
   * creating the pipelines that read it, so a new source size replaces the texture without
   * replacing pipelines: `bindings.share(root, { blur: blur.part })` for the final pass, or
   * `{ shared: { blur: blur.part } }` on a `passes.create` whose drawings read it.
   */
  readonly part: bindings.Part;
  /** Fills `dof.circle` for pipelines created from the configuration it is piped into. */
  inject(): Inject;
  /**
   * Focuses on what the depth holds at a point of the picture, from 0 to 1 over its region, such
   * as under a tap. The next `record` reads the distance there, and from then on each `record`
   * moves `settings.focus` towards it on the GPU at `pull`.
   */
  aim(x: number, y: number): void;
  /** Compiles the depth of field pipelines. Calling it is optional. */
  initAsync(): Promise<void>;
  /**
   * Records the prepare, tiles, gather and composite passes after the source is written. After an
   * `aim`, it first moves the focus by `dt` seconds. Optional timer writes measure the passes
   * together.
   */
  record(target: Frame | TgpuCommandEncoder, options?: RecordOptions): void;
  /** Destroys the textures and the focus buffer. */
  destroy(): void;
}

export interface RecordOptions extends profiler.Timed {
  /** Sets the seconds the focus moves by. Default the frame's `dt`, or all the way. */
  readonly dt?: number;
}

/** Where `aim` asked to focus, and the focus that follows it. */
const Focus = d.struct({
  /** The point `aim` asked for, from 0 to 1 over the picture. */
  aim: d.vec2f,
  /** 1 while an `aim` waits for the kernel to read the distance there. */
  asked: d.u32,
  /** 1 once the kernel has read an aimed distance, so the focus moves towards it. */
  aimed: d.u32,
  /** The distance the focus moves towards, and the distance in focus now. */
  goal: d.f32,
  focus: d.f32,
});
const FOCUS_OFFSET = d.memoryLayoutOf(Focus, (focus) => focus.focus).offset;
const SETTINGS_FOCUS = d.memoryLayoutOf(Settings, (settings) => settings.focus).offset;

/** Sizes in texels: the source region, its half, its tiles, and the depth's share. */
const Regions = d.struct({
  source: d.vec2f,
  half: d.vec2f,
  tiles: d.vec2f,
  depth: d.vec2f,
});

/**
 * Steps through the gather's rings: ring `r` of `rings` holds `8r` taps, odd rings turned half a
 * step. A quarter turn maps a ring onto itself, so each step reads four taps a quarter turn apart
 * and turns its direction by one tap. Gives each ring's steps, share of the reach, start direction
 * and one tap's turn as cosine and sine.
 */
const ringSteps = (rings: number) =>
  Array.from({ length: rings }, (_ring, r) => {
    const ring = r + 1;
    const turn = (2 * Math.PI) / (ring * 8);
    const start = (ring % 2) * 0.5 * turn;
    return [
      ring * 2,
      ring / rings,
      Math.cos(start),
      Math.sin(start),
      Math.cos(turn),
      Math.sin(turn),
    ] as const;
  });

const prepareLayout = tgpu.bindGroupLayout({ source: { texture: d.texture2d(d.f32) } });
const depthLayout = tgpu.bindGroupLayout({ depth: { texture: d.textureDepth2d() } });
const tilesLayout = tgpu.bindGroupLayout({ prepared: { texture: d.texture2d(d.f32) } });
const gatherLayout = tgpu.bindGroupLayout({
  prepared: { texture: d.texture2d(d.f32) },
  tiles: { texture: d.texture2d(d.f32) },
});
const compositeLayout = tgpu.bindGroupLayout({
  source: { texture: d.texture2d(d.f32) },
  prepared: { texture: d.texture2d(d.f32) },
  blurred: { texture: d.texture2d(d.f32) },
  sampler: { sampler: 'filtering' },
});

/** Holds what `dof.circle` reads, for pipelines that share it. */
const shown = bindings.layout({
  prepared: { texture: d.texture2d(d.f32) },
  regions: { uniform: Regions },
});

const outside = (): never => {
  throw new Error(
    'Call `dof.thinLens`, `dof.distanceAt` and `dof.focus` only from the `radius` callback of `dof.create`.',
  );
};
const settingsOf = tgpu.accessor(Settings, outside);
const regionsOf = tgpu.accessor(Regions, outside);
const radiusOf = tgpu.slot<Radius>();

const depthAt = (uv: d.v2f) => {
  'use gpu';
  const region = regionsOf.$.depth;
  const texel = std.clamp(d.vec2i(uv * region), d.vec2i(), d.vec2i(region) - 1);
  return std.textureLoad(depthLayout.$.depth, texel, 0);
};

/**
 * Returns the distance along the view direction to what the depth holds at a point of the
 * picture, from 0 to 1 over its region. Call it only from the `radius` callback, such as
 * `dof.thinLens(distance, dof.distanceAt(cursor.$))` to focus under a pointer.
 */
export const distanceAt = (uv: d.v2f) => {
  'use gpu';
  return cameras.distanceOf(cameras.current.$, depthAt(uv));
};

/**
 * Returns the thin-lens blur radius at an eye distance with `focus` in focus and the settings'
 * aperture, as a fraction of the picture height. It is negative in front of the focus. It reads
 * the settings, so call it only from the `radius` callback.
 */
export const thinLens = (distance: number, focus: number) => {
  'use gpu';
  // The focal length over the sensor height
  const focal = cameras.lensOf(cameras.current.$).y / 2;
  return 0.5 * settingsOf.$.aperture * focal * (1 / focus - 1 / distance);
};

/**
 * Returns the distance in focus, `settings.focus`, which follows `aim`. Call it only from the
 * `radius` callback.
 */
export const focus = () => {
  'use gpu';
  return settingsOf.$.focus;
};

const atFocus = (distance: number, _uv: d.v2f) => {
  'use gpu';
  return thinLens(distance, focus());
};

/**
 * Returns the signed blur radius `record` last prepared at a point of the picture, from 0 to 1
 * over its region, as a fraction of the picture height. Call it from a pipeline piped with
 * `blur.inject()`, such as a final pass's `grade` showing the focus.
 */
export const circle = (uv: d.v2f) => {
  'use gpu';
  const half = shown.$.regions.half;
  const texel = std.clamp(d.vec2i(uv * half), d.vec2i(), d.vec2i(half) - 1);
  return std.textureLoad(shown.$.prepared, texel, 0).w;
};

const radiusAt = (distance: number, uv: d.v2f) => {
  'use gpu';
  const limit = settingsOf.$.limit;
  return std.clamp(radiusOf.$(distance, uv), -limit, limit);
};

/** Averages 2 × 2 pixels and stores the radius at the nearest of their depths in alpha. */
const prepare = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const region = regionsOf.$.source;
  const corner = d.vec2i(position.xy) * 2;
  const last = d.vec2i(region) - 1;
  let color = d.vec3f();
  let z = d.f32(cameras.farDepth());
  for (const y of tgpu.unroll([0, 1])) {
    for (const x of tgpu.unroll([0, 1])) {
      const texel = std.min(corner + d.vec2i(x, y), last);
      color += std.textureLoad(prepareLayout.$.source, texel, 0).xyz;
      z = cameras.nearerDepth()(z, depthAt((d.vec2f(texel) + 0.5) / region));
    }
  }

  const uv = (d.vec2f(corner) + 1) / region;
  return d.vec4f(color * 0.25, radiusAt(cameras.distanceOf(cameras.current.$, z), uv));
});

/** Takes the largest radius in the half-resolution texels of a tile. */
const tile = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const first = d.vec2i(position.xy) * HALF_TILE;
  const last = d.vec2i(regionsOf.$.half) - 1;
  let most = d.f32(0);
  for (const y of std.range(HALF_TILE)) {
    for (const x of std.range(HALF_TILE)) {
      const texel = std.min(first + d.vec2i(d.i32(x), d.i32(y)), last);
      most = std.max(most, std.abs(std.textureLoad(tilesLayout.$.prepared, texel, 0).w));
    }
  }
  return d.vec4f(most, 0, 0, 0);
});

/**
 * Adds the source's detail above the half-resolution picture back onto the blur, where the radius
 * at the pixel is under a pixel and the near field leaves it uncovered.
 */
const composite = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const regions = regionsOf.$;
  const sharp = std.textureLoad(compositeLayout.$.source, d.vec2i(position.xy), 0);
  const uv = position.xy / regions.source;
  const radius = radiusAt(distanceAt(uv), uv);

  const size = d.vec2f(std.textureDimensions(compositeLayout.$.blurred));
  const at = within((uv * regions.half) / size, regions.half, size);
  const blurred = std.textureSampleLevel(
    compositeLayout.$.blurred,
    compositeLayout.$.sampler,
    at,
    0,
  );
  const half = std.textureSampleLevel(
    compositeLayout.$.prepared,
    compositeLayout.$.sampler,
    at,
    0,
  ).xyz;
  // Below a pixel the source is sharp, and from 3 pixels the half-resolution blur is
  const pixels = std.abs(radius) * regions.source.y;
  const sharpness = (1 - std.saturate((pixels - 1) / 2)) * (1 - blurred.w);
  const color = blurred.xyz + (sharp.xyz - half) * sharpness;
  return d.vec4f(std.max(color, d.vec3f()), sharp.w);
});

/**
 * Creates depth of field. A prepare pass halves the source and stores a signed blur radius from
 * `radius`. A tiles pass finds the largest radius in each 16 × 16 pixels. A gather pass scales
 * rings of taps to the largest radius around each tile, keeps the far field from bleeding onto
 * what is sharper and spreads the near field over it, as Jimenez describes it in 2014. A composite
 * pass mixes the blur into the full-size source by the radius at each pixel.
 *
 * @injects hook the half-resolution picture and regions `dof.circle` reads, from the part
 * @binds 0 storage, 1 uniform: `dof.circle`, the regions, beside the texture
 * @allocates on resize: targets for the new size and the bind groups of its passes; so does a new
 * source or depth
 * @owns targets, 2 buffers, a sampler, 4 pipelines, 5 bind groups: the half-resolution picture,
 * the tiles, the blur and the output; the settings and the regions; a group for each pass and one
 * for the depth. With `pull` or after `aim`, also the focus and its step, the kernel moving it and
 * its groups
 * @records `record` after the TAA resolve or whatever wrote the source, before bloom and the final
 * pass; then `output` holds the picture.
 * @param view the camera that drew `depth`
 * @param options.source the linear picture, read at every `record`
 * @param options.depth the depth of the same picture, read at every `record`
 * @param options.region the part of the source that holds the picture, read at every `record`
 * @param options.pull how fast the focus follows `aim`, per second
 * @returns depth of field you record every frame and destroy
 */
export function create(root: TgpuRoot, view: CameraBase, options: Options): Dof {
  const { depth, rings = 3, pull = Infinity } = options;
  const source = textureOf(options.source);
  const region = regionOf(options.source, options.region);
  if (rings !== 2 && rings !== 3) {
    throw new RangeError(`Depth of field takes 2 or 3 rings, got ${rings}.`);
  }
  if (!(pull > 0)) {
    throw new RangeError(`Depth of field takes a pull above 0 per second, got ${pull}.`);
  }

  const full: SizeSource = {
    get width() {
      return source().props.size[0];
    },
    get height() {
      return source().props.size[1] ?? 1;
    },
  };

  const chain = targets.create(root, full, {
    prepared: { format: FORMAT, scale: 0.5, sampled: true },
    tiles: { format: 'r16float', scale: 1 / TILE, sampled: true },
    blurred: { format: FORMAT, scale: 0.5, sampled: true },
    output: { format: FORMAT, sampled: true },
  });
  const settings = root.createUniform(Settings, { ...defaults, ...options.settings });
  const regions = root.createBuffer(Regions).$usage('uniform');
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const part = shown.bind({ prepared: chain.textures.prepared, regions });
  const state = root.createBuffer(Focus).$usage('storage');
  const focusing = state.as('mutable');
  const pace = root.createUniform(d.f32);
  const steps = ringSteps(rings);
  // The centre and 8r taps on each ring r
  const taps = 1 + 4 * rings * (rings + 1);

  /** Gathers rings of taps around each half-resolution texel, near and far fields apart. */
  const gather = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })(({ position }) => {
    'use gpu';
    const regionsNow = regionsOf.$;
    const pixel = d.vec2i(position.xy);
    const last = d.vec2i(regionsNow.half) - 1;
    const center = std.textureLoad(gatherLayout.$.prepared, pixel, 0);

    // Radii in half-resolution texels
    const scale = regionsNow.half.y;

    // The largest radius in the tiles that the largest circle `limit` allows can reach from here
    const tileAt = d.vec2i(std.floor(position.xy / HALF_TILE));
    const lastTile = d.vec2i(regionsNow.tiles) - 1;
    const tiles = d.i32(std.max(std.ceil((settingsOf.$.limit * scale) / HALF_TILE), 1));
    let most = d.f32(0);
    for (let y = -tiles; y <= tiles; y++) {
      for (let x = -tiles; x <= tiles; x++) {
        const around = std.clamp(tileAt + d.vec2i(x, y), d.vec2i(), lastTile);
        most = std.max(most, std.textureLoad(gatherLayout.$.tiles, around, 0).x);
      }
    }
    const spread = most * scale;
    if (spread < 0.5) return d.vec4f(center.xyz, 0);

    const self = std.abs(center.w) * scale;
    const selfWeight = 1 / (Math.PI * std.max(self, 0.5) * std.max(self, 0.5));
    let far = d.vec4f();
    let near = d.vec4f();
    if (center.w < 0) near = d.vec4f(center.xyz, 1) * selfWeight;
    else far = d.vec4f(center.xyz, 1) * selfWeight;

    for (const step of tgpu.unroll(steps)) {
      const reach = step[1] * spread;
      let direction = d.vec2f(step[2], step[3]);
      for (let j = d.u32(0); j < step[0]; j++) {
        const quarters = [
          d.vec2f(direction),
          d.vec2f(-direction.y, direction.x),
          std.neg(direction),
          d.vec2f(direction.y, -direction.x),
        ];
        for (const quarter of tgpu.unroll([0, 1, 2, 3])) {
          const offset = quarters[quarter] * reach;
          const at = std.clamp(d.vec2i(std.floor(position.xy + offset)), d.vec2i(), last);
          const tap = std.textureLoad(gatherLayout.$.prepared, at, 0);
          const radius = std.abs(tap.w) * scale;
          // A far tap counts where the smaller of its circle and the centre's reaches the centre,
          // so a sharper centre keeps its edge. A near tap counts where its own circle reaches
          const spot = std.select(std.min(radius, self), radius, tap.w < 0);
          const cover = std.saturate(spot - reach + 0.5);
          const weight = cover / (Math.PI * std.max(spot, 0.5) * std.max(spot, 0.5));
          const sample = d.vec4f(tap.xyz, 1) * weight;
          if (tap.w < 0) near += sample;
          else far += sample;
        }
        direction = d.vec2f(
          step[4] * direction.x - step[5] * direction.y,
          step[5] * direction.x + step[4] * direction.y,
        );
      }
    }

    // Each tap stands for an equal share of the disc, which the near field covers this much of
    const alpha = std.saturate((near.w * Math.PI * spread * spread) / taps);
    const front = near.xyz / std.max(near.w, 1e-6);
    const back = std.select(far.xyz / std.max(far.w, 1e-6), front, far.w <= 0);
    return d.vec4f(std.mix(back, front, alpha), alpha);
  });

  /** Reads the distance under an asked aim, then moves the focus towards it, evenly in blur. */
  const move = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    if (focusing.$.asked === 1) {
      focusing.$.goal = distanceAt(focusing.$.aim);
      focusing.$.asked = 0;
      focusing.$.aimed = 1;
    }
    const now = settingsOf.$.focus;
    if (focusing.$.aimed === 0) {
      focusing.$.focus = now;
      return;
    }
    let share = d.f32(1);
    if (pull < Infinity) share = 1 - std.exp(-pace.$ * pull);
    focusing.$.focus = 1 / std.mix(1 / now, 1 / focusing.$.goal, share);
  });

  const configured = root
    .pipe(view.inject())
    .with(settingsOf, settings)
    .with(regionsOf, regions.as('uniform'))
    .with(radiusOf, options.radius ?? atFocus);
  const pipeline = (fragment: typeof prepare, format: 'rgba16float' | 'r16float' = FORMAT) =>
    configured.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment,
      targets: { format },
    });
  const preparing = pipeline(prepare);
  const tiling = pipeline(tile, 'r16float');
  const gathering = pipeline(gather);
  const compositing = pipeline(composite);
  const moving = root
    .pipe(view.inject())
    .with(settingsOf, settings)
    .with(regionsOf, regions.as('uniform'))
    .createComputePipeline({ compute: move });

  // Keep each bind group on its pipeline so recording only sets the pipeline on the pass
  const depthGroup = memo((deep: Depth) =>
    root.createBindGroup(depthLayout, { depth: deep.createView(d.textureDepth2d()) }),
  );
  const moved = memo((deep: Depth) => moving.with(depthGroup(deep)));
  const prepared = memo((picture: FloatTexture, deep: Depth) =>
    preparing.with(root.createBindGroup(prepareLayout, { source: picture })).with(depthGroup(deep)),
  );
  const tiled = memo((half: TgpuTexture & SampledFlag) =>
    tiling.with(root.createBindGroup(tilesLayout, { prepared: half })),
  );
  const gathered = memo((half: TgpuTexture & SampledFlag, tiles: TgpuTexture & SampledFlag) =>
    gathering.with(root.createBindGroup(gatherLayout, { prepared: half, tiles })),
  );
  const composited = memo(
    (
      picture: FloatTexture,
      half: TgpuTexture & SampledFlag,
      blurred: TgpuTexture & SampledFlag,
      deep: Depth,
    ) =>
      compositing
        .with(
          root.createBindGroup(compositeLayout, {
            source: picture,
            prepared: half,
            blurred,
            sampler,
          }),
        )
        .with(depthGroup(deep)),
  );
  const prepareInto = passOf();
  const tilesInto = passOf();
  const gatherInto = passOf();
  const compositeInto = passOf();

  const sizes = {
    source: [0, 0] as [number, number],
    half: [0, 0] as [number, number],
    tiles: [0, 0] as [number, number],
    depth: [0, 0] as [number, number],
  };
  const measure = (deep: Depth) => {
    const sourceWidth = full.width;
    const sourceHeight = full.height;
    const width = Math.min(sourceWidth, region?.width ?? sourceWidth);
    const height = Math.min(sourceHeight, region?.height ?? sourceHeight);
    const depthWidth = (deep.props.size[0] * width) / sourceWidth;
    const depthHeight = ((deep.props.size[1] ?? 1) * height) / sourceHeight;

    const same =
      sizes.source[0] === width &&
      sizes.source[1] === height &&
      sizes.depth[0] === depthWidth &&
      sizes.depth[1] === depthHeight;
    if (same) return;

    sizes.source = [width, height];
    sizes.half = [Math.ceil(width / 2), Math.ceil(height / 2)];
    sizes.tiles = [Math.ceil(width / TILE), Math.ceil(height / TILE)];
    sizes.depth = [depthWidth, depthHeight];
    regions.write(sizes);
  };

  // With a pull, the focus moves from the first frame, so a later `aim` creates nothing
  let aimed = options.pull !== undefined;

  const follow = () => {
    if (!chain.fit()) return;
    try {
      part.replace({ prepared: chain.textures.prepared });
    } catch (error) {
      throw new Error(
        'Depth of field follows the size of its source, which needs its picture in a layout you own: `bindings.share(root, { blur: blur.part })` before creating the pipelines that read it.',
        { cause: error },
      );
    }
  };
  measure(depth());

  const destroy = () => {
    chain.destroy();
    state.destroy();
  };

  const blur: Dof = {
    settings,

    get output() {
      follow();
      return chain.textures.output;
    },

    region: {
      get width() {
        return sizes.source[0];
      },

      get height() {
        return sizes.source[1];
      },
    },

    part,

    inject: () => part.inject(),

    initAsync: () => pipelines.compile([preparing, tiling, gathering, compositing, moving]),

    aim(x, y) {
      aimed = true;
      state.patch({ aim: d.vec2f(x, y), asked: 1 });
    },

    record(target, recording) {
      follow();
      const encoder = encoderOf(target);
      const picture = source();
      const deep = depth();
      measure(deep);
      const { opening, closing } = profiler.split(recording?.timestampWrites);
      const { textures, views } = chain;

      if (aimed) {
        pace.write(dtOf(target, recording?.dt));
        const movePass = encoder.beginComputePass();
        movePass.setPipeline(moved(deep));
        movePass.dispatchWorkgroups(1);
        movePass.end();
        // Copy the focus into the settings through the encoder, so the passes below read it
        root
          .unwrap(encoder)
          .copyBufferToBuffer(
            root.unwrap(state),
            FOCUS_OFFSET,
            root.unwrap(settings.buffer),
            SETTINGS_FOCUS,
            d.sizeOf(d.f32),
          );
      }

      const first = prepareInto(views.prepared);
      first.timestampWrites = opening;
      const preparePass = encoder.beginRenderPass(first);
      preparePass.setViewport(0, 0, sizes.half[0], sizes.half[1], 0, 1);
      preparePass.setPipeline(prepared(picture, deep));
      preparePass.draw(3);
      preparePass.end();

      const tilesPass = encoder.beginRenderPass(tilesInto(views.tiles));
      tilesPass.setViewport(0, 0, sizes.tiles[0], sizes.tiles[1], 0, 1);
      tilesPass.setPipeline(tiled(textures.prepared));
      tilesPass.draw(3);
      tilesPass.end();

      const gatherPass = encoder.beginRenderPass(gatherInto(views.blurred));
      gatherPass.setViewport(0, 0, sizes.half[0], sizes.half[1], 0, 1);
      gatherPass.setPipeline(gathered(textures.prepared, textures.tiles));
      gatherPass.draw(3);
      gatherPass.end();

      const last = compositeInto(views.output);
      last.timestampWrites = closing;
      const compositePass = encoder.beginRenderPass(last);
      compositePass.setViewport(0, 0, sizes.source[0], sizes.source[1], 0, 1);
      compositePass.setPipeline(composited(picture, textures.prepared, textures.blurred, deep));
      compositePass.draw(3);
      compositePass.end();
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
  return blur;
}
