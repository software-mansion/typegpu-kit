import * as cameras from '../cameras.ts';
import * as passes from '../passes.ts';
import * as targets from '../targets.ts';
import { type Part as BindingPart } from '../bindings.ts';
import { type CpuView } from '../cameras.ts';
import { type Drawn } from '../passes.ts';
import { type Frame } from '../frames.ts';
import { type Inject } from '../inject.ts';
import { type SizeSource } from '../targets.ts';
import { opening, untimedFrame, type TimestampWrites } from '../profiler.ts';
import { d, std, tgpu, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';
import { shadowMap, shadowSampler, shadowShape, shadowView, SunView } from './shared.ts';

type Vec3 = readonly [number, number, number];
type Triple = [number, number, number];

/** Configures the sun's shadow, a fixed map or one fitted to a view. */
export interface ShadowMapOptions {
  /** Sets the width and height of each layer of the map in texels. Change it with `resize`. Default 2048. */
  readonly size?: number;
  /**
   * Sets how far the map reaches towards the sun in world units: from the point it follows, or
   * beyond each cascade with an `eye`. Casters beyond this distance towards the sun cast no
   * shadow. Default twice the extent, or `far` with an `eye`.
   */
  readonly reach?: number;
  /**
   * Sets the number of comparisons per shaded point to 1, 4 or 9. Each comparison filters four
   * texels. Default 4.
   */
  readonly filter?: 1 | 4 | 9;
  /** Sets the depth format of the map. Default `'depth16unorm'`. */
  readonly format?: 'depth16unorm' | 'depth24plus' | 'depth32float';
  /**
   * Offsets depth written by caster pipelines to prevent self-shadowing. `constant` uses steps
   * of the depth format, and `slope` scales the depth slope. Default `{ constant: 2, slope: 2
   * }`.
   */
  readonly bias?: { readonly constant?: number; readonly slope?: number };
  /**
   * Moves a shaded point along its normal before comparison, in texels. This prevents shadow
   * acne on surfaces facing away from the sun. Default 1.
   */
  readonly lift?: number;
  /**
   * Controls whether `record` draws the shadow in the current frame. When false, it records no
   * depth pass and shading treats every point as sunlit. The map and pipelines stay allocated.
   * Default always.
   * @cpu
   */
  readonly enabled?: () => boolean;
}

/** Configures a fixed map, a square that `follow` centres. */
export interface FixedShadowOptions extends ShadowMapOptions {
  /**
   * Sets the width of the square covered by the map in world units. The square is centred on the
   * point passed to `follow`. Change it with `setExtent`. A shadow with an `eye` takes none.
   * Default 20.
   */
  readonly extent?: number;
  /** Centres the square here at first. Move it with `follow`. Default the origin. */
  readonly center?: Vec3;
  readonly eye?: never;
  readonly cascades?: never;
  readonly far?: never;
  readonly splits?: never;
  readonly fit?: never;
}

/** Configures a shadow fitted to the view of an `eye`, in cascades. */
export interface FittedShadowOptions extends ShadowMapOptions {
  /**
   * Fits the map to this camera's view, in `cascades` layers that reach `far` from the eye. Each
   * `record` fits them again, so the map follows the camera without `follow`. Default none, a
   * fixed map.
   */
  readonly eye: CpuView;
  /**
   * Splits the view into 1 to 4 cascades, each a layer of the map drawn in a pass of its own.
   * Nearer cascades cover less ground, so their shadows are sharper. Needs an `eye`. Default 1.
   */
  readonly cascades?: 1 | 2 | 3 | 4;
  /**
   * Sets how far from the eye the cascades reach, in world units. Change it with `setFar`. Needs an
   * `eye`. Default 50.
   */
  readonly far?: number;
  /**
   * Places the splits between cascades as fractions of `far`, `cascades - 1` rising values between
   * 0 and 1. Needs an `eye`. Default `(k / cascades)²` after cascade `k`, 0.11 and 0.44 for three.
   */
  readonly splits?: readonly number[];
  /**
   * Places the cascades in the sun's light space whenever the shadow fits them: on each `record`,
   * after `aim` and after `setFar`. Needs an `eye`. Default `lighting.fitSlices`, a sphere around
   * each slice of the view, snapped to whole texels.
   * @cpu
   */
  readonly fit?: CascadeFit;
  readonly extent?: never;
  readonly center?: never;
}

/**
 * What a `fit` places cascades from: the eye, the sun's light space and the shadow's settings.
 * Light space is the world turned so that `right`, `up` and `sun` are its axes.
 */
export interface CascadeFitInput {
  /** Gives the camera the shadow fits, its `eye`. */
  readonly eye: CpuView;
  /** Gives light space's first axis in world space, a unit vector across the sun's rays. */
  readonly right: Vec3;
  /** Gives light space's second axis in world space, a unit vector across the sun's rays. */
  readonly up: Vec3;
  /** Gives light space's third axis in world space, the unit vector towards the sun. */
  readonly sun: Vec3;
  /** Gives how far from the eye the cascades reach, in world units. */
  readonly far: number;
  /**
   * Gives where cascades start and end as fractions of `far`, `cascades + 1` rising values from 0
   * to 1: cascade `k` covers the view from `splits[k]` to `splits[k + 1]`.
   */
  readonly splits: readonly number[];
  /** Gives the width and height of each layer of the map in texels. */
  readonly size: number;
}

/**
 * Places one cascade: a cube in light space around `x`, `y` and `z`, `radius` from its centre to
 * each face. The shadow also reaches `reach` beyond it towards the sun.
 */
export interface CascadePlace {
  /** Sets the cube's centre along `right`, in world units. */
  x: number;
  /** Sets the cube's centre along `up`, in world units. */
  y: number;
  /** Sets the cube's centre along `sun`, in world units. */
  z: number;
  /** Sets the distance from the cube's centre to each face, above 0, in world units. */
  radius: number;
}

/**
 * Places each cascade, writing `places[k]` for cascade `k`, and returns whether it placed them.
 * When it returns false, as when the eye has no view yet, every point is sunlit until it places
 * them. `lighting.fitSlices` is the default.
 */
export type CascadeFit = (input: CascadeFitInput, places: CascadePlace[]) => boolean;

/**
 * Configures the sun's shadow. A fixed map covers a square that `follow` centres and takes
 * `extent`. A shadow with an `eye` fits its cascades to that view and takes `cascades`, `far` and
 * `splits`.
 */
export type ShadowOptions = FixedShadowOptions | FittedShadowOptions;

/** Describes depth alone, as every shadow map does. */
export type DepthOnly = { readonly depth: typeof passes.depth };

/** Configures one frame's `record` of the sun's shadow. */
export interface ShadowRecordOptions extends Omit<passes.BeginOptions<DepthOnly>, 'layer'> {
  /**
   * Chooses which cascades draw their layer this frame, by index from the nearest. A cascade it
   * skips keeps the layer it drew last, so still casters under a still sun can skip. A cascade
   * still draws on the first `record` and when its place changes, as a fitted shadow's does when
   * the eye moves, after `aim`, `follow`, `setExtent`, `setFar` and `resize`, and after a frame
   * with `enabled` false, so `() => false` draws casters that never move once. Default every
   * cascade.
   * @cpu
   */
  readonly redraw?: (cascade: number) => boolean;
}

/**
 * The members shadows share: they draw depth from a light into layers of a map, with casters that
 * `add` builds with depth bias.
 */
export interface ShadowPasses extends Disposable {
  /** Builds a caster's depth pipeline with the shadow's bias and returns its draw methods. */
  add<V extends passes.Drawing['Vertex'], S>(drawing: passes.Drawing<V, S>): Drawn;
  add<S, X>(drawing: passes.Compound<S, X>): Drawn;
  /** Reports whether `record` draws this frame, from the `enabled` option. */
  readonly enabled: boolean;
  /** Reports how many layers the map has. */
  readonly layers: number;
  /** Compiles the casters' pipelines. */
  initAsync(): Promise<void>;
  /** Destroys the casters `add` built. Destroying the lighting frees the map. */
  destroy(): void;
}

/**
 * The sun's shadow: depth from the sun, a layer per cascade. Use `light.shadow.add(renderer)` to
 * build caster pipelines with depth bias, and `record` to draw them each frame. The `enabled`
 * option controls whether it records.
 */
export interface SunShadow extends ShadowPasses {
  /** Gives the width and height of each layer of the map in texels. */
  readonly size: number;
  /** Gives the reach towards the sun in world units. */
  readonly reach: number;
  /**
   * Provides bindings for the map. Share them before creating shading pipelines, so `resize`
   * replaces the map without replacing pipelines: give `passes.create`
   * `{ shared: { shadow: light.shadow.part } }`, or call `bindings.share(root, { shadow:
   * light.shadow.part })` for passes of your own.
   */
  readonly part: BindingPart;
  /**
   * Aims the map along `direction` towards the sun. The vector can have any length; zero leaves
   * the direction unchanged. The lighting aims it along `sunDirection` at creation, and `record`
   * aims it again when `light.settings` has received a new one. Call this yourself to cast a
   * shadow from another direction.
   */
  aim(direction: Vec3): void;
  /**
   * Clears the map in a depth pass per cascade that `redraw` picks and draws the casters built by
   * `add`. Call this while recording each frame, before drawing shaded surfaces. When disabled, it
   * draws nothing and makes every point sunlit. A frame that draws no cascade writes a timer's
   * timestamps from an empty compute pass.
   */
  record(
    frame: Frame | TgpuCommandEncoder,
    casters: Drawn | readonly Drawn[],
    options?: ShadowRecordOptions,
  ): void;
  /**
   * Resizes each layer of the map to `size` texels per side without replacing pipelines. Call this
   * between frames, with `part` shared in your layout. It destroys the old map.
   */
  resize(size: number): void;
}

/** This sun shadow covers a square that `follow` centres. */
export interface Shadow extends SunShadow {
  /** Gives the width of the covered square in world units. */
  readonly extent: number;
  /**
   * Centres the square on `point`. It snaps to whole texels to keep shadow edges stable as the
   * point moves. The initial centre is the origin.
   */
  follow(point: Vec3): void;
  /**
   * Sets the square width to `extent` world units without replacing pipelines. `reach` defaults
   * to twice the extent. Increasing the width covers more of the scene with the same texel
   * count.
   */
  setExtent(extent: number, reach?: number): void;
}

/** This sun shadow fits its cascades to the view of the `eye` it was given in each `record`. */
export interface Cascades extends SunShadow {
  /** Gives the number of cascades, each a layer of the map. */
  readonly cascades: number;
  /** Gives how far from the eye the cascades reach, in world units. */
  readonly far: number;
  /**
   * Sets how far from the eye the cascades reach without replacing pipelines. `reach` defaults to
   * `far`. The next `record` fits the cascades to it.
   */
  setFar(far: number, reach?: number): void;
}

const FILTERS = [1, 4, 9];
const UNSET: ShadowRecordOptions = Object.freeze({});
const Y: Vec3 = [0, 1, 0];
const Z: Vec3 = [0, 0, 1];
const CORNERS = [
  [-1, -1],
  [1, -1],
  [-1, 1],
  [1, 1],
] as const;
/** The screen corners, then the middle, whose rays bound the view. */
const RAYS = [...CORNERS, [0, 0]] as const;
/** Where a disabled shadow puts every point, beyond the far end of each cascade. */
const BEYOND = 2;
const STRIDE = d.sizeOf(SunView) / 4;
/** Floats before the first cascade, and in each, of `SunView`. */
const LIGHT = 16;
const CASCADE = 8;

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const copy = (target: Triple, source: Vec3) => {
  for (let i = 0; i < 3; i++) target[i] = source[i];
};

const cross = (a: Vec3, b: Vec3, out: Triple) => {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
};

const positive = (what: string, value: number) => {
  if (!(value > 0 && Number.isFinite(value))) {
    throw new RangeError(`A shadow's ${what} is above 0, got ${value}.`);
  }
};

export const checkSize = (root: TgpuRoot, size: number) => {
  const largest = root.device.limits.maxTextureDimension2D;
  if (!Number.isInteger(size) || size < 1 || size > largest) {
    throw new RangeError(
      `A shadow's size is a whole number of texels up to ${largest}, got ${size}.`,
    );
  }
};

export const checkFilter = (filter: number, lift: number) => {
  if (!FILTERS.includes(filter)) {
    throw new RangeError(`A shadow's filter is 1, 4 or 9 comparisons, got ${filter}.`);
  }
  if (!(lift >= 0 && Number.isFinite(lift))) {
    throw new RangeError(`A shadow's lift is 0 or more texels, got ${lift}.`);
  }
};

/**
 * Returns the filter averaging `side × side` comparisons of a shadow map, `step` apart around
 * `uv`, each of which the sampler filters over four texels. Call it in a shader with a side known
 * at resolution: `pcf(side)(map, uv, layer, depth, step)`.
 */
export const pcf = tgpu.comptime((side: number) => {
  const percentageCloser = (
    map: d.textureDepth2dArray,
    uv: d.v2f,
    layer: number,
    depth: number,
    step: number,
  ) => {
    'use gpu';
    let lit = d.f32(0);
    for (const y of tgpu.unroll(std.range(side))) {
      for (const x of tgpu.unroll(std.range(side))) {
        const offset = (d.vec2f(x, y) - (side - 1) / 2) * step;
        lit += std.textureSampleCompareLevel(map, shadowSampler.$, uv + offset, layer, depth);
      }
    }
    return lit / (side * side);
  };
  return percentageCloser;
});

/**
 * Rounds a radius up to 1/256 of its power of two, so the float noise of a turning camera never
 * moves the texel grid.
 */
const steady = (radius: number) => {
  const step = 2 ** (Math.floor(Math.log2(radius)) - 8);
  return Math.ceil(radius / step) * step;
};

const splitsOf = (cascades: number, given: readonly number[] | undefined) => {
  if (given === undefined) {
    return Array.from({ length: cascades + 1 }, (_, k) => (k / cascades) ** 2);
  }
  const rising = given.every((at, i) => at > (given[i - 1] ?? 0) && at < 1);
  if (given.length !== cascades - 1 || !rising) {
    throw new RangeError(
      `A shadow of ${cascades} cascades takes ${cascades - 1} rising splits between 0 and 1, got [${given.join(', ')}].`,
    );
  }
  return [0, ...given, 1];
};

const snap = (value: number, texel: number) => Math.round(value / texel) * texel;

// Scratch for `fitSlices`, filled anew by each call
const rays = RAYS.map(() => ({ origin: [0, 0, 0], direction: [0, 0, 0] }));
const axis = rays[CORNERS.length];
const middlePoint: Triple = [0, 0, 0];

/** Measures how far the view's corners lie from its axis at `depth` from the eye, squared. */
const across = (depth: number) => {
  let widest = 0;
  for (let c = 0; c < CORNERS.length; c++) {
    const { origin, direction } = rays[c];
    const along = depth / dot(direction as Triple, axis.direction as Triple);
    let squared = 0;
    for (let i = 0; i < 3; i++) {
      const offset =
        origin[i] + direction[i] * along - (axis.origin[i] + axis.direction[i] * depth);
      squared += offset * offset;
    }
    widest = Math.max(widest, squared);
  }
  return widest;
};

/**
 * Fits a sphere around each cascade's slice of the eye's view and snaps its centre to whole
 * texels across the sun's rays. The sphere keeps its size as the eye turns, so shadow edges hold
 * still. This is a fitted shadow's default `fit`; a fit of your own can call it and move what it
 * placed.
 *
 * @param input the eye, the sun's light space and the shadow's settings
 * @param places the places it writes, one per cascade
 * @returns whether it placed them, false while the eye has no view
 */
export function fitSlices(input: CascadeFitInput, places: CascadePlace[]): boolean {
  const { eye, right, up, sun, far, splits, size } = input;
  // A camera never written has no view to fit
  let finite = true;
  for (let c = 0; c < rays.length; c++) {
    const { origin, direction } = rays[c];
    eye.ray(RAYS[c][0], RAYS[c][1], rays[c]);
    for (let i = 0; i < 3; i++) finite &&= Number.isFinite(origin[i] + direction[i]);
  }
  const [x, y, z] = axis.direction as Triple;
  if (!finite || !(x * x + y * y + z * z > 0)) return false;

  for (let k = 0; k < places.length; k++) {
    const near = far * splits[k];
    const farther = far * splits[k + 1];
    const nearWidth = across(near);
    const farWidth = across(farther);
    const middle = Math.min(
      Math.max((near + farther) / 2 + (farWidth - nearWidth) / (2 * (farther - near)), near),
      farther,
    );
    const radius = steady(
      Math.sqrt(Math.max((middle - near) ** 2 + nearWidth, (farther - middle) ** 2 + farWidth)),
    );
    for (let i = 0; i < 3; i++) middlePoint[i] = axis.origin[i] + axis.direction[i] * middle;
    const texel = (radius * 2) / size;
    const place = places[k];
    place.x = snap(dot(right, middlePoint), texel);
    place.y = snap(dot(up, middlePoint), texel);
    place.z = dot(sun, middlePoint);
    place.radius = radius;
  }
  return true;
}

interface DepthLayersOptions {
  readonly size: number;
  readonly layers: number;
  readonly format: 'depth16unorm' | 'depth24plus' | 'depth32float';
  readonly bias: { readonly constant?: number; readonly slope?: number };
  readonly enabled: () => boolean;
  /** Names the pass in captures. */
  readonly label: string;
  /** The layout shading reads the map through, and the name of the map's view. */
  readonly layout: typeof shadowMap;
  readonly name: string;
  /** The error `resize` throws when it can't replace the map in pipelines that read it. */
  readonly unshared: string;
}

/**
 * Makes the square depth map a shadow draws into, a camera per layer and the pass that draws
 * casters into one layer, for the sun's shadow and the lamps'.
 */
export function depthLayers(root: TgpuRoot, options: DepthLayersOptions) {
  const { layers, format, bias, enabled, label, layout, name } = options;
  let size = options.size;
  checkSize(root, size);

  const square: SizeSource = {
    get width() {
      return size;
    },

    get height() {
      return size;
    },
  };
  const map = targets.create(root, square, { depth: { format, sampled: true, layers } });
  const sampled = () => map.textures.depth.createView(d.textureDepth2dArray()).$name(name);
  const part = layout.bind({ map: sampled() });
  const views = cameras.layered(root, layers);
  const into = passes
    .create(
      root,
      map,
      { depth: passes.depth },
      { bias: { constant: bias.constant ?? 2, slope: bias.slope ?? 2 }, enabled, label },
    )
    .pipe(views.inject());

  return {
    part,
    views,
    into,

    get size() {
      return size;
    },

    add: (drawing: never): Drawn => (into.add as (drawing: never) => Drawn)(drawing),

    /** Replaces the map at a new size, and returns whether the size changed. */
    resize(next: number) {
      if (next === size) return false;
      checkSize(root, next);
      // Check whether bindings allow replacement before changing the map
      try {
        part.replace({});
      } catch (error) {
        throw new Error(options.unshared, { cause: error });
      }
      size = next;
      map.fit();
      part.replace({ map: sampled() });
      return true;
    },

    destroy() {
      into.destroy();
      map.destroy();
    },
  };
}

/**
 * Creates the sun's shadow pass and the injection that fills its shading resources. `sunward`
 * views the sun direction in the settings' CPU copy, which `record` follows when it changes.
 */
export function createShadow(root: TgpuRoot, options: ShadowOptions, sunward: Float32Array) {
  const {
    filter = 4,
    format = 'depth16unorm',
    bias = {},
    lift = 1,
    enabled = () => true,
    eye,
    fit = fitSlices,
  } = options;
  checkFilter(filter, lift);

  if (!eye && (options.cascades ?? options.far ?? options.splits ?? options.fit) !== undefined) {
    throw new Error(
      "A shadow's `cascades`, `far`, `splits` and `fit` fit it to a view. Give it an `eye`.",
    );
  }
  if (eye && (options.extent ?? options.center) !== undefined) {
    throw new Error(
      'A shadow with an `eye` fits itself to the view, so it takes no `extent` or `center`.',
    );
  }
  const count = options.cascades ?? 1;
  if (!(Number.isInteger(count) && count >= 1 && count <= 4)) {
    throw new RangeError(`A shadow has 1 to 4 cascades, got ${count}.`);
  }
  const splits = splitsOf(count, options.splits);

  const shape = { side: Math.sqrt(filter), lift, cascades: count };
  let extent = options.extent ?? 20;
  let far = options.far ?? 50;
  let reach = options.reach ?? (eye ? far : extent * 2);
  positive('extent', extent);
  positive('far', far);
  positive('reach', reach);

  let covering = true;

  const layered = depthLayers(root, {
    size: options.size ?? 2048,
    layers: count,
    format,
    bias,
    enabled,
    label: 'shadow',
    layout: shadowMap,
    name: 'shadowMap',
    unshared:
      'Resizing a shadow in place needs its map shared: give `passes.create` `{ shared: { shadow: light.shadow.part } }`, or `bindings.share(root, { shadow: light.shadow.part })` for passes of your own, before creating the pipelines that shade.',
  });
  const { part, views, into } = layered;
  const uniform = root.createUniform(SunView);

  const sun: Triple = [0, 1, 0];
  const center: Triple = [...(options.center ?? [0, 0, 0])];
  const right: Triple = [1, 0, 0];
  const up: Triple = [0, 0, -1];
  const fitting: CascadeFitInput = {
    eye: eye!,
    right,
    up,
    sun,
    get far() {
      return far;
    },
    splits,
    get size() {
      return layered.size;
    },
  };
  const places = Array.from({ length: count }, (): CascadePlace => ({
    x: 0,
    y: 0,
    z: 0,
    radius: 1,
  }));

  /** The sun direction last read from the settings; their CPU copy stays zero until written. */
  const seen: Triple = [0, 0, 0];
  const sunMoved = () => {
    const x = sunward[0];
    const y = sunward[1];
    const z = sunward[2];
    if ((x === 0 && y === 0 && z === 0) || (x === seen[0] && y === seen[1] && z === seen[2])) {
      return false;
    }
    seen[0] = x;
    seen[1] = y;
    seen[2] = z;
    return true;
  };

  // The matrices are column-major and updated in place
  const written = new Float32Array(STRIDE);
  const next = new Float32Array(STRIDE);
  const view = new Float32Array(16);
  const projection = new Float32Array(16);
  view[15] = 1;
  projection[15] = 1;

  /** The placement each cascade's layer was drawn with, NaN where it holds nothing usable. */
  const drawnFrom = new Float32Array(STRIDE).fill(Number.NaN);
  const redrawn = new Uint8Array(count);
  const begin = {
    clear: true as boolean | readonly 'depth'[],
    layer: 0,
    timestampWrites: undefined as TimestampWrites | undefined,
  };

  /** Whether cascade `k`'s layer holds the placement written now: the light's axes and its own. */
  const current = (k: number) => {
    for (let i = 0; i < LIGHT; i++) if (drawnFrom[i] !== written[i]) return false;
    for (let i = LIGHT + k * CASCADE; i < LIGHT + (k + 1) * CASCADE; i++) {
      if (drawnFrom[i] !== written[i]) return false;
    }
    return true;
  };
  // Every cascade whose light axes changed redraws in the same frame, so they can be kept here
  const keep = (k: number) => {
    for (let i = 0; i < LIGHT; i++) drawnFrom[i] = written[i];
    for (let i = LIGHT + k * CASCADE; i < LIGHT + (k + 1) * CASCADE; i++) {
      drawnFrom[i] = written[i];
    }
  };

  /**
   * Places cascade `k`: a square of half-width `radius` around the light-space point `x, y`, its
   * depth from `top` away from the sun across `span`.
   */
  const place = (k: number, x: number, y: number, top: number, radius: number, span: number) => {
    const texel = (radius * 2) / layered.size;

    // View rows hold the light axes; the eye sits at the top of the cascade's depth
    for (let i = 0; i < 3; i++) {
      view[i * 4] = right[i];
      view[i * 4 + 1] = up[i];
      view[i * 4 + 2] = sun[i];
    }
    view[12] = -x;
    view[13] = -y;
    view[14] = -top;
    projection[0] = 1 / radius;
    projection[5] = 1 / radius;
    projection[10] = -1 / span;
    views.write(k, view, projection);

    const at = LIGHT + k * CASCADE;
    next[at] = 1 / radius;
    next[at + 1] = 1 / radius;
    next[at + 2] = covering ? -1 / span : 0;
    next[at + 3] = texel;
    next[at + 4] = -x / radius;
    next[at + 5] = -y / radius;
    next[at + 6] = covering ? top / span : BEYOND;
  };

  const write = () => {
    for (let i = 0; i < 3; i++) {
      next[i * 4] = right[i];
      next[i * 4 + 1] = up[i];
      next[i * 4 + 2] = sun[i];
    }
    next[15] = 1;

    if (!eye) {
      const half = extent / 2;
      const texel = (half * 2) / layered.size;
      const x = snap(dot(right, center), texel);
      place(0, x, snap(dot(up, center), texel), dot(sun, center) + reach, half, reach * 2);
    } else if (fit(fitting, places)) {
      for (let k = 0; k < count; k++) {
        const { x, y, z, radius } = places[k];
        place(k, x, y, z + radius + reach, radius, radius * 2 + reach);
      }
    } else {
      for (let k = 0; k < count; k++) {
        next[LIGHT + k * CASCADE + 2] = 0;
        next[LIGHT + k * CASCADE + 6] = BEYOND;
      }
    }

    let same = true;
    for (let i = 0; i < STRIDE; i++) same &&= next[i] === written[i];
    if (same) return;
    written.set(next);
    uniform.buffer.write(written.buffer);
  };

  const inject: Inject = (cfg) =>
    cfg.with(shadowShape, shape).with(shadowView, uniform).pipe(part.inject());

  const common = {
    add(drawing: never): Drawn {
      return layered.add(drawing);
    },

    get size() {
      return layered.size;
    },

    get reach() {
      return reach;
    },

    part,

    aim(direction: Vec3) {
      const length = Math.hypot(direction[0], direction[1], direction[2]);
      if (!(length > 0)) return;
      for (let i = 0; i < 3; i++) sun[i] = direction[i] / length;

      // Choose an axis away from the sun direction so the cross product is nonzero
      cross(Math.abs(sun[1]) > 0.99 ? Z : Y, sun, right);
      const width = Math.hypot(right[0], right[1], right[2]);
      for (let i = 0; i < 3; i++) right[i] = right[i] / width;
      cross(sun, right, up);

      write();
    },

    record(
      frame: Frame | TgpuCommandEncoder,
      casters: Drawn | readonly Drawn[],
      given: ShadowRecordOptions = UNSET,
    ) {
      if (sunMoved()) common.aim(seen);
      const on = enabled();
      if (on !== covering || eye) {
        covering = on;
        write();
      }
      if (!on) {
        // A frame without the shadow leaves layers that casters may have moved past since
        drawnFrom.fill(Number.NaN);
        if (given.timestampWrites) untimedFrame(frame, given.timestampWrites);
        return;
      }

      let last = -1;
      for (let k = 0; k < count; k++) {
        redrawn[k] = !current(k) || !given.redraw || given.redraw(k) ? 1 : 0;
        if (redrawn[k]) last = k;
      }
      if (last < 0) {
        if (given.timestampWrites) untimedFrame(frame, given.timestampWrites);
        return;
      }
      begin.clear = given.clear ?? true;
      let first = true;
      for (let k = 0; k <= last; k++) {
        if (!redrawn[k]) continue;
        keep(k);
        begin.layer = k;
        begin.timestampWrites =
          given.timestampWrites &&
          (opening(given.timestampWrites, first, k === last) as TimestampWrites);
        first = false;
        into.record(frame, casters, begin as passes.BeginOptions);
      }
    },

    resize(nextSize: number) {
      if (!layered.resize(nextSize)) return;
      drawnFrom.fill(Number.NaN);
      write();
    },

    get enabled() {
      return into.enabled;
    },

    layers: count,

    initAsync: () => into.initAsync(),

    destroy: () => into.destroy(),

    [Symbol.dispose]: () => into.destroy(),
  };

  const fixed = {
    get extent() {
      return extent;
    },

    follow(target: Vec3) {
      copy(center, target);
      write();
    },

    setExtent(nextExtent: number, nextReach = nextExtent * 2) {
      positive('extent', nextExtent);
      positive('reach', nextReach);
      extent = nextExtent;
      reach = nextReach;
      write();
    },
  };

  const fitted = {
    cascades: count,

    get far() {
      return far;
    },

    setFar(nextFar: number, nextReach = nextFar) {
      positive('far', nextFar);
      positive('reach', nextReach);
      far = nextFar;
      reach = nextReach;
      write();
    },
  };

  write();

  const shadow: Shadow | Cascades = Object.defineProperties({} as Shadow | Cascades, {
    ...Object.getOwnPropertyDescriptors(common),
    ...Object.getOwnPropertyDescriptors(eye ? fitted : fixed),
  });

  return {
    shadow,
    inject,

    destroy: () => layered.destroy(),
  };
}
