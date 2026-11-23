/**
 * Point and spot shadows for a lighting: `define` describes them, and `lighting.create` makes
 * them for the lights of its table as `light.pointShadow`, compiles them in `initAsync`, fills
 * their shading and destroys them. A lighting without them never imports this module.
 *
 * @module
 */
import * as bindings from '../bindings.ts';
import * as passes from '../passes.ts';
import { type Part as BindingPart } from '../bindings.ts';
import { type Drawn } from '../passes.ts';
import { type Frame } from '../frames.ts';
import { type Inject } from '../inject.ts';
import { afterEncoderSubmit } from '../encoderSubmit.ts';
import { untimedFrame, type TimestampWrites } from '../profiler.ts';
import { type RowId, type Table } from '../table.ts';
import { d, std, tgpu, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';
import { mat4, vec3 } from 'wgpu-matrix';
import { pcf, pointShadowing } from './shared.ts';
import { checkFilter, depthLayers, type DepthOnly, type ShadowPasses } from './shadow.ts';
import type { PointLight, PointLights, SpotLight, SpotLights } from './lighting.ts';
import type { Point } from './shading.ts';

type Vec3 = readonly [number, number, number];

export interface PointShadowOptions {
  /**
   * Sets how many lights can cast a shadow at once, each with six layers of the map. Shading
   * compares each in-range light with every shadow. Default 1.
   */
  readonly count?: number;
  /**
   * Sets the width and height of each layer in texels, above twice the filter's reach: 2 for
   * filter 1, 3 for 4 and 4 for 9. Change it with `resize`. Default 512.
   */
  readonly size?: number;
  /**
   * Sets the number of comparisons per shaded point to 1, 4 or 9. Each comparison filters four
   * texels. Default 4.
   */
  readonly filter?: 1 | 4 | 9;
  /** Sets the depth format of the map. Default `'depth16unorm'`. */
  readonly format?: 'depth16unorm' | 'depth24plus' | 'depth32float';
  /**
   * Offsets depth written by caster pipelines to prevent self-shadowing. `constant` uses steps
   * of the depth format, and `slope` scales the depth slope. Default `{ constant: 2, slope: 2 }`.
   */
  readonly bias?: { readonly constant?: number; readonly slope?: number };
  /**
   * Moves a shaded point along its normal before comparison, in texels at its distance from the
   * light. Default 1.
   */
  readonly lift?: number;
  /**
   * Controls whether `record` draws the shadows in the current frame. When false, it records no
   * pass and shading treats every point as lit. The map and pipelines stay allocated. Default
   * always.
   * @cpu
   */
  readonly enabled?: () => boolean;
}

/** Chooses the lights that draw this frame, for `prepare` or `record`. */
export interface PointShadowPrepareOptions {
  /**
   * Chooses which cast lights draw their layers this frame. A light it skips keeps the layers it
   * drew last, so a still light over still casters can skip until something it sees moves. A
   * light still draws when it gains a shadow, when its row moves, turns or changes range, after
   * `resize` and after a frame with `enabled` false. Layers count as drawn once their frame
   * submits, so a light recorded into an encoder alone draws on every `record` until `resize`.
   * Default every light.
   * @cpu
   */
  readonly redraw?: (light: RowId) => boolean;
}

/** Configures one frame's `record`. */
export interface PointShadowRecordOptions
  extends Omit<passes.BeginOptions<DepthOnly>, 'layer'>, PointShadowPrepareOptions {}

/**
 * The shadows of the point and spot lights that `cast` picks: depth from each light, six layers
 * per shadow. Use `light.pointShadow.add(renderer)` to build caster pipelines with depth bias, and
 * `record` to draw them each frame.
 */
export interface PointShadow extends ShadowPasses {
  /** Gives the number of shadows, each six layers of the map. */
  readonly count: number;
  /** Gives the width and height of each layer of the map in texels. */
  readonly size: number;
  /**
   * Provides bindings for the map. Share them before creating shading pipelines, so `resize`
   * replaces the map without replacing pipelines: give `passes.create`
   * `{ shared: { pointShadow: light.pointShadow.part } }`, or call `bindings.share(root, {
   * pointShadow: light.pointShadow.part })` for passes of your own.
   */
  readonly part: BindingPart;
  /**
   * Gives a shadow to each light in `lights`, ids in the lighting's table, at most `count`. A light
   * that already casts keeps its shadow, a light left out frees its own, and a new one takes a free
   * shadow, so casting the same lights again writes nothing. A spot whose outer half-angle is at
   * most 60° draws one layer along its aim, and a wider spot or a point light draws six faces. A
   * light of no range or with a cone of no width draws nothing and shadows nothing. Throws for
   * more lights than `count`.
   */
  cast(lights: readonly RowId[]): void;
  /**
   * Chooses the layers that draw this frame before `renderer.cull`, which then culls only those:
   * the layers of each light `redraw` picks, and of a narrow spot only its one. The next `record`
   * draws exactly these and ignores its own `redraw`. Call it after `cast`, in the frame that
   * records: a change before that `record`, such as `cast` or `resize`, drops the choice. Without
   * it, culling covers every layer.
   */
  prepare(options?: PointShadowPrepareOptions): void;
  /**
   * Places each shadow at its light from the table's CPU rows, then clears and draws the layers of
   * each shadowed light that `redraw` picks with the casters built by `add`, a depth pass per
   * layer. Call this while recording each frame, before drawing shaded surfaces. A removed light
   * frees its shadow. A frame that draws no layer writes a timer's timestamps from an empty
   * compute pass.
   */
  record(
    frame: Frame | TgpuCommandEncoder,
    casters: Drawn | readonly Drawn[],
    options?: PointShadowRecordOptions,
  ): void;
  /**
   * Resizes each layer of the map to `size` texels per side without replacing pipelines. Call this
   * between frames, with `part` shared in your layout. It destroys the old map.
   */
  resize(size: number): void;
}

/** Point shadows as `define` describes them, for the `pointShadow` option of `lighting.create`. */
export interface PointShadowDefinition {
  /**
   * Makes the shadows for `lighting.create`, so a lighting without them never imports this module.
   * @hidden
   */
  readonly create: (root: TgpuRoot, table: Lamps) => ReturnType<typeof createPointShadow>;
}

/**
 * Places one point or spot light's shadow: a view-projection per face, the light's slot, the
 * shadow's first layer and face count, and its texel sizes.
 */
const LightShadow = d.struct({
  views: d.arrayOf(d.mat4x4f, 6),
  /** Holds the slot of the light casting this shadow, or `NO_SLOT`. */
  slot: d.u32,
  first: d.u32,
  /** Holds 6 for a cube of faces and 1 for a narrow spot. */
  faces: d.u32,
  /** Holds one texel's width in texture coordinates. */
  texel: d.f32,
  /** Holds one texel's width in world units per unit of distance from the light. */
  spread: d.f32,
});

/** Marks a shadow cast by no light. */
const NO_SLOT = 0xffffffff;

/** Binds the depth map of point and spot shadows, six layers per shadow. */
const pointShadowMap = bindings.layout({ map: { texture: d.textureDepth2dArray() } });

const FACES = 6;
const UNSET: PointShadowRecordOptions = Object.freeze({});
/** The cube faces, `+x, -x, +y, -y, +z, -z`, each with an up direction off its axis. */
const AXES: readonly (readonly [Vec3, Vec3])[] = [
  [
    [1, 0, 0],
    [0, 1, 0],
  ],
  [
    [-1, 0, 0],
    [0, 1, 0],
  ],
  [
    [0, 1, 0],
    [0, 0, -1],
  ],
  [
    [0, -1, 0],
    [0, 0, 1],
  ],
  [
    [0, 0, 1],
    [0, 1, 0],
  ],
  [
    [0, 0, -1],
    [0, 1, 0],
  ],
];
/** Up directions for a spot's single layer, unless it points almost straight up or down. */
const UP: Vec3 = [0, 1, 0];
const SIDEWAYS: Vec3 = [0, 0, -1];
/** Spots at most this wide, as the cosine of their outer half-angle, draw one layer. */
const NARROW = 0.5;
/** Near is this share of a light's range. */
const NEAR = 0.02;
const STRIDE = d.sizeOf(LightShadow) / 4;
const MATRIX = d.sizeOf(d.mat4x4f) / 4;
/** Where the views, slot, first layer, face count, texel and spread sit in each `LightShadow`. */
const VIEWS = d.memoryLayoutOf(LightShadow, (shadow) => shadow.views).offset / 4;
const SLOT = d.memoryLayoutOf(LightShadow, (shadow) => shadow.slot).offset / 4;
const FIRST = d.memoryLayoutOf(LightShadow, (shadow) => shadow.first).offset / 4;
const COUNT = d.memoryLayoutOf(LightShadow, (shadow) => shadow.faces).offset / 4;
const TEXEL = d.memoryLayoutOf(LightShadow, (shadow) => shadow.texel).offset / 4;
const SPREAD = d.memoryLayoutOf(LightShadow, (shadow) => shadow.spread).offset / 4;

const IDENTITY = mat4.identity();
/** Puts every point behind the eye, so culling keeps nothing in an unused layer. */
const NOWHERE = new Float32Array(16);
NOWHERE[15] = -1;

const faceOf = (away: d.v3f, faces: number) => {
  'use gpu';
  if (faces < 2) return d.u32(0);
  const a = std.abs(away);
  if (a.x >= a.y && a.x >= a.z) return std.select(d.u32(1), d.u32(0), away.x > 0);
  if (a.y >= a.z) return std.select(d.u32(3), d.u32(2), away.y > 0);
  return std.select(d.u32(5), d.u32(4), away.z > 0);
};

/** A table of lights the CPU places, with or without cones. */
type Lamps = PointLights | SpotLights;

/**
 * Reads lights from the CPU rows of a point or spot table into one scratch lamp: position and
 * range from either, and the cone from a spot table alone.
 */
const lampsOf = (table: Lamps) => {
  // A spot row holds every field of a point row, so both read positions and ranges alike
  const points = table as Table<typeof PointLight>;
  const spots = 'aim' in table.fields ? (table as Table<typeof SpotLight>) : undefined;
  const ramp = d.vec2f();
  const lamp = {
    position: d.vec3f(),
    aim: d.vec3f(),
    range: 0,
    /** Holds the cosine of the cone's outer half-angle, or -1 for light in every direction. */
    edge: -1,
  };
  return {
    has: (id: RowId) => points.has(id as RowId<typeof PointLight>),
    /** Throws for an id that names no live light in the table. */
    slotOf: (id: RowId) => points.slotOf(id as RowId<typeof PointLight>),
    read(id: RowId) {
      points.fields.position.get(id as RowId<typeof PointLight>, lamp.position);
      lamp.range = points.fields.range.get(id as RowId<typeof PointLight>);
      lamp.edge = -1;
      if (!spots) return lamp;
      spots.fields.aim.get(id as RowId<typeof SpotLight>, lamp.aim);
      spots.fields.cone.get(id as RowId<typeof SpotLight>, ramp);
      // glTF's offset is `-cos(outer) * scale`, and a scale of 0 lights every direction
      if (ramp.x > 0) lamp.edge = -ramp.y / ramp.x;
      return lamp;
    },
  };
};

/**
 * Describes point and spot shadows for `lighting.create`, which gives up to `count` lights of its
 * `points.table` a shadow each as `light.pointShadow`. The lighting checks `count` and `size` when
 * it makes them.
 *
 * @param options the shadows' count, layer size, filter and depth format
 * @returns a definition to give `lighting.create` as `pointShadow`
 */
export function define(options: PointShadowOptions = {}): PointShadowDefinition {
  return { create: (root, table) => createPointShadow(root, options, table) };
}

/** Creates the point and spot shadows' passes and the injection that fills their shading. */
function createPointShadow(root: TgpuRoot, options: PointShadowOptions, table: Lamps) {
  const {
    count = 1,
    filter = 4,
    format = 'depth16unorm',
    bias = {},
    lift = 1,
    enabled = () => true,
  } = options;
  checkFilter(filter, lift);
  // Six layers a shadow in one map, and its placement in one uniform array
  const { maxTextureArrayLayers, maxUniformBufferBindingSize } = root.device.limits;
  const most = Math.min(
    Math.floor(maxTextureArrayLayers / FACES),
    Math.floor(maxUniformBufferBindingSize / d.sizeOf(LightShadow)),
  );
  if (!(Number.isInteger(count) && count >= 1 && count <= most)) {
    throw new RangeError(
      `A point shadow's count is a whole number from 1 to ${most}, got ${count}.`,
    );
  }
  const side = Math.sqrt(filter);
  // A face covers 90° plus the filter's reach, so a filter at its edge stays inside it
  const margin = (side + 1) / 2;
  const layers = count * FACES;
  const fits = (size: number) => {
    if (size <= 2 * margin) {
      throw new RangeError(
        `A point shadow's size is above ${2 * margin} texels with filter ${filter}, got ${size}.`,
      );
    }
  };
  const initial = options.size ?? 512;
  fits(initial);
  const lamps = lampsOf(table);
  /**
   * The layers that draw, which culling and recording follow while `selected`: picked by `prepare`
   * for the next `record`, or by `record` itself. Culling covers every layer otherwise.
   */
  const picked = new Uint8Array(layers);
  let selected = false;
  /** Set when the shadows change after `prepare`, so `record` picks again and confirms nothing. */
  let unsure = false;

  const layered = depthLayers(root, {
    size: initial,
    layers,
    format,
    bias,
    enabled,
    active: (layer) => !selected || picked[layer] === 1,
    label: 'point shadow',
    layout: pointShadowMap,
    name: 'pointShadowMap',
    unshared:
      'To resize point shadows in place, share their map before creating pipelines that shade. Give `passes.create` the option `{ shared: { pointShadow: light.pointShadow.part } }`, or call `bindings.share(root, { pointShadow: light.pointShadow.part })` for your own passes.',
  });
  const { part, views, into } = layered;
  const Shadows = d.arrayOf(LightShadow, count);
  const uniform = root.createUniform(Shadows);

  const written = new ArrayBuffer(d.sizeOf(Shadows));
  const next = new ArrayBuffer(written.byteLength);
  const floats = new Float32Array(next);
  const words = new Uint32Array(next);
  const before = new Uint32Array(written);

  const casting: (RowId | undefined)[] = Array.from({ length: count }, () => undefined);
  const faces = new Uint32Array(count);
  let covering = true;
  /**
   * The light whose depth each shadow's layers hold and the placement it was drawn from, kept once
   * the frame that drew them submits.
   */
  const drawn: (RowId | undefined)[] = Array.from({ length: count }, () => undefined);
  const drawnFrom = new Uint32Array(words.length);
  /** The shadows recorded into a frame's encoder since the placement last changed. */
  const pending = new Uint8Array(count);
  /** The shadows recorded into an encoder alone, which may submit after any later frame. */
  const raw = new Uint8Array(count);

  const forward = new Float32Array(3);
  const target = new Float32Array(3);
  const up = new Float32Array(3);
  const view = new Float32Array(16);
  const projection = new Float32Array(16);
  const viewProjection = new Float32Array(16);

  const free = (k: number) => {
    casting[k] = undefined;
    drawn[k] = undefined;
    faces[k] = 0;
  };

  /** Whether shadow `k`'s layers hold its light as placed now, drawn since it was given. */
  const current = (k: number) => {
    if (drawn[k] !== casting[k]) return false;
    for (let i = k * STRIDE; i < (k + 1) * STRIDE; i++) {
      if (drawnFrom[i] !== words[i]) return false;
    }
    return true;
  };

  /** Drops what was recorded but not yet submitted, and a selection culling may have followed. */
  const moved = () => {
    pending.fill(0);
    if (!selected) return;
    selected = false;
    unsure = true;
  };

  const forget = () => {
    drawn.fill(undefined);
    moved();
  };

  /** Keeps the shadows recorded into the encoder that just submitted, drawn from `words`. */
  const confirm = () => {
    for (let k = 0; k < count; k++) {
      if (!pending[k]) continue;
      pending[k] = 0;
      if (raw[k]) continue;
      drawn[k] = casting[k];
      for (let i = k * STRIDE; i < (k + 1) * STRIDE; i++) drawnFrom[i] = words[i];
    }
  };

  /**
   * Writes shadow `k`'s faces from the CPU row of its light, and returns false, writing nothing,
   * for a light that lights nothing: no range, or a cone of no width.
   */
  const place = (k: number, id: RowId) => {
    const { size } = layered;
    const { position, aim, range, edge } = lamps.read(id);
    if (!(range > 0 && range < Infinity) || !(edge < 1)) return false;
    const narrow = edge >= NARROW && vec3.length(aim) > 0;
    const used = narrow ? 1 : FACES;
    const tangent = narrow ? Math.sqrt(1 - edge * edge) / edge : 1;
    const half = tangent * (size / (size - 2 * margin));
    mat4.perspective(2 * Math.atan(half), 1, range * NEAR, range, projection);

    const at = k * STRIDE;
    for (let f = 0; f < FACES; f++) {
      const layer = k * FACES + f;
      if (f >= used) {
        views.write(layer, IDENTITY, NOWHERE);
        continue;
      }
      if (narrow) {
        vec3.normalize(aim, forward);
        vec3.copy(Math.abs(forward[1]) > 0.99 ? SIDEWAYS : UP, up);
      } else {
        vec3.copy(AXES[f][0], forward);
        vec3.copy(AXES[f][1], up);
      }
      vec3.add(position, forward, target);
      mat4.lookAt(position, target, up, view);
      views.write(layer, view, projection);
      floats.set(mat4.multiply(projection, view, viewProjection), at + VIEWS + f * MATRIX);
    }
    words[at + SLOT] = lamps.slotOf(id);
    words[at + FIRST] = k * FACES;
    words[at + COUNT] = used;
    floats[at + TEXEL] = 1 / size;
    floats[at + SPREAD] = (2 * half) / size;
    faces[k] = used;
    return true;
  };

  const write = () => {
    for (let k = 0; k < count; k++) {
      const id = casting[k];
      if (id !== undefined && !lamps.has(id)) free(k);
      const held = casting[k];
      if (held !== undefined && covering && place(k, held)) continue;
      words[k * STRIDE + SLOT] = NO_SLOT;
      faces[k] = 0;
      for (let f = 0; f < FACES; f++) views.write(k * FACES + f, IDENTITY, NOWHERE);
    }

    let same = true;
    for (let i = 0; i < words.length; i++) same &&= words[i] === before[i];
    if (same) return;
    before.set(words);
    uniform.buffer.write(written);
    moved();
  };

  const shadowedAt = (point: Point, toLight: d.v3f, distance: number, k: number) => {
    'use gpu';
    const spread = uniform.$[k].spread;
    const shown = uniform.$[k].faces;
    const lifted = point.position + point.normal * (distance * spread * lift);
    const face = faceOf(lifted - (point.position + toLight * distance), shown);
    const clip = uniform.$[k].views[face] * d.vec4f(lifted, 1);
    if (clip.w <= 0) return d.f32(1);
    const depth = clip.z / clip.w;
    if (depth > 1) return d.f32(1);

    const uv = (clip.xy / clip.w) * d.vec2f(0.5, -0.5) + 0.5;
    const layer = uniform.$[k].first + face;
    return pcf(side)(pointShadowMap.$.map, uv, layer, depth, uniform.$[k].texel);
  };

  const pointShadowed = (point: Point, toLight: d.v3f, distance: number, slot: number) => {
    'use gpu';
    for (const k of tgpu.unroll(std.range(count))) {
      if (uniform.$[k].slot === d.u32(slot)) return shadowedAt(point, toLight, distance, k);
    }
    return d.f32(1);
  };

  const inject: Inject = (cfg) => cfg.with(pointShadowing, pointShadowed).pipe(part.inject());

  /** Places the shadows and picks the layers that draw. */
  const choose = (redraw: PointShadowPrepareOptions['redraw']) => {
    covering = enabled();
    write();
    // A frame without shadows leaves maps that casters may have moved past since
    if (!covering) forget();

    for (let k = 0; k < count; k++) {
      const draws = faces[k] > 0 && (!current(k) || !redraw || redraw(casting[k]!));
      for (let f = 0; f < FACES; f++) picked[k * FACES + f] = draws && f < faces[k] ? 1 : 0;
    }
  };

  const recording = {
    clear: true as boolean | readonly 'depth'[],
    timestampWrites: undefined as TimestampWrites | undefined,
  };
  const pointShadow: PointShadow = {
    count,

    get size() {
      return layered.size;
    },

    part,

    add(drawing: never): Drawn {
      return layered.add(drawing);
    },

    cast(ids: readonly RowId[]) {
      if (ids.length > count) {
        throw new RangeError(
          `The lighting has ${count} point shadows, so ${ids.length} lights can't cast at once.`,
        );
      }
      // Indexed loops keep a call each frame free of allocations
      for (let i = 0; i < ids.length; i++) lamps.slotOf(ids[i]);
      let changed = false;
      for (let k = 0; k < count; k++) {
        const held = casting[k];
        if (held !== undefined && !ids.includes(held)) {
          free(k);
          changed = true;
        }
      }
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i];
        if (casting.includes(id)) continue;
        casting[casting.indexOf(undefined)] = id;
        changed = true;
      }
      if (changed) write();
    },

    prepare(given: PointShadowPrepareOptions = UNSET) {
      choose(given.redraw);
      selected = true;
      unsure = false;
    },

    record(
      frame: Frame | TgpuCommandEncoder,
      casters: Drawn | readonly Drawn[],
      given: PointShadowRecordOptions = UNSET,
    ) {
      if (!selected) choose(given.redraw);
      const framed = 'encoder' in frame;
      const kept = framed && !unsure;
      unsure = false;
      if (!covering) {
        selected = false;
        if (given.timestampWrites) untimedFrame(frame, given.timestampWrites);
        return;
      }

      // Layers about to be overwritten hold nothing usable until their encoder submits
      for (let k = 0; k < count; k++) {
        if (!picked[k * FACES]) continue;
        drawn[k] = undefined;
        if (!framed) raw[k] = 1;
        else if (kept) pending[k] = 1;
      }
      if (kept) afterEncoderSubmit(frame, confirm);

      recording.clear = given.clear ?? true;
      recording.timestampWrites = given.timestampWrites;
      selected = true;
      try {
        into.record(frame, casters, recording);
      } finally {
        selected = false;
      }
    },

    resize(nextSize: number) {
      fits(nextSize);
      if (!layered.resize(nextSize)) return;
      forget();
      // An encoder recorded before draws into the map it replaced
      raw.fill(0);
      write();
    },

    get enabled() {
      return into.enabled;
    },

    layers,

    initAsync: () => into.initAsync(),

    destroy: () => into.destroy(),

    [Symbol.dispose]: () => into.destroy(),
  };

  write();

  return {
    pointShadow,
    inject,

    destroy: () => layered.destroy(),
  };
}
