import * as passes from '../passes.ts';
import { type Part as BindingPart } from '../bindings.ts';
import { type Drawn } from '../passes.ts';
import { type Frame } from '../frames.ts';
import { type Inject } from '../inject.ts';
import { opening, untimedFrame, type TimestampWrites } from '../profiler.ts';
import { type RowId, type Table } from '../table.ts';
import { d, std, tgpu, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';
import { mat4, vec3 } from 'wgpu-matrix';
import { LightShadow, NO_SLOT, pointShadowing, pointShadowMap, type SpotLight } from './shared.ts';
import { checkFilter, depthLayers, pcf, type DepthOnly, type ShadowPasses } from './shadow.ts';
import type { Point } from './shading.ts';

type Vec3 = readonly [number, number, number];

export interface PointShadowOptions {
  /**
   * Sets how many lights can cast a shadow at once, each with six layers of the map. Shading
   * compares each in-range light with every shadow. Default 1.
   */
  readonly count?: number;
  /** Sets the width and height of each layer in texels. Change it with `resize`. Default 512. */
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

/** Configures one frame's `record`. */
export interface PointShadowRecordOptions extends Omit<passes.BeginOptions<DepthOnly>, 'layer'> {
  /**
   * Chooses which cast lights draw their layers this frame. A light it skips keeps the layers it
   * drew last, so a still light over still casters can skip until something it sees moves. A
   * light still draws when it gains a shadow, when its row moves, turns or changes range, after
   * `resize` and after a frame with `enabled` false. Default every light.
   * @cpu
   */
  readonly redraw?: (light: RowId) => boolean;
}

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
   * most 60° draws one layer along its aim, and a wider spot or a point light draws six faces.
   * Throws for more lights than `count`.
   */
  cast(lights: readonly RowId[]): void;
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
const VIEWS = 16 * FACES;
/** Where the slot, first layer, face count, texel and spread sit in each `LightShadow`. */
const SLOT = VIEWS;
const FIRST = VIEWS + 1;
const COUNT = VIEWS + 2;
const TEXEL = VIEWS + 3;
const SPREAD = VIEWS + 4;

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

/** Creates the point and spot shadows' passes and the injection that fills their shading. */
export function createPointShadow(
  root: TgpuRoot,
  options: PointShadowOptions,
  table: Table<typeof SpotLight>,
) {
  const {
    count = 1,
    filter = 4,
    format = 'depth16unorm',
    bias = {},
    lift = 1,
    enabled = () => true,
  } = options;
  checkFilter(filter, lift);
  const most = Math.floor(root.device.limits.maxTextureArrayLayers / FACES);
  if (!(Number.isInteger(count) && count >= 1 && count <= most)) {
    throw new RangeError(
      `A point shadow's count is a whole number from 1 to ${most}, got ${count}.`,
    );
  }
  const side = Math.sqrt(filter);
  // A face covers 90° plus the filter's reach, so a filter at its edge stays inside it
  const margin = (side + 1) / 2;
  const layers = count * FACES;

  const layered = depthLayers(root, {
    size: options.size ?? 512,
    layers,
    format,
    bias,
    enabled,
    label: 'point shadow',
    layout: pointShadowMap,
    name: 'pointShadowMap',
    unshared:
      'Resizing point shadows in place needs their map shared: give `passes.create` `{ shared: { pointShadow: light.pointShadow.part } }`, or `bindings.share(root, { pointShadow: light.pointShadow.part })` for passes of your own, before creating the pipelines that shade.',
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
  /** The light whose depth each shadow's layers hold, and the placement it was drawn from. */
  const drawn: (RowId | undefined)[] = Array.from({ length: count }, () => undefined);
  const drawnFrom = new Uint32Array(words.length);
  const redrawn = new Uint8Array(count);

  const position = d.vec3f();
  const aim = d.vec3f();
  const ramp = d.vec2f();
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

  /** Writes shadow `k`'s faces from the CPU row of its light. */
  const place = (k: number, id: RowId) => {
    const { size } = layered;
    const lightId = id as RowId<typeof SpotLight>;
    table.fields.position.get(lightId, position);
    const range = table.fields.range.get(lightId);
    const cones = 'aim' in table.fields;
    if (cones) {
      table.fields.aim.get(lightId, aim);
      table.fields.cone.get(lightId, ramp);
    }
    // glTF's offset is `-cos(outer) * scale`, and a scale of 0 lights every direction
    const edge = cones && ramp.x > 0 ? -ramp.y / ramp.x : -1;
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
      floats.set(mat4.multiply(projection, view, viewProjection), at + f * 16);
    }
    words[at + SLOT] = table.slotOf(lightId as never);
    words[at + FIRST] = k * FACES;
    words[at + COUNT] = used;
    floats[at + TEXEL] = 1 / size;
    floats[at + SPREAD] = (2 * half) / size;
    faces[k] = used;
  };

  const write = () => {
    for (let k = 0; k < count; k++) {
      const id = casting[k];
      if (id !== undefined && !table.has(id as never)) free(k);
      const held = casting[k];
      if (held !== undefined && covering) {
        place(k, held);
        continue;
      }
      words[k * STRIDE + SLOT] = NO_SLOT;
      faces[k] = 0;
      for (let f = 0; f < FACES; f++) views.write(k * FACES + f, IDENTITY, NOWHERE);
    }

    let same = true;
    for (let i = 0; i < words.length; i++) same &&= words[i] === before[i];
    if (same) return;
    before.set(words);
    uniform.buffer.write(written);
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

  const begin = {
    clear: true as boolean | readonly 'depth'[],
    layer: 0,
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
      for (let i = 0; i < ids.length; i++) table.slotOf(ids[i] as never);
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

    record(
      frame: Frame | TgpuCommandEncoder,
      casters: Drawn | readonly Drawn[],
      given: PointShadowRecordOptions = UNSET,
    ) {
      covering = enabled();
      write();

      let last = -1;
      for (let k = 0; k < count; k++) {
        const id = casting[k];
        // A frame without shadows leaves maps that casters may have moved past since
        if (!covering) drawn[k] = undefined;
        redrawn[k] =
          covering && id !== undefined && (!current(k) || !given.redraw || given.redraw(id))
            ? 1
            : 0;
        if (redrawn[k]) last = k * FACES + faces[k] - 1;
      }

      if (last < 0) {
        // A timer passed for a frame that draws nothing still gets both timestamps
        if (given.timestampWrites) untimedFrame(frame, given.timestampWrites);
        return;
      }

      let first = true;
      begin.clear = given.clear ?? true;
      for (let k = 0; k < count; k++) {
        if (!redrawn[k]) continue;
        drawn[k] = casting[k];
        for (let i = k * STRIDE; i < (k + 1) * STRIDE; i++) drawnFrom[i] = words[i];
        for (let f = 0; f < faces[k]; f++) {
          begin.layer = k * FACES + f;
          begin.timestampWrites =
            given.timestampWrites &&
            (opening(given.timestampWrites, first, begin.layer === last) as TimestampWrites);
          first = false;
          into.record(frame, casters, begin as passes.BeginOptions);
        }
      }
    },

    resize(nextSize: number) {
      if (layered.resize(nextSize)) write();
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
