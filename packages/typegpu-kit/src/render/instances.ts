/**
 * This module stores mesh instances and builds renderers for colour and depth passes.
 *
 * @module
 * @injects definition its kind's table, which your kernels read and write
 * @binds 1 storage: a field of its kind's table, per half, as a table's getters bind
 * @binds 1 uniform: the mesh of an indirect draw, in the vertex stage, without
 * `indirect-first-instance`
 * @binds 1 storage: a renderer's vertex stage, the live rows per mesh or a culling renderer's
 * visible rows
 * @binds 2 storage, 1 uniform: a culling renderer's marking kernel, where `place`, `visible` and
 * `bounds` run: the listing, its scratch and the listing's length
 * @requires indirect-first-instance option `cull: true`, and a render bundle, draw indirectly;
 * without it, each indirect draw binds a group holding its mesh, or draws through a pipeline per
 * mesh where the pipeline has no group left. A renderer drawn into a pass without `cull` draws
 * directly and needs none.
 * @owns a table, 2 buffers, culling buffers, culling pipelines, bind groups: the rows, the live
 * ones per mesh and, once a render bundle draws them, their draws; with culling, a uniform of how
 * many rows it lists from; without `indirect-first-instance`, a uniform and a bind group per mesh
 * with instances; and a culling renderer's 3 buffers and 2 kernels for each description it is
 * added to, which own its pipelines, colour, with a depth pipeline before it for a kind with a
 * surface, or depth alone
 * @records what each description built of a renderer, in its pass; `flush()` once a frame, before
 * them, uploads what changed; a culling renderer's `cull` in a compute pass after the flush.
 */

import * as cameras from '../cameras.ts';
import * as passes from '../passes.ts';
import * as pipelines from '../pipelines.ts';
import * as table from '../table.ts';
import { type CameraBase } from '../cameras.ts';
import { type Props, type RowId, type TableDef } from '../table.ts';
import { type Inject } from '../inject.ts';
import { type Into } from '../passes.ts';
import type { meshes } from '../geometry/index.ts';
import {
  d,
  std,
  tgpu,
  type IndirectFlag,
  type TgpuBindGroup,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuRenderCommands,
  type TgpuRenderPass,
  type TgpuRenderPipeline,
  type TgpuRoot,
  type TgpuUniform,
  type WithBinding,
} from 'typegpu';
import type { Frame } from '../frames.ts';
import { createCulling, RECORD, type Culling } from './culling.ts';
import { antialiased, colorAt, cutBelow, mappedNormal, masked, texturedBy } from './materials.ts';
import {
  boundsOf,
  positionAt,
  uv1At,
  vertexAt,
  type Mesh,
  type MeshPool,
  type Model,
} from './meshPool.ts';
import { fog, Material, Point, shade, shadeClear } from './shading.ts';
import { Placement, skinned, skinnedPosition } from './skinning.ts';

// Pack each vec3f with a scalar to fill a 16-byte row in both layouts

/** The host row uses 80 bytes per instance. */
const placedHere = {
  position: d.vec3f,
  mesh: d.u32,
  /** Stores rotation as a unit quaternion. */
  rotation: d.vec4f,
  scale: d.vec3f,
  roughness: d.f32,
  color: d.vec3f,
  metallic: d.f32,
  emission: d.vec3f,
  opacity: d.f32,
};

/** The host row uses 48 bytes when kernels own the transform. */
const looks = {
  /** Holds the base colour in linear RGB, which the mesh's textures multiply. */
  color: d.vec3f,
  /** Holds the roughness, from 0 for a mirror to 1. */
  roughness: d.f32,
  /** Holds the emitted light in linear RGB. */
  emission: d.vec3f,
  /** Holds the metalness, from 0 to 1. */
  metallic: d.f32,
  /** Holds the pool index of the mesh it draws. */
  mesh: d.u32,
  /** Holds how much the instance covers what lies behind it under a translucent renderer. */
  opacity: d.f32,
};

/** The device transform uses 48 bytes per instance. */
const placement = {
  /** Holds the rotation as a unit quaternion. */
  rotation: d.vec4f,
  /** Holds the position in world space. */
  position: d.vec3f,
  /** Holds the scale factor per axis. */
  scale: d.vec3f,
};

type Looks = typeof looks;
type Placed = typeof placement;
type Every = Placed & Looks;

/**
 * This type selects whether the CPU or GPU kernels write instance positions, rotations and
 * scales.
 */
export type Moved = 'host' | 'device';

type HostOf<H, M> = (M extends 'device' ? Looks : Placed & Looks) & H;
type DeviceOf<D, M> = (M extends 'device' ? Placed : {}) & D;

/** This type holds the transform, material and custom fields passed to kind callbacks. */
export type InstanceOf<H, D, M> = d.InferGPU<d.WgslStruct<HostOf<H, M> & DeviceOf<D, M>>>;

/** This type describes a vertex decoded by `meshPool.vertexAt`. */
export type Vertex = d.InferGPU<typeof meshes.Vertex>;

export { Placement } from './skinning.ts';

/** This schema describes a world-space sphere for testing instance visibility. */
export const Sphere = d.struct({ center: d.vec3f, radius: d.f32 });
export type Sphere = d.InferGPU<typeof Sphere>;

/**
 * This schema describes the surface at a pixel: a point, which can lie off the mesh, and a
 * material.
 */
export const Shaded = d.struct({ point: Point, material: Material });
export type Shaded = d.InferGPU<typeof Shaded>;

/** @hidden A surface with its coverage, which an impostor works out from the same samples. */
export const Covered = d.struct({ point: Point, material: Material, coverage: d.f32 });

/** This type describes a pixel of an instance, as `coverage`, `material` and `surface` read it. */
export interface Pixel<I> {
  /** Holds the point on the mesh. */
  readonly point: Point;
  /** Holds the transform, material and custom fields of the instance. */
  readonly instance: I;
  /** Holds the values interpolated from `carry`, or zero without it. */
  readonly carried: d.v4f;
}

type Reserved = 'host' | 'device' | 'row' | 'alive' | 'span' | 'slotOf';

type Base = { readonly [K in keyof Placed | keyof Looks]?: 'is a field of every instance' } & {
  readonly [K in Reserved]?: 'is reserved by table.define';
};

export interface KindOptions<H extends Props, D extends Props, M extends Moved> {
  /** Adds fields written on the CPU, as in `table.define`. Default none. */
  readonly host?: H & Base;
  /** Adds fields written by GPU kernels. Default none. */
  readonly device?: D & Base;
  /**
   * Stores position, rotation and scale in device fields when set to `'device'`, so kernels can
   * write them. A kind typed as moved on the device requires it. Default `'host'`.
   */
  readonly transform?: M;
  /**
   * Returns the instance transform from custom data, such as a simulation buffer. All passes use
   * it, including culling and distance limits from `from` and `to`. It runs for each vertex in
   * every pass; when it does heavy work, a kernel that writes device rows once a frame costs less.
   * Attach its fills with `pipe` or the renderer `pipe` method. Default the instance `position`,
   * `rotation` and `scale`.
   * @gpu
   */
  readonly place?: (instance: InstanceOf<H, D, M>) => Placement;
  /**
   * Controls visibility in every renderer and pass, including shadows, without removing the
   * instance. Culling renderers omit hidden instances from their lists. Other renderers collapse
   * their triangles. Default always.
   * @gpu
   */
  readonly visible?: (instance: InstanceOf<H, D, M>) => boolean;
  /**
   * Selects the pose that moves the vertices by their joints. Return an id from the
   * `poses.create` instance injected into the renderer. Both passes apply the pose and rotate
   * normals before `deform`. Culling uses the mesh bounds in the bind pose, so supply `bounds`
   * if animation extends beyond them. Meshes must have joints. Default none, which uses the bind
   * pose.
   * @gpu
   */
  readonly pose?: (instance: InstanceOf<H, D, M>) => number;
  /**
   * Moves a vertex in object space before the instance transform. Colour and shadow passes both
   * apply this function. The shadow pass must then read the whole vertex instead of its position
   * alone. Normals remain as the mesh or pose supplies them. Default none, which uses the mesh
   * or posed position.
   * @gpu
   */
  readonly deform?: (vertex: Vertex, instance: InstanceOf<H, D, M>) => d.v3f;
  /**
   * Controls whether the shadow pass applies `deform`. Set this to `false` for motion too small
   * to affect the shadow, such as a tail beat. The shadow pass can then read positions alone.
   * Default `true`.
   */
  readonly deformShadow?: boolean;
  /**
   * Returns the world-space sphere for culling. `base` surrounds the mesh box after the instance
   * transform. A kind with `deform` must provide this function to support culling. Default
   * `base`.
   * @gpu
   */
  readonly bounds?: (instance: InstanceOf<H, D, M>, base: Sphere) => Sphere;
  /**
   * Returns four values to interpolate between vertices and hand to the fragment hooks as
   * `pixel.carried`. For example, read an extra attribute with `pool.extra.tint(vertexIndex)`.
   * `vertexIndex` indexes the pool, as in `meshPool.vertexAt`. Default none, which omits four
   * floats per vertex and sets `carried` to zero.
   * @gpu
   */
  readonly carry?: (vertexIndex: number, instance: InstanceOf<H, D, M>) => d.v4f;
  /**
   * Returns the material at a pixel. `base` combines the instance material fields with the mesh
   * textures, as `materials.textured` does. Default `base`.
   * @gpu
   */
  readonly material?: (pixel: Pixel<InstanceOf<H, D, M>>, base: Material) => Material;
  /**
   * Replaces the mesh surface at each pixel, as an impostor does. `pixel.point` is the point on
   * the mesh and `base` its material. The colour pass shades the returned point at the mesh
   * depth. The shadow pass calls this after `deform` and writes the returned point depth, so
   * shadows follow the shaded surface. A kind can supply either `material` or `surface`. Default
   * none, which uses the mesh surface.
   * @gpu
   */
  readonly surface?: (pixel: Pixel<InstanceOf<H, D, M>>, base: Material) => Shaded;
  /**
   * Returns how much of the pixel the instance covers, from 0 to 1, to cut holes in its shape,
   * such as lace, or leaves on a card. It runs before `material` or `surface`, in every pass,
   * shadows included, and reads no material, so it costs no texture samples of its own. With
   * multisampling, coverage selects a share of samples; otherwise, pixels below one half are
   * discarded. Blended renderers scale opacity by it. Default none: the instance covers every
   * pixel it rasterises, and opaque renderers keep early depth testing.
   * @gpu
   */
  readonly coverage?: (pixel: Pixel<InstanceOf<H, D, M>>) => number;
}

interface Shaders {
  /** Describes the `Corner` returned by `at`, including values from `carry`. */
  readonly Vertex: typeof Corner;
  readonly at: ReturnType<typeof shadersOf>['at'];
  readonly cast: ReturnType<typeof shadersOf>['cast'];
  readonly lit: ReturnType<typeof shadersOf>['lit'];
  /** Provides surface data for colour passes, and the surface point for depth-only passes. */
  readonly surfaced: ReturnType<typeof shadersOf>['surfaced'];
  /** Gives a depth prepass the coverage of each pixel. */
  readonly covered: ReturnType<typeof shadersOf>['covered'];
  /** Gives a multisampled pixel in a fade the samples it keeps. */
  readonly kept: ReturnType<typeof shadersOf>['kept'];
  /** Tells a depth-only pass which pixels the mesh mask or `coverage` cut away. */
  readonly cutAway: ReturnType<typeof shadersOf>['cutAway'];
  /** Indicates whether the kind has `coverage`. */
  readonly cuts: boolean;
  /** Tests whether the pipeline camera can see an instance row. */
  readonly seen: (row: number) => boolean;
  readonly meshOf: (row: number) => number;
  /** Indicates whether `deform` can move vertices beyond the mesh boxes without custom bounds. */
  readonly unbounded: boolean;
  /** Indicates whether a pose moves vertices and therefore requires meshes with joints. */
  readonly posed: boolean;
}

/**
 * This type combines an instance table definition with its drawing shaders. It needs no root.
 */
export type Kind<
  N extends string = string,
  H extends Props = {},
  D extends Props = {},
  M extends Moved = Moved,
> = TableDef<N, HostOf<H, M>, DeviceOf<D, M>> & {
  readonly '~kind': Shaders & {
    readonly moved: M;
    /** Describes custom fields for type checking only. */
    readonly own?: d.InferInput<d.WgslStruct<H & D>>;
  };
};

// oxlint-disable-next-line typescript/no-explicit-any -- matches a kind of any fields
export type AnyKind = Kind<string, any, any, Moved>;

/** Any kind, with the fields it declares; its own code needs no wildcard. */
type SomeKind = Kind<string, Props, Props, Moved>;

type Plain = Kind<'Instance', {}, {}, 'host'>;

/**
 * This id identifies an instance and carries its kind, as in `Id<typeof Fish>`. Plain sets use
 * `Id`. Removing the instance makes its id stale.
 */
export type Id<K extends AnyKind = Plain> = RowId<K>;

type OwnOf<K extends AnyKind> = NonNullable<K['~kind']['own']>;

/**
 * These values set the transform, material and custom fields in `add` and `write`. All are
 * optional. Vectors also take any array of numbers, such as a `Float32Array` from a math library.
 * You can spread in a value from `materials`. Omitted fields keep their values. On creation, the
 * transform is the identity and material factors come from the mesh, with white chalk for shapes.
 * Mesh textures multiply these factors.
 */
export type Values<K extends AnyKind = Plain> = {
  readonly position?: table.FieldInput<d.Vec3f>;
  /** Sets rotation as a unit quaternion `[x, y, z, w]`. */
  readonly rotation?: table.FieldInput<d.Vec4f>;
  /** Sets one scale factor for all axes or a separate factor per axis. */
  readonly scale?: number | table.FieldInput<d.Vec3f>;
  readonly color?: table.FieldInput<d.Vec3f>;
  readonly roughness?: number;
  readonly metallic?: number;
  readonly emission?: table.FieldInput<d.Vec3f>;
  /**
   * Sets how much the instance covers objects behind it when the renderer uses `blend:
   * 'translucent'` or `'clear'`.
   */
  readonly opacity?: number;
} & ([keyof OwnOf<K>] extends [never] ? unknown : Partial<OwnOf<K>>);

/**
 * Provides one handle per field, as a table's `fields` does, except the mesh. A handle given a
 * model's id writes every mesh of the model and reads its first.
 */
export type Fields<K extends AnyKind = Plain> = Omit<table.Fields<K>, 'mesh'>;

export interface Options<K extends AnyKind = Plain> {
  /**
   * Selects a kind from `instances.define`. Default a plain kind with a transform and material.
   */
  readonly kind?: K;
  /**
   * Limits the number of rows. A mesh instance uses one row; a model instance uses one row per
   * mesh.
   */
  readonly capacity: number;
  /**
   * Sets values for fields that `add` omits. They replace the mesh's material factors. Default
   * none: the identity transform, the mesh's material and zero for custom fields.
   */
  readonly defaults?: Values<K>;
  /**
   * Keeps last frame's rows, as a table's `history` does, so motion follows every instance that
   * moves, whether the CPU or a kernel moves it. `create` then returns `HistoryInstances`, whose
   * `flush` takes the frame. Default false.
   */
  readonly history?: boolean;
}

/** This schema holds a mesh corner position and the values interpolated across pixels. */
export const Corner = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  normal: d.vec3f,
  uv: d.vec2f,
  uv1: d.vec2f,
  row: d.u32,
});
export type Corner = d.Infer<typeof Corner>;

const Carrying = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  normal: d.vec3f,
  uv: d.vec2f,
  uv1: d.vec2f,
  row: d.u32,
  carried: d.vec4f,
});
type Carrying = d.Infer<typeof Carrying>;

/** This schema holds the shaded pixel colour. Blended renderers use premultiplied colour. */
export const Lit = d.struct({ color: d.vec4f });
export type Lit = d.Infer<typeof Lit>;

const Cast = d.struct({ position: d.vec4f });

type Look = Pick<Mesh, 'doubleSided' | 'masked' | 'textured'>;

const primitiveOf = (look: Look) => ({ cullMode: look.doubleSided ? 'none' : 'back' }) as const;

export interface RendererOptions {
  /**
   * Selects how instances write colour and depth. `'opaque'` writes depth. `'translucent'` draws
   * from premultiplied opacity, as a drawing with `translucent: true` does: in any order into a
   * description with translucency targets, over in draw order elsewhere. It leaves depth unchanged
   * and draws only in colour passes, after the opaque renderers. `'clear'` draws in the same way
   * but adds reflections at full strength through `shading.shadeClear`, so glass and clear
   * plastic retain highlights at any opacity. With `transmit`, it blends over in draw order.
   * Default `'opaque'`.
   */
  readonly blend?: 'opaque' | 'translucent' | 'clear';
  /**
   * Sets the linear fraction of each background colour channel that passes through under `blend:
   * 'clear'`. It replaces `1 - opacity` with one tint for the renderer. Instance opacity still
   * scales diffuse light and emission. The renderer compiles this colour into shaders and sets
   * it as the blend constant on every draw. Pixels with partial `surface` coverage receive the
   * same tint. Default none, which uses instance opacity.
   */
  readonly transmit?: readonly [number, number, number];
  /**
   * Computes the surface colour for an opaque or `'translucent'` renderer. Supply this for unlit,
   * toon or debug shading. Default `shading.shade`.
   * @gpu
   */
  readonly shade?: (material: Material, point: Point) => d.v3f;
  /**
   * Set to `false` to ignore mesh textures: textured meshes then draw with their untextured
   * factors and without alpha masks, and a custom `material` function receives those factors as
   * `base`. Default `true`, which reads textures on meshes whose `textured` is set.
   */
  readonly textured?: boolean;
  /**
   * Widens roughness where the normal turns within a pixel with `materials.antialiased`, so smooth
   * highlights stay steady as they move. Set it to `false` for flat-shaded art whose faces should
   * keep sharp highlights. Default `true`.
   */
  readonly specularAntialiasing?: boolean;
  /**
   * Enables GPU culling against each pass camera using spheres around mesh boxes. Call `cull`
   * each frame to list visible instances in set order. Throws if the kind has `deform` without
   * `bounds`. Default `false`.
   */
  readonly cull?: boolean;
  /**
   * Sets the minimum distance from `eye` in metres, measured to the instance position. Within
   * `fade` metres around each endpoint, the renderer fades in or out: by whole samples of each
   * instance with multisampling, and by an ordered pattern of pixels without. Renderers with
   * matching endpoints and `fade` exchange them without a gap. Depth-only passes, including
   * shadows, switch at the endpoints without fading.
   *
   * A shader callback can return a distance for each row slot and carry fills with `pipe`.
   * Values must keep `from` below `to`. Requires `cull: true`. Default 0.
   * @gpu
   */
  readonly from?: number | ((row: number) => number);
  /**
   * Sets the exclusive maximum distance from `eye` in metres, with the same rules as `from`.
   * Default every distance.
   * @gpu
   */
  readonly to?: number | ((row: number) => number);
  /** Sets the fade width around `from` and `to` in metres. Default 0. */
  readonly fade?: number;
  /**
   * Provides the camera for measuring distance for `from` and `to` in every pass, including
   * shadows. Default none, which throws if the renderer uses `from` or `to`.
   */
  readonly eye?: CameraBase;
  /**
   * Draws an opaque renderer twice: first its depth, running only `coverage` and the mesh masks,
   * then its colour where the depth is equal, so each pixel shades once. It costs a second pass
   * over the vertices, and pays off for a heavy `material` or `surface` drawn over itself many
   * times, such as a forest of impostors. Requires passes with depth. Default `false`, which
   * draws once.
   */
  readonly prepass?: boolean;
}

/**
 * This renderer draws a set into pass descriptions. Add it to a colour description with
 * `scene.add(renderer)` or a depth-only description with `light.shadow.add(renderer)`. Each
 * description builds pipelines for its camera and, with `cull`, keeps its own visibility list.
 *
 * A renderer draws the meshes that have instances, in the order they gained them: without
 * culling one direct draw each, with culling one indirect draw each. Render bundles also draw
 * indirectly so replay reads live instance counts. A bundle draws the meshes that had instances
 * when it was recorded; record it again after a mesh without instances gains one, and once a
 * model's `ready` resolves. Meshes with
 * the same back-face, alpha-mask and texture settings share pipelines. Indirect draws without
 * `indirect-first-instance` use a pipeline of their own that reads the mesh from a bind group, or
 * one per mesh where the pipeline has no group left.
 */
export interface Renderer extends passes.Compound<Lit, Corner> {
  /**
   * Records visible-instance lists for each enabled description that contains this renderer, for
   * the layers it reports `active`, such as those a shadow's `prepare` picked. Call this after
   * `flush()` and any kernels that move instances, before drawing. Throws unless the renderer has
   * `cull: true`.
   */
  cull(pass: passes.AnyComputePass): void;
  /**
   * Returns this renderer with `fill` applied to every pipeline and kernel it builds. The fill
   * runs after lighting and before the pass camera. Use it for resources read by kind callbacks,
   * as in `school.renderer().pipe(water.inject())`.
   */
  pipe(fill: Inject): Renderer;
}

/**
 * A set of instances is a table of its kind whose `add` takes a mesh: `fields`, `live()`,
 * `buffers`, `inject()` and `flush` work as they do on a table. An id names an instance; a model
 * instance takes a row per mesh under one id, so `count` counts instances while `capacity` and
 * `span` count rows.
 */
export interface Instances<K extends AnyKind = AnyKind> extends Omit<
  table.Table<K>,
  'add' | 'write' | 'fields'
> {
  /** Gives the kind of its rows, `options.kind` or `instances.Instance`. */
  readonly kind: K;
  /** Gives the number of live mesh or model instances. */
  readonly count: number;
  /**
   * Adds an instance of a mesh, model or group of meshes from the pool. Each mesh uses a row,
   * and all rows move and change together under one id. `values` may be a function of each mesh,
   * to give a model's meshes values of their own. Throws if a mesh belongs to another pool, the
   * group is empty, a posed kind receives a mesh without joints or the set is full. When `values`
   * or a write throws, it adds no row and rethrows.
   */
  add(
    mesh: Mesh | Pick<Model, 'meshes'>,
    values?: Values<K> | ((mesh: Mesh, index: number) => Values<K>),
  ): Id<K>;
  /**
   * Writes the supplied fields and keeps the rest. At the next `flush`, device fields written
   * here overwrite kernel writes.
   */
  write(id: Id<K>, values: Values<K>): void;
  /**
   * Provides one handle per field, which writes without allocating. Keep a handle outside a loop
   * instead of reading `fields.x` inside it.
   */
  readonly fields: Fields<K>;
  /** Returns the id of the instance that holds `slot`, such as a slot a picker found. */
  idAt(slot: number): Id<K>;
  /**
   * Uploads changed rows. When instances are added or removed, it also uploads live rows per
   * mesh and updates draw counts. Call this once per frame before drawing. `HistoryInstances`
   * take the frame or its encoder and throw without one.
   */
  flush(target?: Frame | TgpuCommandEncoder): void;
  /**
   * Creates a renderer for colour and shadow pass descriptions, lit by the lighting each
   * description gives through `beneath`. Use `pipe` to fill resources read by kind callbacks, or
   * another lighting.
   */
  renderer(options?: RendererOptions): Renderer;
}

/** A set of instances created with `history`, which keeps last frame's rows. */
export interface HistoryInstances<K extends AnyKind = AnyKind> extends Omit<Instances<K>, 'flush'> {
  /**
   * Uploads changed rows, as `Instances.flush` does, and records into `target`, a frame or an
   * encoder, the copies that keep last frame's rows. Call this once per frame before drawing.
   */
  flush(target: Frame | TgpuCommandEncoder): void;
}

/** The blend constant tints the background before adding the fragment colour. */
const THROUGH: GPUBlendState = Object.freeze({
  color: Object.freeze({ srcFactor: 'one', dstFactor: 'constant', operation: 'add' }),
  alpha: pipelines.blend.over.alpha,
});

const rowOf = tgpu.slot<(instanceIndex: number) => number>();
/** Indirect draws without `firstInstance` read the mesh they draw from a group set per draw. */
// Only the vertex stage reads the mesh, so the fragment stage keeps its uniform for others
const Meshing = tgpu
  .bindGroupLayout({ mesh: { uniform: d.u32, visibility: ['vertex'] } })
  .$name('meshing');
const blended = tgpu.slot(false);
const clearing = tgpu.slot(false);
const transmitting = tgpu.slot(false);
const transmittance = tgpu.slot(d.vec3f(1));

/** The transmitted share must also pass through the fog in front of the background. */
const clearAt = (made: Material, point: Point, opacity: number) => {
  'use gpu';
  const cleared = shadeClear(made, point, opacity);
  if (!transmitting.$) return cleared;
  const behind = d.vec3f(1 - opacity) - transmittance.$;
  return d.vec4f(cleared.xyz + fog.$(point).added * behind, cleared.w);
};
const texturing = tgpu.slot(true);
const smoothing = tgpu.slot(true);
const shading = /* @__PURE__ */ tgpu.slot<(material: Material, point: Point) => d.v3f>(shade);
const twoSided = tgpu.slot(false);
const cutting = tgpu.slot(false);
const covering = tgpu.slot(false);
const prepassed = tgpu.slot(false);
/** Whether the pipeline hands samples over through its sample mask rather than discarding. */
const masking = tgpu.slot(false);
const band = tgpu.slot<{
  near: boolean;
  far: boolean;
  low: (row: number) => number;
  high: (row: number) => number;
  fades: boolean;
  fade: number;
}>();
const eye = tgpu.accessor(d.vec3f);

const thresholdAt = (pixel: d.v2f) => {
  'use gpu';
  const x = d.u32(pixel.x) & 3;
  const y = d.u32(pixel.y) & 3;
  const a = x ^ y;
  const index = ((a & 1) << 3) | ((y & 1) << 2) | (a & 2) | ((y >>> 1) & 1);
  return (d.f32(index) + 0.5) / 16;
};

/** The thresholds the renderer keeps at `distance`, from `x` up to but not including `y`. */
const keptRange = (distance: number, row: number) => {
  'use gpu';
  let range = d.vec2f(0, 1);
  if (band.$.far) range.y = std.saturate((band.$.high(row) - distance) / band.$.fade);
  if (band.$.near) range.x = std.saturate((band.$.low(row) - distance) / band.$.fade + 1);
  return range;
};

/**
 * The share of `kept` samples that `coverage` selects, one bit each, as alpha to coverage would.
 * Choosing among the kept samples leaves no edge without samples while a renderer hands over.
 */
const coveredSamples = (kept: number, coverage: number) => {
  'use gpu';
  const count = d.u32(passes.sampleCount.$);
  const total = d.f32(std.countOneBits(kept & ((d.u32(1) << count) - 1)));
  let covered = d.u32(0);
  let seen = d.f32(0);
  for (let sample = d.u32(0); sample < count; sample++) {
    const bit = d.u32(1) << sample;
    if ((kept & bit) !== 0) {
      if (coverage * total > seen + 0.5) covered = covered | bit;
      seen += 1;
    }
  }
  return covered;
};

/**
 * The share of the pixel an instance covers. Opaque drawings discard what they cover too little
 * of; with multisampling, the share, sharpened to one pixel, selects samples by alpha to
 * coverage, and otherwise it is whole. Blended drawings scale their opacity by it.
 */
const shareOf = (raw: number) => {
  'use gpu';
  let coverage = d.f32(raw);
  if (blended.$) return coverage;
  if (covering.$) {
    // Sharpen coverage to one pixel before alpha-to-coverage to avoid a dithered band on
    // magnified atlases
    const across = std.max(std.fwidth(coverage), 1e-4);
    coverage = std.saturate((coverage - 0.5) / across + 0.5);
    if (coverage <= 0) std.discard();
  } else {
    if (coverage < 0.5) std.discard();
    coverage = 1;
  }
  return coverage;
};

const pointOf = (corner: Corner) => {
  'use gpu';
  const world = corner.world;
  const view = cameras.towards(cameras.current.$, world);
  let bent = std.normalize(corner.normal);
  // Light back faces from their visible side
  if (twoSided.$) {
    if (std.dot(bent, view) < 0) bent = bent * -1;
  }
  if (texturing.$) bent = mappedNormal(bent, world, corner.uv, corner.uv1);
  return Point({ position: world, normal: bent, view, uv: corner.uv, uv1: corner.uv1 });
};

const depthOf = (position: d.v3f) => {
  'use gpu';
  const clip = cameras.toClip(position);
  return std.saturate(clip.z / clip.w);
};

/** Rotates `v` by unit quaternion `q`. Use `d.vec4f(-q.xyz, q.w)` to undo the rotation. */
export const rotate = (q: d.v4f, v: d.v3f) => {
  'use gpu';
  const twice = std.cross(q.xyz, v) * 2;
  return v + twice * q.w + std.cross(q.xyz, twice);
};

/**
 * Returns a unit quaternion that points mesh +z along `direction` and keeps +y up. glTF models
 * face +z. `direction` can have any length.
 */
export const toward = (direction: d.v3f) => {
  'use gpu';
  const forward = std.normalize(direction);
  const yaw = std.atan2(forward.x, forward.z) * 0.5;
  const pitch = std.asin(std.clamp(forward.y, -1, 1)) * -0.5;
  const cy = std.cos(yaw);
  const sy = std.sin(yaw);
  const cp = std.cos(pitch);
  const sp = std.sin(pitch);

  return d.vec4f(cy * sp, sy * cp, -sy * sp, cy * cp);
};

const placed = (where: Placement, local: d.v3f) => {
  'use gpu';
  return rotate(where.rotation, local * where.scale) + where.position;
};

type ShaderRow<I> = { readonly [K in keyof Every]: (slot: number) => d.InferGPU<Every[K]> } & {
  readonly row: (slot: number) => I;
};

function shadersOf<I>(
  Row: ShaderRow<I>,
  PixelOf: d.WgslStruct,
  hooks: {
    readonly pose: ((instance: I) => number) | undefined;
    readonly deform: ((vertex: Vertex, instance: I) => d.v3f) | undefined;
    readonly carry: ((vertexIndex: number, instance: I) => d.v4f) | undefined;
    readonly material: ((pixel: Pixel<I>, base: Material) => Material) | undefined;
    readonly surface: ((pixel: Pixel<I>, base: Material) => Shaded) | undefined;
    readonly coverage: ((pixel: Pixel<I>) => number) | undefined;
    readonly coveredSurface:
      | ((pixel: Pixel<I>, base: Material) => d.Infer<typeof Covered>)
      | undefined;
    readonly bounds: ((instance: I, base: Sphere) => Sphere) | undefined;
    readonly place: ((instance: I) => Placement) | undefined;
    readonly visible: ((instance: I) => boolean) | undefined;
    readonly deformShadow: boolean;
  },
) {
  const { pose, deform, carry, material, surface, coverage: share, bounds, place, visible } = hooks;
  const { coveredSurface } = hooks;
  const { deformShadow } = hooks;

  const placementOf = (row: number) => {
    'use gpu';
    if (place !== undefined) return Placement(place(Row.row(row)));
    return Placement({
      rotation: Row.rotation(row),
      position: Row.position(row),
      scale: Row.scale(row),
    });
  };

  /** Hidden instances collapse beyond the far plane so they produce no fragments. */
  const hidden = d.vec4f(0, 0, 2, 1);

  const posedAt = (vertexIndex: number, row: number) => {
    'use gpu';
    if (pose !== undefined) {
      return skinned(vertexIndex, vertexAt(vertexIndex), pose(Row.row(row)));
    }
    return vertexAt(vertexIndex);
  };

  const shaped = (vertex: Vertex, row: number) => {
    'use gpu';
    if (deform !== undefined) return deform(vertex, Row.row(row));
    return d.vec3f(vertex.position);
  };

  const castAt = (vertexIndex: number, row: number) => {
    'use gpu';
    if (deform !== undefined && deformShadow) return shaped(posedAt(vertexIndex, row), row);
    if (pose !== undefined) {
      return skinnedPosition(vertexIndex, positionAt(vertexIndex), pose(Row.row(row)));
    }
    return positionAt(vertexIndex);
  };

  /** Samples the colour texture once for the mask and the material, or gives white without one. */
  const colorOf = (corner: Corner) => {
    'use gpu';
    let sampled = d.vec4f(1);
    if (texturing.$ || cutting.$) sampled = colorAt(corner.uv, corner.uv1);
    return sampled;
  };

  const standard = (point: Point, row: number, color: d.v4f) => {
    'use gpu';
    const factors = Material({
      color: Row.color(row),
      roughness: Row.roughness(row),
      emission: Row.emission(row),
      metallic: Row.metallic(row),
      occlusion: 1,
    });
    if (texturing.$) return texturedBy(factors, point, color);
    return factors;
  };

  const carriedOf = (corner: Corner) => {
    'use gpu';
    if (carry !== undefined) return d.vec4f((corner as Carrying).carried);
    return d.vec4f();
  };

  const pixelOf = (corner: Corner, point: Point) => {
    'use gpu';
    const instance = Row.row(corner.row);
    return PixelOf({ point, instance, carried: carriedOf(corner) }) as unknown as Pixel<I>;
  };

  const materialAt = (corner: Corner, point: Point, color: d.v4f) => {
    'use gpu';
    const base = standard(point, corner.row, color);
    if (material !== undefined) return material(pixelOf(corner, point), base);
    return base;
  };

  const at = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const row = rowOf.$(instanceIndex);
    const where = placementOf(row);
    const local = posedAt(vertexIndex, row);
    const world = placed(where, shaped(local, row));
    let position = cameras.toClip(world);
    if (visible !== undefined && !visible(Row.row(row))) position = d.vec4f(hidden);
    const normal = rotate(where.rotation, local.normal / where.scale);
    const uv1 = uv1At(vertexIndex);
    if (carry !== undefined) {
      const carried = carry(vertexIndex, Row.row(row));
      return Carrying({ position, world, normal, uv: local.uv, uv1, row, carried });
    }
    return Corner({ position, world, normal, uv: local.uv, uv1, row });
  };

  const cast = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const row = rowOf.$(instanceIndex);
    const world = placed(placementOf(row), castAt(vertexIndex, row));
    let position = cameras.toClip(world);
    if (visible !== undefined && !visible(Row.row(row))) position = d.vec4f(hidden);
    return Cast({ position });
  };

  const distanceOf = (row: number) => {
    'use gpu';
    return std.distance(eye.$, placementOf(row).position);
  };

  const handedOver = (pixel: d.v4f, row: number) => {
    'use gpu';
    const range = keptRange(distanceOf(row), row);
    const threshold = thresholdAt(pixel.xy);
    return threshold < range.x || threshold >= range.y;
  };

  /**
   * The samples the renderer keeps, one bit each. The instance's share rounds to whole samples,
   * the same in every pixel, so the resolve blends the whole instance alike, without a pattern.
   */
  const keptSamples = (row: number) => {
    'use gpu';
    const range = keptRange(distanceOf(row), row);
    if (range.x <= 0 && range.y >= 1) return d.u32(0xffffffff);
    const count = d.u32(passes.sampleCount.$);
    let kept = d.u32(0);
    for (let sample = d.u32(0); sample < count; sample++) {
      const threshold = (d.f32(sample) + 0.5) / d.f32(count);
      if (threshold >= range.x && threshold < range.y) kept = kept | (d.u32(1) << sample);
    }
    return kept;
  };

  const coverageOf = (corner: Corner, point: Point) => {
    'use gpu';
    if (share !== undefined) return d.f32(share(pixelOf(corner, point)));
    return d.f32(1);
  };
  const cuts = share !== undefined;

  /** Discards what fades out of range. */
  const handOver = (corner: Corner) => {
    'use gpu';
    if (band.$.fades && !masking.$) {
      if (handedOver(corner.position, corner.row)) std.discard();
    }
  };

  /** Discards what fades out of range or its alpha cuts away, and returns its sampled colour. */
  const keptColor = (corner: Corner) => {
    'use gpu';
    handOver(corner);
    const sampled = colorOf(corner);
    if (cutting.$) {
      if (cutBelow(sampled.a, Row.opacity(corner.row))) std.discard();
    }
    return sampled;
  };

  const shadedLit = (corner: Corner, point: Point, sampled: d.v4f, coverage: number) => {
    'use gpu';
    let made = materialAt(corner, point, sampled);
    if (smoothing.$) made = antialiased(made, point.normal);
    if (clearing.$) {
      return Lit({ color: clearAt(made, point, Row.opacity(corner.row)) * coverage });
    }

    const color = shading.$(made, point);
    if (blended.$) {
      const opacity = Row.opacity(corner.row) * coverage;
      return Lit({ color: d.vec4f(color * opacity, opacity) });
    }
    return Lit({ color: d.vec4f(color, coverage) });
  };

  const lit = (corner: Corner) => {
    'use gpu';
    // Coverage reads the point, which otherwise waits until the discards have run
    if (cuts && !prepassed.$) {
      const point = pointOf(corner);
      const coverage = shareOf(coverageOf(corner, point));
      const sampled = keptColor(corner);
      return shadedLit(corner, point, sampled, coverage);
    }
    const sampled = keptColor(corner);
    return shadedLit(corner, pointOf(corner), sampled, d.f32(1));
  };

  const surfaceAt = (corner: Corner, point: Point) => {
    'use gpu';
    const pixel = pixelOf(corner, point);
    const base = standard(point, corner.row, colorOf(corner));
    if (coveredSurface !== undefined) {
      const here = coveredSurface(pixel, base);
      return Shaded({ point: here.point, material: here.material });
    }
    return Shaded(surface!(pixel, base));
  };

  /**
   * A depth prepass writes what the instance covers, with its coverage as alpha for alpha to
   * coverage, so the colour pass shades each pixel once where the depth is equal.
   */
  const covered = (corner: Corner) => {
    'use gpu';
    let coverage = d.f32(1);
    if (cuts) coverage = shareOf(coverageOf(corner, pointOf(corner)));
    handOver(corner);
    if (cutting.$) {
      if (masked(corner.uv, corner.uv1, Row.opacity(corner.row))) std.discard();
    }
    return Lit({ color: d.vec4f(0, 0, 0, coverage) });
  };

  /** Blended surfaces scale their opacity by coverage. */
  const showing = (corner: Corner) => {
    'use gpu';
    // Discard by coverage before the surface, so pixels a card leaves empty skip its work
    const point = pointOf(corner);
    let coverage = d.f32(1);
    if (cuts && !prepassed.$) coverage = shareOf(coverageOf(corner, point));
    handOver(corner);
    const here = surfaceAt(corner, point);
    let made = Material(here.material);
    if (smoothing.$) made = antialiased(made, here.point.normal);
    if (clearing.$) {
      const cleared = clearAt(made, here.point, Row.opacity(corner.row));
      return Lit({ color: cleared * coverage });
    }
    const color = shading.$(made, here.point);
    if (blended.$) {
      const opacity = Row.opacity(corner.row) * coverage;
      return Lit({ color: d.vec4f(color * opacity, opacity) });
    }
    return Lit({ color: d.vec4f(color, coverage) });
  };

  /** The samples an opaque pixel writes in its fade, with its coverage folded in. */
  const kept = (corner: Corner, shown: Lit) => {
    'use gpu';
    let samples = keptSamples(corner.row);
    if (cuts && !blended.$) samples = coveredSamples(samples, shown.color.w);
    return samples;
  };

  /** Tells a depth-only pass of a mesh what its mask or `coverage` cut away. */
  const cutAway = (corner: Corner) => {
    'use gpu';
    // Take the coverage first, so its derivatives run for every pixel
    let coverage = d.f32(1);
    if (cuts) coverage = coverageOf(corner, pointOf(corner));
    let away = false;
    if (cutting.$) away = masked(corner.uv, corner.uv1, Row.opacity(corner.row));
    if (cuts) away = away || coverage < 0.5;
    return away;
  };

  /**
   * Places a depth-only pixel of a surface, discarding what its coverage cuts away. A surface that
   * works out its own coverage takes both from one set of samples, which costs less here than
   * discarding first.
   */
  const placing = (corner: Corner) => {
    'use gpu';
    const point = pointOf(corner);
    if (coveredSurface !== undefined) {
      const here = coveredSurface(
        pixelOf(corner, point),
        standard(point, corner.row, colorOf(corner)),
      );
      if (here.coverage < 0.5) std.discard();
      return d.vec3f(here.point.position);
    }
    if (cuts && coverageOf(corner, point) < 0.5) std.discard();
    return d.vec3f(surfaceAt(corner, point).point.position);
  };

  const sphereOf = (row: number) => {
    'use gpu';
    const box = boundsOf(Row.mesh(row));
    const where = placementOf(row);
    const sphere = Sphere({
      center: placed(where, box.center),
      radius: std.length(box.half * where.scale),
    });
    if (bounds !== undefined) return bounds(Row.row(row), sphere);
    return sphere;
  };

  const seen = (row: number) => {
    'use gpu';
    if (visible !== undefined && !visible(Row.row(row))) return false;
    if (band.$.near && distanceOf(row) < band.$.low(row)) return false;
    if (band.$.far && distanceOf(row) >= band.$.high(row)) return false;
    const sphere = sphereOf(row);
    return cameras.sees(cameras.current.$, sphere.center, sphere.radius);
  };

  const meshOf = (row: number) => {
    'use gpu';
    return Row.mesh(row);
  };

  return {
    Vertex: carry ? (Carrying as unknown as typeof Corner) : Corner,
    at,
    cast,
    lit,
    surfaced: (surface ?? coveredSurface) && { showing, placing, depthOf },
    covered,
    kept,
    cutAway,
    cuts,
    seen,
    meshOf,
  };
}

/** Lets an impostor hand over a surface that works out its own coverage. */
interface Hidden<I> {
  readonly '~covered'?: (pixel: Pixel<I>, base: Material) => d.Infer<typeof Covered>;
}

const BASE = new Set([...Object.keys(placement), ...Object.keys(looks)]);

/**
 * Defines an instance kind with custom fields, transform ownership and callbacks for vertex
 * motion or materials. Each kind uses colour and shadow pipelines per renderer. `name` brands
 * set ids, as in `Id<typeof Fish>`, and labels the table and WGSL. Throws if both `material` and
 * `surface` are supplied, or a custom field duplicates a built-in or reserved name.
 *
 * @param name a label for ids and WGSL
 * @returns a kind to give `instances.create` as `kind`
 */
export function define<
  const N extends string,
  H extends Props = {},
  D extends Props = {},
  M extends 'host' = 'host',
>(name: N, options?: KindOptions<H, D, M>): Kind<N, H, D, M>;
// A kind whose transform kernels own says so at run time too
export function define<
  const N extends string,
  H extends Props = {},
  D extends Props = {},
  M extends Moved = Moved,
>(name: N, options: KindOptions<H, D, M> & { readonly transform: M }): Kind<N, H, D, M>;
export function define<
  const N extends string,
  H extends Props = {},
  D extends Props = {},
  M extends Moved = 'host',
>(name: N, options: KindOptions<H, D, M> = {}): Kind<N, H, D, M> {
  const { host = {}, device = {}, transform = 'host' as M, deform, material, surface } = options;
  const { coverage } = options;
  const { '~covered': coveredSurface } = options as Hidden<InstanceOf<H, D, M>>;
  const { pose, bounds, place, visible, carry, deformShadow = true } = options;
  if (material && surface)
    throw new Error(`The kind '${name}' must supply either material or surface, not both.`);

  for (const key of [...Object.keys(host), ...Object.keys(device)]) {
    if (BASE.has(key))
      throw new Error(`'${key}' is a built-in instance field. Choose another name.`);
  }

  // Either placement has the same fields as the hooks, with the transform in the chosen part.
  // TypeScript cannot narrow the generic transform with the runtime branch.
  const Row = (transform === 'device'
    ? table.define(name, { host: { ...looks, ...host }, device: { ...placement, ...device } })
    : table.define(name, { host: { ...placedHere, ...host }, device })) as unknown as TableDef<
    N,
    HostOf<H, M>,
    DeviceOf<D, M>
  > &
    ShaderRow<InstanceOf<H, D, M>>;

  const PixelOf = d
    .struct({ point: Point, instance: Row['~parts'].row, carried: d.vec4f })
    .$name(`${name}Pixel`);
  const shaders = shadersOf<InstanceOf<H, D, M>>(Row, PixelOf, {
    pose,
    deform,
    carry,
    material,
    surface,
    coverage,
    coveredSurface,
    bounds,
    place,
    visible,
    deformShadow,
  });
  const unbounded = deform !== undefined && bounds === undefined;

  return Object.assign(Row, {
    '~kind': { ...shaders, unbounded, posed: pose !== undefined, moved: transform },
  });
}

/**
 * The plain kind, with a transform and material, that a set without `kind` stores. Its getters
 * read a plain set in shaders filled by the set's `inject()`.
 */
export const Instance: Plain = /* @__PURE__ */ define('Instance');

/** A table's field handle with its types erased. */
interface Handle<K extends SomeKind> {
  write(id: Id<K>, value: unknown): void;
  set?(id: Id<K>, x: number, y?: number, z?: number, w?: number): void;
  get?(id: Id<K>, out?: unknown): unknown;
}

type RowInput = { readonly [F in keyof Values]?: Values[F] | undefined } & {
  readonly mesh?: number;
};

/**
 * Creates a set of mesh instances with transforms, materials and optional custom fields. Each
 * instance uses a row, and renderers draw one list of live rows per mesh.
 *
 * @param pool the mesh pool whose meshes it draws
 * @param options.kind a kind from `instances.define`
 * @returns a set of instances you flush every frame and destroy
 */
export function create(
  root: TgpuRoot,
  pool: MeshPool,
  options: Options & { readonly kind?: Plain; readonly history: true },
): HistoryInstances<Plain>;
export function create(
  root: TgpuRoot,
  pool: MeshPool,
  options: Options & { readonly kind?: Plain },
): Instances<Plain>;
// A set typed with a kind of its own is given that kind at run time too
export function create<K extends AnyKind>(
  root: TgpuRoot,
  pool: MeshPool,
  options: Options<K> & { readonly kind: K; readonly history: true },
): HistoryInstances<K>;
export function create<K extends AnyKind>(
  root: TgpuRoot,
  pool: MeshPool,
  options: Options<K> & { readonly kind: K },
): Instances<K>;
export function create<K extends SomeKind>(
  root: TgpuRoot,
  pool: MeshPool,
  options: Options<K>,
): Instances<K> | Instances<Plain> {
  const kind = options.kind;
  return kind ? createSet(root, pool, kind, options) : createSet(root, pool, Instance, options);
}

function createSet<K extends SomeKind>(
  root: TgpuRoot,
  pool: MeshPool,
  kind: K,
  options: Omit<Options<K>, 'kind'>,
): Instances<K> {
  const { name } = kind['~parts'];
  const { capacity, history = false } = options;
  const shaders = kind['~kind'];

  // Material defaults replace the mesh's factors, so they apply after them in `add`
  const { color, roughness, metallic, emission, opacity, ...placing } = options.defaults ?? {};
  const looking = { color, roughness, metallic, emission, opacity };

  // The kind adds fields to the built-ins but cannot redeclare them. Normalize scalar scale
  // before passing these values through the table's mapped type.
  const rowValues = (given: RowInput): table.Values<K> => {
    const { scale } = given;
    const values = typeof scale === 'number' ? { ...given, scale: [scale, scale, scale] } : given;
    return values as table.Values<K>;
  };
  const looksDefault = Object.values(looking).some((value) => value !== undefined)
    ? rowValues(looking)
    : undefined;
  const rows = table.create(root, kind, {
    capacity,
    history,
    defaults: rowValues({
      rotation: [0, 0, 0, 1],
      scale: [1, 1, 1],
      color: [1, 1, 1],
      roughness: 1,
      opacity: 1,
      ...placing,
    }),
  });
  const firstInstance = root.device.features.has('indirect-first-instance');
  const meshCount = pool.capacity;

  // Upload mesh start offsets before grouped live rows. Use firstInstance to select each range,
  // or read the start through the mesh's group in indirect draws without that feature
  const runs: number[][] = Array.from({ length: meshCount }, () => []);
  const meshAt = new Uint32Array(capacity);
  const orderAt = new Uint32Array(capacity);
  const packed = new Uint32Array(meshCount + capacity);
  const listing = root
    .createBuffer(d.arrayOf(d.u32, meshCount + capacity))
    .$usage('storage')
    .$name(`${name}_listing`);
  const listed = listing.as('readonly');

  const args = new Uint32Array(meshCount * RECORD);
  const staged = new Uint32Array(meshCount * RECORD);
  const queue = root.device.queue;
  const listingGpu = root.unwrap(listing);

  // Use indirect draws in bundles so replay reads live instance counts
  let replays: (TgpuBuffer<d.WgslArray<d.U32>> & IndirectFlag) | undefined;

  let listedRows = 0;
  let entries: TgpuUniform<d.U32> | undefined;
  const lists: Culling[] = [];
  /** Meshes with rows in the order they gained them, and those emptied since the last `flush`. */
  const drawn: number[] = [];
  const isDrawn = new Uint8Array(meshCount);
  // An indirect draw without firstInstance starts at instance 0, so a group names its mesh
  const meshings: TgpuBindGroup<(typeof Meshing)['entries']>[] = [];
  let changed = false;

  const list = (slot: number, mesh: number) => {
    const run = runs[mesh];
    meshAt[slot] = mesh;
    orderAt[slot] = run.length;
    run.push(slot);
  };

  const unlist = (slot: number) => {
    const run = runs[meshAt[slot]];
    const order = orderAt[slot];
    const last = run.pop()!;
    if (last !== slot) {
      run[order] = last;
      orderAt[last] = order;
    }
  };

  const uploadReplays = (into: TgpuBuffer<d.WgslArray<d.U32>> & IndirectFlag) => {
    staged.set(args);
    if (!firstInstance) for (let mesh = 0; mesh < meshCount; mesh++) staged[mesh * RECORD + 4] = 0;
    queue.writeBuffer(root.unwrap(into), 0, staged.buffer);
  };

  const replayed = () => {
    if (!replays) {
      replays = root
        .createBuffer(d.arrayOf(d.u32, meshCount * RECORD))
        .$usage('indirect')
        .$name(`${name}_replays`);
      uploadReplays(replays);
    }
    return replays;
  };

  const flush = (target?: Frame | TgpuCommandEncoder) => {
    rows.flush(target);
    if (!changed) return;
    changed = false;

    let next = meshCount;
    let kept = 0;
    for (const mesh of drawn) {
      const run = runs[mesh];
      // A bundle that drew this mesh keeps reading its record, so an empty mesh draws nothing
      if (run.length === 0) {
        args[mesh * RECORD + 1] = 0;
        isDrawn[mesh] = 0;
        continue;
      }
      drawn[kept++] = mesh;
      packed[mesh] = next;
      packed.set(run, next);
      args[mesh * RECORD + 1] = run.length;
      args[mesh * RECORD + 4] = next;
      next += run.length;
    }
    drawn.length = kept;
    queue.writeBuffer(listingGpu, 0, packed.buffer, 0, next * 4);
    listedRows = next - meshCount;
    if (replays) uploadReplays(replays);
    if (!entries || lists.length === 0) return;
    entries.write(listedRows);
    staged.set(args);
    for (let mesh = 0; mesh < meshCount; mesh++) {
      staged[mesh * RECORD + 1] = 0;
      staged[mesh * RECORD + 4] = 0;
    }
    for (const each of lists) {
      const draws = root.unwrap(each.draws);
      for (let k = 0; k < each.layers; k++) {
        queue.writeBuffer(draws, k * staged.byteLength, staged.buffer);
      }
    }
  };

  /** A model's rows after its first, by the first's id. */
  const together = new Map<Id<K>, Id<K>[]>();
  /** The slot of the first row of the instance that holds each slot. */
  const firstOf = new Uint32Array(capacity);
  let count = 0;

  const write = (id: Id<K>, given: Values<K>) => {
    const values = rowValues(given);
    rows.write(id, values);

    const others = together.get(id);
    if (others) for (const other of others) rows.write(other, values);
  };

  /** Adds and lists a row for `mesh`, into `added` before writing it, so a failed add can undo it. */
  const addRow = (mesh: Mesh, values: table.Values<K>, added: Id<K>[]) => {
    const id = rows.add(rowValues({ mesh: mesh.index }));
    list(rows.slotOf(id), mesh.index);
    added.push(id);
    changed = true;
    // Write mesh material first so omitted instance fields retain mesh factors
    rows.write(id, rowValues(mesh.material));
    if (looksDefault) rows.write(id, looksDefault);
    rows.write(id, values);

    if (!isDrawn[mesh.index]) {
      isDrawn[mesh.index] = 1;
      drawn.push(mesh.index);
      if (!firstInstance && !meshings[mesh.index]) {
        const index = root.createBuffer(d.u32, mesh.index).$usage('uniform');
        meshings[mesh.index] = root.createBindGroup(Meshing, {
          mesh: index.$name(`${name}_mesh_${mesh.index}`),
        });
      }
      args[mesh.index * RECORD] = mesh.indexCount;
      args[mesh.index * RECORD + 2] = mesh.firstIndex;
      args[mesh.index * RECORD + 3] = mesh.baseVertex;
    }
  };

  const add = (
    target: Mesh | Pick<Model, 'meshes'>,
    given?: Values<K> | ((mesh: Mesh, index: number) => Values<K>),
  ) => {
    const meshes = 'meshes' in target ? target.meshes : [target];
    if (meshes.length === 0) throw new Error('The model has no meshes.');
    if (meshes.some((mesh) => pool.meshes[mesh.index] !== mesh)) {
      throw new Error('The mesh belongs to another pool.');
    }
    if (shaders.posed && meshes.some((mesh) => !mesh.skinned)) {
      throw new Error(`The kind '${name}' uses pose and requires a mesh with joints.`);
    }
    if (rows.count + meshes.length > capacity) {
      throw new Error(
        `The instance set cannot fit this model within its ${capacity}-row capacity. A model takes one row per mesh.`,
      );
    }

    const each = typeof given === 'function' ? given : undefined;
    const shared = rowValues(typeof given === 'function' || given === undefined ? {} : given);
    const added: Id<K>[] = [];
    try {
      for (let i = 0; i < meshes.length; i++) {
        addRow(meshes[i], each ? rowValues(each(meshes[i], i)) : shared, added);
      }
    } catch (error) {
      // A model adds whole or not at all
      for (const row of added) {
        unlist(rows.slotOf(row));
        rows.remove(row);
      }
      throw error;
    }
    const [id, ...others] = added;
    if (others.length > 0) together.set(id!, others);
    const first = rows.slotOf(id!);
    firstOf[first] = first;
    for (const other of others) firstOf[rows.slotOf(other)] = first;
    count += 1;
    return id!;
  };

  /** Wraps a table's field handle so a model's id reaches every row of the model. */
  const fanned = ({ write: each, set, get }: Handle<K>) => {
    const result: Handle<K> = {
      write(id, value) {
        each(id, value);
        const others = together.get(id);
        if (others) for (const other of others) each(other, value);
      },
    };
    if (set) {
      // Pass every component on so the table's handle for the field's arity takes what it needs
      result.set = (id, x, y, z, w) => {
        set(id, x, y, z, w);
        const others = together.get(id);
        if (others) for (const other of others) set(other, x, y, z, w);
      };
    }
    if (get) result.get = get;
    return result;
  };
  const fields = Object.fromEntries(
    Object.entries(rows.fields as Record<string, Handle<K>>)
      .filter(([key]) => key !== 'mesh')
      .map(([key, handle]) => [key, fanned(handle)]),
  ) as unknown as Fields<K>;

  // Pipelines that read the mesh from its group, rather than compiling it in
  const grouped = new WeakSet<TgpuRenderPipeline>();
  const fits = (pipeline: TgpuRenderPipeline) =>
    tgpu.resolveWithContext([pipeline]).usedBindGroupLayouts.length <=
    root.device.limits.maxBindGroups;

  const byMesh = (
    culling: Culling | undefined,
    keyOf: (look: Look) => string,
    make: (row: (instanceIndex: number) => number, look: Look) => TgpuRenderPipeline,
  ) => {
    const from = culling?.culled.as('readonly') ?? listed;
    // Each layer's list starts after the lists of the layers before it
    const perLayer = culling !== undefined && culling.layers > 1;
    const span = meshCount + capacity;
    const listedRow = (instanceIndex: number) => {
      'use gpu';
      return from.$[instanceIndex];
    };
    /** Reads rows from the list of `mesh`, or of the mesh in the draw's group without one. */
    const rowsOf = (mesh?: number) => {
      const meshRow = (instanceIndex: number) => {
        'use gpu';
        const start = perLayer ? passes.layer.$ * span : d.u32(0);
        const which = mesh === undefined ? Meshing.$.mesh : d.u32(mesh);
        return from.$[from.$[start + which] + instanceIndex];
      };
      return meshRow;
    };
    const direct = new Map<string, TgpuRenderPipeline>();
    const meshed = new Map<string, TgpuRenderPipeline>();
    // Looks whose pipelines have no room for one more group
    const crowded = new Set<string>();
    const perMesh: TgpuRenderPipeline[] = [];

    return (mesh: number, indirect: boolean) => {
      const look = pool.meshes[mesh];
      const key = keyOf(look);
      if (!indirect || firstInstance) {
        let pipeline = direct.get(key);
        if (!pipeline) direct.set(key, (pipeline = make(listedRow, look)));
        return pipeline;
      }
      if (!meshed.has(key) && !crowded.has(key)) {
        const made = make(rowsOf(), look);
        if (fits(made)) {
          grouped.add(made);
          meshed.set(key, made);
        } else {
          crowded.add(key);
        }
      }
      const pipeline = meshed.get(key);
      if (pipeline) return pipeline;
      // A crowded look compiles each mesh's start into a pipeline of its own
      let alone = perMesh[mesh];
      if (!alone) perMesh[mesh] = alone = make(rowsOf(mesh), look);
      return alone;
    };
  };

  /**
   * Keep bindings on the pipeline so later draws in the pass retain their own resources. Each mesh
   * sets its bound pipeline on the pass; consecutive meshes with the same pipeline and textures
   * share one, so the pass binds it once, unless an indirect draw binds its mesh's group too. Bind
   * again after texture upload. Depth-only passes need textures only for alpha masks. Culling and
   * bundle draws use indirect counts; direct draws use CPU counts from `flush`.
   */
  const drawer = (
    pipelineOf: (mesh: number, indirect: boolean) => TgpuRenderPipeline,
    withTextures: (mesh: number) => boolean,
    culling: Culling | undefined,
  ) => {
    type Group = ReturnType<MeshPool['texturesOf']> | undefined;
    type Indexed = ReturnType<TgpuRenderPipeline['withIndexBuffer']>;
    const bound: Indexed[] = [];
    const groups: Group[] = [];
    const shared = new Map<TgpuRenderPipeline, Map<Group, Indexed>>();

    const boundOf = (pipeline: TgpuRenderPipeline, group: Group) => {
      let byGroup = shared.get(pipeline);
      if (!byGroup) shared.set(pipeline, (byGroup = new Map()));

      let result = byGroup.get(group);
      if (!result) {
        const indexed = pipeline.withIndexBuffer(pool.indices);
        result = group ? indexed.with(group) : indexed;
        byGroup.set(group, result);
      }
      return result;
    };

    const meshedBy = new Map<Indexed, Indexed[]>();
    const bind = (mesh: number, group: Group, indirect: boolean) => {
      const pipeline = pipelineOf(mesh, indirect);
      const made = boundOf(pipeline, group);
      if (!grouped.has(pipeline)) return made;
      let ofMesh = meshedBy.get(made);
      if (!ofMesh) meshedBy.set(made, (ofMesh = []));
      let withMesh = ofMesh[mesh];
      if (!withMesh) ofMesh[mesh] = withMesh = made.with(meshings[mesh]);
      return withMesh;
    };

    return (pass: TgpuRenderCommands, layer: number) => {
      const replaying = !culling && pass.resourceType === 'render-bundle-encoder';
      const records = replaying ? replayed() : culling?.draws;
      const layered = culling ? layer * meshCount * RECORD : 0;

      for (const mesh of drawn) {
        const at = mesh * RECORD;
        if (!records && args[at + 1] === 0) continue;

        const group = withTextures(mesh) ? pool.texturesOf(mesh) : undefined;
        let drawing: Indexed;
        if (replaying) {
          drawing = bind(mesh, group, true);
        } else {
          if (!bound[mesh] || groups[mesh] !== group) {
            bound[mesh] = bind(mesh, group, records !== undefined);
            groups[mesh] = group;
          }
          drawing = bound[mesh];
        }

        pass.setPipeline(drawing);
        if (records) pass.drawIndexedIndirect(records, (layered + at) * 4);
        else pass.drawIndexed(args[at], args[at + 1], args[at + 2], args[at + 3], args[at + 4]);
      }
    };
  };

  const compile = async (
    kinds: readonly ((mesh: number, indirect: boolean) => TgpuRenderPipeline)[],
    culling: Culling | undefined,
  ) => {
    const indirect = culling !== undefined;
    const compiled = new Set(
      pool.meshes.flatMap(({ index }) => kinds.map((of) => of(index, indirect))),
    );
    await pipelines.compile(culling ? [...compiled, culling] : [...compiled]);
  };

  const make = (settings: RendererOptions, fills: Inject): Renderer => {
    const { blend = 'opaque', cull = false, eye: viewer, transmit, prepass = false } = settings;
    const { from = 0, to = Infinity, fade = 0 } = settings;
    const translucent = blend !== 'opaque';
    const ranged = settings.from !== undefined || settings.to !== undefined;
    const { surfaced } = shaders;

    if (cull && shaders.unbounded) {
      throw new Error(`The kind '${name}' uses deform. Supply bounds before enabling culling.`);
    }

    if (transmit && blend !== 'clear') {
      throw new Error(`A renderer with transmit requires blend: 'clear'.`);
    }
    if (transmit && !transmit.every((share) => share >= 0 && share <= 1)) {
      throw new RangeError(
        `Each transmit channel must be from 0 to 1. Received [${transmit.join(', ')}].`,
      );
    }
    if (ranged && !cull) throw new Error('A renderer with from or to requires cull: true.');
    if (ranged && !viewer) {
      throw new Error('A renderer with from or to requires eye.');
    }
    if (
      !(fade >= 0) ||
      (typeof from === 'number' && !(from >= 0)) ||
      (typeof to === 'number' && !(to > 0)) ||
      (typeof from === 'number' && typeof to === 'number' && !(to > from))
    ) {
      throw new RangeError(
        `Renderer distances require from >= 0, to > from and fade >= 0. Received from ${from}, to ${to}, fade ${fade}.`,
      );
    }
    // Extend colour and culling distance limits by half the fade width. Depth-only passes switch
    // at the limits
    const near = typeof from === 'function' || from > 0;
    const far = typeof to === 'function' || to < Infinity;
    const lowAt = typeof from === 'function' ? from : undefined;
    const highAt = typeof to === 'function' ? to : undefined;
    const low = (row: number) => {
      'use gpu';
      if (lowAt !== undefined) return lowAt(row);
      return d.f32(from as number);
    };
    const high = (row: number) => {
      'use gpu';
      if (highAt !== undefined) return highAt(row);
      return d.f32(to as number);
    };
    const shown = {
      near,
      far,
      low: (row: number) => {
        'use gpu';
        return low(row) - fade / 2;
      },
      high: (row: number) => {
        'use gpu';
        return high(row) + fade / 2;
      },
      fades: fade > 0,
      fade,
    };
    const cast = { near, far, low, high, fades: false, fade };
    const eyeOf = () => {
      'use gpu';
      if (viewer !== undefined) return d.vec3f(viewer.uniform.$.position);
      return d.vec3f(cameras.current.$.position);
    };

    // Apply caller fills after lighting so they can override hooks, then apply the pass camera
    const start = root.pipe(pool.inject()).pipe(rows.inject()).with(eye, eyeOf);
    const reads = (look: Look) => settings.textured !== false && look.textured;
    // Keep colour textures for alpha masking only when the mesh reads textures
    const masks = (look: Look) => reads(look) && look.masked;
    const castKey = (look: Look) => `${look.doubleSided} ${masks(look)}`;
    const colorKey = (look: Look) => `${castKey(look)} ${reads(look)}`;
    const bindsTextures = (mesh: number) => reads(pool.meshes[mesh]);
    // Read colour textures in depth-only passes only for alpha masks
    const castsMasked = (mesh: number) => !surfaced && masks(pool.meshes[mesh]);
    const labelOf = (look: Look, what?: string) =>
      [name, what, look.doubleSided && 'two-sided', masks(look) && 'masked']
        .filter(Boolean)
        .join(' ');

    const culled: { readonly culling: Culling; readonly into: Into }[] = [];
    const listOf = (cfg: WithBinding, into: Into) => {
      entries ??= root.createUniform(d.u32);
      const culling = createCulling(root, cfg, {
        meshes: meshCount,
        capacity,
        layers: into.layers,
        firstInstance,
        listed,
        entries,
        meshOf: shaders.meshOf,
        seen: shaders.seen,
        active: into.active,
      });
      lists.push(culling);
      culled.push({ culling, into });
      changed = true;
      return culling;
    };

    const dropped = (culling: Culling | undefined) => {
      if (!culling) return;
      culled.splice(
        culled.findIndex((each) => each.culling === culling),
        1,
      );
      lists.splice(lists.indexOf(culling), 1);
      culling.destroy();
    };

    const casting = (into: Into): passes.Built => {
      const cfg = start.pipe(fills).with(band, cast);
      const culling = cull ? listOf(cfg.pipe(into.inject()), into) : undefined;
      const castOf = byMesh(culling, castKey, (row, look) => {
        const label = labelOf(look, 'cast');
        const primitive = primitiveOf(look);
        // A surface reads the untextured factors here, and cuts its own coverage
        if (surfaced) {
          return into.pipeline(
            {
              label,
              Vertex: shaders.Vertex,
              at: shaders.at,
              pixelOf: surfaced.placing,
              depthOf: surfaced.depthOf,
              invariant: true,
              primitive,
            },
            cfg.with(rowOf, row).with(texturing, false),
          );
        }
        if (masks(look) || shaders.cuts) {
          return into.pipeline(
            {
              label,
              Vertex: shaders.Vertex,
              at: shaders.at,
              cutout: shaders.cutAway,
              invariant: true,
              primitive,
            },
            cfg.with(rowOf, row).with(texturing, false).with(cutting, masks(look)),
          );
        }
        return into.pipeline(
          { label, Vertex: Cast, at: shaders.cast, primitive },
          cfg.with(rowOf, row),
        );
      });
      const drawCast = drawer(castOf, castsMasked, culling);

      return {
        draw: drawCast,
        initAsync: () => compile([castOf], culling),
        destroy: () => dropped(culling),
      };
    };

    const lighting = (into: Into, colored: boolean): passes.Built => {
      // A prepass writes the depth the colour pass shades where it is equal
      const covers = colored && prepass && !translucent;
      if (covers && !into.depthFormat) {
        throw new Error('A renderer with prepass: true draws into passes with depth.');
      }

      const seeing = start.pipe(fills).with(band, shown);
      const culling = cull ? listOf(seeing.pipe(into.inject()), into) : undefined;
      const described = start.pipe(into.inject());
      const lit = (settings.shade ? described.with(shading, settings.shade) : described)
        .pipe(fills)
        .with(blended, translucent)
        .with(clearing, blend === 'clear')
        .with(transmitting, transmit !== undefined)
        .with(transmittance, d.vec3f(...(transmit ?? [1, 1, 1])))
        .with(smoothing, settings.specularAntialiasing ?? true)
        .with(band, shown);
      const corners = {
        Vertex: shaders.Vertex,
        at: shaders.at,
        color: passes.shaded,
        invariant: true,
      };

      const multisampled = into.samples > 1;
      // A multisampled fade hands over samples, which WebGPU won't mix with alpha to coverage
      const handing = multisampled && shown.fades;
      // Alpha to coverage reads the colour target, so outputs alone take their samples by mask
      const coverage = colored
        ? { writeMask: 0, ...(handing ? { sampleMask: shaders.kept } : { alphaToCoverage: true }) }
        : multisampled && { sampleMask: shaders.kept };
      const coverOf =
        (covers || !colored) &&
        byMesh(culling, colorKey, (row, look) =>
          into.pipeline(
            {
              ...corners,
              label: labelOf(look, 'cover'),
              primitive: primitiveOf(look),
              pixelOf: shaders.covered,
              ...coverage,
            },
            lit
              .with(rowOf, row)
              .with(texturing, reads(look))
              .with(twoSided, look.doubleSided)
              .with(cutting, masks(look))
              .with(covering, multisampled)
              .with(masking, handing),
          ),
        );
      const colorOf =
        colored &&
        byMesh(culling, colorKey, (row, look) =>
          into.pipeline(
            {
              ...corners,
              label: labelOf(look),
              primitive: primitiveOf(look),
              pixelOf: surfaced ? surfaced.showing : shaders.lit,
              ...(translucent && (transmit ? { blend: THROUGH } : { translucent: true })),
              ...(shaders.cuts && !translucent && !covers && !handing && { alphaToCoverage: true }),
              ...(handing && !translucent && { sampleMask: shaders.kept }),
              depthStencil: covers
                ? { depthWriteEnabled: false, depthCompare: 'equal' }
                : { depthWriteEnabled: !translucent },
            },
            lit
              .with(rowOf, row)
              .with(texturing, reads(look))
              .with(twoSided, look.doubleSided)
              .with(cutting, masks(look) && !covers)
              .with(covering, multisampled)
              .with(prepassed, covers)
              .with(masking, handing && !translucent),
          ),
        );

      const drawCover = coverOf && drawer(coverOf, bindsTextures, culling);
      const drawColor = colorOf && drawer(colorOf, bindsTextures, culling);

      // Set the blend constant on the parent pass because bundles do not own one
      const constant = transmit && ([...transmit, 1] as const);
      return {
        draw(pass, layer) {
          if (drawCover) drawCover(pass, layer);
          if (constant && pass.resourceType === 'render-pass')
            (pass as TgpuRenderPass).setBlendConstant(constant);
          if (drawColor) drawColor(pass, layer);
        },
        initAsync: () =>
          compile(
            [colorOf, coverOf].filter((each) => each !== false),
            culling,
          ),
        destroy: () => dropped(culling),
      };
    };

    return {
      build(into) {
        const { depthOnly } = into;
        const colored = !depthOnly && Object.values(into.writes).includes(passes.color);
        if (translucent && !colored) {
          throw new Error(
            `A renderer with blend: '${blend}' leaves depth alone; add it to passes with colour.`,
          );
        }
        return depthOnly ? casting(into) : lighting(into, colored);
      },

      cull(pass) {
        if (!cull) throw new Error('This renderer was created without `cull: true`.');
        for (const { culling, into } of culled) {
          if (into.enabled) culling.record(pass, listedRows);
        }
      },

      pipe: (fill) => make(settings, (cfg) => fill(fills(cfg))),
    };
  };

  const destroy = () => {
    rows.destroy();
    listing.destroy();
    replays?.destroy();
    for (const each of lists) each.destroy();
  };

  return {
    kind,
    capacity,

    get count() {
      return count;
    },

    get span() {
      return rows.span;
    },

    buffers: rows.buffers,
    fields,

    add,
    write,

    remove(id) {
      for (const row of [id, ...(together.get(id) ?? [])]) {
        unlist(rows.slotOf(row));
        rows.remove(row);
      }
      together.delete(id);
      count -= 1;
      changed = true;
    },

    clear() {
      rows.clear();
      for (const run of runs) run.length = 0;
      together.clear();
      count = 0;
      changed = true;
    },

    has: rows.has,

    slotOf: rows.slotOf,

    idAt: (slot) => {
      rows.idAt(slot); // throws for a free slot
      return rows.idAt(firstOf[slot]);
    },

    flush,

    gpu: rows.gpu,

    inject: () => rows.inject(),

    live: () => rows.live(),

    renderer: (settings = {}) => make(settings, (cfg) => cfg),

    destroy,
    [Symbol.dispose]: destroy,
  };
}
