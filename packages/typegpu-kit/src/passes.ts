/**
 * Describes the targets and outputs for passes that draw a scene. A renderer supplies vertex data,
 * data at each pixel and a draw call. The description builds its pipelines to write colour, depth
 * and outputs such as motion vectors. `pipe` supplies fills shared by those pipelines, such as a
 * camera. Passes can draw into the canvas alone, a set of targets or a depth-only shadow map.
 *
 * @module
 * @injects member what was piped into it, such as the camera, for a drawing's kernels per
 * description
 * @binds 0 storage: what is piped in binds what its own `@binds` say
 * @binds 1 uniform: `passes.layer`, the layer drawn, in a pipeline of a description drawn into
 * layers that reads it
 * @binds 1 uniform: `passes.viewport`, the drawn region, in a pipeline that reads it
 * @owns the pipelines `add` builds, until the drawing or description is destroyed, a pipeline per
 * description that composites, bind groups, a buffer per layer, a buffer per viewport its
 * pipelines read, bindings when given parts as `shared`: WebGPU pipelines and bind groups need no
 * destruction.
 * Compound drawings free the resources they built for the description. `composite` lays
 * translucent drawings over the colour with that pipeline and a bind group per pair of textures.
 * A description drawn into layers holds each layer's index in a uniform with a bind group
 * @records `begin` from the frame, with the `shared` group set, then each drawing's `draw` from
 * `add` into the pass; into targets with layers, `record` does so once per layer. `composite`
 * records a pass of its own.
 */

import {
  common as shells,
  d,
  std,
  tgpu,
  type IndexFlag,
  type SampledFlag,
  type TgpuBindGroup,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuFragmentFn,
  type TgpuRenderCommands,
  type TgpuRenderPass,
  type TgpuRenderPipeline,
  type TgpuRoot,
  type TgpuTexture,
  type TgpuVertexFn,
  type UniformFlag,
  type WithBinding,
} from 'typegpu';
import { share, type AnyPart, type Shared } from './bindings.ts';
import { encoderOf, type Frame } from './frames.ts';
import { epoch } from './history.ts';
import { unfilled, type Inject, type Injectable } from './inject.ts';
import { memo } from './memo.ts';
import { freeEach } from './owned.ts';
import * as pipelines from './pipelines.ts';
import { opening, untimedFrame, type Timed, type TimestampWrites } from './profiler.ts';
import type { Ordered } from './workgroup.ts';
import type { SizeSource, Spec, Surface, Targets } from './targets.ts';

type Schema = d.AnyWgslData;
type Struct = d.WgslStruct;
/** A scalar or vector schema of numbers, which an output holds. */
type Held = Ordered;
type Carried<C> = C extends Struct ? d.Infer<C> : undefined;

/**
 * Defines how a pass writes one target, as returned by `output`. Give it to a target in
 * `passes.create` or to a drawing as its `color`. It has no members to read.
 */
export interface Output<in P = unknown, in X = unknown> {
  /**
   * Carries the pixel and vertex types that the output's `write` and `vary` read. Used only for
   * type checking.
   */
  readonly '~reads'?: (pixel: P, vertex: X) => void;
}

/** What `output` keeps of its spec for the stages a description builds. */
interface Written extends Output<never, never> {
  readonly schema: Held;
  /** Sets the value where no drawing covers the target. */
  readonly clear: GPUColor;
  /** Writes `clear` when a renderer leaves this output out, as if nothing drew there. */
  readonly unwritten: () => unknown;
  readonly varyings: Struct | undefined;
  /** Reports whether `vary` receives the previous frame's vertex. */
  readonly previous: boolean;
  readonly vary: ((current: never, previous: never) => unknown) | undefined;
  readonly write: (pixel: never, carried: never) => unknown;
}

const recordOf = (out: Output<never, never>) => out as Written;

const clearOf = (value: d.InferGPU<Held> | undefined): GPUColor => {
  if (value === undefined) return [0, 0, 0, 0];
  if (typeof value === 'number') return [value, 0, 0, 0];
  return [value[0], value[1], value[2] ?? 0, value[3] ?? 0];
};

function unwrittenOf<V extends Held>(schema: V, clear: d.InferGPU<V> | undefined) {
  const make = schema as unknown as (value?: d.InferGPU<V>) => d.InferGPU<V>;
  const unwritten = () => {
    'use gpu';
    if (clear === undefined) return make();
    return make(clear);
  };
  return unwritten;
}

/**
 * Defines how a pass writes one target from renderer data. `write` takes the renderer's data at
 * a pixel, and its parameter type declares the fields it needs. For vertex data, supply `varyings` and
 * `vary`. `vary` runs per vertex, and its interpolated result reaches `write`. With `previous:
 * true`, `vary` also receives the previous frame's vertex. Throws if `vary` has no `varyings`
 * schema, `previous` has no `vary`, or `vary` takes two parameters without `previous: true`.
 *
 * @param schema what the target holds per pixel, such as `d.vec2f` for an `rg16float` target
 * @param spec `write`, which gives the value at a pixel, and what it clears to and carries from the
 * vertex
 * @returns an output to give a target in `passes.create`
 */
export function output<
  V extends Held,
  P = unknown,
  X = unknown,
  C extends Struct | undefined = undefined,
>(
  schema: V,
  spec: {
    /**
     * Sets the value where nothing drew, as a value of `schema`. A drawing's `color` writes into
     * the description's `color` target, which uses the description's `clear` instead. Default
     * zero.
     */
    readonly clear?: d.InferGPU<V>;
    readonly varyings?: C;
    readonly vary?: (current: X, previous: X) => Carried<C>;
    /**
     * Passes the previous frame's vertex as the second argument to `vary`. Without it, the second
     * argument is zero. This must be explicit because `tgpu.fn` and `pipe` hide the parameters of
     * `vary`. Default `false`.
     */
    readonly previous?: boolean;
    readonly write: (pixel: P, carried: Carried<C>) => d.Infer<V>;
  },
): Output<P, X> {
  if (spec.vary && !spec.varyings) {
    throw new Error('passes.output requires the `varyings` struct returned by `vary`.');
  }
  if (spec.previous && !spec.vary) {
    throw new Error('passes.output requires `vary` when `previous` is true.');
  }
  if (!spec.previous && (spec.vary?.length ?? 0) >= 2) {
    throw new Error(
      "passes.output: this `vary` takes a second parameter; say `previous: true` to give it last frame's vertex.",
    );
  }

  const made: Written & Output<P, X> = {
    schema,
    clear: clearOf(spec.clear),
    unwritten: unwrittenOf(schema, spec.clear),
    varyings: spec.varyings,
    previous: spec.previous ?? false,
    vary: spec.vary,
    write: spec.write,
  };
  return made;
}

/** Writes the pixel's `color` unchanged as a renderer's `color` output. */
export const shaded: Output<{ readonly color: d.v4f }> = /* @__PURE__ */ output(d.vec4f, {
  write: (pixel: { readonly color: d.v4f }) => {
    'use gpu';
    return d.vec4f(pixel.color);
  },
});

const Behind = d.struct({ depth: d.f32 });

/**
 * Marks the target that translucent drawings add their colour into, weighted by their distance
 * from the eye, so that the nearest layers count most. Use an `rgba16float` target with
 * `sampled: true`, beside targets marked `passes.color` and `passes.revealage`. Clears to zero.
 */
export const accumulation: Output<unknown, { readonly position: d.v4f }> = /* @__PURE__ */ output(
  d.vec4f,
  {
    varyings: Behind,
    vary: (corner: { readonly position: d.v4f }) => {
      'use gpu';
      return Behind({ depth: corner.position.w });
    },
    write: () => {
      'use gpu';
      return d.vec4f();
    },
  },
);

/**
 * Marks the target where translucent drawings multiply the share of what lies behind them that
 * shows through. Use an `r16float` or `r8unorm` target with `sampled: true`, beside targets marked
 * `passes.color` and `passes.accumulation`. Clears to one.
 */
export const revealage: Output = /* @__PURE__ */ output(d.f32, {
  clear: 1,
  write: () => {
    'use gpu';
    return d.f32(1);
  },
});

const is = (write: Write | undefined, marker: Output<never, never>) => write === marker;
const weighted = (write: Write | undefined) => is(write, accumulation) || is(write, revealage);

const ADD: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
const REVEAL: GPUBlendComponent = {
  srcFactor: 'zero',
  dstFactor: 'one-minus-src',
  operation: 'add',
};

/**
 * Weights a translucent pixel by its distance from the eye, so nearer layers dominate the average
 * (McGuire and Bavoil 2013, equation 7).
 */
const weightOf = (alpha: number, depth: number) => {
  'use gpu';
  const near = std.abs(depth) / 5;
  const far = std.abs(depth) / 200;
  const far2 = far * far;
  return alpha * std.clamp(10 / (1e-5 + near * near + far2 * far2 * far2), 0.01, 3000);
};

/**
 * Describes a vertex of the full-screen triangle and its UV coordinates measured from the top
 * left.
 */
export const ScreenVertex = d.struct({ position: d.vec4f, uv: d.vec2f });

/**
 * Provides `Vertex` and `at` for a renderer that covers the screen with one triangle at depth 0.
 * Draw it with `draw(3)`. Use `{ ...passes.fullScreen, pixelOf, color, draw }`.
 */
export const fullScreen = {
  Vertex: ScreenVertex,
  at: (vertexIndex: number, _instanceIndex: number) => {
    'use gpu';
    const x = std.select(d.f32(-1), d.f32(3), vertexIndex === 1);
    const y = std.select(d.f32(-1), d.f32(3), vertexIndex === 2);
    return ScreenVertex({
      position: d.vec4f(x, y, 0, 1),
      uv: d.vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5),
    });
  },
};

/**
 * Holds the layer that a pass of a layered description draws into, from 0, in every pipeline the
 * description builds. Reads 0 in pipelines of other descriptions. `cameras.layered` reads it to
 * pick the layer's view, and per-layer data of your own can too.
 */
export const layer = tgpu.accessor(d.u32, 0);

/** This schema holds the size in pixels of the region a pass draws and its pixels per point. */
export const Region = d.struct({
  /** Holds the width and height of the drawn region in pixels. */
  size: d.vec2f,
  /**
   * Holds pixels per point: the surface's `pixelRatio`, scaled with a target's resolution, so a
   * one-point line keeps its size in a low-resolution target. 1 for targets that follow a size
   * without a ratio, such as a shadow map.
   */
  pixelRatio: d.f32,
});

/**
 * Holds the region the pass draws into, in every pipeline and kernel a description builds. Each
 * description writes it as a pass begins, when the region has changed. Drawings that size things
 * in pixels or points read it, such as lines and screen text.
 */
export const viewport = tgpu.accessor(
  Region,
  unfilled(
    'the `passes.viewport`',
    'scene',
    'the `passes.create` description the pipeline draws into',
  ),
);

/**
 * Holds the samples per pixel of the description that builds a pipeline, 1 outside descriptions. A
 * drawing that turns coverage into samples with `alphaToCoverage` can cut its edges itself where
 * there is one sample.
 */
export const sampleCount = tgpu.slot(1);

/**
 * Holds whether the description that builds a pipeline draws into depth alone, such as a shadow
 * map, false outside descriptions. A drawing can leave out there what shows only in colour, such
 * as an antialiased fringe.
 */
export const depthOnly = tgpu.slot(false);

/**
 * Holds how many layers the description of a pipeline draws, 1 outside layered descriptions.
 * `cameras.layered` checks its count against it.
 *
 * @hidden
 */
export const layerCount = tgpu.slot(1);

const Layering = tgpu.bindGroupLayout({ layer: { uniform: d.u32 } }).$name('layering');
const layered = tgpu
  .fn(
    [],
    d.u32,
  )(() => Layering.$.layer)
  .$name('layered');

/** Marks the target that receives each renderer's colour. */
export const color: unique symbol = Symbol('passes.color');
/** Marks the depth target. */
export const depth: unique symbol = Symbol('passes.depth');

export type Write = typeof color | typeof depth | Output<never, never>;
export type Writes = { readonly [key: string]: Write | undefined };

type OutputKeys<W> = {
  [K in keyof W]: W[K] extends Output<never, never> ? K : never;
}[keyof W] &
  string;

type Lacks<O, P, X, K> =
  O extends Output<infer NP, infer NX>
    ? [P] extends [NP]
      ? [X] extends [NX]
        ? never
        : K
      : K
    : never;
type Needs<O> =
  O extends Output<infer NP, infer NX> ? { readonly pixel: NP; readonly vertex: NX } : never;
/** The targets of `W` whose outputs read a field that pixel `P` or vertex `X` lacks. */
type Unfed<W, P, X> = { [K in keyof W]-?: Lacks<W[K], P, X, K> }[keyof W] & string;

/**
 * Checks a drawing with pixel `P` and vertex `X` against a description's outputs. Adds nothing
 * when it can write them all. Otherwise it requires a property named for each target the drawing
 * cannot write, so the error names the target and the pixel and vertex fields its output reads.
 * A description typed without its targets checks nothing.
 */
export type Fits<W, P, X> = string extends keyof W
  ? unknown
  : [Unfed<W, P, X>] extends [never]
    ? unknown
    : { readonly [K in Unfed<W, P, X> as `target '${K}' needs`]: Needs<W[K]> };

type VertexStruct = d.WgslStruct<{ position: d.Vec4f }>;

/** Describes one renderer pipeline whose stages a pass description builds. */
export interface Shaders<V extends VertexStruct = VertexStruct, P = never> {
  /** Labels the pipelines in GPU captures. Default none. */
  readonly label?: string;
  /** Defines a vertex with its `position` in clip space and data carried to pixels. */
  readonly Vertex: V;
  /** Computes the vertex for a vertex and instance index, as a `'use gpu'` function. */
  readonly at: (vertexIndex: number, instanceIndex: number) => d.Infer<V>;
  /**
   * Computes data at a pixel from the interpolated vertex. `position` is the pixel position.
   * Required for descriptions that write colour or outputs, and for `depthOf` and `sampleMask`.
   */
  readonly pixelOf?: (vertex: d.Infer<V>) => P;
  /**
   * Supplies the output for the target marked `passes.color`. Required when the description has
   * one.
   */
  readonly color?: Output<P, d.Infer<V>>;
  /** Sets blending for the `color` target. Default none. */
  readonly blend?: GPUBlendState;
  /**
   * Draws the `color` output, premultiplied by its alpha, as a translucent layer. With targets
   * marked `passes.accumulation` and `passes.revealage`, it adds into them in any order and
   * `composite` lays the result over the colour target. Elsewhere it blends over the colour target
   * in draw order. Tests depth without writing it, so draw it after the opaque drawings. Takes no
   * `blend`, and a description of depth alone refuses it. Default false.
   */
  readonly translucent?: boolean;
  /**
   * Selects the channels written to the `color` target. With `0`, writes no output target, for a
   * pass of depth alone. The `color` output still runs for alpha to coverage. Default every channel.
   */
  readonly writeMask?: GPUColorWriteFlags;
  /**
   * Controls depth testing and writing when the description has a depth target. Default `{
   * depthWriteEnabled: true }` with the description's `depthCompare` and bias.
   */
  readonly depthStencil?: Omit<GPUDepthStencilState, 'format'>;
  /**
   * Computes pixel depth from the pixel's data instead of using rasterised depth, such as the depth of
   * a point shown by an impostor. The description adds its depth bias. A description without a
   * depth target leaves it out. A depth-only drawing without this function or `cutout` has no
   * fragment stage.
   */
  readonly depthOf?: (pixel: P) => number;
  /**
   * Returns true where the pixel is cut out of the shape, such as where a sprite's texture is
   * clear. Every description discards those pixels before `pixelOf`. Into depth alone, such as
   * a shadow map, it runs without `pixelOf` unless `depthOf` needs the pixel, so a cut-out
   * shape casts its shadow without the fills its shading reads. Default none.
   */
  readonly cutout?: (vertex: d.Infer<V>) => boolean;
  /**
   * Computes the same vertex `position` in every pipeline built from this vertex. Allows later
   * passes to test earlier depth with `equal`. Default false.
   */
  readonly invariant?: boolean;
  /**
   * Carries these vertex fields to pixels without interpolation. All vertices of a primitive use
   * the same value. Integer fields always use flat interpolation, and a field whose schema sets
   * `d.interpolate` keeps it. Default none.
   */
  readonly flat?: readonly (Exclude<keyof d.Infer<V>, 'position'> & string)[];
  /**
   * Interpolates these vertex fields linearly in screen space, without perspective correction,
   * such as a distance measured across a shape extruded on screen. Default none.
   */
  readonly linear?: readonly (Exclude<keyof d.Infer<V>, 'position'> & string)[];
  /**
   * Sets the primitive state, such as the topology and culling. Default a triangle list without
   * culling.
   */
  readonly primitive?: GPUPrimitiveState;
  /**
   * Uses the alpha of the `color` output to select the covered samples in a multisampled target.
   * Default false.
   */
  readonly alphaToCoverage?: boolean;
  /**
   * Returns the samples a pixel writes in a multisampled target, bit `i` for sample `i`, such as
   * the share of a pixel one drawing hands to another. Requires `pixelOf`. A drawing takes this
   * or `alphaToCoverage`, not both, so fold its coverage into the mask. Default every sample.
   */
  readonly sampleMask?: (vertex: d.Infer<V>, pixel: P) => number;
}

/**
 * Describes one pipeline and draw call for `scene.add(system.drawing)`. Holds no pipeline. Each
 * description it is added to builds a pipeline for its own targets, jitter, sample count and
 * fills.
 */
export interface Drawing<V extends VertexStruct = VertexStruct, P = never> extends Shaders<V, P> {
  /**
   * Fills shader accessors over the fills given to the description's `beneath` and under those
   * piped into it, which take precedence.
   */
  inject?(): Inject;
  /** Supplies an index buffer bound to each pipeline built for this drawing. */
  readonly indices?: TgpuBuffer<d.BaseData> & IndexFlag;
  /**
   * Records a draw into the pass, on which the description has set the pipeline it built. That
   * pipeline binds `indices` if supplied. `drawIndexed` throws without an index buffer. A group
   * the drawing sets with `pass.setBindGroup` stays for later drawings in the pass that read its
   * layout, so give such a layout to one drawing.
   */
  draw(pass: TgpuRenderCommands): void;
  /**
   * Compiles resources besides the render pipeline, such as kernels. The description's `initAsync`
   * awaits it.
   */
  initAsync?(): Promise<void>;
}

/**
 * Builds several pipelines or resources for each description, such as a list of visible objects.
 * Each description calls `build` with itself. The drawing creates its pipelines through
 * `into.pipeline`.
 */
export interface Compound<P = never, X = never> {
  /**
   * Carries the pixel and vertex types that description outputs read. Used only for type
   * checking.
   */
  readonly '~fed'?: (pixel: P, vertex: X) => void;
  /**
   * Builds the pipelines and resources for one description, which calls it when the drawing is
   * added.
   */
  build(into: Into): Built;
}

export type Drawable<V extends VertexStruct = VertexStruct, P = never> =
  | Drawing<V, P>
  | Compound<P, d.Infer<V>>;

/** Accepts any drawing, for a function that takes one without naming its vertex and pixel. */
// oxlint-disable-next-line typescript/no-explicit-any -- the vertex and pixel types vary both ways
export type AnyDrawing = Drawable<any, any>;

/**
 * Returns `spec` unchanged, inferring its vertex and pixel types, so a drawing written as a value
 * needs no annotations on `at`, `pixelOf` and `draw`.
 *
 * @param spec the drawing, as `add` takes it
 * @returns the same drawing, typed by its vertex and pixel
 */
export const drawing = <V extends VertexStruct, P>(spec: Drawing<V, P>): Drawing<V, P> => spec;

/** Provides a drawing built for one description. */
export interface Drawn extends Disposable {
  /** Records the drawing into a pass returned by the description's `begin`. */
  draw(pass: TgpuRenderCommands): void;
  /** Compiles the drawing's pipelines. The description's `initAsync` also compiles them. */
  initAsync(): Promise<void>;
  /**
   * Removes the drawing from the description and frees resources built for it. The description
   * stops compiling it.
   */
  destroy(): void;
}

/**
 * Returns the drawing and compilation methods built by a compound drawing, with a method to free
 * its resources, such as a list of visible objects.
 */
export interface Built {
  /**
   * Records the drawing into a pass. `layer` is the layer the pass draws into, 0 without layers,
   * for per-layer work such as a visible list for each.
   */
  draw(pass: TgpuRenderCommands, layer: number): void;
  initAsync(): Promise<void>;
  destroy?(): void;
}

type Fragment = TgpuFragmentFn<Record<string, d.BaseData>, Record<string, d.AnyWgslData>>;

/** The stages and state of one pipeline, as `createRenderPipeline` takes them. */
interface Stages {
  readonly vertex: TgpuVertexFn;
  readonly fragment: Fragment;
  /**
   * Lists pipeline targets with the `color` target first, then the remaining targets in
   * description order.
   */
  readonly targets: Record<string, GPUColorTargetState>;
  /** The drawing's `depthStencil` state, or undefined if the description has no depth target. */
  readonly depthStencil: GPUDepthStencilState | undefined;
  readonly multisample: GPUMultisampleState;
  readonly primitive: GPUPrimitiveState | undefined;
}

type Depthless = Omit<Stages, 'fragment' | 'targets'> & {
  readonly fragment?: undefined;
  readonly targets?: undefined;
};

/**
 * What a description promises a drawing: its targets and outputs, the fills its pipelines take,
 * and the methods that build pipelines for it. A compound drawing's `build` receives one.
 */
export interface Into<W extends Writes = Writes> extends Injectable {
  /** Describes what renderers write into each target. */
  readonly writes: W;
  /** Reports the depth target format, or undefined without a depth target. */
  readonly depthFormat: GPUTextureFormat | undefined;
  /** Reports whether the description writes only depth, as a shadow map does. */
  readonly depthOnly: boolean;
  /** Reports the depth comparison drawings use unless their `depthStencil` overrides it. */
  readonly depthCompare: GPUCompareFunction;
  /** Reports the number of samples per pixel for every target. */
  readonly samples: number;
  /**
   * Reports whether this description draws in the current frame, from `enabled`. Compound drawings
   * skip work such as culling when false.
   */
  readonly enabled: boolean;
  /**
   * Reports whether layer `layer` draws in the current frame, from `active`. Compound drawings
   * prepare only the layers that draw, as an instances renderer culls only those.
   */
  active(layer: number): boolean;
  /**
   * Reports how many layers `record` draws, one pass each: the `layers` of the targets it draws
   * into, or 1.
   * Compound drawings keep per-layer work, such as a visible list for each layer, when they build,
   * and their `draw` receives the layer of the pass.
   */
  readonly layers: number;
  /**
   * Fills shader accessors for kernels built by a drawing for this description: the fills given to
   * `beneath`, then the piped fills such as the camera.
   */
  inject(): Inject;
  /**
   * Builds a pipeline for `drawing` that writes every described output and returns its draw
   * methods. If the drawing cannot supply the fields an output reads, type checking fails with
   * that target's name and those fields. Use `without` to leave it out or `instead` to replace
   * it.
   */
  add<V extends VertexStruct, P>(drawing: Drawing<V, P> & Fits<W, P, d.Infer<V>>): Drawn;
  add<P, X>(drawing: Compound<P, X> & Fits<W, P, X>): Drawn;
  /**
   * Builds a pipeline from `drawing` using `from` and the description's piped fills. `from`
   * defaults to the root with the fills given to `beneath`. Compound drawings use this method, and
   * their `initAsync` compiles the resulting pipelines.
   */
  pipeline<V extends VertexStruct, P>(
    drawing: Shaders<V, P>,
    from?: WithBinding,
  ): TgpuRenderPipeline;
  /**
   * Returns a description with `keys` left out for a renderer that cannot write them. Opaque
   * drawings write each omitted output's `clear`. Drawings with a `blend` preserve the value
   * underneath. No varyings are carried for omitted outputs.
   */
  without<K extends OutputKeys<W>>(...keys: K[]): Into<Omit<W, K>>;
  /** Returns a description with the given outputs replaced by target key. */
  instead<const I extends { readonly [K in OutputKeys<W>]?: Output<never, never> }>(
    outputs: I,
  ): Into<Omit<W, keyof I> & I>;
}

/**
 * Describes the targets and outputs of passes that draw a scene, as `passes.create` returns it.
 * Drawings are added to it, and it begins the passes that draw them.
 */
export interface Description<W extends Writes = Writes> extends Into<W>, Disposable {
  /**
   * Adds `fill` after existing fills and returns a description of the same passes. Piped fills
   * take precedence over a drawing's own. For example, `passes.create(…).pipe(view.inject())` uses
   * the camera from `view`.
   */
  pipe(fill: Inject): Description<W>;
  /**
   * Adds `fill` after the fills given to earlier `beneath` calls and returns a description of the
   * same passes. A drawing's own fills take precedence over these, and piped fills over both, so
   * `beneath` supplies defaults such as a lighting. A compound drawing that passes its own `from`
   * to `pipeline` gets them by starting `from` with `inject()`.
   */
  beneath(fill: Inject): Description<W>;
  /**
   * Compiles every pipeline built for this description and the resources required by each added
   * drawing, including drawings added while it compiles.
   */
  initAsync(): Promise<void>;
  /**
   * Destroys drawings added to this description or descriptions derived with `pipe`, `without` and
   * `instead`. Frees resources that compound drawings built for them, such as lists of visible
   * objects. Keeps the caller's targets alive. When a drawing throws, destroys the rest, then
   * throws its error, or an `AggregateError` of all of them with the first one's message.
   */
  destroy(): void;
  /**
   * Begins a render pass on the targets using the frame encoder. Clears each target to its
   * output's `clear`. With `clear: false`, preserves target contents. With a list, clears those
   * targets and preserves the rest. Throws when preserving a transient target, because each pass
   * discards its contents. Sets a viewport for targets drawn below full resolution, and the
   * `shared` group when the description has one. Descriptions without a surface also accept an
   * encoder directly.
   */
  begin(frame: Frame | TgpuCommandEncoder, options?: BeginOptions<W>): TgpuRenderPass;
  /**
   * Begins a pass, draws `drawn` or each drawing in a list in order, then ends the pass. Reuse the
   * list to avoid allocating each frame. Call `begin` yourself when you need to choose drawings
   * per frame. Records nothing while `enabled` is false.
   */
  record(
    frame: Frame | TgpuCommandEncoder,
    drawn: Drawn | readonly Drawn[],
    options?: BeginOptions<W>,
  ): void;
  /**
   * Lays what translucent drawings added to the targets marked `passes.accumulation` and
   * `passes.revealage` over the colour target, or over its resolve destination, in a pass of its
   * own. Call it after the pass that drew them and before anything reads the colour. Records
   * nothing while `enabled` is false. Throws without those targets.
   */
  composite(frame: Frame | TgpuCommandEncoder, options?: Timed): void;
}

/** Configures the pass started by `begin` or `record`. */
export interface BeginOptions<W extends Writes = Writes> extends Timed {
  /**
   * Selects targets to clear. Use true for all, false to preserve all, or a list of target keys to
   * clear. Default `true`.
   */
  readonly clear?: boolean | readonly (keyof W & string)[];
  /**
   * Draws into this layer of a description drawn into layers. `begin` requires it there unless
   * there is one layer, and `record` without it records a pass for every layer. Default none.
   */
  readonly layer?: number;
}

const BEGIN: BeginOptions = Object.freeze({});

const INTEGERS = new Set(['u32', 'i32', 'vec2u', 'vec3u', 'vec4u', 'vec2i', 'vec3i', 'vec4i']);
type Interpolation = 'flat' | 'linear' | undefined;
type Interpolations = (key: string) => Interpolation;

/** Integer values require flat interpolation between shader stages. A schema's own stays. */
const io = (schema: Schema, how: Interpolation = undefined) => {
  const decorated = d.isDecorated(schema);
  if (decorated && schema.attribs.some(d.isInterpolateAttrib)) return schema;
  return how === 'flat' || INTEGERS.has(decorated ? schema.inner.type : schema.type)
    ? d.interpolate('flat', schema as d.U32)
    : how === 'linear'
      ? d.interpolate('linear', schema as d.F32)
      : schema;
};

const fieldsOf = (schema: Struct) => schema.propTypes as Record<string, Schema>;

// `add` checks each output against the drawing through `Fits`. The stages use `as never` because
// the stored outputs have different pixel and varying types
type Outputs = Readonly<Record<string, Written>>;

const DEPTH = 'fragDepth';
const MASK = 'sampleMask';

/** What a drawing's stages pass between them, built once for all its shaders. */
type Plan = ReturnType<typeof planOf>;

function planOf(Vertex: Struct, outputs: Outputs, how: Interpolations, fragmented: boolean) {
  const vertex = fieldsOf(Vertex);
  for (const field of [DEPTH, MASK]) {
    if (Object.hasOwn(vertex, field)) {
      throw new Error(`passes reserves vertex field '${field}'. Rename the renderer's field.`);
    }
  }
  const own = fragmented ? Object.keys(vertex).filter((key) => key !== 'position') : [];
  const between = Object.fromEntries(own.map((key) => [key, io(vertex[key], how(key))]));
  const carried = Object.entries(outputs).map(([key, out]) => {
    const varyings = out.varyings ? fieldsOf(out.varyings) : {};
    // A varying goes by its output and position, numbered on past a vertex field of that name
    const fields = Object.keys(varyings).map((name, at) => {
      let field = `${key}_${at}`;
      for (let n = 1; Object.hasOwn(vertex, field) || Object.hasOwn(between, field); n++) {
        field = `${key}_${at}_${n}`;
      }
      between[field] = io(varyings[name]);
      return { name, field };
    });
    return { key, out, fields };
  });
  return { own, carried, between };
}

type Position = (position: d.v4f) => d.v4f;

// Pipeline construction determines field names at runtime, so entry functions read and write
// fields by key
type Shell<F, In> = (options: {
  readonly in: Record<string, unknown>;
  readonly out: Record<string, unknown> | d.Void;
}) => (body: (input: In, Out: () => Record<string, unknown>) => unknown) => F;
const vertexShell = tgpu.vertexFn as unknown as Shell<
  TgpuVertexFn,
  { readonly vertexIndex: number; readonly instanceIndex: number }
>;
const fragmentShell = tgpu.fragmentFn as unknown as Shell<
  Fragment,
  Readonly<Record<string, unknown> & { position: d.v4f }>
>;

function vertexOf<V extends VertexStruct>(
  Vertex: V,
  at: (vertexIndex: number, instanceIndex: number) => d.Infer<V>,
  { own, carried, between }: Plan,
  jitter: Position | undefined,
  invariant: boolean,
): TgpuVertexFn {
  const varied = carried.filter(({ out }) => out.vary);
  const current = tgpu.fn(at);
  const previous = current.with(epoch, 'previous');
  const before = varied.some(({ out }) => out.previous);

  return vertexShell({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      position: invariant ? d.invariant(d.builtin.position) : d.builtin.position,
      ...between,
    },
  })((input, Out) => {
    'use gpu';
    const here = current(input.vertexIndex, input.instanceIndex);
    let past = Vertex();
    if (before) past = previous(input.vertexIndex, input.instanceIndex);

    const out = Out();
    // Only the rasteriser sees jitter. Outputs need the original vertex position
    out.position = jitter === undefined ? d.vec4f(here.position) : jitter(here.position);
    for (const key of tgpu.unroll(own)) out[key] = std.copy((here as Record<string, never>)[key]);
    for (const entry of tgpu.unroll(varied)) {
      const value = entry.out.vary!(here as never, past as never) as Record<string, unknown>;
      for (const each of tgpu.unroll(entry.fields)) {
        out[each.field] = std.copy(value[each.name]);
      }
    }
    return out;
  }).$name('vertex');
}

/** Fragment-written depth needs the same bias that the rasteriser applies to rasterised depth. */
interface Bias {
  readonly constant: number;
  readonly slope: number;
}

/** Target keys of a drawing that adds into the translucency targets. */
interface Translucency {
  readonly color: string;
  readonly accumulation: string;
  readonly revealage: string;
}

function fragmentOf<V extends VertexStruct, P>(
  Vertex: V,
  pixelOf: (vertex: d.Infer<V>) => P,
  plan: Plan,
  depthOf: ((pixel: P) => number) | undefined,
  bias: Bias,
  translucency: Translucency | undefined,
  cutout: ((vertex: d.Infer<V>) => boolean) | undefined,
  sampleMask: ((vertex: d.Infer<V>, pixel: P) => number) | undefined,
): Fragment {
  const translucent = translucency !== undefined;
  const all = plan.carried.filter(({ out }) => !translucent || !weighted(out));
  const shown = translucency?.color ?? '';
  const summed = translucency?.accumulation ?? '';
  const revealed = translucency?.revealage ?? '';
  const behind = plan.carried.find(({ key }) => key === summed)?.fields[0].field ?? '';
  const biased = bias.constant !== 0 || bias.slope !== 0;

  return fragmentShell({
    in: { position: d.builtin.position, ...plan.between },
    out: {
      ...Object.fromEntries(plan.carried.map(({ key, out }) => [key, out.schema])),
      ...(depthOf && { [DEPTH]: d.builtin.fragDepth }),
      ...(sampleMask && { [MASK]: d.builtin.sampleMask }),
    },
  })((input, Out) => {
    'use gpu';
    const vertex = Vertex();
    vertex.position = std.copy(input.position);
    for (const key of tgpu.unroll(plan.own)) {
      (vertex as Record<string, unknown>)[key] = std.copy(input[key]);
    }
    if (cutout !== undefined) {
      if (cutout(vertex as d.Infer<V>)) std.discard();
    }
    const pixel = pixelOf(vertex as d.Infer<V>);

    const out = Out();
    for (const entry of tgpu.unroll(all)) {
      // Use an unread placeholder for outputs without varyings
      let carried = entry.out.varyings ? entry.out.varyings() : d.u32(0);
      for (const each of tgpu.unroll(entry.fields)) {
        (carried as Record<string, unknown>)[each.name] = std.copy(input[each.field]);
      }
      out[entry.key] = entry.out.write(pixel as never, carried as never);
    }
    if (translucent) {
      const shade = d.vec4f(out[shown] as d.v4f);
      out[summed] = shade * weightOf(shade.w, input[behind] as number);
      out[revealed] = shade.w;
    }
    if (depthOf !== undefined) {
      // Fragment-written depth bypasses pipeline bias, so add the same bias here
      const deep = depthOf(pixel);
      let biasedDepth = deep;
      if (biased) {
        const slope = std.max(std.abs(std.dpdx(deep)), std.abs(std.dpdy(deep)));
        biasedDepth = std.min(deep + slope * bias.slope + bias.constant, 1);
      }
      out[DEPTH] = biasedDepth;
    }
    if (sampleMask !== undefined) out[MASK] = sampleMask(vertex as d.Infer<V>, pixel);
    return out;
  }).$name('fragment');
}

/** A depth-only fragment stage that only discards what `cutout` cuts out of the shape. */
function cutoutOf<V extends VertexStruct>(
  Vertex: V,
  plan: Plan,
  cutout: (vertex: d.Infer<V>) => boolean,
): Fragment {
  return fragmentShell({
    in: { position: d.builtin.position, ...plan.between },
    out: d.Void,
  })((input) => {
    'use gpu';
    const vertex = Vertex();
    vertex.position = std.copy(input.position);
    for (const key of tgpu.unroll(plan.own)) {
      (vertex as Record<string, unknown>)[key] = std.copy(input[key]);
    }
    if (cutout(vertex as d.Infer<V>)) std.discard();
  }).$name('cutout');
}

const cleared = (out: Written): Written => ({
  ...out,
  varyings: undefined,
  previous: false,
  vary: undefined,
  write: out.unwritten,
});

const biasOf = (format: GPUTextureFormat, constant: number, slope: number): Bias => {
  const step =
    format === 'depth16unorm' ? 2 ** -16 : format.startsWith('depth24') ? 2 ** -24 : 2 ** -23;
  return { constant: constant * step, slope };
};

export interface Options {
  /**
   * Moves clip positions for rasterisation. Outputs read the original vertex positions. Use for a
   * sub-pixel offset in temporal antialiasing. Default none.
   * @gpu
   */
  readonly jitter?: Position;
  /**
   * Sets the `color` target value where nothing drew. The pass reads it at each begin, so writing
   * into it changes the next pass. Default transparent black.
   */
  readonly clear?: readonly number[] | GPUColorDict;
  /**
   * Adds a depth bias to every pipeline, in depth-format steps and per unit of slope, as
   * `depthBias` and `depthBiasSlopeScale`. Use for shadow maps to prevent surfaces from shadowing
   * themselves. Default none.
   */
  readonly bias?: { readonly constant?: number; readonly slope?: number };
  /**
   * Sets the depth clear value. For reversed depth, use `depthClear: 0` and `depthCompare:
   * 'greater'`, with a camera created with `reversed: true`. Default 1.
   */
  readonly depthClear?: number;
  /**
   * Sets the depth comparison. Drawings test depth with it unless their `depthStencil` state
   * overrides it. With `'greater'` or `'greater-equal'`, drawings that place themselves at the far
   * plane, such as `environment.background`, place themselves at depth 0. Default `'less'`.
   */
  readonly depthCompare?: GPUCompareFunction;
  /** Sets the stencil clear value for formats with stencil. Default 0. */
  readonly stencilClear?: number;
  /**
   * Reports whether the description should draw in the current frame, exposed as `enabled`.
   * `record` and `composite` record nothing when false, and compound drawings skip work such as
   * culling. `begin` does not check it. Default always.
   * @cpu
   */
  readonly enabled?: () => boolean;
  /**
   * Reports whether a layer draws in the current frame, exposed as `active`. Compound drawings
   * prepare only the layers that draw, and `record` without a `layer` records only those. A
   * shadow's `prepare` sets it for its own passes. Default every layer.
   * @cpu
   */
  readonly active?: (layer: number) => boolean;
  /** Labels passes in the GPU debug panel and captures. Default none. */
  readonly label?: string;
  /**
   * Shares these parts, by key, in one layout the description owns: every pass `begin` and
   * `record` start sets its group, and every pipeline the description builds reads the parts
   * through it. Takes the result of `bindings.share` instead when another description or a pass of
   * your own reads the same parts. Create the description before any other pipeline that reads
   * the parts. Default none.
   */
  readonly shared?: Shared | Readonly<Record<string, AnyPart>>;
}

export interface TargetOptions<T extends Record<string, Spec>> extends Options {
  /**
   * Maps multisampled targets to their resolve destinations at the end of a pass. A destination
   * can be another target in the set or a surface such as the canvas. Default none.
   */
  readonly resolve?: { readonly [K in keyof T]?: Exclude<keyof T, K> | Surface };
}

interface Attachment {
  readonly key: string;
  readonly write: typeof color | Written;
  readonly format: GPUTextureFormat;
  readonly view: (frame: Frame | undefined, layer: number) => GPUTextureView;
  readonly resolve: ((frame: Frame | undefined) => GPUTextureView) | undefined;
  readonly store: GPUStoreOp;
}

interface Layout {
  readonly root: TgpuRoot;
  readonly writes: Writes;
  /** The colour attachment must come first for alpha to coverage. */
  readonly attachments: readonly Attachment[];
  readonly depth:
    | {
        readonly key: string;
        readonly format: GPUTextureFormat;
        readonly view: (layer: number) => GPUTextureView;
        store: GPUStoreOp;
      }
    | undefined;
  readonly samples: number;
  readonly clear: readonly number[] | GPUColorDict;
  readonly depthClear: number;
  readonly depthCompare: GPUCompareFunction;
  readonly stencilClear: number;
  readonly jitter: Position | undefined;
  readonly bias: { readonly constant: number; readonly slope: number };
  readonly enabled: () => boolean;
  readonly active: (layer: number) => boolean;
  readonly label: string | undefined;
  readonly shared: Shared | undefined;
  /** A group per layer holding its index, and the labels of its passes. */
  readonly layering:
    | {
        readonly groups: readonly TgpuBindGroup<(typeof Layering)['entries']>[];
        readonly labels: readonly (string | undefined)[];
      }
    | undefined;
  /** The layer of each pass begun, shared by the descriptions `pipe`, `without` and `instead` make. */
  readonly begun: WeakMap<object, number>;
  /** Applies these fills after the drawing's own fills. */
  readonly fill: Inject;
  /** Applies these fills before the drawing's own fills. */
  readonly beneath: Inject;
  /** Keeps both the drawn region and full target size when they differ. */
  readonly region:
    | { readonly drawn: SizeSource; readonly full: () => readonly [number, number] }
    | undefined;
  /** A surface view requires a frame when `begin` acquires it. */
  readonly framed: boolean;
  /** Writes the drawn region into the `viewport` fill when it changed. */
  readonly place: () => void;
  /** Reports whether a target set matches its source's size and resolution. */
  readonly fitted?: () => boolean;
  readonly warming: Set<() => Promise<void>>;
  /** The `destroy` of each drawing added, to run when the description is destroyed. */
  readonly held: Set<() => void>;
  /** Lays the translucency targets over the colour target, when the description has them. */
  readonly composite:
    | ((encoder: TgpuCommandEncoder, frame: Frame | undefined, options: Timed) => void)
    | undefined;
}

const drawInto = (begun: TgpuRenderPass, drawn: Drawn | readonly Drawn[]) => {
  if ('draw' in drawn) drawn.draw(begun);
  else for (const each of drawn) each.draw(begun);
  begun.end();
};

const UNLAYERED =
  'passes: a layered description begins one layer at a time, `begin(frame, { layer })`; `record` draws them all.';
const UNFRAMED =
  'passes: a description that draws into a surface begins from the frame, `begin(frame)`.';

function loadOf(
  clearing: NonNullable<BeginOptions['clear']>,
  key: string,
  store: GPUStoreOp,
): GPULoadOp {
  if (clearing === true || (clearing !== false && clearing.includes(key))) return 'clear';
  if (store === 'discard') {
    throw new Error(
      `This pass tries to load '${key}', a transient target whose contents are discarded at the end of each pass. Set \`transient: false\` for this target in \`targets.create\` to keep its contents.`,
    );
  }
  return 'load';
}

function describe(
  layout: Layout,
  left: ReadonlySet<string>,
  replaced: ReadonlyMap<string, Written>,
): Description {
  const { attachments, samples, jitter, root, warming, held } = layout;
  const outputKeys = new Set(
    attachments.filter(({ write }) => write !== color).map(({ key }) => key),
  );
  const depthAlone = attachments.length === 0;
  const bias = layout.depth
    ? biasOf(layout.depth.format, layout.bias.constant, layout.bias.slope)
    : { constant: 0, slope: 0 };

  const checked = (keys: readonly string[], what: string) => {
    for (const key of keys) {
      if (!outputKeys.has(key)) {
        throw new Error(`passes: a renderer ${what} '${key}', which isn't one of its outputs.`);
      }
      if (weighted(layout.writes[key])) {
        throw new Error(`passes: '${key}' holds translucent drawings, so it can't be ${what}.`);
      }
    }
  };

  const keyOf = (write: Write) => attachments.find((each) => each.write === write)?.key;
  const summed = keyOf(accumulation);
  const revealed = keyOf(revealage);
  const shown = keyOf(color);
  const translucency: Translucency | undefined =
    summed && revealed && shown
      ? { color: shown, accumulation: summed, revealage: revealed }
      : undefined;

  const writes = Object.fromEntries(
    Object.entries(layout.writes).flatMap(([key, write]): [string, Write | undefined][] =>
      left.has(key) ? [] : [[key, replaced.get(key) ?? write]],
    ),
  );

  const stagesOf = <V extends VertexStruct, P>(shaders: Shaders<V, P>): Stages | Depthless => {
    const { Vertex, at, pixelOf, color: own, blend, writeMask, depthStencil: state } = shaders;
    const { invariant = false, flat = [], linear = [], primitive, cutout } = shaders;
    const { alphaToCoverage = false, sampleMask } = shaders;
    // Without a depth target there is no depth to write
    const depthOf = layout.depth ? shaders.depthOf : undefined;
    const flatKeys: readonly string[] = flat;
    const linearKeys: readonly string[] = linear;
    const how = (key: string): Interpolation =>
      flatKeys.includes(key) ? 'flat' : linearKeys.includes(key) ? 'linear' : undefined;
    const { translucent = false } = shaders;
    if (translucent && depthAlone) {
      throw new Error(
        'A translucent drawing leaves depth unchanged. Add it to a pass with a colour target.',
      );
    }
    if (translucent && blend) {
      throw new Error('passes: a translucent drawing blends on its own; leave out `blend`.');
    }
    const into = translucent ? translucency : undefined;
    const blending = blend ?? (translucent && !into ? pipelines.blend.over : undefined);
    // Omitted outputs write their clear for opaque drawings and preserve contents for blended
    // drawings
    const masked = (key: string) =>
      writeMask === 0 || (left.has(key) && (blending !== undefined || into !== undefined));
    const fragmented =
      !depthAlone || depthOf !== undefined || cutout !== undefined || sampleMask !== undefined;
    // Into depth alone, a cutout runs without `pixelOf` unless the depth or mask needs it
    const cutAlone =
      depthAlone && depthOf === undefined && sampleMask === undefined && cutout !== undefined;

    if (!depthAlone && !pixelOf) {
      throw new Error('passes requires `pixelOf` for a drawing into colour or output targets.');
    }
    if (shown !== undefined && !own) {
      throw new Error(
        `passes requires \`color\` for a drawing into '${shown}', the target marked \`passes.color\`.`,
      );
    }
    if (shaders.depthOf && !pixelOf)
      throw new Error('passes requires `pixelOf` for a drawing with `depthOf`.');
    if (sampleMask && !pixelOf)
      throw new Error('passes requires `pixelOf` for a drawing with `sampleMask`.');
    if (sampleMask && alphaToCoverage) {
      throw new Error('passes: a drawing takes `sampleMask` or `alphaToCoverage`, not both.');
    }

    const drawn = Object.fromEntries(
      attachments.map(({ key, write }) => {
        if (write === color) return [key, recordOf(own as Output<never, never>)];
        if (weighted(write)) return [key, into ? write : cleared(write)];
        if (masked(key) || left.has(key)) return [key, cleared(write)];
        return [key, replaced.get(key) ?? write];
      }),
    );

    const plan = planOf(Vertex, drawn, how, fragmented);

    const vertex = {
      vertex: vertexOf(Vertex, at, plan, jitter, invariant),
      depthStencil: layout.depth && {
        format: layout.depth.format,
        depthWriteEnabled: true,
        depthCompare: layout.depthCompare,
        ...(layout.bias.constant !== 0 && { depthBias: layout.bias.constant }),
        ...(layout.bias.slope !== 0 && { depthBiasSlopeScale: layout.bias.slope }),
        ...(translucent && { depthWriteEnabled: false }),
        ...state,
      },
      multisample: {
        count: samples,
        ...(alphaToCoverage && samples > 1 && { alphaToCoverageEnabled: true }),
      },
      primitive,
    } satisfies Depthless;
    if (!fragmented) return vertex;

    return {
      ...vertex,
      fragment: cutAlone
        ? cutoutOf(Vertex, plan, cutout)
        : fragmentOf(Vertex, pixelOf!, plan, depthOf, bias, into, cutout, sampleMask),
      targets: Object.fromEntries(
        attachments.map(({ key, write, format }) => [
          key,
          {
            format,
            ...(write === color && blending ? { blend: blending } : {}),
            ...(write === color && writeMask !== undefined ? { writeMask } : {}),
            ...(is(write, accumulation) && into ? { blend: { color: ADD, alpha: ADD } } : {}),
            ...(is(write, revealage) && into ? { blend: { color: REVEAL, alpha: REVEAL } } : {}),
            ...((write === color ? into !== undefined : weighted(write) && !into)
              ? { writeMask: 0 }
              : {}),
            ...(masked(key) ? { writeMask: 0 } : {}),
          },
        ]),
      ),
    };
  };

  const pipeline = <V extends VertexStruct, P>(
    shaders: Shaders<V, P>,
    from: WithBinding = root.pipe(layout.beneath),
  ): TgpuRenderPipeline => {
    const made = from.pipe(layout.fill).createRenderPipeline(stagesOf(shaders));
    return shaders.label === undefined ? made : made.$name(shaders.label);
  };

  const layers = layout.layering?.groups.length ?? 1;

  const open = (
    target: Frame | TgpuCommandEncoder,
    clearing: NonNullable<BeginOptions['clear']>,
    at: number,
    timestampWrites: TimestampWrites | GPURenderPassTimestampWrites | undefined,
  ) => {
    if (!Number.isInteger(at) || at < 0 || at >= layers) {
      throw new RangeError(`passes: the description has ${layers} layers, so it can't draw ${at}.`);
    }
    const frame = 'encoder' in target ? target : undefined;
    if (layout.framed && !frame) throw new Error(UNFRAMED);
    const encoder = encoderOf(target);

    // Attachment order must match the pipeline target order
    const colorAttachments = attachments.map(({ key, write, view, resolve, store }) => ({
      view: view(frame, at),
      ...(resolve && { resolveTarget: resolve(frame) }),
      loadOp: loadOf(clearing, key, store),
      storeOp: store,
      // WebGPU only reads the clear, so the caller's readonly array passes as is
      clearValue: write === color ? (layout.clear as GPUColor) : write.clear,
    }));

    // Acquiring a surface resolve view can commit a resize
    if (layout.fitted?.() === false) throw unfitted();
    const begun = encoder.beginRenderPass({
      label: layout.layering ? layout.layering.labels[at] : layout.label,
      colorAttachments,
      ...(layout.depth && {
        depthStencilAttachment: {
          view: layout.depth.view(at),
          depthLoadOp: loadOf(clearing, layout.depth.key, layout.depth.store),
          depthStoreOp: layout.depth.store,
          depthClearValue: layout.depthClear,
          ...(layout.depth.format.includes('stencil') && {
            stencilLoadOp: loadOf(clearing, layout.depth.key, layout.depth.store),
            stencilStoreOp: layout.depth.store,
            stencilClearValue: layout.stencilClear,
          }),
        },
      }),
      ...(timestampWrites && { timestampWrites }),
    });
    layout.place();
    if (layout.region) {
      const { drawn, full } = layout.region;
      const [width, height] = full();
      if (drawn.width < width || drawn.height < height) {
        begun.setViewport(0, 0, drawn.width, drawn.height, 0, 1);
      }
    }
    if (layout.shared) begun.setBindGroup(layout.shared.layout, layout.shared.group());
    if (layout.layering) {
      begun.setBindGroup(Layering, layout.layering.groups[at]);
      layout.begun.set(begun, at);
    }

    return begun;
  };

  const track = (built: Built): Drawn => {
    const warm = () => built.initAsync();
    warming.add(warm);

    const destroy = () => {
      if (!held.delete(destroy)) return;
      warming.delete(warm);
      built.destroy?.();
    };
    held.add(destroy);

    return {
      draw(into) {
        if (layout.layering && into.resourceType === 'render-bundle-encoder') {
          throw new Error(
            `passes: a render bundle can't draw into a description with ${layers} layers, because each layer's pass binds its index. Draw into the passes that \`record\` or \`begin\` opens.`,
          );
        }
        built.draw(into, layout.begun.get(into) ?? 0);
      },
      initAsync: () => built.initAsync(),
      destroy,
      [Symbol.dispose]: destroy,
    };
  };

  const release = () => freeEach([...held]);

  const pass: Description = {
    writes,
    depthFormat: layout.depth?.format,
    depthOnly: depthAlone,
    depthCompare: layout.depthCompare,
    samples,

    get enabled() {
      return layout.enabled();
    },

    active: (at) => layout.active(at),

    layers,

    inject: () => (cfg) => layout.fill(layout.beneath(cfg)),

    pipe(fill) {
      const before = layout.fill;
      return describe({ ...layout, fill: (cfg) => fill(before(cfg)) }, left, replaced);
    },

    beneath(fill) {
      const before = layout.beneath;
      return describe({ ...layout, beneath: (cfg) => fill(before(cfg)) }, left, replaced);
    },

    add(item: Drawable<VertexStruct, unknown>): Drawn {
      if ('build' in item) return track(item.build(pass));

      const under = root.pipe(layout.beneath);
      const made = pipeline(item, item.inject ? under.pipe(item.inject()) : under);
      const bound = item.indices ? made.withIndexBuffer(item.indices) : made;

      return track({
        draw: (into) => {
          into.setPipeline(bound);
          item.draw(into);
        },
        initAsync: async () => {
          const own = item.initAsync?.bind(item);
          await pipelines.compile(own ? [made, { initAsync: own }] : [made]);
        },
      });
    },

    pipeline,

    without(...keys) {
      checked(keys, 'left out');
      return describe(layout, new Set([...left, ...keys]), replaced);
    },

    instead(outputs) {
      const entries = Object.entries(outputs as Record<string, Written | undefined>);
      checked(
        entries.map(([key]) => key),
        'replaced',
      );
      const next = new Map(replaced);
      for (const [key, out] of entries) if (out) next.set(key, out);
      return describe(layout, left, next);
    },

    async initAsync() {
      // Include drawings added while compilation is pending
      const done = new Set<() => Promise<void>>();
      for (let batch = [...warming]; batch.length > 0;) {
        for (const warm of batch) done.add(warm);
        await pipelines.compile(batch.map((warm) => ({ initAsync: warm })));
        batch = [...warming].filter((warm) => !done.has(warm));
      }
    },

    destroy: release,
    [Symbol.dispose]: release,

    begin(target, options = BEGIN) {
      const { clear = true, timestampWrites, layer: at } = options;
      if (layout.layering && layers > 1 && at === undefined) throw new Error(UNLAYERED);
      return open(target, clear, at ?? 0, timestampWrites);
    },

    composite(target, options = BEGIN) {
      if (!layout.composite) {
        throw new Error(
          'passes: composite needs targets marked `passes.accumulation` and `passes.revealage`.',
        );
      }
      if (!layout.enabled()) return;
      const frame = 'encoder' in target ? target : undefined;
      if (layout.framed && !frame) throw new Error(UNFRAMED);
      layout.composite(encoderOf(target), frame, options);
    },

    record(target, drawn, options = BEGIN) {
      if (!layout.enabled()) return;
      if (options.layer !== undefined) return drawInto(pass.begin(target, options), drawn);

      // One pass per active layer, timed from the start of the first to the end of the last
      const { clear = true, timestampWrites } = options;
      let first = -1;
      let last = -1;
      for (let at = 0; at < layers; at++) {
        if (!layout.active(at)) continue;
        if (first < 0) first = at;
        last = at;
      }
      if (last < 0) {
        if (timestampWrites) untimedFrame(target, timestampWrites);
        return;
      }
      for (let at = first; at <= last; at++) {
        if (!layout.active(at)) continue;
        const timed = timestampWrites && opening(timestampWrites, at === first, at === last);
        drawInto(open(target, clear, at, timed), drawn);
      }
    },
  } as Description;

  return pass;
}

const Layered = tgpu.bindGroupLayout({
  accumulated: { texture: d.texture2d(d.f32) },
  revealed: { texture: d.texture2d(d.f32) },
});

const laid = /* @__PURE__ */ tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  const at = d.vec2i(position.xy);
  const revealed = std.textureLoad(Layered.$.revealed, at, 0).x;
  if (revealed >= 1) std.discard();
  const sum = std.textureLoad(Layered.$.accumulated, at, 0);
  const covered = 1 - revealed;
  return d.vec4f((sum.xyz / std.clamp(sum.w, 1e-4, 5e4)) * covered, covered);
});

interface Compositing {
  readonly accumulation: string;
  readonly revealage: string;
  readonly color: string;
  readonly resolve: { readonly [key: string]: string | Surface | undefined };
  readonly region: Layout['region'];
  readonly label: string | undefined;
  readonly warming: Set<() => Promise<void>>;
}

type Layer = TgpuTexture<{ size: [number, number]; format: GPUTextureFormat }> & SampledFlag;

const unfitted = () =>
  new Error(
    'The targets do not match the size or resolution they follow. Call `fit()` on the set once a frame, before recording into it.',
  );

function compositorOf(root: TgpuRoot, screen: Targets, keys: Compositing): Layout['composite'] {
  const readOf = (key: string) => {
    const into = keys.resolve[key];
    const read = into === undefined ? key : into;
    if (typeof read !== 'string') {
      throw new Error(
        `passes.create: '${key}' resolves into a surface, which composite can't read.`,
      );
    }
    if (!(screen.specs[read] as { sampled?: boolean }).sampled) {
      throw new Error(
        `Set \`sampled: true\` on target '${read}' in \`targets.create\` so \`composite\` can read it.`,
      );
    }
    return read;
  };
  const summed = readOf(keys.accumulation);
  const revealed = readOf(keys.revealage);

  const onto = keys.resolve[keys.color] ?? keys.color;
  if (typeof onto === 'string' && screen.transient[onto]) {
    throw new Error(
      `The \`composite\` pass draws over '${onto}', which is transient. Set \`transient: false\` on that target in \`targets.create\`, or resolve it.`,
    );
  }
  const format = typeof onto === 'string' ? screen.textures[onto].props.format : onto.format;

  const pipeline = root
    .createRenderPipeline({
      vertex: shells.fullScreenTriangle,
      fragment: laid,
      targets: { format, blend: pipelines.blend.over },
    })
    .$name('composite');
  keys.warming.add(async () => {
    await pipeline.initAsync();
  });
  const bound = memo((accumulated: Layer, revealedBy: Layer) =>
    pipeline.with(root.createBindGroup(Layered, { accumulated, revealed: revealedBy })),
  );

  return (encoder, frame, { timestampWrites }) => {
    const view = typeof onto === 'string' ? screen.views[onto] : onto.view(frame!);
    if (!screen.fitted) throw unfitted();
    const pass = encoder.beginRenderPass({
      ...(keys.label !== undefined && { label: `${keys.label} composite` }),
      colorAttachments: [{ view, loadOp: 'load', storeOp: 'store' }],
      ...(timestampWrites && { timestampWrites }),
    });
    if (keys.region) {
      const [width, height] = keys.region.full();
      const { drawn } = keys.region;
      if (drawn.width < width || drawn.height < height) {
        pass.setViewport(0, 0, drawn.width, drawn.height, 0, 1);
      }
    }
    const textures = screen.textures as unknown as Record<string, Layer>;
    pass.setPipeline(bound(textures[summed], textures[revealed]));
    pass.draw(3);
    pass.end();
  };
}

/** Accepts a surface, or any destination with a format, a view and a size in pixels. */
export type Screen = Pick<Surface, 'format' | 'view'> & SizeSource;

const isSurface = (screen: object): screen is Screen => !('textures' in screen);

const sharedOf = (root: TgpuRoot, given: Options['shared']): Shared | undefined =>
  given === undefined || isShared(given) ? given : share(root, given);

const isShared = (given: NonNullable<Options['shared']>): given is Shared =>
  typeof given.group === 'function';

/** Types `writes`, reporting a key that names no target of the set at the key. */
type Named<W, T> = [Exclude<keyof W, keyof T>] extends [never]
  ? W
  : {
      readonly [K in keyof W]: K extends keyof T
        ? W[K]
        : { readonly 'is not a target of this set, which has': keyof T & string };
    };

/**
 * Describes the targets and outputs for scene passes. With a surface such as the canvas, writes
 * only each renderer's colour. With a target set, writes colour into the target marked `color`,
 * depth into the target marked `depth`, and other outputs into their named targets. Multisampled
 * targets resolve according to `resolve`. Targets with `layers` draw one pass per layer, and every
 * target the description draws into needs the same `layers`. `add` builds a renderer's drawing.
 * `begin` starts a pass. `pipe` supplies fills such as a camera.
 *
 * @param screen the surface, or the targets, which give the formats; each one the description
 * names is drawn into
 * @param writes what fills each target, by its key in `screen`
 * @returns a description you destroy, which holds the pipelines it builds
 */
export function create(
  root: TgpuRoot,
  screen: Screen,
  options?: Options,
): Description<{ readonly color: typeof color }>;
export function create<
  T extends Record<string, Spec>,
  const W extends { readonly [K in keyof T]?: Write },
>(
  root: TgpuRoot,
  screen: Targets<T>,
  writes: Named<W, T>,
  options?: TargetOptions<T>,
): Description<W>;
export function create(
  root: TgpuRoot,
  screen: Screen | Targets,
  ...rest: readonly unknown[]
): Description {
  const common = (
    options: Options,
    measure: (into: Float32Array) => void,
    drawn: {
      readonly samples: number;
      readonly colored: boolean;
      readonly layers?: number | undefined;
    },
  ) => {
    const { samples, colored, layers } = drawn;
    const { jitter, clear = [0, 0, 0, 0], bias = {}, enabled = () => true } = options;
    const { active = () => true } = options;
    const { label } = options;
    const shared = sharedOf(root, options.shared);
    const { depthClear = 1, depthCompare = 'less', stencilClear = 0 } = options;
    const held = new Set<() => void>();

    // Made when a shader first reads `viewport`, so descriptions whose drawings don't own nothing
    let region: (TgpuBuffer<typeof Region> & UniformFlag) | undefined;
    const measured = new Float32Array(4);
    const written = new Float32Array(4);
    const write = (into: TgpuBuffer<typeof Region>) => {
      written.set(measured);
      into.write(written.buffer);
    };
    const place = () => {
      if (!region) return;
      measure(measured);
      if (measured[0] !== written[0] || measured[1] !== written[1] || measured[2] !== written[2]) {
        write(region);
      }
    };
    const regionOf = () => {
      if (!region) {
        const made = root.createBuffer(Region).$usage('uniform').$name('viewport');
        // A pipeline built inside a pass reads the region before the next `begin` places it
        measure(measured);
        write(made);
        region = made;
      }
      return region.as('uniform');
    };
    const placed: Inject = (cfg) =>
      cfg.with(viewport, regionOf).with(sampleCount, samples).with(depthOnly, !colored);

    let layering: Layout['layering'];
    if (layers !== undefined && layers > 1) {
      const indices = Array.from({ length: layers }, (_, at) =>
        root.createBuffer(d.u32, at).$usage('uniform'),
      );
      layering = {
        groups: indices.map((index) => root.createBindGroup(Layering, { layer: index })),
        labels: indices.map((_, at) => (label === undefined ? undefined : `${label} ${at}`)),
      };
    }
    const sharing = shared ? shared.inject() : (cfg: Parameters<Inject>[0]) => cfg;

    return {
      root,
      jitter,
      clear,
      bias: { constant: bias.constant ?? 0, slope: bias.slope ?? 0 },
      depthClear,
      depthCompare,
      stencilClear,
      enabled,
      active,
      label,
      shared,
      layering,
      begun: new WeakMap(),
      fill: layering
        ? (cfg: Parameters<Inject>[0]) =>
            placed(sharing(cfg)).with(layer, layered).with(layerCount, layering.groups.length)
        : (cfg: Parameters<Inject>[0]) => placed(sharing(cfg)),
      place,
      beneath: (cfg: Parameters<Inject>[0]) => cfg,
      warming: new Set<() => Promise<void>>(),
      held,
    };
  };

  if (isSurface(screen)) {
    const options = (rest[0] ?? {}) as Options;
    const view = (frame: Frame | undefined) => screen.view(frame!);
    return describe(
      {
        ...common(
          options,
          (into) => {
            into[0] = screen.width;
            into[1] = screen.height;
            into[2] = screen.pixelRatio ?? 1;
          },
          { samples: 1, colored: true },
        ),
        writes: { color },
        attachments: [
          {
            key: 'color',
            write: color,
            format: screen.format,
            view,
            resolve: undefined,
            store: 'store',
          },
        ],
        depth: undefined,
        samples: 1,
        region: undefined,
        framed: true,
        composite: undefined,
      },
      new Set(),
      new Map(),
    );
  }

  const writes = rest[0] as Writes;
  const options = (rest[1] ?? {}) as TargetOptions<Record<string, Spec>>;
  const { resolve = {} } = options;
  const entries = (Object.entries(writes) as [string, Write | undefined][]).filter(
    (entry): entry is [string, Write] => entry[1] !== undefined,
  );
  const colors = entries.filter(([, each]) => each === color);
  const depths = entries.filter(([, each]) => each === depth);
  const sums = entries.filter(([, each]) => each === accumulation);
  const reveals = entries.filter(([, each]) => each === revealage);

  if (colors.length > 1 || depths.length > 1 || sums.length > 1 || reveals.length > 1) {
    throw new Error(
      'passes.create accepts at most one target marked `passes.color`, `passes.depth`, `passes.accumulation` and `passes.revealage` each.',
    );
  }
  if (sums.length !== reveals.length || (sums.length > 0 && colors.length === 0)) {
    throw new Error(
      'passes.create requires `passes.accumulation` and `passes.revealage` together, beside `passes.color`.',
    );
  }

  const samplesOf = (key: string) =>
    (screen.textures[key].props as { sampleCount?: number }).sampleCount ?? 1;
  for (const [key] of entries) {
    if (!screen.textures[key]) throw new Error(`passes.create: the set has no target '${key}'.`);
  }
  const counts = new Set(entries.map(([key]) => samplesOf(key)));
  if (counts.size > 1) {
    throw new Error('passes.create requires the same sample count for every target it draws into.');
  }
  const samples = [...counts][0] ?? 1;

  const drawnLayers = entries.map(([key]) => ({
    key,
    count: (screen.textures[key].props.size as readonly number[])[2],
  }));
  if (new Set(drawnLayers.map((each) => each.count)).size > 1) {
    const listed = drawnLayers
      .map(({ key, count }) => `'${key}' ${count === undefined ? 'none' : count}`)
      .join(', ');
    throw new Error(
      `passes.create requires the same \`layers\` in targets.create for every target it draws into, got ${listed}.`,
    );
  }
  const layers = drawnLayers[0]?.count;
  if (layers !== undefined && sums.length > 0) {
    throw new Error(
      'passes.create cannot draw translucency into layered targets. Leave `layers` off the targets of a pass that draws translucency.',
    );
  }

  const formatOf = (key: string) => screen.textures[key].props.format;
  const storeOf = (key: string): GPUStoreOp => (screen.transient[key] ? 'discard' : 'store');

  for (const [key, into] of Object.entries(resolve)) {
    if (into === undefined) continue;
    if (!entries.some(([each, write]) => each === key && write !== depth)) {
      throw new Error(
        `passes.create cannot resolve '${key}' because it is not a colour target drawn by this description.`,
      );
    }
    if (samplesOf(key) === 1) {
      throw new Error(`passes.create cannot resolve '${key}' because it is not multisampled.`);
    }
    const format = typeof into === 'string' ? screen.textures[into]?.props.format : into.format;
    if (typeof into === 'string' && (!screen.textures[into] || samplesOf(into) > 1)) {
      throw new Error(
        `passes.create requires resolve destination '${into}' for '${key}' to be an existing target with one sample.`,
      );
    }
    if (format !== formatOf(key)) {
      throw new Error(
        `passes.create requires the resolve destination for '${key}' to have the same format.`,
      );
    }
  }

  const resolverOf = (key: string) => {
    const into = resolve[key];
    if (into === undefined) return undefined;
    if (typeof into === 'string') return () => screen.views[into];
    return (frame: Frame | undefined) => into.view(frame!);
  };

  // Alpha to coverage reads colour at location 0
  const ordered = [...colors, ...entries.filter(([, each]) => each !== color && each !== depth)];
  const attachments = ordered.map(([key, write]): Attachment => ({
    key,
    write: write === color ? color : recordOf(write as Output<never, never>),
    format: formatOf(key),
    view: (_frame, at) => screen.layers[key][at],
    resolve: resolverOf(key),
    store: storeOf(key),
  }));

  const depthKey = depths[0]?.[0];
  const first = ordered[0]?.[0] ?? depthKey;
  const region = first
    ? {
        drawn: screen.regions[first],
        full: () => screen.textures[first].props.size as [number, number],
      }
    : undefined;
  const warming = new Set<() => Promise<void>>();
  const composite =
    sums[0] && reveals[0] && colors[0]
      ? compositorOf(root, screen, {
          accumulation: sums[0][0],
          revealage: reveals[0][0],
          color: colors[0][0],
          resolve,
          region,
          label: options.label,
          warming,
        })
      : undefined;

  return describe(
    {
      ...common(
        options,
        (into) => {
          const drawn = region?.drawn ?? screen;
          into[0] = drawn.width;
          into[1] = drawn.height;
          into[2] = (screen.pixelRatio * drawn.width) / screen.width;
        },
        { samples, colored: attachments.length > 0, layers },
      ),
      fitted: () => screen.fitted,
      writes,
      attachments,
      depth: depthKey
        ? {
            key: depthKey,
            format: formatOf(depthKey),
            view: (at) => screen.layers[depthKey][at],
            store: storeOf(depthKey),
          }
        : undefined,
      samples,
      region,
      framed: Object.values(resolve).some((into) => into !== undefined && typeof into !== 'string'),
      warming,
      composite,
    },
    new Set(),
    new Map(),
  );
}
