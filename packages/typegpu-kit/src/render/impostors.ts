/**
 * This module captures meshes into an atlas and draws them as camera-facing cards. Each card
 * samples views near the camera direction and receives lighting and shadows like a mesh.
 *
 * @module
 * @unstable
 * @injects definition the atlas that `Impostor` and kinds made by `define` read, for a renderer
 * piped `atlas.inject(sampling)`
 * @binds 1 storage: the subject's sphere, in the impostors' renderer's pipelines
 * @binds 1 storage: the kind's device rows, where its surface reads the captured `phase`, beside
 * the host rows its transform reads
 * @owns 7 textures, 3 buffers, 1 or 2 samplers, 5 to 8 pipelines, mipmaps: the atlas's colours
 * and its normals with depth, the multisampled targets that draw them, the two that fill their
 * mips for copies to the atlas, the subject's sphere, the extents it is fitted from and its
 * pieces; the atlas's sampler, and the pool's the first time a capture binds its textures; four
 * kernels that fit the sphere, a pipeline for each look of the subject's meshes, one-sided or
 * two-sided, masked or not, that capture it; the mipmaps that fill the two textures, for colour
 * and for normals
 * @records `atlas.record(frame)` once while loading, after what moves the subject's poses and
 * before the passes that draw the atlas
 */

import { linearToSrgb, srgbToLinear } from '@typegpu/color';
import * as cameras from '../cameras.ts';
import * as mipmaps from '../mipmaps.ts';
import * as passes from '../passes.ts';

import { encoderOf, type Frame } from '../frames.ts';
import { unfilled, type Inject } from '../inject.ts';
import { type Props } from '../table.ts';
import {
  d,
  std,
  tgpu,
  type TgpuCommandEncoder,
  type TgpuRenderPipeline,
  type TgpuRoot,
} from 'typegpu';
import {
  define as defineInstances,
  Placement,
  rotate,
  Sphere,
  type InstanceOf,
  type Kind,
  type KindOptions as InstanceKindOptions,
  type Moved,
  Covered,
  type Pixel,
  type Vertex,
} from './instances.ts';
import { mappedNormal, masked, matte, textured } from './materials.ts';
import {
  boundsOf,
  uv1At,
  vertexAt,
  type Mesh,
  type MeshPool,
  type Model,
  type ModelData,
} from './meshPool.ts';
import * as octahedral from './octahedral.ts';
import { ordered, unordered } from './ordered.ts';
import { Material, Point } from './shading.ts';
import { skinned, skinnedPosition, type Id, type Poses } from './poses.ts';

type Triple = readonly [number, number, number];

/**
 * This piece places a mesh within the captured subject and sets the colour multiplied by its
 * textures.
 */
export interface Piece {
  /** Selects a mesh from the pool supplied to the capture. */
  readonly mesh: Mesh;
  /** Sets the mesh origin in subject space. Default `[0, 0, 0]`. */
  readonly position?: Triple;
  /**
   * Rotates the mesh with unit quaternion `[x, y, z, w]`. Default `[0, 0, 0, 1]`, the identity
   * rotation.
   */
  readonly rotation?: readonly [number, number, number, number];
  /** Scales the mesh before rotation, with one factor or one per axis. Default 1. */
  readonly scale?: number | Triple;
  /**
   * Sets the mesh colour in linear colour space. Default the mesh material colour, which is
   * white for shapes.
   */
  readonly color?: Triple;
  /**
   * Selects a pose from `options.poses`, or an array with one pose per captured phase. All pose
   * arrays must have the same length. A single pose or no pose repeats in every phase. Default
   * none, which uses the bind pose.
   */
  readonly pose?: Id | readonly Id[];
}

export interface Options {
  /**
   * Selects the card mesh the atlas draws on: `impostors.CARD` added to the same pool. Atlases
   * share one card.
   */
  readonly card: Mesh;
  /**
   * Supplies the poses selected by pieces. Animate them before `record`, in the same encoder or
   * one submitted earlier. Default none.
   */
  readonly poses?: Poses;
  /** Sets the number of views per side of the atlas grid, from 2 to 16. Default 12. */
  readonly views?: number;
  /**
   * Sets the width and height of each view in texels. It must be a power of two from 16 to 512.
   * Default 128.
   */
  readonly size?: number;
  /**
   * Limits capture directions to the upper hemisphere when true, for subjects viewed from above
   * or the side, such as trees. A value of false captures the full sphere. Default `true`.
   */
  readonly hemisphere?: boolean;
}

/**
 * This type selects atlas sampling. `'nearest'` takes 2 samples from the closest view and
 * switches views as the camera turns. `'blended'` takes 6 samples from the three closest views
 * and weights them by distance. `'parallax'` takes 18 samples and uses stored depth to move each
 * view sample along the pixel ray, for subjects seen close up.
 */
export type Sampling = 'nearest' | 'blended' | 'parallax';

/** This atlas stores the captured views used by `Impostor` or a kind created with `define`. */
export interface Atlas extends Disposable {
  /** Gives the number of views per side of the grid. */
  readonly views: number;
  /** Gives the width and height of each view in texels. */
  readonly size: number;
  /**
   * Gives the captured phase count. Each instance selects one with its zero-based `phase` field.
   */
  readonly phases: number;
  /**
   * Records the subject bounds, captured views and atlas mipmaps into `target`, a frame or an
   * encoder. The atlas stays blank until submission executes these commands. Record again after
   * poses or textures change.
   */
  record(target: Frame | TgpuCommandEncoder): void;
  /**
   * Fills atlas resources for pipelines using `Impostor` or a kind made by `define`, read with
   * `sampling`. Default `'blended'`.
   */
  inject(sampling?: Sampling): Inject;
  /** Destroys the atlas textures, the capture targets and the buffers. */
  destroy(): void;
}

const MSAA = 4;
/** Atlas texels a card's shadow sits behind what it shows. */
const LAG = 12;
/** Threads per workgroup that fit the subject's sphere. */
const FIT = 64;
const SAMPLINGS: readonly Sampling[] = ['nearest', 'blended', 'parallax'];

/**
 * This model is the card every atlas draws on: add it to the pool once, with
 * `pool.add(impostors.CARD).meshes[0]`, give the mesh to each `create` and instances of it to the
 * cards. Count it in the pool's room, as in `meshPool.roomFor([horse, impostors.CARD])`.
 */
export const CARD: ModelData = Object.freeze({
  meshes: [
    {
      positions: new Float32Array([-0.5, 0, 0.5, 0.5, 0, 0.5, -0.5, 0, -0.5, 0.5, 0, -0.5]),
      normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]),
      indices: new Uint32Array([0, 1, 3, 0, 3, 2]),
      material: {
        color: [1, 1, 1, 1] as const,
        roughness: 1,
        metallic: 0,
        emission: [0, 0, 0] as const,
        textures: {},
      },
    },
  ],
  images: [],
});

const oneLayer = (layer: number) => ({
  baseMipLevel: 0,
  mipLevelCount: 1,
  baseArrayLayer: layer,
  arrayLayerCount: 1,
});

/** Averages four linear texels and encodes the result as sRGB. */
const meanToSrgb: mipmaps.Combine = (a, b, c, e) => {
  'use gpu';
  const mean = (a + b + c + e) * 0.25;
  return d.vec4f(linearToSrgb(mean.xyz), mean.w);
};

/** Averages four sRGB-encoded texels in linear light and encodes the result. */
const srgbMean: mipmaps.Combine = (a, b, c, e) => {
  'use gpu';
  const sum = srgbToLinear(a.xyz) + srgbToLinear(b.xyz) + srgbToLinear(c.xyz) + srgbToLinear(e.xyz);
  return d.vec4f(linearToSrgb(sum * 0.25), (a.w + b.w + c.w + e.w) * 0.25);
};

const missing = unfilled("an impostor's atlas", 'atlas', 'an `impostors.create`');
const albedoAtlas = tgpu.accessor(d.texture2dArray(d.f32), missing);
const normalAtlas = tgpu.accessor(d.texture2dArray(d.f32), missing);
const atlasSampler = tgpu.accessor(d.sampler(), missing);
const Subject = d.struct({ center: d.vec3f, radius: d.f32, half: d.vec3f });
const subjectShape = tgpu.accessor(Subject, missing);
const views = tgpu.accessor(d.u32, missing);
const phases = tgpu.accessor(d.u32, missing);
const hemispheric = tgpu.slot(true);
const blending = tgpu.slot(true);
const refining = tgpu.slot(false);
const twoSided = tgpu.slot(false);
const cutting = tgpu.slot(false);

const Basis = d.struct({ right: d.vec3f, up: d.vec3f });

const cornerOf = (corner: number) => {
  'use gpu';
  return d.vec3f(d.vec3u(corner & 1, (corner >>> 1) & 1, (corner >>> 2) & 1)) * 2 - 1;
};

const basisOf = (direction: d.v3f) => {
  'use gpu';
  const pole = std.select(d.vec3f(0, 1, 0), d.vec3f(0, 0, 1), std.abs(direction.y) > 0.999);
  const right = std.normalize(std.cross(pole, direction));
  return Basis({ right, up: std.cross(direction, right) });
};

/**
 * Octahedral encoding folds the lower sphere into the square corners. For a hemisphere, rotate
 * the diamond to fill the square and clamp directions below the horizon to the horizon.
 */
const encode = (direction: d.v3f) => {
  'use gpu';
  if (hemispheric.$) {
    const above = d.vec3f(direction.x, std.max(direction.y, 0), direction.z);
    const n = above / std.max(std.abs(above.x) + above.y + std.abs(above.z), 1e-6);
    return d.vec2f(n.x + n.z, n.z - n.x) * 0.5 + 0.5;
  }
  return octahedral.encode(direction.xzy) * 0.5 + 0.5;
};

const decode = (uv: d.v2f) => {
  'use gpu';
  const f = uv * 2 - 1;
  if (hemispheric.$) {
    const x = (f.x - f.y) * 0.5;
    const z = (f.x + f.y) * 0.5;
    return std.normalize(d.vec3f(x, 1 - std.abs(x) - std.abs(z), z));
  }
  return octahedral.decode(f).xzy;
};

const directionOf = (cell: d.v2f) => {
  'use gpu';
  return decode(cell / d.f32(views.$ - 1));
};

const layerOf = (cell: d.v2f, phase: number) => {
  'use gpu';
  return (
    std.min(d.u32(phase), phases.$ - 1) * views.$ * views.$ +
    d.u32(cell.y) * views.$ +
    d.u32(cell.x)
  );
};

const Views = d.struct({ cells: d.arrayOf(d.vec2f, 3), weights: d.vec3f });

const nearestTo = (direction: d.v3f) => {
  'use gpu';
  return std.round(encode(direction) * d.f32(views.$ - 1));
};

const viewsAround = (direction: d.v3f) => {
  'use gpu';
  const last = d.f32(views.$ - 1);
  const grid = encode(direction) * last;
  const cell = std.min(std.floor(grid), d.vec2f(last - 1));
  const f = grid - cell;
  const lower = f.x + f.y < 1;
  const corner = std.select(d.vec2f(1), d.vec2f(0), lower);
  const weights = std.select(
    d.vec3f(f.x + f.y - 1, 1 - f.y, 1 - f.x),
    d.vec3f(1 - f.x - f.y, f.x, f.y),
    lower,
  );
  return Views({
    cells: d.arrayOf(d.vec2f, 3)([cell + corner, cell + d.vec2f(1, 0), cell + d.vec2f(0, 1)]),
    weights,
  });
};

const inView = (point: d.v3f, direction: d.v3f, basis: d.Infer<typeof Basis>, radius: number) => {
  'use gpu';
  return d.vec3f(
    std.dot(point, basis.right) / (2 * radius) + 0.5,
    0.5 - std.dot(point, basis.up) / (2 * radius),
    0.5 - std.dot(point, direction) / (2 * radius),
  );
};

const probe = (uv: d.v2f, layer: number, dx: d.v2f, dy: d.v2f) => {
  'use gpu';
  const covered = std.textureSampleGrad(albedoAtlas.$, atlasSampler.$, uv, layer, dx, dy).a;
  const kept = std.textureSampleGrad(normalAtlas.$, atlasSampler.$, uv, layer, dx, dy).a;
  const depth = kept / std.max(covered, 1e-4);
  return d.vec2f(std.select(0, (0.5 - depth) * 2 * subjectShape.$.radius, covered > 0.01), covered);
};

/** What a view shows along a ray, and how much of the pixel it covers, as `coveredIn` finds. */
const Seen = d.struct({ color: d.vec4f, shape: d.vec4f, at: d.vec3f, covered: d.f32 });

/** Where a ray crosses a view's middle plane, in that view's atlas layer. */
const Crossing = d.struct({
  direction: d.vec3f,
  basis: Basis,
  /** Holds the ray's slope towards the view, below zero. */
  facing: d.f32,
  middle: d.vec2f,
  layer: d.u32,
});

const crossingOf = (cell: d.v2f, start: d.v3f, ray: d.v3f, phase: number) => {
  'use gpu';
  const direction = directionOf(cell);
  const basis = basisOf(direction);
  // Measure height towards the view from its middle plane
  const height = std.dot(start, direction);
  const facing = std.min(std.dot(ray, direction), -1e-3);
  const at = start - ray * (height / facing);
  const middle = inView(at, direction, basis, subjectShape.$.radius).xy;
  return Crossing({ direction, basis, facing, middle, layer: layerOf(cell, phase) });
};

/** How much of the pixel a view covers where the ray crosses its middle plane. */
const coveredIn = (cell: d.v2f, start: d.v3f, ray: d.v3f, phase: number) => {
  'use gpu';
  const crossing = crossingOf(cell, start, ray, phase);
  const middle = crossing.middle;
  const dx = std.dpdx(middle);
  const dy = std.dpdy(middle);
  return std.textureSampleGrad(albedoAtlas.$, atlasSampler.$, middle, crossing.layer, dx, dy).a;
};

const seenIn = (cell: d.v2f, start: d.v3f, ray: d.v3f, phase: number) => {
  'use gpu';
  const radius = subjectShape.$.radius;
  const crossing = crossingOf(cell, start, ray, phase);
  const basis = crossing.basis;
  const middle = crossing.middle;
  const layer = crossing.layer;
  const dx = std.dpdx(middle);
  const dy = std.dpdy(middle);

  let uv = d.vec2f(middle);
  let covered = d.f32(-1);
  if (refining.$) {
    // Intersect the ray with the line through two sampled heights so planar surfaces resolve
    // exactly. A ray point at a height above the middle plane projects to `middle + shift * height`
    const shift =
      d.vec2f(std.dot(ray, basis.right), -std.dot(ray, basis.up)) / (2 * radius * crossing.facing);
    const first = probe(middle, layer, dx, dy);
    covered = first.y;
    const near = first.x;
    const next = probe(middle + shift * near, layer, dx, dy).x;
    const bend = 2 * near - next;
    const met = std.select(next, (near * near) / bend, std.abs(bend) > radius * 1e-5);
    uv = middle + shift * std.clamp(met, -radius, radius);
  }

  const color = std.textureSampleGrad(albedoAtlas.$, atlasSampler.$, uv, layer, dx, dy);
  const shape = std.textureSampleGrad(normalAtlas.$, atlasSampler.$, uv, layer, dx, dy);

  const across = (uv - 0.5) * (2 * radius);
  const depth = (0.5 - shape.a / std.max(color.a, 1e-4)) * (2 * radius);
  const at = basis.right * across.x - basis.up * across.y + crossing.direction * depth;
  if (!refining.$) covered = color.a;
  return Seen({ color, shape, at, covered });
};

const unplace = (v: d.v3f, rotation: d.v4f, scale: d.v3f) => {
  'use gpu';
  return rotate(d.vec4f(rotation.xyz * -1, rotation.w), v) / scale;
};

type Placed = { position: d.v3f; rotation: d.v4f; scale: d.v3f };

/** Where a pixel's ray enters the subject's frame, its direction there and the card's view. */
const Sight = d.struct({ start: d.vec3f, ray: d.vec3f, toward: d.vec3f });

const towardsCamera = (impostor: Placement) => {
  'use gpu';
  return towardsFrom(impostor, subjectShape.$.center);
};

const sightOf = (impostor: Placement, point: Point) => {
  'use gpu';
  const rotation = impostor.rotation;
  const scale = impostor.scale;
  return Sight({
    start: unplace(point.position - impostor.position, rotation, scale) - subjectShape.$.center,
    ray: unplace(point.view, rotation, scale) * -1,
    toward: towardsCamera(impostor),
  });
};

const towardsFrom = (impostor: Placement, local: d.v3f) => {
  'use gpu';
  const world = impostor.position + rotate(impostor.rotation, local * impostor.scale);
  const back = cameras.towards(cameras.current.$, world);
  return std.normalize(unplace(back, impostor.rotation, impostor.scale));
};

export interface KindOptions<H extends Props = {}, D extends Props = {}, M extends Moved = 'host'> {
  /** Adds CPU-written fields, as in `instances.define`. Default none. */
  readonly host?: NonNullable<InstanceKindOptions<H, D, M>['host']> & { readonly phase?: never };
  /** Adds kernel-written fields beside the built-in `phase` field. Default none. */
  readonly device?: NonNullable<InstanceKindOptions<H, D, M>['device']> & {
    readonly phase?: never;
  };
  /**
   * Selects who writes position, rotation and scale, as in `instances.define`. Default `'host'`.
   */
  readonly transform?: M;
  /**
   * Returns the card's transform from custom data, such as a simulation buffer, as in
   * `instances.define`. The card faces the camera from it and its sphere follows it. Default the
   * card's `position`, `rotation` and `scale`.
   * @gpu
   */
  readonly place?: (impostor: InstanceOf<H, D & { phase: d.U32 }, M>) => Placement;
  /** Hides the card in every pass, as in `instances.define`. Default always shown. @gpu */
  readonly visible?: (impostor: InstanceOf<H, D & { phase: d.U32 }, M>) => boolean;
}

type Phased = {
  /** Holds the captured phase the instance shows, from 0. */
  phase: d.U32;
};

/**
 * Defines an instance kind that draws an atlas with custom host and device fields. Its device
 * fields include a zero-based `phase`, initially 0. Values beyond the captured range select the
 * last phase. Inject the atlas into the renderer. Instance colour multiplies atlas colour;
 * roughness, metallic and emission control shading as for a mesh.
 *
 * @param name a label for ids and WGSL, as `table.define` takes it
 * @returns a kind to give `instances.create` as `kind`
 */
export function define<
  N extends string,
  H extends Props = {},
  D extends Props = {},
  M extends 'host' = 'host',
>(name: N, options?: KindOptions<H, D, M>): Kind<N, H, D & Phased, M>;
export function define<
  N extends string,
  H extends Props = {},
  D extends Props = {},
  M extends Moved = Moved,
>(name: N, options: KindOptions<H, D, M> & { readonly transform: M }): Kind<N, H, D & Phased, M>;
export function define<
  N extends string,
  H extends Props = {},
  D extends Props = {},
  M extends Moved = 'host',
>(name: N, options: KindOptions<H, D, M> = {}): Kind<N, H, D & Phased, M> {
  const place = options.place as ((impostor: Placed) => Placement) | undefined;
  const placementOf = (impostor: Placed) => {
    'use gpu';
    if (place !== undefined) return Placement(place(impostor));
    return Placement({
      rotation: impostor.rotation,
      position: impostor.position,
      scale: impostor.scale,
    });
  };
  const hooks = {
    // Project the subject box corners onto a camera-facing plane through its centre to size the
    // card
    deform: (vertex: Vertex, row: Placed) => {
      'use gpu';
      const impostor = placementOf(row);
      const shape = subjectShape.$;
      const toward = towardsCamera(impostor);
      const basis = basisOf(toward);
      let low = d.vec2f(3.4e38);
      let high = d.vec2f(-3.4e38);
      for (let corner = d.u32(0); corner < 8; corner++) {
        const offset = shape.half * cornerOf(corner);
        const back = towardsFrom(impostor, shape.center + offset);
        const crossing =
          offset - back * (std.dot(offset, toward) / std.max(std.dot(back, toward), 1e-3));
        const at = d.vec2f(std.dot(crossing, basis.right), std.dot(crossing, basis.up));
        low = std.min(low, at);
        high = std.max(high, at);
      }
      const at = std.mix(low, high, vertex.uv);
      return shape.center + basis.right * at.x + basis.up * at.y;
    },

    // The surface and its coverage from one set of samples. Depth-only passes read both; colour
    // passes discard by `coverage` first, then read the surface
    '~covered': (pixel: Pixel<Placed & { phase: number }>, base: Material) => {
      'use gpu';
      const point = pixel.point;
      const row = pixel.instance;
      const impostor = placementOf(row);
      const rotation = impostor.rotation;
      const scale = impostor.scale;
      const sight = sightOf(impostor, point);
      const start = sight.start;
      const ray = sight.ray;
      const toward = sight.toward;

      let color = d.vec4f();
      let encoded = d.vec4f();
      let shown = d.vec3f();
      let coverage = d.f32();
      if (blending.$) {
        const around = viewsAround(toward);
        for (const k of tgpu.unroll([0, 1, 2])) {
          const seen = seenIn(around.cells[k], start, ray, row.phase);
          const weight = around.weights[k];
          color += seen.color * weight;
          encoded += seen.shape * weight;
          shown += seen.at * (seen.color.a * weight);
          coverage += seen.covered * weight;
        }
      } else {
        const seen = seenIn(nearestTo(toward), start, ray, row.phase);
        color = d.vec4f(seen.color);
        encoded = d.vec4f(seen.shape);
        shown = seen.at * seen.color.a;
        coverage = seen.covered;
      }

      const kept = std.max(color.a, 1e-4);
      const local = (encoded.xyz / kept) * 2 - 1;
      const normal = std.normalize(rotate(rotation, local / scale));

      const material = Material(base);
      material.color = base.color * (color.rgb / kept);
      let position =
        impostor.position + rotate(rotation, (subjectShape.$.center + shown / kept) * scale);
      if (passes.depthOnly.$) {
        // Views of a point disagree on its depth by a few texels, so a card casts that far behind
        // what it shows and doesn't shadow itself
        const texel = (2 * subjectShape.$.radius) / d.f32(std.textureDimensions(albedoAtlas.$).x);
        const widest = std.max(std.max(scale.x, scale.y), scale.z);
        position = position - point.view * (texel * widest * LAG);
      }
      const moved = Point(point);
      moved.position = d.vec3f(position);
      moved.normal = d.vec3f(normal);
      return Covered({ point: moved, material, coverage });
    },

    // The coverage alone, which colour passes discard by before the surface
    coverage: (pixel: Pixel<Placed & { phase: number }>) => {
      'use gpu';
      const row = pixel.instance;
      const sight = sightOf(placementOf(row), pixel.point);
      const start = sight.start;
      const ray = sight.ray;
      const toward = sight.toward;
      if (blending.$) {
        const around = viewsAround(toward);
        let coverage = d.f32(0);
        for (const k of tgpu.unroll([0, 1, 2])) {
          coverage += coveredIn(around.cells[k], start, ray, row.phase) * around.weights[k];
        }
        return coverage;
      }
      return coveredIn(nearestTo(toward), start, ray, row.phase);
    },

    bounds: (row: Placed, _base: Sphere) => {
      'use gpu';
      const impostor = placementOf(row);
      const shape = subjectShape.$;
      const scale = impostor.scale;
      return Sphere({
        center: impostor.position + rotate(impostor.rotation, shape.center * scale),
        radius: shape.radius * std.max(std.max(scale.x, scale.y), scale.z),
      });
    },
  };
  // The merge preserves base fields that generic conditional types cannot establish
  return defineInstances<N, H, D & Phased, M>(name, {
    ...options,
    ...hooks,
    device: { ...options.device, phase: d.u32 },
  } as unknown as InstanceKindOptions<H, D & Phased, M> & { readonly transform: M });
}

/**
 * This impostor kind stores its transform and material on the host and its `phase` on the
 * device.
 */
export const Impostor = define('Impostor');

const PieceData = d.struct({
  position: d.vec3f,
  mesh: d.u32,
  scale: d.vec3f,
  rotation: d.vec4f,
  color: d.vec3f,
  opacity: d.f32,
  pose: d.u32,
  vertices: d.u32,
  firstVertex: d.u32,
  posed: d.u32,
});

const piecesOf = (subject: Model | Mesh | readonly Piece[]): readonly Piece[] => {
  if (Array.isArray(subject)) return subject as readonly Piece[];
  if ('meshes' in subject) return subject.meshes.map((mesh) => ({ mesh }));
  return [{ mesh: subject as Mesh }];
};

/**
 * Creates an atlas that `record` fills with views around `subject`. Each view stores textured
 * colour, coverage, normal and depth. A model is captured whole; a piece list places each mesh.
 * Back faces receive light from their side on double-sided meshes, and masked meshes apply their
 * alpha cutoff.
 *
 * Record after shape baking in the same encoder or an earlier submission. Wait for
 * `model.ready` to include model textures; otherwise, the views use untextured factors. Invalid
 * view counts or sizes, empty subjects, meshes or a card from another pool, invalid poses and
 * excess texture layers throw.
 *
 * @param pool the mesh pool that holds the subject and the card
 * @param subject a mesh, a model, or a list of pieces, each a mesh of `pool`
 * @returns an atlas you record once and destroy
 */
export function create(
  root: TgpuRoot,
  pool: MeshPool,
  subject: Model | Mesh | readonly Piece[],
  options: Options,
): Atlas {
  const { views: count = 12, size = 128, hemisphere = true } = options;
  if (pool.meshes[options.card.index] !== options.card) {
    throw new Error('The card must be `impostors.CARD` added to the pool supplied to create.');
  }
  if (!Number.isInteger(count) || count < 2 || count > 16) {
    throw new RangeError(`An atlas has 2 to 16 views along each side, got ${count}.`);
  }
  if (!Number.isInteger(size) || size < 16 || size > 512 || (size & (size - 1)) !== 0) {
    throw new RangeError(
      `An atlas's \`size\` is a power of two from 16 to 512 texels, got ${size}.`,
    );
  }
  const pieces = piecesOf(subject);
  if (pieces.length === 0) throw new Error('A capture needs at least one mesh.');
  for (const { mesh } of pieces) {
    if (pool.meshes[mesh.index] !== mesh)
      throw new Error('A piece mesh must belong to the pool supplied to create.');
  }
  const counts = pieces.flatMap(({ pose }) => (typeof pose === 'object' ? [pose.length] : []));
  const phaseCount = counts[0] ?? 1;
  if (phaseCount === 0 || counts.some((length) => length !== phaseCount)) {
    throw new Error(
      'Give each capture with a `pose` array the same nonzero phase count. Each entry is one phase.',
    );
  }
  for (const { mesh, pose } of pieces) {
    if (pose === undefined) continue;
    if (!mesh.skinned) throw new Error('A posed capture needs a mesh with joints.');
    const ids = typeof pose === 'object' ? pose : [pose];
    if (ids.some((id) => !options.poses?.has(id)))
      throw new Error('A captured pose must be live in options.poses.');
  }
  const posed = pieces.some((piece) => piece.pose !== undefined);
  const layers = count * count * phaseCount;
  if (layers > root.device.limits.maxTextureArrayLayers) {
    throw new RangeError(
      `The impostor atlas needs ${layers} layers. The device's \`maxTextureArrayLayers\` is ${root.device.limits.maxTextureArrayLayers}. Reduce \`views\` or the number of captured phases.`,
    );
  }
  const levels = Math.log2(size) + 1;
  const atlasSize = [size, size, layers] as const;
  const albedo = root
    .createTexture({ size: atlasSize, format: 'rgba8unorm-srgb', mipLevelCount: levels })
    .$usage('render', 'sampled');
  const normals = root
    .createTexture({ size: atlasSize, format: 'rgba8unorm', mipLevelCount: levels })
    .$usage('render', 'sampled');
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });

  const partBuffer = root
    .createBuffer(
      d.arrayOf(PieceData, pieces.length * phaseCount),
      Array.from({ length: phaseCount }, (_, phase) =>
        pieces.map((piece) => {
          const { position = [0, 0, 0], rotation = [0, 0, 0, 1], scale = 1 } = piece;
          const sizes = typeof scale === 'number' ? [scale, scale, scale] : scale;
          const color = piece.color ?? piece.mesh.material.color;
          return {
            position: d.vec3f(...position),
            mesh: piece.mesh.index,
            scale: d.vec3f(sizes[0], sizes[1], sizes[2]),
            rotation: d.vec4f(...rotation),
            color: d.vec3f(...color),
            opacity: piece.mesh.material.opacity,
            pose: (typeof piece.pose === 'object' ? piece.pose[phase] : piece.pose) ?? 0,
            vertices: piece.mesh.vertexCount,
            firstVertex: piece.mesh.baseVertex,
            posed: piece.pose === undefined ? 0 : 1,
          };
        }),
      ).flat(),
    )
    .$usage('storage');
  const subjectBuffer = root.createBuffer(Subject).$usage('storage');
  const pieceList = partBuffer.as('readonly');
  const shaped = subjectBuffer.as('mutable');

  const placedAt = (piece: number, point: d.v3f) => {
    'use gpu';
    const at = pieceList.$[piece];
    return rotate(at.rotation, point * at.scale) + at.position;
  };
  const positionOf = (piece: number, vertex: number) => {
    'use gpu';
    const local = vertexAt(vertex).position;
    if (posed && pieceList.$[piece].posed !== 0) {
      return skinnedPosition(vertex, local, pieceList.$[piece].pose);
    }
    return d.vec3f(local);
  };

  const items = pieces.length * phaseCount;
  const pointsOf = (piece: Piece) => (piece.pose === undefined ? 8 : piece.mesh.vertexCount);
  const most = Math.max(...pieces.map(pointsOf));
  const fitted = root.createBuffer(d.arrayOf(d.atomic(d.u32), 7)).$usage('storage');
  const fitting = fitted.as('mutable');

  /** How many points item `piece` fits the subject to: its posed vertices or its box corners. */
  const countOf = (piece: number) => {
    'use gpu';
    const item = pieceList.$[piece];
    if (posed && item.posed !== 0) return item.vertices;
    return d.u32(8);
  };
  const pointOf = (piece: number, index: number) => {
    'use gpu';
    const item = pieceList.$[piece];
    if (posed && item.posed !== 0) {
      return placedAt(piece, positionOf(piece, item.firstVertex + index));
    }
    const box = boundsOf(item.mesh);
    return placedAt(piece, box.center + box.half * cornerOf(index));
  };
  const centerOf = () => {
    'use gpu';
    const low = d.vec3f(
      unordered(std.atomicLoad(fitting.$[0])),
      unordered(std.atomicLoad(fitting.$[1])),
      unordered(std.atomicLoad(fitting.$[2])),
    );
    const high = d.vec3f(
      unordered(std.atomicLoad(fitting.$[3])),
      unordered(std.atomicLoad(fitting.$[4])),
      unordered(std.atomicLoad(fitting.$[5])),
    );
    return Subject({ center: (low + high) * 0.5, radius: 0, half: (high - low) * 0.5 });
  };

  const empty = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    for (const axis of tgpu.unroll([0, 1, 2])) {
      std.atomicStore(fitting.$[axis], 0xffffffff);
      std.atomicStore(fitting.$[axis + 3], 0);
    }
    std.atomicStore(fitting.$[6], 0);
  });
  // One thread per point of each piece in each phase: x is the point, y the piece and phase
  const box = tgpu.computeFn({ workgroupSize: [FIT], in: { gid: d.builtin.globalInvocationId } })(({
    gid,
  }) => {
    'use gpu';
    if (gid.y >= items || gid.x >= countOf(gid.y)) return;
    const at = pointOf(gid.y, gid.x);
    for (const axis of tgpu.unroll([0, 1, 2])) {
      std.atomicMin(fitting.$[axis], ordered(at[axis]));
      std.atomicMax(fitting.$[axis + 3], ordered(at[axis]));
    }
  });
  const sphere = tgpu.computeFn({
    workgroupSize: [FIT],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    if (gid.y >= items || gid.x >= countOf(gid.y)) return;
    // Distances are not negative, so their bits order as the values do
    const reach = std.length(pointOf(gid.y, gid.x) - centerOf().center);
    std.atomicMax(fitting.$[6], std.bitcast(d.f32, d.u32)(reach));
  });
  const enclose = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const fit = centerOf();
    const radius = std.bitcast(d.u32, d.f32)(std.atomicLoad(fitting.$[6]));
    shaped.$ = Subject({ center: fit.center, radius: std.max(radius, 1e-4), half: fit.half });
  });

  // Encode view and piece as `view * pieces + piece` in the instance index
  const vertex = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      position: d.builtin.position,
      world: d.vec3f,
      normal: d.vec3f,
      uv: d.vec2f,
      uv1: d.vec2f,
      color: d.vec3f,
      opacity: d.f32,
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    const layer = std.intdiv(instanceIndex, d.u32(pieces.length));
    const phase = std.intdiv(layer, views.$ * views.$);
    const piece = phase * pieces.length + (instanceIndex % pieces.length);
    const view = layer % (views.$ * views.$);
    const cell = d.vec2f(d.f32(view % views.$), d.f32(std.intdiv(view, views.$)));
    const direction = directionOf(cell);
    const basis = basisOf(direction);
    let local = vertexAt(vertexIndex);
    const at = pieceList.$[piece];
    if (posed && at.posed !== 0) local = skinned(vertexIndex, local, at.pose);
    const world = placedAt(piece, local.position);
    const seen = inView(world - subjectShape.$.center, direction, basis, subjectShape.$.radius);

    return {
      position: d.vec4f(seen.xy * d.vec2f(2, -2) + d.vec2f(-1, 1), seen.z, 1),
      world,
      normal: rotate(at.rotation, local.normal / at.scale),
      uv: local.uv,
      uv1: uv1At(vertexIndex),
      color: at.color,
      opacity: at.opacity,
    };
  });

  const fragment = tgpu.fragmentFn({
    in: {
      position: d.builtin.position,
      front: d.builtin.frontFacing,
      world: d.vec3f,
      normal: d.vec3f,
      uv: d.vec2f,
      uv1: d.vec2f,
      color: d.vec3f,
      opacity: d.f32,
    },
    out: { albedo: d.vec4f, normal: d.vec4f },
  })(({ position, front, world, normal, uv, uv1, color, opacity }) => {
    'use gpu';
    if (cutting.$) {
      if (masked(uv, uv1, opacity)) std.discard();
    }
    // Light back faces from their visible side
    let outward = std.normalize(normal);
    if (twoSided.$) {
      if (!front) outward = outward * -1;
    }
    const facing = mappedNormal(outward, world, uv, uv1);
    const point = Point({ position: world, normal: facing, view: facing, uv, uv1 });
    return {
      albedo: d.vec4f(textured(matte(color), point).color, 1),
      normal: d.vec4f(facing * 0.5 + 0.5, position.z),
    };
  });

  const readAs =
    (sampling: Sampling): Inject =>
    (cfg) =>
      cfg
        .with(albedoAtlas, () => albedoView.$)
        .with(normalAtlas, () => normalView.$)
        .with(atlasSampler, () => sampler.$)
        .with(subjectShape, subjectBuffer.as('readonly'))
        .with(views, count)
        .with(phases, phaseCount)
        .with(hemispheric, hemisphere)
        .with(blending, sampling !== 'nearest')
        .with(refining, sampling === 'parallax');
  const inject = readAs('blended');
  const reads = new Map(
    SAMPLINGS.map((sampling) => [sampling, sampling === 'blended' ? inject : readAs(sampling)]),
  );
  const albedoView = albedo.createView(d.texture2dArray(d.f32));
  const normalView = normals.createView(d.texture2dArray(d.f32));

  let source = root.pipe(pool.inject());
  if (options.poses) source = source.pipe(options.poses.inject());
  const configured = source.pipe(inject);
  const fitters = [
    source.createComputePipeline({ compute: empty }),
    source.createComputePipeline({ compute: box }),
    source.createComputePipeline({ compute: sphere }),
    source.createComputePipeline({ compute: enclose }),
  ];
  const across = [Math.ceil(most / FIT), items] as const;

  const looks = new Map<string, TgpuRenderPipeline>();
  const capturing = pieces.map(({ mesh }) => {
    const key = `${mesh.doubleSided} ${mesh.masked}`;
    let made = looks.get(key);
    if (!made) {
      made = configured
        .with(twoSided, mesh.doubleSided)
        .with(cutting, mesh.masked)
        .createRenderPipeline({
          vertex,
          fragment,
          targets: { albedo: { format: 'rgba8unorm-srgb' }, normal: { format: 'rgba8unorm' } },
          primitive: { cullMode: mesh.doubleSided ? 'none' : 'back' },
          depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
          multisample: { count: MSAA },
        });
      looks.set(key, made);
    }
    return made;
  });

  const target = (format: GPUTextureFormat) =>
    root.createTexture({ size: [size, size], format, sampleCount: MSAA }).$usage('render');
  const albedoSamples = target('rgba8unorm-srgb');
  const normalSamples = target('rgba8unorm');
  const depthSamples = target('depth24plus');
  const albedoTarget = albedoSamples.createView('render');
  const normalTarget = normalSamples.createView('render');
  const depthTarget = depthSamples.createView('render');

  const atlases = [albedo, normals];
  const resolves = atlases.map((texture) =>
    Array.from({ length: layers }, (_, layer) => texture.createView('render', oneLayer(layer))),
  );

  // A render pass per layer and mip level can reach 2048 passes in one submit, where Firefox loses
  // the device. Kernels fill every layer of a level at once in storage textures, which the atlases
  // lack to keep their sampling compressed, and the levels are copied over. `first` holds mip 1 and
  // `rest` the mips below it, for one atlas texture at a time, with albedo sRGB-encoded.
  const scratch = (side: number, mipLevelCount: number) =>
    root
      .createTexture({ size: [side, side, layers], format: 'rgba8unorm', mipLevelCount })
      .$usage('storage', 'sampled');
  const first = scratch(size / 2, 1);
  const rest = scratch(size / 4, levels - 2);
  const normalRest = mipmaps.create(root, rest, { combine: mipmaps.mean, source: first });
  const mipmapping = [
    {
      atlas: albedo,
      first: mipmaps.create(root, first, { combine: meanToSrgb, source: albedo }),
      rest: mipmaps.create(root, rest, { combine: srgbMean, source: first }),
    },
    { atlas: normals, first: normalRest.of(first, normals), rest: normalRest },
  ];
  const copies = Array.from({ length: levels - 1 }, (_, level) => {
    const side = size >> (level + 1);
    return {
      from: level === 0 ? { texture: first, mipLevel: 0 } : { texture: rest, mipLevel: level - 1 },
      mipLevel: level + 1,
      extent: [side, side, layers] as const,
    };
  });

  const record = (into: Frame | TgpuCommandEncoder) => {
    const encoder = encoderOf(into);
    const compute = encoder.beginComputePass();
    const [emptying, boxing, reaching, enclosing] = fitters;
    compute.setPipeline(emptying);
    compute.dispatchWorkgroups(1);
    compute.setPipeline(boxing);
    compute.dispatchWorkgroups(...across);
    compute.setPipeline(reaching);
    compute.dispatchWorkgroups(...across);
    compute.setPipeline(enclosing);
    compute.dispatchWorkgroups(1);
    compute.end();
    for (let layer = 0; layer < layers; layer++) {
      const pass = encoder.beginRenderPass({
        // Match fragment output order, with colour before normal and depth
        colorAttachments: [
          {
            view: albedoTarget,
            resolveTarget: resolves[0][layer],
            clearValue: [0, 0, 0, 0],
            loadOp: 'clear',
            storeOp: 'discard',
          },
          {
            view: normalTarget,
            resolveTarget: resolves[1][layer],
            clearValue: [0, 0, 0, 0],
            loadOp: 'clear',
            storeOp: 'discard',
          },
        ],
        depthStencilAttachment: {
          view: depthTarget,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
        },
      });
      pass.setIndexBuffer(pool.indices, 'uint32');
      for (const [index, { mesh }] of pieces.entries()) {
        pass.setPipeline(capturing[index]);
        pass.setBindGroup(pool.texturesOf(mesh.index));
        pass.drawIndexed(
          mesh.indexCount,
          1,
          mesh.firstIndex,
          mesh.baseVertex,
          layer * pieces.length + index,
        );
      }
      pass.end();
    }

    const raw = root.unwrap(encoder);
    for (const mips of mipmapping) {
      const filling = encoder.beginComputePass();
      mips.first.record(filling);
      mips.rest.record(filling);
      filling.end();
      for (const { from, mipLevel, extent } of copies) {
        raw.copyTextureToTexture(
          { texture: root.unwrap(from.texture), mipLevel: from.mipLevel },
          { texture: root.unwrap(mips.atlas), mipLevel },
          extent,
        );
      }
    }
  };

  const destroy = () => {
    for (const texture of [...atlases, albedoSamples, normalSamples, depthSamples, first, rest]) {
      texture.destroy();
    }
    partBuffer.destroy();
    subjectBuffer.destroy();
    fitted.destroy();
  };

  return {
    views: count,
    size,
    phases: phaseCount,
    record,

    inject(sampling = 'blended') {
      const read = reads.get(sampling);
      if (!read) {
        throw new RangeError(
          `Atlas sampling must be 'nearest', 'blended' or 'parallax'. Received '${sampling}'.`,
        );
      }
      return read;
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
