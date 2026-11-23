import { encoderOf, type Frame } from '../frames.ts';
import { unfilled, type Inject } from '../inject.ts';
import * as pipelines from '../pipelines.ts';
import { meshes } from '../geometry/index.ts';
import {
  d,
  std,
  tgpu,
  type IndexFlag,
  type StorageFlag,
  type TgpuBindGroup,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuComputePipeline,
  type TgpuReadonly,
  type TgpuRoot,
} from 'typegpu';
import { decode as fromOctahedral, encode as octahedral } from './octahedral.ts';
import { ordered, orderedOnCpu, unordered } from './ordered.ts';
import * as workgroup from '../workgroup.ts';
import { createTextures, keepsSecondSet, type layout, type TexturesData } from './textures.ts';
import { decodesItself, HDR, type Decode, type ImageSource } from '../images.ts';

/**
 * This layout binds mesh textures, a sampler and sampling settings through
 * `pool.texturesOf(index)`, to the fragment stage. Use it in custom fragment shaders that sample
 * textures, as `materials.textured` does.
 */
export { layout as textureLayout, Reading, TextureUvs } from './textures.ts';

export type { SamplerData, TextureData, TexturesData, TextureTransformData } from './textures.ts';

/**
 * This type accepts indexed or unindexed geometry from `@typegpu/kit/geometry` with the standard
 * vertex struct.
 */
export type Shape = meshes.Geometry<typeof meshes.Vertex>;

type Rgb = [r: number, g: number, b: number];

/**
 * These factors set the default instance material. Shapes use white chalk; model meshes use
 * their material factors. Mesh textures multiply the factors.
 */
export interface Factors {
  /** Holds colour in linear colour space. */
  readonly color: Rgb;
  readonly roughness: number;
  readonly metallic: number;
  /** Holds emitted light in linear colour space. */
  readonly emission: Rgb;
  readonly opacity: number;
}

/** This type describes the vertex and index ranges of a mesh in its pool. */
export interface Mesh {
  /** Gives the pool index stored by instances. */
  readonly index: number;
  readonly firstIndex: number;
  readonly indexCount: number;
  readonly baseVertex: number;
  readonly vertexCount: number;
  readonly material: Factors;
  /** Indicates whether the mesh has joints and weights for a kind with `pose`. */
  readonly skinned: boolean;
  /**
   * Indicates whether back faces draw and receive light from their side, as specified by glTF
   * `doubleSided`.
   */
  readonly doubleSided: boolean;
  /**
   * Indicates whether the material uses glTF `alpha: 'mask'` to discard pixels below its alpha
   * cutoff.
   */
  readonly masked: boolean;
  /**
   * Indicates whether the mesh reads textures because its material names one or masks alpha.
   * Shapes are untextured.
   */
  readonly textured: boolean;
}

/** This model contains one pooled mesh per material. */
export interface Model {
  readonly meshes: readonly Mesh[];
  /**
   * Resolves when all textures have decoded and uploaded, at once for a model whose materials
   * name none. Until then, meshes draw without textures. Rejects if image decoding fails or the
   * pool is destroyed before upload. A destroyed pool uploads nothing.
   */
  readonly ready: Promise<void>;
}

/** This type describes the glTF metallic-roughness material returned by `gltf.read`. */
export interface MaterialData {
  /** Names the material, as in its file or `''` from `gltf.read`. The pool ignores it. */
  readonly name?: string;
  /** Holds linear colour channels and alpha. */
  readonly color: readonly [r: number, g: number, b: number, a: number];
  readonly roughness: number;
  readonly metallic: number;
  /** Holds emitted light in linear colour space. */
  readonly emission: readonly [r: number, g: number, b: number];
  readonly textures: TexturesData;
  /** Scales the x and y components of the normal texture. Default 1. */
  readonly normalScale?: number;
  /** Sets how much the occlusion texture darkens, from 0 for none to 1 for all of it. Default 1. */
  readonly occlusionStrength?: number;
  /**
   * Selects glTF alpha handling. `'mask'` discards pixels when colour alpha times texture alpha
   * is below `alphaCutoff`. Renderers treat `'blend'` as `'opaque'` unless their own blending
   * option is enabled. Default `'opaque'`.
   */
  readonly alpha?: 'opaque' | 'mask' | 'blend';
  /** Sets the alpha threshold below which masked pixels are discarded. Default 0.5. */
  readonly alphaCutoff?: number;
  /** Enables back-face rendering. Default false. */
  readonly doubleSided?: boolean;
}

/** This type accepts a 32-bit scalar or vector for an extra vertex attribute. */
export type ExtraType =
  | d.F32
  | d.U32
  | d.I32
  | d.Vec2f
  | d.Vec3f
  | d.Vec4f
  | d.Vec2u
  | d.Vec3u
  | d.Vec4u
  | d.Vec2i
  | d.Vec3i
  | d.Vec4i;

/** These names and types declare extra pool attributes, for example `{ tint: d.vec4f }`. */
export type Extras = Record<string, ExtraType>;

type ArrayOf<T> = T extends d.F32 | d.Vec2f | d.Vec3f | d.Vec4f
  ? Float32Array
  : T extends d.U32 | d.Vec2u | d.Vec3u | d.Vec4u
    ? Uint32Array
    : Int32Array;

/**
 * These arrays hold extra attributes in vertex order. A `d.vec4f` needs four floats per vertex.
 * The pool keeps attributes named in `extra` and drops the others.
 */
export type ExtraData<E extends Extras = {}> = { readonly [K in keyof E]?: ArrayOf<E[K]> } & {
  readonly [name: string]: Float32Array | Uint32Array | Int32Array | undefined;
};

/**
 * These functions read extra attributes for vertex `i`, as `vertexAt` reads standard attributes.
 */
export type ExtraReaders<E extends Extras> = {
  readonly [K in keyof E]: (i: number) => d.InferGPU<E[K]>;
};

/** This type describes mesh arrays returned by `gltf.read`. */
export interface MeshData<E extends Extras = {}> {
  /** Holds three position floats per vertex. */
  readonly positions: Float32Array;
  /** Holds three normal floats per vertex. */
  readonly normals: Float32Array;
  /** Holds two texture coordinates per vertex. */
  readonly uvs: Float32Array;
  /**
   * Holds a second uv set, two coordinates per vertex, such as glTF's `TEXCOORD_1` for a lightmap.
   * Textures with `uvSet: 1` read it. A pool keeps it only when created with `uvs1`. Default none.
   */
  readonly uvs1?: Float32Array;
  /** Holds three indices per triangle, counter-clockwise when viewed from the front. */
  readonly indices: Uint32Array;
  readonly material: MaterialData;
  /**
   * Holds four joint indices per vertex. Indices refer to the skin joints and cannot exceed 255.
   */
  readonly joints?: Uint16Array;
  /** Holds four weights per vertex, corresponding to `joints` and summing to 1. */
  readonly weights?: Float32Array;
  /**
   * Supplies extra vertex attributes, such as glTF vertex colours in `color`. Default none,
   * which reads as zero in pools that keep these attributes.
   */
  readonly extra?: ExtraData<E>;
}

/**
 * This type describes a model returned by `gltf.read`, with one mesh per material and its
 * images.
 */
export interface ModelData<E extends Extras = {}> {
  /** Lists the meshes, each with its material. */
  readonly meshes: readonly MeshData<E>[];
  /** Holds the images that material textures reference by index. */
  readonly images: readonly ImageSource[];
}

export interface Options<E extends Extras = {}> {
  /** Limits the number of meshes to at most 256. */
  readonly capacity: number;
  /**
   * Declares custom vertex attributes, for example `{ tint: d.vec4f }`. Model meshes supply
   * their data in `extra`; shaders read it through `pool.extra.tint(i)`. These attributes use a
   * separate buffer with one struct per vertex. Shapes read as zero. Default none.
   */
  readonly extra?: E;
  /** Limits the total vertex count across all meshes. Default 65 536. */
  readonly vertices?: number;
  /** Limits the total index count across all meshes. Default 262 144. */
  readonly indices?: number;
  /**
   * Keeps each mesh's second uv set, `uvs1`, for textures with `uvSet: 1`, in 8 more bytes per
   * vertex. Shapes and meshes without one read zero. Default false.
   */
  readonly uvs1?: boolean;
  /**
   * Decodes encoded images when `createImageBitmap` is unavailable. The pool decodes Radiance HDR,
   * `images.HDR`, itself on any thread. On React Native,
   * react-native-webgpu supplies it only on the JS thread. Worklet runtimes, including hosted
   * scenes, need this function or images decoded already, as bitmaps from `images.decodeBitmap` on
   * the JS thread or as `{ width, height, pixels }`. Default `createImageBitmap`.
   * @cpu
   */
  readonly decode?: Decode;
  /**
   * Sets the texture sample count at grazing angles, up to 16. More samples sharpen distant
   * surfaces at the cost of memory reads. A sampler with nearest filtering uses one sample.
   * Default 1.
   */
  readonly anisotropy?: number;
  /**
   * Builds the bake of each shape now, for `initAsync` to compile without blocking, so adding one
   * of them later compiles nothing. Default none.
   */
  readonly shapes?: readonly Shape[];
}

export interface MeshPool<E extends Extras = {}> extends Disposable {
  /** Lists added meshes by pool index. */
  readonly meshes: readonly Mesh[];
  /** Gives the maximum mesh count. */
  readonly capacity: number;
  /** Holds all mesh indices, relative to each mesh `baseVertex`. */
  readonly indices: TgpuBuffer<d.WgslArray<d.U32>> & IndexFlag & StorageFlag;
  /**
   * Bakes a shape into the next free ranges in a compute pass. Given a frame or an encoder, it
   * records there; otherwise, it submits at once. The bake's pipeline inlines the shape's
   * `vertexAt` and `indexAt`, and compiles, blocking, the first time the pool bakes those
   * functions, unless `initAsync` compiled it for the `shapes` option. Throws when the pool is
   * full.
   */
  add(shape: Shape, target?: Frame | TgpuCommandEncoder): Mesh;
  /**
   * Uploads each mesh into the next free ranges and decodes images in the background. Each image
   * decodes once per pool and format, and pixels and bitmaps upload during the call. The first skinned mesh allocates a joint-and-weight buffer
   * for the full vertex capacity. Throws if the pool is full, the model has no meshes, its
   * arrays disagree or encoded images other than HDR have no decoder.
   */
  add(model: ModelData<E>): Model;
  /**
   * Reads the attributes named in `extra` for pool vertex `i`. Create calling shaders from
   * `root.pipe(pool.inject())`.
   */
  readonly extra: ExtraReaders<E>;
  /** Fills mesh resources for pipelines created from the configuration you pipe it into. */
  inject(): Inject;
  /**
   * Compiles the bakes of the `shapes` option. Add those shapes once it resolves, as TypeGPU
   * throws for a pipeline still compiling. Resolves at once without them.
   */
  initAsync(): Promise<void>;
  /**
   * Returns the texture bind group for mesh `index`. Bind it in the pass before calling
   * `materials.textured`.
   */
  texturesOf(index: number): TgpuBindGroup<(typeof layout)['entries']>;
  /** Destroys the buffers and textures. */
  destroy(): void;
}

const WORKGROUP = 64;

const boxes = workgroup.bounds(d.vec3f);
const enclose = workgroup.reduce(boxes, WORKGROUP);

const empty = unfilled("the mesh pool's vertices", 'pool', 'a `meshPool.create`');

const positions = tgpu.accessor(d.arrayOf(d.f32), empty);
const attributes = tgpu.accessor(d.arrayOf(d.vec2u), empty);
/**
 * Each vertex keeps its second uv set as two full floats, in a pool made with `uvs1`, because half
 * floats miss lightmap texels by up to one at 4096 texels.
 */
const secondUvs = tgpu.accessor(d.arrayOf(d.vec2f), empty);
/** Each vertex packs four byte-sized joint indices and four `unorm8x4` weights into two words. */
const influences = tgpu.accessor(d.arrayOf(d.vec2u), empty);
/**
 * Each mesh stores six floats in `ordered` bit form, with minimum x, y and z followed by maximum
 * x, y and z.
 */
const extents = tgpu.accessor(d.arrayOf(d.u32), empty);

/** Where one bake writes: its mesh's ranges and counts, and the first word of its box. */
const Bake = d.struct({
  baseVertex: d.u32,
  firstIndex: d.u32,
  vertexCount: d.u32,
  indexCount: d.u32,
  box: d.u32,
});
const baking = tgpu.bindGroupLayout({ bake: { uniform: Bake } }).$name('baking');

/** This schema describes a mesh-space box by its centre and half-size along each axis. */
export const Box = d.struct({ center: d.vec3f, half: d.vec3f });

/** This schema holds four joint indices and their weights, which sum to 1. */
export const Influences = d.struct({ joints: d.vec4u, weights: d.vec4f });

const CHALK: Factors = {
  color: [1, 1, 1],
  roughness: 1,
  metallic: 0,
  emission: [0, 0, 0],
  opacity: 1,
};

const UNIFORM = { skinned: false, doubleSided: false, masked: false, textured: false } as const;

const half = (() => {
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);

  return (value: number) => {
    float[0] = value;
    const x = bits[0];
    const sign = (x >>> 16) & 0x8000;
    const exponent = ((x >>> 23) & 0xff) - 112;
    const mantissa = x & 0x7fffff;

    if (exponent >= 31) return sign | 0x7c00;
    if (exponent <= 0) {
      if (exponent < -10) return sign;
      return sign | ((((mantissa | 0x800000) >>> (1 - exponent)) + 0x1000) >>> 13);
    }

    return sign | ((exponent << 10) + ((mantissa + 0x1000) >>> 13));
  };
})();

const snorm16 = (value: number) =>
  Math.floor(0.5 + 32767 * Math.min(Math.max(value, -1), 1)) & 0xffff;

function layoutOf(type: ExtraType) {
  const components = 'componentCount' in type ? type.componentCount : 1;
  const scalar = 'primitive' in type ? type.primitive.type : type.type;
  const array = scalar === 'f32' ? Float32Array : scalar === 'u32' ? Uint32Array : Int32Array;
  return { components, array };
}

/**
 * Validate indices before upload because an out-of-range index could draw vertices from another
 * mesh.
 */
function check(mesh: MeshData<Extras>, i: number, extra: Extras, keepsUvs1: boolean) {
  const vertexCount = mesh.positions.length / 3;
  const what = `Mesh ${i} of the model`;

  if (!Number.isInteger(vertexCount)) {
    throw new Error(
      `${what} has ${mesh.positions.length} position floats. Supply three per vertex.`,
    );
  }
  if (mesh.normals.length !== vertexCount * 3) {
    throw new Error(
      `${what} has ${mesh.normals.length} normal floats for ${vertexCount} vertices. Supply three per vertex.`,
    );
  }
  if (mesh.uvs.length !== vertexCount * 2) {
    throw new Error(
      `${what} has ${mesh.uvs.length} UV floats for ${vertexCount} vertices. Supply two per vertex.`,
    );
  }
  if (mesh.uvs1 && mesh.uvs1.length !== vertexCount * 2) {
    throw new Error(
      `${what} has ${mesh.uvs1.length} uvs1 floats for ${vertexCount} vertices. Supply two per vertex.`,
    );
  }
  for (const [slot, texture] of Object.entries(mesh.material.textures)) {
    if (texture?.uvSet !== 1) continue;
    const reads = `${what} reads uv set 1 in its ${slot} texture`;
    if (!mesh.uvs1) throw new Error(`${reads}. Supply \`uvs1\` on the mesh.`);
    if (!keepsUvs1) throw new Error(`${reads}. Create the pool with \`uvs1: true\` to keep it.`);
  }

  for (const [name, type] of Object.entries(extra)) {
    const data = mesh.extra?.[name];
    if (!data) continue;
    const { components, array } = layoutOf(type);
    if (!(data instanceof array)) {
      throw new Error(
        `${what} supplies extra '${name}' as ${data.constructor.name}. Use ${array.name} for the pool type ${type.type}.`,
      );
    }
    if (data.length !== vertexCount * components) {
      throw new Error(
        `${what} has ${data.length} values for extra '${name}' and ${vertexCount} vertices. Supply ${components} per vertex.`,
      );
    }
  }

  let largest = -1;
  for (let at = 0; at < mesh.indices.length; at++) largest = Math.max(largest, mesh.indices[at]);
  if (largest >= vertexCount) {
    throw new Error(`${what} has an index of ${largest}, past its ${vertexCount} vertices.`);
  }

  const { joints, weights } = mesh;
  if (!joints !== !weights) {
    throw new Error(`${what} has joints without weights, or weights without joints.`);
  }
  if (!joints || !weights) return;
  if (joints.length !== vertexCount * 4 || weights.length !== vertexCount * 4) {
    throw new Error(
      `${what} has ${joints.length} joints and ${weights.length} weights for ${vertexCount} vertices. Supply four of each per vertex.`,
    );
  }
  const last = joints.reduce((most, joint) => Math.max(most, joint), 0);
  if (last > 255) {
    throw new Error(`${what} is bound to joint ${last}, and a pool keeps 256 joints at most.`);
  }
  for (let v = 0; v < vertexCount; v++) {
    const at = v * 4;
    const w0 = weights[at];
    const w1 = weights[at + 1];
    const w2 = weights[at + 2];
    const w3 = weights[at + 3];
    const sum = w0 + w1 + w2 + w3;
    if (!(w0 >= 0 && w1 >= 0 && w2 >= 0 && w3 >= 0 && Math.abs(sum - 1) < 1e-3)) {
      throw new Error(
        `${what}'s vertex ${v} has weights ${w0}, ${w1}, ${w2}, ${w3}, which don't add up to 1.`,
      );
    }
  }
}

/** Pack four weights in 255ths and preserve their total of 255 so unpacked weights sum to 1. */
function packInfluences(joints: Uint16Array, weights: Float32Array) {
  const out = new Uint32Array(joints.length / 2);
  const shares = [0, 0, 0, 0];
  const behind = [0, 0, 0, 0];

  for (let i = 0; i < joints.length / 4; i++) {
    // Distribute remaining 255ths to weights with the largest rounding loss to preserve the
    // total
    let left = 255;
    for (let k = 0; k < 4; k++) {
      const exact = weights[i * 4 + k] * 255;
      shares[k] = Math.floor(exact);
      behind[k] = exact - shares[k];
      left -= shares[k];
    }
    for (; left > 0; left--) {
      const most = behind.indexOf(Math.max(...behind));
      shares[most] = shares[most] + 1;
      behind[most] = -1;
    }

    const at = i * 4;
    out[i * 2] =
      (joints[at] | (joints[at + 1] << 8) | (joints[at + 2] << 16) | (joints[at + 3] << 24)) >>> 0;
    out[i * 2 + 1] = (shares[0] | (shares[1] << 8) | (shares[2] << 16) | (shares[3] << 24)) >>> 0;
  }
  return out;
}

interface Field {
  readonly name: string;
  /** Measures the offset in 4-byte words. */
  readonly offset: number;
  readonly components: number;
}

function packExtras(mesh: MeshData<Extras>, fields: readonly Field[], stride: number) {
  const vertexCount = mesh.positions.length / 3;
  const words = new Uint32Array((vertexCount * stride) / 4);

  for (const { name, offset, components } of fields) {
    const data = mesh.extra?.[name];
    if (!data) continue;
    const bits = new Uint32Array(data.buffer, data.byteOffset, data.length);
    for (let v = 0; v < vertexCount; v++) {
      const at = (v * stride) / 4 + offset;
      for (let c = 0; c < components; c++) words[at + c] = bits[v * components + c];
    }
  }
  return words;
}

function typeFor(data: Float32Array | Uint32Array | Int32Array, count: number) {
  const scalar = data instanceof Float32Array ? 'f' : data instanceof Uint32Array ? 'u' : 'i';
  const components = data.length / count;
  return components === 1 ? `d.${scalar}32` : `d.vec${components}${scalar}`;
}

/** Reports whether every component is 1 within 1/255, colours that multiply nothing. */
function allWhite(data: Float32Array | Uint32Array | Int32Array) {
  if (!(data instanceof Float32Array)) return false;
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i] - 1) > 1 / 255) return false;
  }
  return true;
}

/** Initialize minima above all vertex values and maxima below them for atomic bounds updates. */
const EMPTY_BOX = new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0]);

function boxOf(points: Float32Array) {
  const least = [Infinity, Infinity, Infinity];
  const most = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < points.length; i++) {
    const axis = i % 3;
    least[axis] = Math.min(least[axis], points[i]);
    most[axis] = Math.max(most[axis], points[i]);
  }
  return Uint32Array.from([...least, ...most], orderedOnCpu);
}

function pack(normals: Float32Array, uvs: Float32Array) {
  const out = new Uint32Array((normals.length / 3) * 2);

  for (let i = 0; i < normals.length / 3; i++) {
    const x = normals[i * 3];
    const y = normals[i * 3 + 1];
    const z = normals[i * 3 + 2];
    const length = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
    let u = x / length;
    let v = y / length;
    if (z < 0) {
      const folded = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1);
      v = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1);
      u = folded;
    }

    out[i * 2] = snorm16(u) | (snorm16(v) << 16);
    out[i * 2 + 1] = half(uvs[i * 2]) | (half(uvs[i * 2 + 1]) << 16);
  }
  return out;
}

/**
 * Reads only the position of pool vertex `i`, for depth-only passes. It uses the same indexing
 * and injection as `vertexAt`.
 */
export const positionAt = (i: number) => {
  'use gpu';
  const at = i * 3;
  return d.vec3f(positions.$[at], positions.$[at + 1], positions.$[at + 2]);
};

/**
 * Decodes pool vertex `i` into an `@typegpu/kit/geometry` vertex. In a mesh draw, the vertex index
 * already includes `baseVertex`. Create calling shaders from `root.pipe(pool.inject())`.
 */
export const vertexAt = (i: number) => {
  'use gpu';
  const packed = attributes.$[i];

  return meshes.Vertex({
    position: positionAt(i),
    normal: fromOctahedral(std.unpack2x16snorm(packed.x)),
    uv: std.unpack2x16float(packed.y),
  });
};

/**
 * Reads the second uv set of pool vertex `i`, from a mesh's `uvs1`, or zero in a pool made
 * without `uvs1`. It uses the same indexing and injection as `vertexAt`.
 */
export const uv1At = (i: number) => {
  'use gpu';
  if (keepsSecondSet.$) return d.vec2f(secondUvs.$[i]);
  return d.vec2f();
};

/**
 * Reads joints and weights for skinned vertex `i`. Create calling shaders from
 * `root.pipe(pool.inject())` with a pool that contains a skinned model.
 */
export const influencesAt = (i: number) => {
  'use gpu';
  const packed = influences.$[i];
  const joints = packed.x;
  return Influences({
    joints: d.vec4u(joints & 0xff, (joints >>> 8) & 0xff, (joints >>> 16) & 0xff, joints >>> 24),
    weights: std.unpack4x8unorm(packed.y),
  });
};

/**
 * Returns the bounding box of mesh `index` in mesh space. Shapes compute it during baking;
 * models compute it when added. Create calling shaders from `root.pipe(pool.inject())`.
 */
export const boundsOf = (index: number) => {
  'use gpu';
  const at = d.u32(index) * 6;
  const low = d.vec3f(
    unordered(extents.$[at]),
    unordered(extents.$[at + 1]),
    unordered(extents.$[at + 2]),
  );
  const high = d.vec3f(
    unordered(extents.$[at + 3]),
    unordered(extents.$[at + 4]),
    unordered(extents.$[at + 5]),
  );

  return Box({ center: (low + high) * 0.5, half: (high - low) * 0.5 });
};

function extrasOf(root: TgpuRoot, extra: Extras, vertices: number) {
  const names = Object.keys(extra);
  if (names.length === 0) return undefined;

  const Extra = d.struct(extra);
  const data = root.createBuffer(d.arrayOf(Extra, vertices)).$usage('storage');
  const read = tgpu.accessor(d.arrayOf(Extra), empty);
  const readerOf = (name: string) => {
    const reader = (i: number) => {
      'use gpu';
      return std.copy(read.$[i][name]);
    };
    return reader;
  };
  const readers = Object.fromEntries(names.map((name) => [name, readerOf(name)]));

  return {
    data,
    readers,
    /** Gives bytes per vertex. */
    stride: d.sizeOf(d.arrayOf(Extra, 2)) / 2,
    fields: names.map((name): Field => ({
      name,
      offset: d.memoryLayoutOf(Extra, (vertex) => vertex[name]).offset / 4,
      components: layoutOf(extra[name]).components,
    })),
    inject: ((cfg) => cfg.with(read, data.as('readonly'))) satisfies Inject,
  };
}

/** Counts room in a pool: meshes, vertices and indices. */
export interface Room {
  readonly meshes: number;
  readonly vertices: number;
  readonly indices: number;
}

/**
 * Holds a pool's `capacity`, `vertices` and `indices`, and `uvs1` when a mesh carries a second uv
 * set, to spread into its options.
 */
export interface Sized {
  /** Gives the number of meshes to hold. */
  readonly capacity: number;
  /** Gives the number of vertices to hold. */
  readonly vertices: number;
  /** Gives the number of indices to hold. */
  readonly indices: number;
  /** Keeps a second uv set in the pool, present when a mesh carries one. */
  readonly uvs1?: true;
}

/**
 * Sizes a pool to hold exactly `items`: models such as `impostors.CARD`, their meshes, shapes, and
 * counts. Spread it into the options: `meshPool.create(root, meshPool.roomFor([...]))`. It keeps a
 * second uv set when one of the meshes carries `uvs1`.
 *
 * @param items what the pool will hold
 * @returns the capacity, vertices and indices they take, and `uvs1: true` when a mesh needs it
 */
export function roomFor(items: readonly (ModelData | MeshData | Shape | Room)[]): Sized {
  let capacity = 0;
  let vertices = 0;
  let indices = 0;
  let uvs1 = false;
  const count = (mesh: MeshData) => {
    capacity += 1;
    vertices += mesh.positions.length / 3;
    indices += mesh.indices.length;
    uvs1 ||= mesh.uvs1 !== undefined;
  };
  for (const item of items) {
    if ('positions' in item) count(item);
    else if ('vertexCount' in item) {
      capacity += 1;
      vertices += item.vertexCount;
      indices += meshes.isIndexed(item) ? item.indexCount : item.vertexCount;
    } else if ('vertices' in item) {
      capacity += item.meshes;
      vertices += item.vertices;
      indices += item.indices;
    } else for (const mesh of item.meshes) count(mesh);
  }
  return uvs1 ? { capacity, vertices, indices, uvs1 } : { capacity, vertices, indices };
}

/**
 * Creates shared vertex storage and an index buffer, with bounds and textures per mesh. One
 * pipeline can draw any mesh. Each vertex uses 12 bytes for position and 8 for normal and UV
 * coordinates, and 8 for a second uv set with `uvs1`. Skinned models add 8 bytes for joints and
 * weights. Extras add the size of their struct per vertex.
 *
 * @unstable
 * @injects definition the vertices `vertexAt`, `positionAt` and `uv1At` read, the joints and
 * weights `influencesAt` reads, and the boxes `boundsOf` reads
 * @injects member the extras `pool.extra` reads
 * @binds 1 storage: `positionAt`, the positions
 * @binds 2 storage: `vertexAt`, the positions and the normals and uvs
 * @binds 1 storage: `uv1At`, the second uvs, in a pool made with `uvs1`
 * @binds 1 storage: `influencesAt`, the joints and weights
 * @binds 1 storage: `boundsOf`, every mesh's box
 * @binds 1 storage: `pool.extra`, every vertex's extras
 * @binds 1 uniform: `pool.texturesOf(index)` for `materials`, beside its textures and sampler, in
 * the fragment stage
 * @submits setup a shape added without an encoder bakes in a compute pass of its own, submitted at
 * once, and a model's images generate their mips as they decode
 * @owns 5 to 7 buffers, a uniform per way of reading textures, a uniform per shape added, a
 * pipeline per pair of shape functions, textures, samplers, bind groups, what `images` makes:
 * positions, normals and uvs, indices and boxes, second uvs with `uvs1`, joints and weights once a
 * skinned model comes, and the extras when `extra` names any; a bake per `vertexAt` and `indexAt`
 * it bakes, and per shape added the uniform of its ranges and their group; each image decoded once
 * per format it is read in, and a group per set of a mesh's textures
 * @returns a pool you add meshes and models to and destroy
 */
export function create<E extends Extras = {}>(root: TgpuRoot, options: Options<E>): MeshPool<E> {
  const { vertices = 1 << 16, indices = 1 << 18, capacity } = options;
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > 256) {
    throw new RangeError(`A mesh pool holds 1 to 256 meshes, got ${capacity}.`);
  }

  const extra: Extras = options.extra ?? {};
  const kept = extrasOf(root, extra, vertices);

  const positionData = root.createBuffer(d.arrayOf(d.f32, vertices * 3)).$usage('storage');
  const attributeData = root.createBuffer(d.arrayOf(d.vec2u, vertices)).$usage('storage');
  const keepsUvs1 = options.uvs1 ?? false;
  const uvs1Data = keepsUvs1
    ? root.createBuffer(d.arrayOf(d.vec2f, vertices)).$usage('storage').$name('uvs1Data')
    : undefined;
  const indexData = root.createBuffer(d.arrayOf(d.u32, indices)).$usage('index', 'storage');
  // Use atomics to update bounds and plain words to read the same memory in shaders
  const extentData = root.createBuffer(d.arrayOf(d.atomic(d.u32), capacity * 6)).$usage('storage');
  // Views the memory of extentData, which frees it
  const extentWords = root
    .createBuffer(d.arrayOf(d.u32, capacity * 6), root.unwrap(extentData))
    .$usage('storage');

  const written = {
    positions: positionData.as('mutable'),
    attributes: attributeData.as('mutable'),
    indices: indexData.as('mutable'),
    extents: extentData.as('mutable'),
  };
  // Allocate influences for the full vertex capacity before resolving shaders that read poses
  let influenceData: (TgpuBuffer<d.WgslArray<d.Vec2u>> & StorageFlag) | undefined;
  let influenceBinding: TgpuReadonly<d.WgslArray<d.Vec2u>> | undefined;
  const influenced = () => {
    if (!influenceBinding) {
      throw new Error(
        'The mesh pool has no skinned model. Add one before compiling pipelines that read joints.',
      );
    }
    return influenceBinding;
  };

  const filled: Inject = (cfg) => {
    const pool = cfg
      .with(positions, positionData.as('readonly'))
      .with(attributes, attributeData.as('readonly'))
      .with(influences, influenced)
      .with(extents, extentWords.as('readonly'))
      // Fill the switch either way, so a pool without the set undoes one piped before it
      .with(keepsSecondSet, uvs1Data !== undefined);
    return uvs1Data ? pool.with(secondUvs, uvs1Data.as('readonly')) : pool;
  };
  const inject: Inject = kept ? (cfg) => kept.inject(filled(cfg)) : filled;

  const anisotropy = Math.min(Math.max(Math.round(options.anisotropy ?? 1), 1), 16);
  const textures = createTextures(root, options.decode, anisotropy);
  const groups: TgpuBindGroup<(typeof layout)['entries']>[] = [];

  const list: Mesh[] = [];
  let usedVertices = 0;
  let usedIndices = 0;
  const dropped = new Set<string>();

  const reserve = (counts: readonly [number, number][], what: string) => {
    const vertexCount = counts.reduce((sum, [each]) => sum + each, 0);
    const indexCount = counts.reduce((sum, [, each]) => sum + each, 0);

    if (list.length + counts.length > capacity) {
      throw new Error(`The mesh pool is full (${capacity} meshes).`);
    }
    if (usedVertices + vertexCount > vertices || usedIndices + indexCount > indices) {
      throw new Error(
        `The mesh pool has room for ${vertices - usedVertices} vertices and ` +
          `${indices - usedIndices} indices. ${what} needs ${vertexCount} vertices and ${indexCount} indices.`,
      );
    }
  };

  const take = (
    vertexCount: number,
    indexCount: number,
    material: Factors,
    looks: Pick<Mesh, 'skinned' | 'doubleSided' | 'masked' | 'textured'> = UNIFORM,
  ): Mesh => {
    const mesh = {
      index: list.length,
      firstIndex: usedIndices,
      indexCount,
      baseVertex: usedVertices,
      vertexCount,
      material,
      ...looks,
    };

    list.push(mesh);
    groups.push(textures.fallback);
    usedVertices += vertexCount;
    usedIndices += indexCount;
    return mesh;
  };

  const { queue } = root.device;
  const gpu = {
    positions: root.unwrap(positionData),
    attributes: root.unwrap(attributeData),
    indices: root.unwrap(indexData),
    extents: root.unwrap(extentData),
  };

  const kernelOf = (vertexOf: Shape['vertexAt'], indexOf?: (i: number) => number) =>
    tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationIndex },
    })(({ gid, lid }) => {
      'use gpu';
      const bake = baking.$.bake;
      const i = gid.x;
      let point = boxes.identity();
      if (i < bake.vertexCount) {
        const vertex = vertexOf(i);
        const at = (bake.baseVertex + i) * 3;
        written.positions.$[at] = vertex.position.x;
        written.positions.$[at + 1] = vertex.position.y;
        written.positions.$[at + 2] = vertex.position.z;

        const normal = std.pack2x16snorm(octahedral(std.normalize(vertex.normal)));
        written.attributes.$[bake.baseVertex + i] = d.vec2u(normal, std.pack2x16float(vertex.uv));

        point = boxes.schema({ low: vertex.position, high: vertex.position });
      }

      if (i < bake.indexCount) {
        written.indices.$[bake.firstIndex + i] = d.u32(indexOf === undefined ? i : indexOf(i));
      }

      const around = enclose(point, lid);
      if (lid < 3) {
        std.atomicMin(written.extents.$[bake.box + lid], ordered(around.low[lid]));
        std.atomicMax(written.extents.$[bake.box + 3 + lid], ordered(around.high[lid]));
      }
    });

  // The functions a bake inlines key its pipeline, weakly, so a shape lives no longer for it
  const bakes = new WeakMap<object, Map<object | undefined, TgpuComputePipeline>>();
  const bakeOf = (shape: Shape) => {
    const indexAt = meshes.isIndexed(shape) ? shape.indexAt : undefined;
    let byIndex = bakes.get(shape.vertexAt);
    if (!byIndex) bakes.set(shape.vertexAt, (byIndex = new Map()));
    let pipeline = byIndex.get(indexAt);
    if (!pipeline) {
      pipeline = root.createComputePipeline({ compute: kernelOf(shape.vertexAt, indexAt) });
      byIndex.set(indexAt, pipeline);
    }
    return pipeline;
  };
  const ahead = (options.shapes ?? []).map(bakeOf);

  const addShape = (shape: Shape, into?: Frame | TgpuCommandEncoder): Mesh => {
    const indexed = meshes.isIndexed(shape);
    const { vertexCount } = shape;
    const indexCount = indexed ? shape.indexCount : vertexCount;

    if (shape.topology !== 'triangle-list') {
      throw new Error(`A mesh pool holds triangle lists, got ${shape.topology}.`);
    }
    reserve([[vertexCount, indexCount]], 'The shape');

    const pipeline = bakeOf(shape);
    const mesh = take(vertexCount, indexCount, CHALK);
    const { baseVertex, firstIndex } = mesh;
    const box = mesh.index * 6;
    queue.writeBuffer(gpu.extents, box * 4, EMPTY_BOX);
    const ranges = root
      .createBuffer(Bake, { baseVertex, firstIndex, vertexCount, indexCount, box })
      .$usage('uniform');

    const encoder = into ? encoderOf(into) : root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(root.createBindGroup(baking, { bake: ranges }));
    pass.dispatchWorkgroups(Math.ceil(Math.max(vertexCount, indexCount) / WORKGROUP));
    pass.end();
    if (!into) encoder.submit();

    return mesh;
  };

  const addModel = (model: ModelData<Extras>): Model => {
    if (model.meshes.length === 0) throw new Error('The model has no meshes.');
    model.meshes.forEach((each, i) => check(each, i, extra, keepsUvs1));
    if (
      !options.decode &&
      !decodesItself() &&
      model.images.some((image) => 'bytes' in image && image.mimeType !== HDR)
    ) {
      throw new Error(
        'This platform lacks `createImageBitmap`. Pass a `decode` function to the mesh pool, ' +
          'or supply decoded images with the model. Use bitmaps from `images.decodeBitmap` on a thread ' +
          'with `createImageBitmap`, or use `{ width, height, pixels }`.',
      );
    }
    for (const each of model.meshes) {
      for (const [name, data] of Object.entries(each.extra ?? {})) {
        if (!data || name in extra || dropped.has(name)) continue;
        if (name === 'color' && allWhite(data)) continue;
        dropped.add(name);
        console.warn(
          `The mesh pool discards extra '${name}'. Create it with \`extra: { ${name}: ${typeFor(data, each.positions.length / 3)} }\` to keep the attribute.`,
        );
      }
    }

    reserve(
      model.meshes.map((each) => [each.positions.length / 3, each.indices.length]),
      'The model',
    );

    if (!influenceData && model.meshes.some((each) => each.joints)) {
      influenceData = root.createBuffer(d.arrayOf(d.vec2u, vertices)).$usage('storage');
      influenceBinding = influenceData.as('readonly');
    }

    const added = model.meshes.map((each) => {
      const { color, roughness, metallic, emission, doubleSided = false } = each.material;
      const { joints, weights } = each;
      const factors: Factors = {
        color: [color[0], color[1], color[2]],
        roughness,
        metallic,
        emission: [...emission],
        opacity: color[3],
      };
      const mesh = take(each.positions.length / 3, each.indices.length, factors, {
        skinned: !!joints,
        doubleSided,
        masked: each.material.alpha === 'mask',
        textured:
          each.material.alpha === 'mask' || Object.values(each.material.textures).some(Boolean),
      });

      queue.writeBuffer(gpu.positions, mesh.baseVertex * 12, each.positions);
      queue.writeBuffer(gpu.attributes, mesh.baseVertex * 8, pack(each.normals, each.uvs));
      if (uvs1Data && each.uvs1) {
        queue.writeBuffer(root.unwrap(uvs1Data), mesh.baseVertex * 8, each.uvs1);
      }
      queue.writeBuffer(gpu.indices, mesh.firstIndex * 4, each.indices);
      queue.writeBuffer(gpu.extents, mesh.index * 24, boxOf(each.positions));
      if (joints && weights) {
        const packed = packInfluences(joints, weights);
        queue.writeBuffer(root.unwrap(influenceData!), mesh.baseVertex * 8, packed);
      }
      if (kept && kept.fields.some(({ name }) => each.extra?.[name])) {
        const packed = packExtras(each, kept.fields, kept.stride);
        queue.writeBuffer(root.unwrap(kept.data), mesh.baseVertex * kept.stride, packed);
      }
      return mesh;
    });

    // A material that names no texture binds its group at once, so its model is ready at once
    const uploading = model.meshes.flatMap((each, i) => {
      const { textures: data, normalScale = 1, occlusionStrength = 1 } = each.material;
      const { alpha, alphaCutoff = 0.5 } = each.material;
      const index = added[i].index;
      const group = textures.groupOf(data, model.images, {
        normalScale,
        occlusionStrength,
        alphaCutoff: alpha === 'mask' ? alphaCutoff : 0,
      });
      if (!(group instanceof Promise)) {
        groups[index] = group;
        return [];
      }
      return [
        group.then((uploaded) => {
          groups[index] = uploaded;
        }),
      ];
    });
    const ready = Promise.all(uploading).then(() => {});

    return { meshes: added, ready };
  };

  function add(shape: Shape, target?: Frame | TgpuCommandEncoder): Mesh;
  function add(model: ModelData<E>): Model;
  function add(source: Shape | ModelData<E>, target?: Frame | TgpuCommandEncoder) {
    return 'meshes' in source ? addModel(source) : addShape(source, target);
  }

  const destroy = () => {
    kept?.data.destroy();
    positionData.destroy();
    attributeData.destroy();
    uvs1Data?.destroy();
    indexData.destroy();
    extentData.destroy();
    influenceData?.destroy();
    textures.destroy();
  };

  return {
    meshes: list,
    capacity,
    indices: indexData,

    add,

    extra: (kept?.readers ?? {}) as ExtraReaders<E>,

    inject: () => inject,

    initAsync: () => pipelines.compile(ahead),

    texturesOf(index) {
      const group = groups[index];
      if (!group) throw new Error(`The mesh pool has no mesh ${index}.`);
      return group;
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
