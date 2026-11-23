import type { d } from 'typegpu';
import { mat4d, quatd, vec3d } from 'wgpu-matrix';
import type { EncodedImage } from '../images.ts';
import type {
  MaterialData,
  MeshData,
  ModelData as PoolModelData,
  TextureData,
  TexturesData,
} from '../render/meshPool.ts';
import type { ChannelData, ClipData, JointData, SkinData } from '../render/poses.ts';
import type { Draco, DracoArray, DracoMesh } from './draco.ts';

export {
  draco,
  type Draco,
  type DracoArray,
  type DracoAttribute,
  type DracoMesh,
  type DracoModule,
} from './draco.ts';

type Vec3 = readonly [number, number, number];
type Vec4 = readonly [number, number, number, number];

/**
 * These extra attributes come with glTF meshes: `color` holds `COLOR_0` as four linear RGBA
 * floats per vertex, for a pool with `extra: { color: d.vec4f }`. A model has them when at least
 * one primitive has colours; vertices of other primitives are white.
 */
export type Extra = { color: d.Vec4f };

/**
 * This type contains binary glTF data as arrays ready for `meshPool.add`. Materials, textures and
 * samplers hold every field, with glTF defaults for omitted values.
 */
export interface ModelData<Name extends string = never> extends PoolModelData<Extra> {
  /**
   * Lists meshes grouped by material with node transforms applied. Nodes selected as parts are
   * omitted. With `placements`, each glTF mesh contributes one entry per material in mesh space.
   */
  readonly meshes: readonly MeshData<Extra>[];
  /** Holds copies of the encoded images that material textures reference. */
  readonly images: readonly EncodedImage[];
  /** Contains the requested named nodes and their descendants. */
  readonly parts: { readonly [N in Name]: PartData };
  /**
   * Provides the skin used by meshes in the default scene. When present, every mesh has `joints`
   * and `weights`. An unskinned mesh follows the joint it is attached to.
   */
  readonly skin?: SkinData;
  /**
   * Contains file animations that target nodes by name, so another file with matching joint
   * names can use them.
   */
  readonly clips: readonly ClipData[];
}

/**
 * This part contains a named node and its descendants in node space. Its origin is the node
 * pivot. Scale stays in the vertices, so placement needs only position and rotation. You can
 * pass it to `meshPool.add` as a model.
 *
 * A part holding a skinned mesh binds to the model's `skin` as `meshes` do instead: every mesh
 * has `joints` and `weights`, vertices stay in model space at the bind pose, and its position and
 * rotation are the model's origin. Pose it with the poses of the whole model.
 */
export interface PartData extends PoolModelData<Extra> {
  /** Lists meshes grouped by material in node space. */
  readonly meshes: readonly MeshData<Extra>[];
  /** Holds model images referenced by these mesh textures. */
  readonly images: readonly EncodedImage[];
  /** Gives the node origin in model space. */
  readonly position: Vec3;
  /** Gives node rotation in model space as a unit quaternion `[x, y, z, w]`. */
  readonly rotation: Vec4;
}

/**
 * This placement describes a node transform for an entry of `meshes` when read with
 * `placements`. It splits the model-space transform into instance fields. Shear is discarded.
 */
export interface PlacementData {
  /** Selects an entry in `meshes`. */
  readonly mesh: number;
  readonly position: [x: number, y: number, z: number];
  /** Gives rotation as a unit quaternion `[x, y, z, w]`. */
  readonly rotation: [x: number, y: number, z: number, w: number];
  /** Gives scale per axis. A negative x scale indicates a mirrored mesh. */
  readonly scale: [x: number, y: number, z: number];
}

/** This model stores each mesh once and lists node transforms in `placements`. */
export interface PlacedModelData<Name extends string = never> extends ModelData<Name> {
  /** Lists mesh placements for nodes outside the selected parts, in scene node order. */
  readonly placements: readonly PlacementData[];
}

// Treat these JSON types as expected shapes only. Validate file values before using them

interface TextureInfo {
  index: number;
  texCoord?: number;
  scale?: number;
  strength?: number;
  extensions?: {
    KHR_texture_transform?: {
      offset?: number[];
      rotation?: number;
      scale?: number[];
      texCoord?: number;
    };
  };
}

interface NodeJson {
  name?: string;
  children?: number[];
  mesh?: number;
  skin?: number;
  matrix?: number[];
  translation?: Vec3;
  rotation?: Vec4;
  scale?: Vec3;
}

interface DracoJson {
  bufferView?: number;
  attributes?: Record<string, unknown>;
}

interface PrimitiveJson {
  attributes?: {
    POSITION?: number;
    NORMAL?: number;
    TEXCOORD_0?: number;
    TEXCOORD_1?: number;
    COLOR_0?: number;
    JOINTS_0?: number;
    WEIGHTS_0?: number;
    JOINTS_1?: number;
  };
  indices?: number;
  material?: number;
  mode?: number;
  extensions?: {
    KHR_draco_mesh_compression?: DracoJson;
  };
}

interface Json {
  asset?: { version?: string };
  extensionsRequired?: string[];
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: NodeJson[];
  meshes?: { primitives?: (PrimitiveJson | null)[] }[];
  accessors?: {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    normalized?: boolean;
    count: number;
    type: string;
    sparse?: {
      count: number;
      indices?: { bufferView: number; byteOffset?: number; componentType: number };
      values?: { bufferView: number; byteOffset?: number };
    };
  }[];
  bufferViews?: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
  buffers?: { byteLength: number; uri?: string }[];
  materials?: {
    name?: string;
    pbrMetallicRoughness?: {
      baseColorFactor?: Vec4;
      baseColorTexture?: TextureInfo;
      metallicFactor?: number;
      roughnessFactor?: number;
      metallicRoughnessTexture?: TextureInfo;
    };
    normalTexture?: TextureInfo;
    emissiveTexture?: TextureInfo;
    occlusionTexture?: TextureInfo;
    emissiveFactor?: Vec3;
    alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND';
    alphaCutoff?: number;
    doubleSided?: boolean;
    extensions?: { KHR_materials_emissive_strength?: { emissiveStrength?: number } };
  }[];
  textures?: {
    source?: number;
    sampler?: number;
    extensions?: { EXT_texture_webp?: { source: number }; KHR_texture_basisu?: { source: number } };
  }[];
  samplers?: { magFilter?: number; minFilter?: number; wrapS?: number; wrapT?: number }[];
  images?: { bufferView?: number; mimeType?: string; uri?: string }[];
  skins?: { joints?: number[]; inverseBindMatrices?: number }[];
  animations?: {
    name?: string;
    channels?: { sampler?: number; target?: { node?: number; path?: string } }[];
    samplers?: { input?: number; output?: number; interpolation?: string }[];
  }[];
}

const GLB = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

const UNDERSTOOD = new Set([
  'KHR_materials_emissive_strength',
  'KHR_mesh_quantization',
  'KHR_texture_transform',
  'EXT_texture_webp',
]);

const DRACO = 'KHR_draco_mesh_compression';

// Use maps for file-supplied keys so names such as constructor cannot match object properties

const CONVERSIONS = new Map([
  [DRACO, 'gltf-transform copy in.glb out.glb'],
  ['EXT_meshopt_compression', 'gltf-transform copy in.glb out.glb'],
  ['KHR_texture_basisu', 'gltf-transform ktxdecompress in.glb out.glb'],
]);

const SIZES = new Map([
  [5120, 1],
  [5121, 1],
  [5122, 2],
  [5123, 2],
  [5125, 4],
  [5126, 4],
]);

type Semantic =
  | 'POSITION'
  | 'NORMAL'
  | 'TEXCOORD_0'
  | 'TEXCOORD_1'
  | 'COLOR_0'
  | 'JOINTS_0'
  | 'WEIGHTS_0'
  | 'indices'
  | 'inverseBindMatrices'
  | 'times'
  | 'translation'
  | 'rotation'
  | 'scale';

/**
 * Validate glTF accessor types and normalization requirements, including vertex types allowed by
 * KHR_mesh_quantization.
 */
const SEMANTICS: Record<
  Semantic,
  {
    readonly types: ReadonlyMap<unknown, number>;
    readonly components: readonly number[];
    readonly normalized?: boolean;
  }
> = {
  POSITION: { types: new Map([['VEC3', 3]]), components: [5120, 5121, 5122, 5123, 5126] },
  NORMAL: { types: new Map([['VEC3', 3]]), components: [5120, 5122, 5126] },
  TEXCOORD_0: { types: new Map([['VEC2', 2]]), components: [5120, 5121, 5122, 5123, 5126] },
  TEXCOORD_1: { types: new Map([['VEC2', 2]]), components: [5120, 5121, 5122, 5123, 5126] },
  COLOR_0: {
    types: new Map([
      ['VEC3', 3],
      ['VEC4', 4],
    ]),
    components: [5121, 5123, 5126],
    normalized: true,
  },
  JOINTS_0: { types: new Map([['VEC4', 4]]), components: [5121, 5123], normalized: false },
  WEIGHTS_0: { types: new Map([['VEC4', 4]]), components: [5121, 5123, 5126], normalized: true },
  indices: { types: new Map([['SCALAR', 1]]), components: [5121, 5123, 5125], normalized: false },
  inverseBindMatrices: { types: new Map([['MAT4', 16]]), components: [5126] },
  times: { types: new Map([['SCALAR', 1]]), components: [5126] },
  translation: { types: new Map([['VEC3', 3]]), components: [5126] },
  rotation: {
    types: new Map([['VEC4', 4]]),
    components: [5120, 5121, 5122, 5123, 5126],
    normalized: true,
  },
  scale: { types: new Map([['VEC3', 3]]), components: [5126] },
};

const INTERPOLATIONS = new Map<unknown, ChannelData['interpolation']>([
  ['LINEAR', 'linear'],
  ['STEP', 'step'],
  ['CUBICSPLINE', 'cubic'],
]);

const TRIANGLES = 4;
const STRIP = 5;
const FAN = 6;

const WRAPS = new Map<unknown, GPUAddressMode>([
  [10497, 'repeat'],
  [33648, 'mirror-repeat'],
  [33071, 'clamp-to-edge'],
]);

export interface ReadOptions<Name extends string = never, P extends boolean = boolean> {
  /**
   * Selects named nodes and their descendants as separate `parts` instead of entries in
   * `meshes`. A selected node nested in another selected node becomes a separate part too. A part
   * holding a skinned mesh stays bound to the skin. Default none.
   */
  readonly parts?: readonly Name[];
  /**
   * Rotates the whole model around its origin with a unit quaternion `[x, y, z, w]`. Use this
   * for models that do not face +z as `instances.toward` expects. For example, `[0,
   * -Math.SQRT1_2, 0, Math.SQRT1_2]` turns +x to +z. Parts, skin and clips rotate together; read
   * separate clip files with the same rotation. Default none.
   */
  readonly rotation?: readonly [x: number, y: number, z: number, w: number];
  /**
   * Reads each glTF mesh once in mesh space when true, with one entry per material. Node
   * transforms are returned in `placements` for instancing. Throws for a skinned model. Default
   * `false`, which copies placements and merges them by material.
   */
  readonly placements?: P;
  /**
   * Limits the bytes of the vertex and index arrays the read returns, counted from accessor counts
   * before reading mesh data. This prevents small files from expanding through repeated mesh
   * placement. A Draco decoder's own memory is outside the limit. Default 256 MiB.
   */
  readonly maxBytes?: number;
  /**
   * Decodes primitives compressed with `KHR_draco_mesh_compression`. Make one with `draco` from a
   * Draco decoder module. Default none, which throws for a model that needs Draco.
   * @cpu
   */
  readonly draco?: Draco;
  /**
   * Keeps every primitive's `TEXCOORD_1` as `uvs1`, such as the unwrap of a lightmap the program
   * samples itself. Default `false`, which keeps it only for materials whose textures read it.
   */
  readonly uvs1?: boolean;
}

/**
 * Reads a binary glTF 2.0 file into meshes, selected parts, a skin and animations. It applies
 * default-scene node transforms and groups meshes by material. Morph targets are omitted.
 *
 * Skinned meshes use their bind pose with four joints and weights per vertex. An unskinned mesh
 * attached to a joint, such as a helmet, binds to that joint. An unskinned mesh without a joint
 * ancestor must be read as a part or this throws. A part holding a skinned mesh binds to the skin
 * as `meshes` do; other parts have no skin bindings.
 *
 * It checks every index, count and range before reading. Invalid glTF, unsupported file versions
 * and unsupported required features throw with details. Use `load` for a model whose buffers or
 * images are separate files. For extensions that need conversion,
 * errors name the `gltf-transform` command.
 *
 * @param source the file's bytes, from anywhere
 * @param options.parts names of nodes to keep apart from the rest, each posed about its own origin
 *   or, holding a skinned mesh, by the skin
 * @param options.rotation a turn of the whole model, for one that doesn't face +z
 * @returns plain arrays and images, for `meshPool.add`
 */
export function read<const Name extends string = never>(
  source: ArrayBuffer | ArrayBufferView,
  options: ReadOptions<Name, true> & { readonly placements: true },
): PlacedModelData<Name>;
export function read<const Name extends string = never>(
  source: ArrayBuffer | ArrayBufferView,
  options?: ReadOptions<Name, false>,
): ModelData<Name>;
export function read<const Name extends string = never>(
  source: ArrayBuffer | ArrayBufferView,
  options: ReadOptions<Name> = {},
): ModelData<Name> | PlacedModelData<Name> {
  const bytes = ArrayBuffer.isView(source)
    ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength)
    : new Uint8Array(source);
  const { json, bin } = chunks(bytes);
  return modelOf(json, [json.buffers?.[0]?.uri === undefined ? bin : undefined], [], options);
}

/** Fetches a model and each file it names, given the bytes or the failure. */
export type Fetch = (url: string) => Promise<{
  readonly ok: boolean;
  readonly status: number;
  arrayBuffer(): Promise<ArrayBuffer>;
}>;

export interface LoadOptions<
  Name extends string = never,
  P extends boolean = boolean,
> extends ReadOptions<Name, P> {
  /**
   * Fetches the model and the files beside it, such as a `fetch` that adds headers or reads app
   * assets. Default the global `fetch`.
   * @cpu
   */
  readonly fetch?: Fetch;
}

/**
 * Fetches a `.glb` or `.gltf` with the buffers and images it keeps in separate files, then reads
 * it as `read` does. It resolves each file's URI against `url` and fetches them all at once.
 * Base64 `data:` URIs are decoded without fetching.
 *
 * @param url the model's address; files it names are fetched from beside it
 * @param options.fetch a fetch of your own, for headers or app assets
 * @returns the model `read` returns, for `meshPool.add`
 */
export function load<const Name extends string = never>(
  url: string,
  options: LoadOptions<Name, true> & { readonly placements: true },
): Promise<PlacedModelData<Name>>;
export function load<const Name extends string = never>(
  url: string,
  options?: LoadOptions<Name, false>,
): Promise<ModelData<Name>>;
export async function load<const Name extends string = never>(
  url: string,
  options: LoadOptions<Name> = {},
): Promise<ModelData<Name> | PlacedModelData<Name>> {
  const get = options.fetch ?? globalThis.fetch;
  if (!get) throw new Error('gltf.load needs a global fetch here; pass one as options.fetch.');

  const bytes = await fetched(get, url, 'the model');
  const binary =
    bytes.byteLength >= 4 &&
    new DataView(bytes.buffer, bytes.byteOffset).getUint32(0, true) === GLB;
  const { json, bin } = binary ? chunks(bytes) : { json: parsed(bytes, url), bin: undefined };

  // Check both lists before any fetch starts, so no fetch is left without a handler
  const bufferList = listOf(json.buffers, "The model's buffers");
  const imageList = listOf(json.images, "The model's images");
  const file = (uri: string, what: string) =>
    uri.startsWith('data:') ? decoded(uri, what) : fetched(get, beside(url, uri), what);
  const buffers = bufferList.map((buffer, index) =>
    typeof buffer?.uri === 'string'
      ? file(buffer.uri, `buffer ${index} of the model`)
      : index === 0
        ? bin
        : undefined,
  );
  const images = imageList.map(async (image, index) => {
    if (image?.bufferView !== undefined || typeof image?.uri !== 'string') return undefined;
    const encoded = await file(image.uri, `image ${index} of the model`);
    return { bytes: encoded, mimeType: mimeTypeOf(image.uri, image.mimeType) };
  });
  const [loadedBuffers, loadedImages] = await Promise.all([
    Promise.all(buffers),
    Promise.all(images),
  ]);
  return modelOf(json, loadedBuffers, loadedImages, options);
}

function modelOf<const Name extends string = never>(
  json: Json,
  buffers: readonly (Uint8Array | undefined)[],
  separate: readonly (EncodedImage | undefined)[],
  options: ReadOptions<Name>,
): ModelData<Name> | PlacedModelData<Name> {
  const { parts = [], placements: placing = false, maxBytes = 256 * 2 ** 20 } = options;
  if (!(maxBytes > 0)) {
    throw new RangeError(`gltf.read requires maxBytes to be greater than 0. Received ${maxBytes}.`);
  }
  const turned = vector(options.rotation, 4, 'The rotation option');
  const root = turned ? mat4d.fromQuat(turned) : IDENTITY;

  for (const name of listOf(json.extensionsRequired, 'extensionsRequired')) {
    if (UNDERSTOOD.has(name) || (name === DRACO && options.draco)) continue;
    const fix = CONVERSIONS.get(name);
    throw new Error(
      name === DRACO
        ? `The model needs ${name}. Pass a decoder made with \`gltf.draco\` as the draco option, or convert it with \`${fix}\`.`
        : fix
          ? `The model needs ${name}, which gltf.read doesn't decode. Convert it with \`${fix}\`.`
          : `The model needs ${name}, which gltf.read doesn't support.`,
    );
  }

  const file = readerOf(json, buffers, separate, options.draco, options.uvs1 ?? false);
  const images = listOf(json.images, "The model's images").map((_, index) => file.imageOf(index));
  const nodes = [...posed(json, root)];
  const skeleton = skeletonOf(json, file, nodes, root, turned);
  if (placing && skeleton) {
    throw new Error('Skinned meshes require the bind pose. Read this model without placements.');
  }

  const wanted = new Set<string>(parts);
  const names: string[] = [];
  const owners = new Map<number, string | undefined>();
  const frames = new Map<string, ReturnType<typeof frameOf>>();

  // Find every node's part before planning meshes, since one skinned mesh binds its whole part
  const skinnedParts = new Set<string>();
  for (const [node, world, index, parent] of nodes) {
    let owner = owners.get(parent);
    if (typeof node.name === 'string') names.push(node.name);

    if (typeof node.name === 'string' && wanted.has(node.name)) {
      if (frames.has(node.name)) {
        throw new Error(`Two nodes are named '${node.name}', so the name can't pick a part.`);
      }
      owner = node.name;
      frames.set(owner, frameOf(world));
    }
    owners.set(index, owner);
    if (owner !== undefined && node.mesh !== undefined && node.skin !== undefined) {
      skinnedParts.add(owner);
    }
  }

  const missing = [...wanted].filter((name) => !frames.has(name));
  if (missing.length) {
    const known = names.length
      ? `Its nodes are named ${names.slice(0, 20).join(', ')}.`
      : 'Its nodes have no names.';
    throw new Error(
      `No node of the default scene is named ${missing.map((name) => `'${name}'`).join(' or ')}. ${known}`,
    );
  }

  // Plan every returned mesh before reading any, so the budget counts exactly what is returned
  const groups = new Map<string | undefined, Map<unknown, Entry[]>>();
  const shared: Entry[][] = [];
  const sharedOf = new Map<number, readonly number[]>();
  const placed: PlacementData[] = [];
  const primitivesOf = (index: number) =>
    listOf(item(json.meshes, index, 'mesh').primitives, `Mesh ${index}'s primitives`).map(
      (primitive, p) => ({ primitive: primitive ?? {}, what: `mesh ${index}'s primitive ${p}` }),
    );

  for (const [node, world, index] of nodes) {
    if (node.mesh === undefined) continue;
    const owner = owners.get(index);

    if (placing && owner === undefined) {
      let meshes = sharedOf.get(node.mesh);
      if (!meshes) {
        const byMaterial = new Map<unknown, number>();
        for (const entry of primitivesOf(node.mesh)) {
          let at = byMaterial.get(entry.primitive.material);
          if (at === undefined) {
            at = shared.push([]) - 1;
            byMaterial.set(entry.primitive.material, at);
          }
          shared[at].push({ ...entry, world: IDENTITY });
        }
        meshes = [...byMaterial.values()];
        sharedOf.set(node.mesh, meshes);
      }
      const { position, rotation, scale } = decompose(world);
      for (const at of meshes) placed.push({ mesh: at, position, rotation, scale });
      continue;
    }

    const primitives = primitivesOf(node.mesh);
    // A skinned part binds to the skin as the model's meshes do, so its frame never applies
    const bound =
      owner === undefined || skinnedParts.has(owner)
        ? skeleton?.bind(node, index, world)
        : undefined;
    const into = owner === undefined ? world : mat4d.multiply(frames.get(owner)!.inverse, world);
    let byMaterial = groups.get(owner);
    if (!byMaterial) groups.set(owner, (byMaterial = new Map()));

    for (const entry of primitives) {
      let list = byMaterial.get(entry.primitive.material);
      if (!list) byMaterial.set(entry.primitive.material, (list = []));
      list.push({ ...entry, world: bound?.world ?? into, binding: bound?.binding });
    }
  }

  let size = 0;
  const planMesh = (entries: readonly Entry[]) => {
    const plan = file.plan(entries);
    size += plan.bytes;
    return plan;
  };
  const plans = new Map<string | undefined, MeshPlan[]>();
  for (const [owner, byMaterial] of groups) {
    plans.set(owner, Array.from(byMaterial.values(), planMesh));
  }
  const sharedPlans = shared.map(planMesh);
  if (size > maxBytes) {
    throw new Error(
      placing
        ? `The model's meshes take ${megabytes(size)}. This exceeds the \`maxBytes\` limit of ${megabytes(maxBytes)}.`
        : `The model's mesh placements take ${megabytes(size)} with a copy for each placement. This exceeds the \`maxBytes\` limit of ${megabytes(maxBytes)}. Set \`placements: true\` to read each mesh once.`,
    );
  }

  const meshesOf = (owner: string | undefined) => (plans.get(owner) ?? []).map(file.meshOf);
  const model = {
    meshes: placing ? sharedPlans.map(file.meshOf) : meshesOf(undefined),
    images,
    // Build own entries, so a node named __proto__ is a part and not the prototype
    parts: Object.fromEntries(
      [...wanted].map((name): [string, PartData] => {
        const { position, rotation } = skinnedParts.has(name) ? MODEL_ORIGIN : frames.get(name)!;
        return [name, { meshes: meshesOf(name), images, position, rotation }];
      }),
    ) as ModelData<Name>['parts'],
    ...(skeleton && { skin: skeleton.skin }),
    clips: clipsOf(json, file, nodes, turned),
  };
  return placing ? { ...model, placements: placed } : model;
}

type Posed = [node: NodeJson, world: Float64Array, index: number, parent: number];

/** The semantics `sourcesOf` reads, which it asks Draco to decode. */
const COMPRESSIBLE = [
  'POSITION',
  'NORMAL',
  'TEXCOORD_0',
  'TEXCOORD_1',
  'COLOR_0',
  'JOINTS_0',
  'WEIGHTS_0',
] as const;

const ARRAYS = new Map<number, new (length: number) => DracoArray>([
  [5120, Int8Array],
  [5121, Uint8Array],
  [5122, Int16Array],
  [5123, Uint16Array],
  [5125, Uint32Array],
  [5126, Float32Array],
]);

/** An accessor's decoded values, read with `componentType` in place of the accessor's. */
type DecodedValues = { readonly array: DracoArray; readonly componentType: number };

type Decoded = {
  readonly values: ReadonlyMap<unknown, DecodedValues>;
  readonly indices: Uint32Array;
};

/** An accessor checked against its buffer view, which writes `width` values per element. */
interface Layout {
  readonly count: number;
  readonly width: number;
  /** Writes element `i`'s component `c` to `out[at + i * stride + c]`. */
  write(out: Values, at: number, stride: number): void;
}

type Values = Float32Array | Uint32Array | Uint16Array;

/** A primitive as a node places it. */
interface Entry {
  readonly primitive: PrimitiveJson;
  /** Names the primitive in errors, such as `mesh 2's primitive 0`. */
  readonly what: string;
  readonly world: Float64Array;
  readonly binding?: Binding | undefined;
}

/** A primitive's counts as its accessors declare them, before reading its data. */
interface Shape {
  readonly mode: number;
  readonly count: number;
  readonly listed: number;
  /** Counts triangle corners, three per triangle once strips and fans are listed. */
  readonly corners: number;
}

/** A returned mesh: one material's entries, and the arrays they fill. */
interface MeshPlan {
  readonly material: number | undefined;
  readonly entries: readonly Entry[];
  readonly shapes: readonly Shape[];
  readonly lengths: Lengths;
  /** Counts the bytes of every array the mesh returns. */
  readonly bytes: number;
}

/** The length of each array a mesh returns, undefined for an optional array it leaves out. */
type Lengths = {
  readonly [K in keyof Arrays]: undefined extends Arrays[K] ? number | undefined : number;
};

type Binding = { readonly remap: Uint16Array } | { readonly joint: number };

/** The frame of a part bound to the skin, whose vertices stay in model space. */
const MODEL_ORIGIN: { readonly position: Vec3; readonly rotation: Vec4 } = {
  position: [0, 0, 0],
  rotation: [0, 0, 0, 1],
};

function skeletonOf(
  json: Json,
  file: Reader,
  nodes: readonly Posed[],
  root: Float64Array,
  turned: Vec4 | undefined,
) {
  const used = new Set(
    nodes.flatMap(([node]) =>
      node.mesh !== undefined && node.skin !== undefined ? [node.skin] : [],
    ),
  );
  if (used.size === 0) return undefined;
  if (used.size > 1) {
    throw new Error(`The default scene's meshes use ${used.size} skins, and gltf.read reads one.`);
  }

  const [index] = used;
  const skin = item(json.skins, index, 'skin');
  const listed = listOf(skin.joints, `Skin ${index}'s joints`);
  const placed = new Map(nodes.map(([node, world, at, parent]) => [at, { node, world, parent }]));
  if (listed.length === 0) throw new Error(`Skin ${index} has no joints.`);
  if (new Set(listed).size !== listed.length) throw new Error(`Skin ${index} lists a joint twice.`);

  // Include ancestor nodes so each joint pose includes all transforms that move it
  const members = new Set<number>();
  for (const joint of listed) {
    if (!placed.has(joint)) {
      throw new Error(
        `Skin ${index}'s joint ${JSON.stringify(joint)} isn't a node of the default scene.`,
      );
    }
    for (let at = joint; at !== -1 && !members.has(at); at = placed.get(at)!.parent) {
      members.add(at);
    }
  }

  const order = nodes.filter(([, , at]) => members.has(at));
  const indexOf = new Map(order.map(([, , at], k) => [at, k]));
  const slotOf = new Map(listed.map((node, k) => [node, k]));
  const binds =
    skin.inverseBindMatrices === undefined
      ? undefined
      : file.accessorOf(skin.inverseBindMatrices, 'inverseBindMatrices', Float32Array);
  if (binds && binds.length !== listed.length * 16) {
    throw new Error(
      `Skin ${index} has ${listed.length} joints and ${binds.length / 16} inverse bind matrices.`,
    );
  }

  // Undo the model rotation in inverse binds before applying joint poses
  const unturn = turned ? mat4d.fromQuat(quatd.conjugate(turned)) : IDENTITY;
  const inverses = order.map(([node, world, at]) => {
    const k = slotOf.get(at);
    if (k === undefined) return invert(world, `Node ${nameOf(node, at)}'s transform`);
    return mat4d.multiply(
      binds ? Float64Array.from(binds.subarray(k * 16, k * 16 + 16)) : IDENTITY,
      unturn,
    );
  });

  const joints = order.map(([node, , at, parent], k): JointData => {
    const rest = restOf(node, at);
    const top = parent === -1;
    return {
      name: nameOf(node, at),
      parent: top ? -1 : indexOf.get(parent)!,
      position: top && turned ? triple(vec3d.transformQuat(rest.position, turned)) : rest.position,
      rotation: top && turned ? quad(quatd.multiply(turned, rest.rotation)) : rest.rotation,
      scale: rest.scale,
      inverseBind: Float32Array.from(inverses[k]),
    };
  });

  const remap = Uint16Array.from(listed, (node) => indexOf.get(node)!);
  const unbinds: Float64Array[] = [];

  return {
    skin: { joints } satisfies SkinData,

    bind(node: NodeJson, at: number, world: Float64Array) {
      // Ignore node transforms on skinned meshes because vertices already use the bind pose
      if (node.skin !== undefined) return { world: root, binding: { remap } };

      let joint = at;
      while (joint !== -1 && !indexOf.has(joint)) joint = placed.get(joint)!.parent;
      if (joint === -1) {
        throw new Error(
          `The mesh on node ${nameOf(node, at)} has no skin and no ancestor joint in this skin, so it cannot move with the skin. Read the node as a separate part with \`parts\`.`,
        );
      }

      // Undo the rest joint transform and inverse bind so their product restores the vertices to
      // their rest positions
      const k = indexOf.get(joint)!;
      const unbound = (unbinds[k] ??= invert(
        inverses[k],
        `Joint ${joints[k].name}'s inverse bind`,
      ));
      const hanging = mat4d.multiply(
        invert(placed.get(joint)!.world, `Node ${joints[k].name}'s transform`),
        world,
      );
      return { world: mat4d.multiply(unbound, hanging), binding: { joint: k } };
    },
  };
}

function clipsOf(
  json: Json,
  file: Reader,
  nodes: readonly Posed[],
  turned: Vec4 | undefined,
): ClipData[] {
  const tops = new Set(nodes.filter(([, , , parent]) => parent === -1).map(([, , at]) => at));

  return listOf(json.animations, "The model's animations").map((_, a) => {
    const animation = item(json.animations, a, 'animation');
    const what = `Animation ${a}`;
    const samplers = listOf(animation.samplers, `${what}'s samplers`);
    const channels: ChannelData[] = [];
    let duration = 0;

    for (const [c, channel] of listOf(animation.channels, `${what}'s channels`).entries()) {
      const target = channel?.target ?? {};
      // Skip node-free extension channels and morph-target weights
      if (target.node === undefined || target.path === 'weights') continue;

      const semantic = target.path;
      if (semantic !== 'translation' && semantic !== 'rotation' && semantic !== 'scale') {
        throw new Error(
          `${what}'s channel ${c} moves ${JSON.stringify(target.path)}, not a node's translation, rotation or scale.`,
        );
      }
      const path = semantic === 'translation' ? 'position' : semantic;
      const node = item(json.nodes, target.node, 'node');
      const index = channel?.sampler;
      const sampler = index !== undefined && Number.isInteger(index) ? samplers[index] : undefined;
      if (typeof sampler !== 'object' || sampler === null) {
        throw new Error(`${what}'s channel ${c} has no sampler ${JSON.stringify(index)}.`);
      }
      const { input, output, interpolation = 'LINEAR' } = sampler;
      const keyed = INTERPOLATIONS.get(interpolation);
      if (!keyed) {
        throw new Error(
          `${what}'s sampler ${index} interpolates ${JSON.stringify(interpolation)}, not LINEAR, STEP or CUBICSPLINE.`,
        );
      }

      const times = file.accessorOf(input, 'times', Float32Array);
      for (let i = 0; i < times.length; i++) {
        if (!Number.isFinite(times[i]) || (i > 0 && !(times[i] > times[i - 1]))) {
          throw new Error(`${what}'s sampler ${index} has times that don't increase.`);
        }
      }
      const values = file.accessorOf(output, semantic, Float32Array);
      const width = path === 'rotation' ? 4 : 3;
      const keys = values.length / width / (keyed === 'cubic' ? 3 : 1);
      if (keys !== times.length) {
        throw new Error(`${what}'s sampler ${index} has ${times.length} times and ${keys} keys.`);
      }
      if (!values.every(Number.isFinite)) {
        throw new Error(`${what}'s sampler ${index} has keys that aren't numbers.`);
      }

      if (turned && path !== 'scale' && tops.has(target.node)) {
        for (let at = 0; at < values.length; at += width) {
          const key = values.subarray(at, at + width);
          if (path === 'rotation') quatd.multiply(turned, key, key);
          else vec3d.transformQuat(key, turned, key);
        }
      }
      channels.push({
        joint: nameOf(node, target.node),
        path,
        interpolation: keyed,
        times,
        values,
      });
      duration = Math.max(duration, times[times.length - 1]);
    }

    const name = typeof animation.name === 'string' ? animation.name : `animation ${a}`;
    return { name, duration, channels };
  });
}

const nameOf = (node: NodeJson, index: number) =>
  typeof node.name === 'string' ? node.name : `node ${index}`;

function restOf(node: NodeJson, index: number) {
  const name = `Node ${index}`;
  const matrix = vector(node.matrix, 16, `${name}'s matrix`);
  if (!matrix) {
    return {
      position: vector(node.translation, 3, `${name}'s translation`) ?? ([0, 0, 0] as const),
      rotation: vector(node.rotation, 4, `${name}'s rotation`) ?? ([0, 0, 0, 1] as const),
      scale: vector(node.scale, 3, `${name}'s scale`) ?? ([1, 1, 1] as const),
    };
  }
  return decompose(matrix);
}

/**
 * Decomposition discards shear because instance transforms hold only position, rotation and
 * scale.
 */
function decompose(matrix: ArrayLike<number>) {
  const axes = [0, 1, 2].map((c) => [matrix[c * 4], matrix[c * 4 + 1], matrix[c * 4 + 2]]);
  const mirrored = vec3d.dot(vec3d.cross(axes[0], axes[1]), axes[2]) < 0;
  const scale = axes.map((axis, c) => vec3d.length(axis) * (mirrored && c === 0 ? -1 : 1));
  const basis = mat4d.identity();
  for (const [c, axis] of axes.entries()) {
    basis.set(unit(vec3d.scale(axis, 1 / (scale[c] || 1)), AXES[c]), c * 4);
  }
  return {
    position: [matrix[12], matrix[13], matrix[14]] as [number, number, number],
    rotation: quad(quatd.fromMat(basis)),
    scale: triple(scale),
  };
}

/**
 * Extract only position and rotation for the part frame. Leave scale, shear and reflection in
 * vertices transformed by its inverse.
 */
function frameOf(world: Float64Array) {
  const column = (c: number) => world.subarray(c * 4, c * 4 + 3);
  // Use Gram-Schmidt to form rotation axes and leave scale and shear in the vertices
  const x = unit(column(0), AXES[0]);
  const second = column(1);
  const y = unit(
    vec3d.subtract(second, vec3d.scale(x, vec3d.dot(second, x))),
    unit(vec3d.cross(x, Math.abs(x[2]) < 0.9 ? AXES[2] : AXES[0]), AXES[1]),
  );
  const frame = mat4d.identity();
  frame.set(x, 0);
  frame.set(y, 4);
  frame.set(vec3d.cross(x, y), 8);
  frame.set(column(3), 12);

  return {
    inverse: mat4d.inverse(frame),
    position: triple(column(3)),
    rotation: quad(quatd.fromMat(frame)),
  };
}

const megabytes = (bytes: number) => `${(bytes / 2 ** 20).toFixed(1)} MiB`;

function chunks(bytes: Uint8Array) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (bytes.byteLength < 20 || view.getUint32(0, true) !== GLB) {
    throw new Error(
      bytes[0] === 0x7b
        ? 'gltf.read reads binary glTF; load a .gltf with `gltf.load`, or pack it with `gltf-transform copy in.gltf out.glb`.'
        : "The bytes aren't a binary glTF.",
    );
  }
  if (view.getUint32(4, true) !== 2) {
    throw new Error(`gltf.read reads glTF 2, got version ${view.getUint32(4, true)}.`);
  }

  // Respect the declared length so trailing bytes cannot become part of the file
  const length = view.getUint32(8, true);
  if (length > bytes.byteLength) {
    throw new Error(
      `The binary glTF is incomplete. Its header declares ${length} bytes, but only ${bytes.byteLength} are available.`,
    );
  }

  let json: Json | undefined;
  let bin: Uint8Array | undefined;
  for (let at = 12; at + 8 <= length;) {
    const size = view.getUint32(at, true);
    const type = view.getUint32(at + 4, true);
    if (at + 8 + size > length) throw new Error('A chunk of the binary glTF runs past its end.');
    const data = bytes.subarray(at + 8, at + 8 + size);

    if (type === JSON_CHUNK) json = jsonOf(data, 'The binary glTF');
    else if (type === BIN_CHUNK) bin ??= data;
    at += 8 + size;
  }

  if (json === undefined) throw new Error('The binary glTF has no JSON chunk.');
  return { json, bin };
}

/** Parses a `.gltf` file, which is JSON. */
function parsed(bytes: Uint8Array, url: string): Json {
  // Skip a byte order mark, then expect the opening brace
  let at = bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf ? 3 : 0;
  while (bytes[at] === 0x20 || bytes[at] === 0x09 || bytes[at] === 0x0a || bytes[at] === 0x0d) {
    at++;
  }
  if (bytes[at] !== 0x7b) {
    throw new Error(`${url} is neither a binary glTF nor a glTF JSON file.`);
  }
  return jsonOf(bytes.subarray(at), url);
}

/**
 * Parses glTF JSON and checks its version. The reader checks every other field where it reads
 * it, and ignores fields it doesn't read.
 */
function jsonOf(bytes: Uint8Array, what: string): Json {
  let json: unknown;
  try {
    json = JSON.parse(utf8(bytes));
  } catch (error) {
    throw new Error(`${what} contains invalid JSON. ${(error as Error).message}`, {
      cause: error,
    });
  }
  if (typeof json !== 'object' || json === null || Array.isArray(json)) {
    throw new Error(`${what}'s JSON isn't an object.`);
  }
  const version = (json as Json).asset?.version;
  if (typeof version === 'string' && !version.startsWith('2.')) {
    throw new Error(`gltf.read reads glTF 2, got version ${version}.`);
  }
  return json as Json;
}

async function fetched(get: Fetch, url: string, what: string): Promise<Uint8Array> {
  let response: Awaited<ReturnType<Fetch>>;
  try {
    response = await get(url);
  } catch (error) {
    throw new Error(`gltf.load couldn't fetch ${what}, ${url}.`, { cause: error });
  }
  if (!response.ok) {
    throw new Error(`gltf.load couldn't fetch ${what}, ${url}: HTTP ${response.status}.`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/** Decodes a base64 `data:` URI with `atob`, which browsers, Node and Hermes provide. */
async function decoded(uri: string, what: string): Promise<Uint8Array> {
  const comma = uri.indexOf(',');
  if (comma < 0 || !uri.slice(0, comma).endsWith(';base64')) {
    throw new Error(`gltf.load decodes base64 data URIs, and ${what} isn't one.`);
  }
  const text = atob(uri.slice(comma + 1));
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}

const SCHEME = /^[a-z][a-z\d+.-]*:/i;

/**
 * Resolves a file's URI against the model's URL. It avoids `URL`, which React Native implements
 * only in part.
 */
function beside(base: string, uri: string): string {
  if (SCHEME.test(uri)) return uri;
  const path = base.replace(/[?#].*$/, '');
  const origin = /^[a-z][a-z\d+.-]*:\/\/[^/]*/i.exec(path)?.[0] ?? '';
  const folder = path.slice(origin.length, path.lastIndexOf('/') + 1);
  const relative = uri.startsWith('/') ? uri : `${origin && !folder ? '/' : ''}${folder}${uri}`;

  const segments: string[] = [];
  for (const segment of relative.split('/')) {
    const last = segments[segments.length - 1];
    if (segment === '.') continue;
    if (segment !== '..') segments.push(segment);
    else if (last !== undefined && last !== '' && last !== '..') segments.pop();
    else if (last !== '') segments.push(segment);
  }
  return origin + segments.join('/');
}

function mimeTypeOf(uri: string, given: unknown): string {
  if (typeof given === 'string') return given;
  if (uri.startsWith('data:')) return uri.slice(5, uri.search(/[;,]/)) || 'image/png';
  const extension = /\.([a-z\d]+)(?:[?#]|$)/i.exec(uri)?.[1]?.toLowerCase();
  return extension === 'jpg' || extension === 'jpeg'
    ? 'image/jpeg'
    : extension === 'webp'
      ? 'image/webp'
      : 'image/png';
}

/** Decode UTF-8 without requiring `TextDecoder` on the runtime. */
function utf8(bytes: Uint8Array) {
  let text = '';
  for (let i = 0; i < bytes.length;) {
    const byte = bytes[i++];
    let code = byte;

    if (byte >= 0xf0) {
      code = ((byte & 0x07) << 18) | ((bytes[i++] & 0x3f) << 12);
      code |= ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
    } else if (byte >= 0xe0) {
      code = ((byte & 0x0f) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
    } else if (byte >= 0xc0) {
      code = ((byte & 0x1f) << 6) | (bytes[i++] & 0x3f);
    }
    text += String.fromCodePoint(code);
  }
  return text;
}

/** Use an explicit stack to avoid overflowing the call stack on deep node trees. */
function* posed(
  json: Json,
  root = IDENTITY,
): Generator<[node: NodeJson, world: Float64Array, index: number, parent: number]> {
  const parents = new Map<number, number>();
  const stack: { index: number; parent: number; above: Float64Array }[] = [];

  // Push in reverse order to preserve node order when popping the stack
  const push = (indices: readonly number[], parent: number, above: Float64Array) => {
    for (let i = indices.length - 1; i >= 0; i--) stack.push({ index: indices[i], parent, above });
  };
  push(rootsOf(json), -1, root);

  for (let next = stack.pop(); next; next = stack.pop()) {
    const { index, parent, above } = next;
    const node = item(json.nodes, index, 'node');

    // Reject repeated nodes because glTF forbids multiple parents and cycles
    if (parents.has(index)) {
      let up = parent;
      while (up !== -1 && up !== index) up = parents.get(up)!;

      throw new Error(
        up === index
          ? `Node ${index} is its own ancestor.`
          : `Node ${index} appears twice in the scene, and a node has one parent at most.`,
      );
    }
    parents.set(index, parent);

    const world = mat4d.multiply(above, localOf(node, index));
    yield [node, world, index, parent];

    push(listOf(node.children, `Node ${index}'s children`), index, world);
  }
}

function rootsOf(json: Json): readonly number[] {
  if (json.scene !== undefined || json.scenes !== undefined) {
    const index = json.scene ?? 0;
    return listOf(item(json.scenes, index, 'scene').nodes, `Scene ${index}'s nodes`);
  }

  const nodes = listOf(json.nodes, "The model's nodes");
  const children = new Set(
    nodes.flatMap((_, index) =>
      listOf(item(nodes, index, 'node').children, `Node ${index}'s children`),
    ),
  );
  return nodes.map((_, index) => index).filter((index) => !children.has(index));
}

type Reader = ReturnType<typeof readerOf>;

function readerOf(
  json: Json,
  buffers: readonly (Uint8Array | undefined)[],
  separate: readonly (EncodedImage | undefined)[],
  draco: Draco | undefined,
  everyUvs1: boolean,
) {
  // Decode a primitive once, though placements may read it again for every node
  const decodedPrimitives = new Map<PrimitiveJson, Decoded>();
  // Check an accessor once, though many nodes may place it
  const layouts = new Map<string, Layout>();

  // Read TEXCOORD_1 only where a texture samples it or the caller asks: exporters write every set
  const secondUvs = new Map<number | undefined, boolean>();
  function readsUvs1(material: number | undefined) {
    let reads = secondUvs.get(material);
    if (reads === undefined) {
      const { textures } = materialOf(material);
      reads = Object.values(textures).some((texture) => texture.uvSet === 1);
      secondUvs.set(material, reads);
    }
    return reads;
  }

  const countOf = (index: unknown) =>
    whole(item(json.accessors, index, 'accessor').count, `Accessor ${index}'s count`, 1);

  function shapeOf(primitive: PrimitiveJson): Shape {
    const mode = primitive.mode ?? TRIANGLES;
    if (mode !== TRIANGLES && mode !== STRIP && mode !== FAN) {
      throw new Error(`gltf.read reads triangles, got a primitive of mode ${mode}.`);
    }
    const position = primitive.attributes?.POSITION;
    if (position === undefined) throw new Error('A primitive has no POSITION.');

    const count = countOf(position);
    const listed = primitive.indices === undefined ? count : countOf(primitive.indices);
    return {
      mode,
      count,
      listed,
      corners: mode === TRIANGLES ? listed : Math.max(listed - 2, 0) * 3,
    };
  }

  /**
   * Count a mesh's arrays from its accessors. A vertex without normals becomes one per corner, and
   * an optional attribute that one entry has takes space for every vertex of the mesh.
   */
  function plan(entries: readonly Entry[]): MeshPlan {
    const { material } = entries[0].primitive;
    const keepUvs1 = everyUvs1 || readsUvs1(material);
    let vertices = 0;
    let corners = 0;
    let uvs1 = false;
    let colors = false;
    const shapes = entries.map(({ primitive }) => {
      const shape = shapeOf(primitive);
      const { NORMAL, TEXCOORD_1, COLOR_0 } = primitive.attributes ?? {};
      vertices += NORMAL === undefined ? shape.corners : shape.count;
      corners += shape.corners;
      uvs1 ||= keepUvs1 && TEXCOORD_1 !== undefined;
      colors ||= COLOR_0 !== undefined;
      return shape;
    });
    const skinned = entries[0].binding !== undefined;
    const lengths: Lengths = {
      positions: vertices * 3,
      normals: vertices * 3,
      uvs: vertices * 2,
      uvs1: uvs1 ? vertices * 2 : undefined,
      colors: colors ? vertices * 4 : undefined,
      indices: corners,
      joints: skinned ? vertices * 4 : undefined,
      weights: skinned ? vertices * 4 : undefined,
    };
    let bytes = 0;
    for (const array of Object.keys(lengths) as (keyof Arrays)[]) {
      bytes += (lengths[array] ?? 0) * ELEMENTS[array].BYTES_PER_ELEMENT;
    }
    return { material, entries, shapes, lengths, bytes };
  }

  /** Reads a planned mesh's entries straight into its arrays. */
  function meshOf(planned: MeshPlan): MeshData<Extra> {
    const { entries, shapes, lengths } = planned;
    // Check every source before allocating, so a range past its buffer throws first
    const sources = entries.map((entry, k) => sourcesOf(entry, shapes[k]));
    const allocated = <K extends keyof Arrays>(array: K) => {
      const length = lengths[array];
      return (length === undefined ? undefined : new ELEMENTS[array](length)) as Arrays[K];
    };
    const out: Arrays = {
      positions: allocated('positions'),
      normals: allocated('normals'),
      uvs: allocated('uvs'),
      // Entries without TEXCOORD_1 read zero, as meshes without uvs1 do in a pool
      uvs1: allocated('uvs1'),
      // Entries without COLOR_0 are white
      colors: allocated('colors')?.fill(1),
      indices: allocated('indices'),
      joints: allocated('joints'),
      weights: allocated('weights'),
    };

    let vertex = 0;
    let corner = 0;
    entries.forEach((entry, k) => {
      vertex += written(entry, sources[k], out, vertex, corner);
      corner += sources[k].corners;
    });

    const { uvs1, colors, joints, weights, ...rest } = out;
    return {
      ...rest,
      ...(uvs1 && { uvs1 }),
      ...(colors && { extra: { color: colors } }),
      ...(joints && weights && { joints, weights }),
      material: materialOf(planned.material),
    };
  }

  function sourcesOf({ primitive, what, binding }: Entry, shape: Shape) {
    const { POSITION, NORMAL, TEXCOORD_0, TEXCOORD_1, COLOR_0 } = primitive.attributes ?? {};
    const compressed = primitive.extensions?.[DRACO];
    if (
      compressed &&
      !draco &&
      item(json.accessors, POSITION, 'accessor').bufferView === undefined
    ) {
      throw new Error(
        `A primitive is compressed with ${DRACO}. Pass a decoder made with \`gltf.draco\` as the draco option, or convert the model with \`${CONVERSIONS.get(DRACO)}\`.`,
      );
    }
    const unpacked =
      compressed && draco ? decodedOf(primitive, compressed, draco, what) : undefined;
    const take = (index: unknown, semantic: Semantic) =>
      layoutOf(index, semantic, unpacked?.values.get(index));

    const { count } = shape;
    const sized = (index: unknown, semantic: Semantic, noun: string) => {
      if (index === undefined) return undefined;
      const layout = take(index, semantic);
      if (layout.count !== count) {
        throw new Error(`A primitive has ${count} positions and ${layout.count} ${noun}.`);
      }
      return layout;
    };

    const positions = take(POSITION, 'POSITION');
    const normals = sized(NORMAL, 'NORMAL', 'normals');
    const uvs = sized(TEXCOORD_0, 'TEXCOORD_0', 'uvs');
    const sampled = readsUvs1(primitive.material);
    if (sampled && TEXCOORD_1 === undefined) {
      throw new Error(
        `Material ${primitive.material} reads TEXCOORD_1, and a primitive drawn with it has none.`,
      );
    }
    const uvs1 =
      sampled || everyUvs1 ? sized(TEXCOORD_1, 'TEXCOORD_1', 'TEXCOORD_1 uvs') : undefined;
    const colors = sized(COLOR_0, 'COLOR_0', 'colours');

    let joints: Layout | undefined;
    let weights: Layout | undefined;
    if (binding && 'remap' in binding) {
      const { JOINTS_0, WEIGHTS_0, JOINTS_1 } = primitive.attributes ?? {};
      if (JOINTS_1 !== undefined) {
        throw new Error('A primitive has JOINTS_1, and gltf.read reads four joints a vertex.');
      }
      if (JOINTS_0 === undefined || WEIGHTS_0 === undefined) {
        throw new Error('A primitive of a skinned mesh has no JOINTS_0 or no WEIGHTS_0.');
      }
      joints = take(JOINTS_0, 'JOINTS_0');
      weights = take(WEIGHTS_0, 'WEIGHTS_0');
      if (joints.count !== count || weights.count !== count) {
        throw new Error(
          `A primitive has ${count} positions, ${joints.count} joints and ${weights.count} weights.`,
        );
      }
    }

    // Draco decodes a list of triangles, whatever the primitive's mode
    let indices: Layout | undefined;
    if (unpacked) {
      const listed = unpacked.indices;
      if (listed.length !== shape.corners) {
        throw new Error(
          `The Draco decoder returned ${listed.length} indices for ${what}, whose accessors list ${shape.corners}.`,
        );
      }
      indices = { count: listed.length, width: 1, write: (out, at) => out.set(listed, at) };
    } else if (primitive.indices !== undefined) {
      indices = take(primitive.indices, 'indices');
    }

    if (shape.mode === TRIANGLES && shape.listed % 3 !== 0) {
      throw new Error(
        `A primitive of triangles lists ${shape.listed} vertices, which isn't a multiple of three.`,
      );
    }
    return {
      count,
      corners: shape.corners,
      mode: unpacked ? TRIANGLES : shape.mode,
      positions,
      normals,
      uvs,
      uvs1,
      colors,
      indices,
      joints,
      weights,
    };
  }

  /**
   * Write one entry into a mesh's arrays from `vertex` and `corner` on, and return how many
   * vertices it took. An entry with normals reads straight into the mesh. One without reads its
   * vertices apart, then gives each triangle corner its own vertex and the triangle's normal.
   */
  function written(
    { world, binding }: Entry,
    from: ReturnType<typeof sourcesOf>,
    out: Arrays,
    vertex: number,
    corner: number,
  ) {
    const { count, corners, mode } = from;
    const flat = from.normals === undefined;
    const floats = (array: Float32Array, width: number) =>
      flat
        ? new Float32Array(count * width)
        : array.subarray(vertex * width, (vertex + count) * width);

    const positions = floats(out.positions, 3);
    const normals = flat ? undefined : floats(out.normals, 3);
    const uvs = floats(out.uvs, 2);
    const uvs1 = out.uvs1 && from.uvs1 && floats(out.uvs1, 2);
    const colors = out.colors && from.colors && floats(out.colors, 4);
    // A three-component colour keeps the opaque alpha it finds
    if (flat && colors) colors.fill(1);
    from.positions.write(positions, 0, 3);
    if (normals) from.normals?.write(normals, 0, 3);
    from.uvs?.write(uvs, 0, 2);
    if (uvs1) from.uvs1?.write(uvs1, 0, 2);
    if (colors) from.colors?.write(colors, 0, 4);

    let joints: Uint16Array | undefined;
    let weights: Float32Array | undefined;
    if (out.joints && out.weights && binding) {
      joints = flat
        ? new Uint16Array(count * 4)
        : out.joints.subarray(vertex * 4, (vertex + count) * 4);
      weights = floats(out.weights, 4);
      skin(binding, from, joints, weights);
    }

    const local = flat ? new Uint32Array(corners) : out.indices.subarray(corner, corner + corners);
    const listed = mode === TRIANGLES ? local : new Uint32Array(from.indices?.count ?? count);
    if (from.indices) from.indices.write(listed, 0, 1);
    else for (let i = 0; i < listed.length; i++) listed[i] = i;
    for (let i = 0; i < listed.length; i++) {
      if (listed[i] >= count) {
        throw new Error(
          `A primitive has ${count} vertices. Index ${listed[i]} is outside that range.`,
        );
      }
    }
    if (mode !== TRIANGLES) triangles(listed, mode, local);

    if (pose(positions, normals, world)) {
      for (let i = 0; i < corners; i += 3) {
        const second = local[i + 1];
        local[i + 1] = local[i + 2];
        local[i + 2] = second;
      }
    }

    if (!flat) {
      for (let i = 0; i < corners; i++) local[i] += vertex;
      return count;
    }
    flatten({ positions, uvs, uvs1, colors, joints, weights }, local, out, vertex, corner);
    return corners;
  }

  /** Bind vertices to joints, and normalize weights so they sum to 1 despite file precision. */
  function skin(
    binding: Binding,
    from: ReturnType<typeof sourcesOf>,
    joints: Uint16Array,
    weights: Float32Array,
  ) {
    const count = joints.length / 4;
    if ('joint' in binding) {
      for (let i = 0; i < count; i++) {
        joints[i * 4] = binding.joint;
        weights[i * 4] = 1;
      }
      return;
    }

    from.joints!.write(joints, 0, 4);
    from.weights!.write(weights, 0, 4);
    const { remap } = binding;
    for (let i = 0; i < count; i++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        const weight = weights[i * 4 + k];
        if (!(weight >= 0 && weight <= 1)) {
          throw new Error(
            `Vertex ${i} of a skinned primitive has a weight of ${weight}. Joint weights must be between 0 and 1, inclusive.`,
          );
        }
        sum += weight;
      }
      if (!(sum > 0)) throw new Error(`Vertex ${i} of a skinned primitive has no weight.`);

      for (let k = 0; k < 4; k++) {
        const at = i * 4 + k;
        const joint = joints[at];
        if (joint >= remap.length) {
          throw new Error(
            `Vertex ${i} of a skinned primitive is bound to joint index ${joint}. The skin has only ${remap.length} joints.`,
          );
        }
        joints[at] = remap[joint];
        weights[at] /= sum;
      }
    }
  }

  /**
   * Decode a compressed primitive's attributes that `sourcesOf` reads, keyed by accessor, and its
   * indices.
   */
  function decodedOf(
    primitive: PrimitiveJson,
    extension: DracoJson,
    decode: Draco,
    what: string,
  ): Decoded {
    const cached = decodedPrimitives.get(primitive);
    if (cached) return cached;

    const { view } = viewOf(extension.bufferView);
    const ids = extension.attributes ?? {};
    const unread = !everyUvs1 && !readsUvs1(primitive.material);
    const wanted = COMPRESSIBLE.flatMap((semantic) => {
      const index = primitive.attributes?.[semantic];
      if (index === undefined || !Object.hasOwn(ids, semantic)) return [];
      if (semantic === 'TEXCOORD_1' && unread) return [];
      const id = whole(ids[semantic], `The Draco id of ${semantic}`);
      const { componentType } = item(json.accessors, index, 'accessor');
      if (!SEMANTICS[semantic].components.includes(componentType)) {
        throw new Error(
          `Accessor ${index} has component type ${componentType}, which ${semantic} can't have.`,
        );
      }
      return [{ index, id, componentType, semantic }];
    });

    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    const mesh: Partial<DracoMesh> | undefined = decode(
      bytes,
      wanted.map(({ id, componentType }) => ({ id, componentType })),
    );
    // Name what a custom decoder got wrong before reading anything it returned
    if (!(mesh?.indices instanceof Uint32Array)) {
      throw new Error(
        `The Draco decoder returned ${described(mesh?.indices)} as the indices of ${what}, which need a Uint32Array.`,
      );
    }
    const arrays: readonly unknown[] = Array.isArray(mesh.attributes) ? mesh.attributes : [];

    const values = new Map<unknown, DecodedValues>();
    for (const [k, { index, componentType, semantic }] of wanted.entries()) {
      const array = arrays[k];
      const Kind = ARRAYS.get(componentType)!;
      if (!(array instanceof Kind)) {
        throw new Error(
          `The Draco decoder returned ${described(array)} for the ${semantic} of ${what}, which needs a ${Kind.name}.`,
        );
      }
      values.set(index, { array, componentType });
    }

    const unpacked = { values, indices: mesh.indices };
    decodedPrimitives.set(primitive, unpacked);
    return unpacked;
  }

  /**
   * Check accessor type, component type and buffer bounds before allocating arrays. Map
   * normalized integers to [-1, 1] or [0, 1]. Read `unpacked` values in place of a buffer view for
   * an accessor Draco compressed.
   */
  function layoutOf(index: unknown, semantic: Semantic, unpacked?: DecodedValues): Layout {
    const key = `${semantic} ${index}`;
    const known = unpacked ? undefined : layouts.get(key);
    if (known) return known;

    const accessor = item(json.accessors, index, 'accessor');
    const name = `Accessor ${index}`;
    const { types, components } = SEMANTICS[semantic];
    const { componentType } = accessor;
    const normalized = accessor.normalized === true;
    const width = types.get(accessor.type);

    if (width === undefined) {
      throw new Error(
        `${name} is ${accessor.type}, and ${semantic} must be ${[...types.keys()].join(' or ')}.`,
      );
    }
    if (!components.includes(componentType)) {
      throw new Error(`${name} has component type ${componentType}, which ${semantic} can't have.`);
    }
    if (normalized && SEMANTICS[semantic].normalized === false) {
      throw new Error(`${name} is normalized, which ${semantic} can't be.`);
    }
    if (SEMANTICS[semantic].normalized && componentType !== 5126 && !normalized) {
      throw new Error(`${name} holds integers that aren't normalized, which ${semantic} can't.`);
    }
    // Reject accessors without views to prevent small files from allocating unbounded
    // zero-filled arrays
    if (accessor.bufferView === undefined && !unpacked) {
      throw new Error(`${name} has no buffer view, and gltf.read reads only accessors with one.`);
    }

    const count = whole(accessor.count, `${name}'s count`, 1);
    const size = SIZES.get(componentType)!;
    const element = size * width;
    let get: (at: number) => number;
    let start = 0;
    let step: number;
    let part = size;

    if (unpacked) {
      const { array } = unpacked;
      if (array.length !== count * width) {
        throw new Error(
          `${name} holds ${count * width} values, and Draco decoded ${array.length}.`,
        );
      }
      part = array.BYTES_PER_ELEMENT;
      step = width * part;
      const view = new DataView(array.buffer, array.byteOffset, array.byteLength);
      get = getter(view, unpacked.componentType, normalized);
    } else {
      const { view, stride = element } = viewOf(accessor.bufferView);
      start = whole(accessor.byteOffset ?? 0, `${name}'s byteOffset`);
      step = stride;
      const end = start + (count - 1) * stride + element;

      if (stride < element) {
        throw new Error(
          `${name}'s elements take ${element} bytes, more than its buffer view's stride of ${stride}.`,
        );
      }
      if (end > view.byteLength) {
        throw new Error(
          `${name} exceeds its buffer view. It needs ${end} bytes, but the view has ${view.byteLength}.`,
        );
      }
      get = getter(view, componentType, normalized);
    }

    const sparse = sparseOf(accessor, name, count, element, normalized);
    const layout: Layout = {
      count,
      width,
      write(out, at, stride) {
        for (let i = 0; i < count; i++) {
          for (let c = 0; c < width; c++)
            out[at + i * stride + c] = get(start + i * step + c * part);
        }
        if (!sparse) return;
        const { slots, value } = sparse;
        for (let k = 0; k < slots.length; k++) {
          for (let c = 0; c < width; c++)
            out[at + slots[k] * stride + c] = value((k * width + c) * size);
        }
      },
    };
    if (!unpacked) layouts.set(key, layout);
    return layout;
  }

  function sparseOf(
    accessor: NonNullable<Json['accessors']>[number],
    name: string,
    count: number,
    element: number,
    normalized: boolean,
  ) {
    const { sparse } = accessor;
    if (!sparse) return undefined;
    const replaced = whole(sparse.count, `${name}'s sparse count`, 1);
    const slotType = sparse.indices?.componentType;

    if (replaced > count) {
      throw new Error(
        `${name} tries to replace ${replaced} elements. Its sparse accessor cannot replace more than the ${count} elements it holds.`,
      );
    }
    if (slotType === undefined || !SEMANTICS.indices.components.includes(slotType)) {
      throw new Error(
        `${name}'s sparse indices have component type ${slotType}, which indices can't have.`,
      );
    }

    const slotSize = SIZES.get(slotType)!;
    const slotAt = getter(
      packed(sparse.indices, replaced * slotSize, `${name}'s sparse indices`),
      slotType,
      false,
    );
    const values = packed(sparse.values, replaced * element, `${name}'s sparse values`);
    const slots = new Uint32Array(replaced);
    for (let k = 0; k < replaced; k++) {
      slots[k] = slotAt(k * slotSize);
      if (slots[k] >= count) {
        throw new Error(
          `${name} has ${count} elements. Sparse index ${slots[k]} is outside that range.`,
        );
      }
    }
    return { slots, value: getter(values, accessor.componentType, normalized) };
  }

  /** Reads a whole accessor into a new array. */
  function accessorOf<T extends Float32Array | Uint32Array>(
    index: unknown,
    semantic: Semantic,
    Kind: new (length: number) => T,
  ): T {
    const layout = layoutOf(index, semantic);
    const out = new Kind(layout.count * layout.width);
    layout.write(out, 0, layout.width);
    return out;
  }

  function materialOf(index: number | undefined): MaterialData {
    const material = index === undefined ? {} : item(json.materials, index, 'material');
    const name = `Material ${index}`;
    const pbr = material.pbrMetallicRoughness ?? {};
    const strength = material.extensions?.KHR_materials_emissive_strength?.emissiveStrength;
    const factor = scalar(strength, `${name}'s emissiveStrength`) ?? 1;
    const [r, g, b] = vector(material.emissiveFactor, 3, `${name}'s emissiveFactor`) ?? [0, 0, 0];

    const textures: { -readonly [K in keyof TexturesData]: TextureData } = {};
    const add = (key: keyof TexturesData, info: TextureInfo | undefined) => {
      if (info) textures[key] = textureOf(info, `${name}'s ${key} texture`);
    };
    add('color', pbr.baseColorTexture);
    add('metallicRoughness', pbr.metallicRoughnessTexture);
    add('normal', material.normalTexture);
    add('emission', material.emissiveTexture);
    add('occlusion', material.occlusionTexture);

    return {
      name: typeof material.name === 'string' ? material.name : '',
      color: vector(pbr.baseColorFactor, 4, `${name}'s baseColorFactor`) ?? [1, 1, 1, 1],
      roughness: scalar(pbr.roughnessFactor, `${name}'s roughnessFactor`) ?? 1,
      metallic: scalar(pbr.metallicFactor, `${name}'s metallicFactor`) ?? 1,
      emission: [r * factor, g * factor, b * factor],
      textures,
      normalScale: scalar(material.normalTexture?.scale, `${name}'s normal scale`) ?? 1,
      occlusionStrength:
        scalar(material.occlusionTexture?.strength, `${name}'s occlusion strength`) ?? 1,
      alpha:
        material.alphaMode === 'MASK'
          ? 'mask'
          : material.alphaMode === 'BLEND'
            ? 'blend'
            : 'opaque',
      alphaCutoff: scalar(material.alphaCutoff, `${name}'s alphaCutoff`) ?? 0.5,
      doubleSided: material.doubleSided === true,
    };
  }

  function imageOf(index: number): EncodedImage {
    const image = item(json.images, index, 'image');
    if (image.bufferView === undefined) {
      const loaded = separate[index];
      if (loaded) return loaded;
      const where =
        typeof image.uri === 'string' && !image.uri.startsWith('data:')
          ? `a separate file, ${image.uri}`
          : 'outside the binary chunk';
      throw new Error(
        `Image ${index} of the model is ${where}; load the model with \`gltf.load\`, which fetches it from beside the model, or pack it in with \`gltf-transform copy in.glb out.glb\`.`,
      );
    }

    const { view } = viewOf(image.bufferView);
    return {
      bytes: new Uint8Array(view.buffer, view.byteOffset, view.byteLength).slice(),
      mimeType: typeof image.mimeType === 'string' ? image.mimeType : 'image/png',
    };
  }

  function textureOf(info: TextureInfo, what: string): TextureData {
    const transform = info.extensions?.KHR_texture_transform;
    const set = transform?.texCoord ?? info.texCoord ?? 0;
    const [u, v] = vector(transform?.offset, 2, `${what}'s offset`) ?? [0, 0];
    const [su, sv] = vector(transform?.scale, 2, `${what}'s scale`) ?? [1, 1];
    const angle = scalar(transform?.rotation, `${what}'s rotation`) ?? 0;

    if (set !== 0 && set !== 1) {
      throw new Error(`${what} reads TEXCOORD_${set}, and gltf.read reads TEXCOORD_0 and 1.`);
    }
    const moves = u !== 0 || v !== 0 || su !== 1 || sv !== 1 || angle !== 0;

    const { index } = info;
    const texture = item(json.textures, index, 'texture');
    const image = texture.source ?? texture.extensions?.EXT_texture_webp?.source;
    if (image === undefined) {
      throw new Error(
        texture.extensions?.KHR_texture_basisu
          ? `A texture is KTX2 only. Convert it with \`${CONVERSIONS.get('KHR_texture_basisu')}\`.`
          : `The model has no image for texture ${index}.`,
      );
    }
    // Validate image indices during reading so malformed files fail before texture upload
    item(json.images, image, 'image');

    const sampler =
      texture.sampler === undefined ? {} : item(json.samplers, texture.sampler, 'sampler');
    const min = sampler.minFilter;
    return {
      image,
      ...(set === 1 && { uvSet: 1 as const }),
      ...(moves && { transform: { offset: [u, v], rotation: angle, scale: [su, sv] } }),
      sampler: {
        addressModeU: WRAPS.get(sampler.wrapS) ?? 'repeat',
        addressModeV: WRAPS.get(sampler.wrapT) ?? 'repeat',
        magFilter: sampler.magFilter === 9728 ? 'nearest' : 'linear',
        // These glTF filter values use nearest minification
        minFilter: min === 9728 || min === 9984 || min === 9986 ? 'nearest' : 'linear',
        // These glTF filter values use nearest mip selection
        mipmapFilter: min === 9984 || min === 9985 ? 'nearest' : 'linear',
      },
    };
  }

  function viewOf(index: unknown) {
    const bufferView = item(json.bufferViews, index, 'buffer view');

    const which = bufferView.buffer;
    const data = Number.isInteger(which) ? buffers[which] : undefined;
    if (!data) {
      throw new Error(
        'The model keeps data outside its binary chunk; load it with `gltf.load`, which fetches it from beside the model, or pack it in with `gltf-transform copy in.glb out.glb`.',
      );
    }

    const name = `Buffer view ${index}`;
    const start = whole(bufferView.byteOffset ?? 0, `${name}'s byteOffset`);
    const end = start + whole(bufferView.byteLength, `${name}'s byteLength`, 1);
    if (end > data.byteLength) {
      throw new Error(
        `${name} exceeds buffer ${which}. It ends at byte ${end}, but the buffer has ${data.byteLength} bytes.`,
      );
    }

    const { byteStride } = bufferView;
    return {
      view: new DataView(data.buffer, data.byteOffset + start, end - start),
      stride: byteStride === undefined ? undefined : whole(byteStride, `${name}'s byteStride`, 1),
    };
  }

  function packed(
    place: { bufferView?: number; byteOffset?: number } | undefined,
    length: number,
    what: string,
  ) {
    const { view } = viewOf(place?.bufferView);
    const start = whole(place?.byteOffset ?? 0, `${what}' byteOffset`);

    if (start + length > view.byteLength) {
      throw new Error(`${what} run past the end of their buffer view.`);
    }
    return new DataView(view.buffer, view.byteOffset + start, length);
  }
  return { plan, meshOf, accessorOf, materialOf, imageOf };
}

function item<T extends object>(list: readonly T[] | undefined, index: unknown, what: string): T {
  const found: T | undefined =
    typeof index === 'number' && Number.isInteger(index) && index >= 0 && Array.isArray(list)
      ? list[index]
      : undefined;

  if (typeof found !== 'object' || found === null) {
    throw new Error(`The model has no ${what} ${index}.`);
  }
  return found;
}

function listOf<T>(value: readonly T[] | undefined, what: string): readonly T[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error(`${what} must be a list.`);
  return value;
}

function scalar(value: number | undefined, what: string): number | undefined {
  if (value !== undefined && !Number.isFinite(value)) throw new Error(`${what} isn't a number.`);
  return value;
}

function vector<T extends readonly number[]>(
  value: T | undefined,
  length: T['length'],
  what: string,
): T | undefined {
  // Check by index, since `every` skips the holes of a sparse array
  let numbers = Array.isArray(value) && value.length === length;
  for (let i = 0; numbers && i < length; i++) numbers = Number.isFinite(value![i]);

  if (value !== undefined && !numbers) throw new Error(`${what} isn't ${length} numbers.`);
  return value;
}

/** Names a value's type for an error, safely for any value. */
const tagOf = (value: unknown) => Object.prototype.toString.call(value).slice(8, -1);
const described = (value: unknown) =>
  value === undefined || value === null ? 'nothing' : `a ${tagOf(value)}`;

function whole(value: unknown, what: string, least: 0 | 1 = 0): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < least) {
    const kind = least ? 'positive whole number' : 'whole number';
    throw new Error(`${what} is ${JSON.stringify(value)}, not a ${kind}.`);
  }
  return value;
}

function getter(view: DataView, type: number, normalized: boolean): (at: number) => number {
  switch (type) {
    case 5120:
      return normalized ? (at) => Math.max(view.getInt8(at) / 127, -1) : (at) => view.getInt8(at);
    case 5121:
      return normalized ? (at) => view.getUint8(at) / 255 : (at) => view.getUint8(at);
    case 5122:
      return normalized
        ? (at) => Math.max(view.getInt16(at, true) / 32767, -1)
        : (at) => view.getInt16(at, true);
    case 5123:
      return normalized
        ? (at) => view.getUint16(at, true) / 65535
        : (at) => view.getUint16(at, true);
    case 5125:
      return (at) => view.getUint32(at, true);
    default:
      return (at) => view.getFloat32(at, true);
  }
}

/** Lists a strip's or fan's triangles in `out`, three corners each. */
function triangles(indices: Uint32Array, mode: number, out: Uint32Array) {
  for (let i = 0; i < out.length / 3; i++) {
    // Swap the first two indices of odd strip triangles to preserve winding
    const first = mode === FAN ? 0 : i % 2 === 0 ? i : i + 1;
    const second = mode === FAN ? i + 1 : i % 2 === 0 ? i + 1 : i;

    out[i * 3] = indices[first];
    out[i * 3 + 1] = indices[second];
    out[i * 3 + 2] = indices[i + 2];
  }
}

// Store 4×4 matrices in column-major order, as glTF requires

const IDENTITY = /* @__PURE__ */ mat4d.identity();
const AXES = [
  /* @__PURE__ */ vec3d.create(1, 0, 0),
  /* @__PURE__ */ vec3d.create(0, 1, 0),
  /* @__PURE__ */ vec3d.create(0, 0, 1),
];

function localOf(node: NodeJson, index: number) {
  const name = `Node ${index}`;
  const matrix = vector(node.matrix, 16, `${name}'s matrix`);
  if (matrix) return new Float64Array(matrix);

  const local = mat4d.fromQuat(vector(node.rotation, 4, `${name}'s rotation`) ?? [0, 0, 0, 1]);
  mat4d.scale(local, vector(node.scale, 3, `${name}'s scale`) ?? [1, 1, 1], local);
  return mat4d.setTranslation(
    local,
    vector(node.translation, 3, `${name}'s translation`) ?? [0, 0, 0],
    local,
  );
}

/** Normals require the inverse transpose. A mirrored transform reverses triangle winding. */
function pose(positions: Float32Array, normals: Float32Array | undefined, m: Float64Array) {
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    positions[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    positions[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    positions[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  }

  // Normalized normals need only the determinant sign when transformed by the cofactor matrix
  const a = m.subarray(0, 3);
  const b = m.subarray(4, 7);
  const c = m.subarray(8, 11);
  const x = vec3d.cross(b, c);
  const y = vec3d.cross(c, a);
  const z = vec3d.cross(a, b);
  const determinant = vec3d.dot(a, x);

  if (normals) {
    const sign = determinant < 0 ? -1 : 1;

    for (let i = 0; i < normals.length; i += 3) {
      const nx = normals[i];
      const ny = normals[i + 1];
      const nz = normals[i + 2];
      const ox = x[0] * nx + y[0] * ny + z[0] * nz;
      const oy = x[1] * nx + y[1] * ny + z[1] * nz;
      const oz = x[2] * nx + y[2] * ny + z[2] * nz;
      const scale = sign / (Math.hypot(ox, oy, oz) || 1);

      normals[i] = ox * scale;
      normals[i + 1] = oy * scale;
      normals[i + 2] = oz * scale;
    }
  }
  return determinant < 0;
}

function unit(v: Float64Array, fallback: Float64Array) {
  const length = vec3d.length(v);
  return length > 1e-12 ? vec3d.scale(v, 1 / length) : vec3d.clone(fallback);
}

const triple = (v: ArrayLike<number>): [number, number, number] => [v[0], v[1], v[2]];
const quad = (v: ArrayLike<number>): [number, number, number, number] => [v[0], v[1], v[2], v[3]];

function invert(m: Float64Array, what: string) {
  if (!(Math.abs(mat4d.determinant(m)) > 1e-30)) throw new Error(`${what} can't be inverted.`);
  return mat4d.inverse(m);
}

/** The arrays of a mesh being read. */
interface Arrays {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs: Float32Array;
  readonly uvs1: Float32Array | undefined;
  readonly colors: Float32Array | undefined;
  readonly indices: Uint32Array;
  readonly joints: Uint16Array | undefined;
  readonly weights: Float32Array | undefined;
}

/** The element type of each array a mesh returns. */
const ELEMENTS = {
  positions: Float32Array,
  normals: Float32Array,
  uvs: Float32Array,
  uvs1: Float32Array,
  colors: Float32Array,
  indices: Uint32Array,
  joints: Uint16Array,
  weights: Float32Array,
} satisfies Record<
  keyof Arrays,
  Float32ArrayConstructor | Uint32ArrayConstructor | Uint16ArrayConstructor
>;

function copy(source: Values, into: Values, v: number, o: number, width: number) {
  for (let c = 0; c < width; c++) into[o * width + c] = source[v * width + c];
}

/**
 * Gives each triangle corner of `local` its own vertex in `out`, from `vertex` and `corner` on,
 * with the triangle's normal.
 */
function flatten(
  from: Omit<Arrays, 'normals' | 'indices'>,
  local: Uint32Array,
  out: Arrays,
  vertex: number,
  corner: number,
) {
  const { positions } = from;
  const along = vec3d.create();
  const across = vec3d.create();
  const n = vec3d.create();
  for (let t = 0; t < local.length; t += 3) {
    const first = local[t] * 3;
    for (let c = 0; c < 3; c++) {
      along[c] = positions[local[t + 1] * 3 + c] - positions[first + c];
      across[c] = positions[local[t + 2] * 3 + c] - positions[first + c];
    }
    vec3d.cross(along, across, n);
    vec3d.scale(n, 1 / (vec3d.length(n) || 1), n);

    for (let k = 0; k < 3; k++) {
      const v = local[t + k];
      const o = vertex + t + k;
      out.indices[corner + t + k] = o;
      copy(positions, out.positions, v, o, 3);
      for (let c = 0; c < 3; c++) out.normals[o * 3 + c] = n[c];
      copy(from.uvs, out.uvs, v, o, 2);
      if (from.uvs1 && out.uvs1) copy(from.uvs1, out.uvs1, v, o, 2);
      if (from.colors && out.colors) copy(from.colors, out.colors, v, o, 4);
      if (from.joints && out.joints) copy(from.joints, out.joints, v, o, 4);
      if (from.weights && out.weights) copy(from.weights, out.weights, v, o, 4);
    }
  }
}
