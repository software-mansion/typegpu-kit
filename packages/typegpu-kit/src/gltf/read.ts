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
import type { Draco, DracoArray } from './draco.ts';

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
    KHR_draco_mesh_compression?: { bufferView?: number; attributes?: Record<string, unknown> };
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

const PATHS = new Map<unknown, ChannelData['path']>([
  ['translation', 'position'],
  ['rotation', 'rotation'],
  ['scale', 'scale'],
]);

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
   * Limits total vertex and index bytes calculated before reading mesh data. This prevents small
   * files from expanding through repeated mesh placement. Default 256 MiB.
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

  const file = (uri: string, what: string) =>
    uri.startsWith('data:') ? decoded(uri, what) : fetched(get, beside(url, uri), what);
  const buffers = listOf(json.buffers, "The model's buffers").map((buffer, index) =>
    typeof buffer?.uri === 'string'
      ? file(buffer.uri, `buffer ${index} of the model`)
      : index === 0
        ? bin
        : undefined,
  );
  const images = listOf(json.images, "The model's images").map(async (image, index) => {
    if (image?.bufferView !== undefined || typeof image?.uri !== 'string') return undefined;
    const encoded = await file(image.uri, `image ${index} of the model`);
    return { bytes: encoded, mimeType: image.mimeType ?? mimeTypeOf(image.uri) };
  });
  return modelOf(json, await Promise.all(buffers), await Promise.all(images), options);
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

  // Count expanded mesh bytes before reading because many nodes can repeat one large mesh
  const everyUvs1 = options.uvs1 ?? false;
  const size = placedBytes(json, everyUvs1, placing ? new Set<string>(parts) : undefined);
  if (size > maxBytes) {
    throw new Error(
      placing
        ? `The model's meshes take ${megabytes(size)}, past the ${megabytes(maxBytes)} of maxBytes.`
        : `The model's nodes place ${megabytes(size)} of meshes, past the ${megabytes(maxBytes)} of maxBytes. Read it with placements: true to read each mesh once.`,
    );
  }

  const file = readerOf(json, buffers, separate, options.draco, everyUvs1);
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

  const groups = new Map<string | undefined, Map<number | undefined, Piece[]>>();

  const shared = new Map<string, { readonly material: number | undefined; pieces: Piece[] }>();
  const indexOf = new Map<string, number>();
  const readMeshes = new Set<number>();
  const placed: PlacementData[] = [];

  // Find every node's part before reading meshes, since one skinned mesh binds its whole part
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

  for (const [node, world, index] of nodes) {
    if (node.mesh === undefined) continue;
    const owner = owners.get(index);

    if (placing && owner === undefined) {
      const mesh = item(json.meshes, node.mesh, 'mesh');
      const primitives = listOf(mesh.primitives, `Mesh ${node.mesh}'s primitives`);
      const keys = new Set(primitives.map((primitive) => `${node.mesh} ${primitive?.material}`));
      if (!readMeshes.has(node.mesh)) {
        readMeshes.add(node.mesh);
        for (const primitive of primitives) {
          const key = `${node.mesh} ${primitive?.material}`;
          let entry = shared.get(key);
          if (!entry) {
            entry = { material: primitive?.material, pieces: [] };
            shared.set(key, entry);
            indexOf.set(key, indexOf.size);
          }
          entry.pieces.push(file.part(primitive ?? {}, IDENTITY));
        }
      }
      const { position, rotation, scale } = decompose(world);
      for (const key of keys) {
        placed.push({ mesh: indexOf.get(key)!, position, rotation, scale });
      }
      continue;
    }

    // A skinned part binds to the skin as the model's meshes do, so its frame never applies
    const bound =
      owner === undefined || skinnedParts.has(owner)
        ? skeleton?.bind(node, index, world)
        : undefined;
    const into = owner === undefined ? world : mat4d.multiply(frames.get(owner)!.inverse, world);
    const byMaterial = groups.get(owner) ?? new Map<number | undefined, Piece[]>();
    groups.set(owner, byMaterial);

    const mesh = item(json.meshes, node.mesh, 'mesh');
    for (const primitive of listOf(mesh.primitives, `Mesh ${node.mesh}'s primitives`)) {
      const list = byMaterial.get(primitive?.material) ?? [];
      list.push(file.part(primitive ?? {}, bound?.world ?? into, bound?.binding));
      byMaterial.set(primitive?.material, list);
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

  const meshOf = (material: number | undefined, list: readonly Piece[]): MeshData => {
    const { uvs1, colors, joints, weights, ...rest } = joined(list);
    return {
      ...rest,
      ...(uvs1 && { uvs1 }),
      ...(colors && { extra: { color: colors } }),
      ...(joints && weights && { joints, weights }),
      material: file.materialOf(material),
    };
  };
  const meshesOf = (owner: string | undefined) =>
    [...(groups.get(owner) ?? [])].map(([material, list]) => meshOf(material, list));

  const named: Record<string, PartData> = {};
  for (const name of wanted) {
    const { position, rotation } = skinnedParts.has(name) ? MODEL_ORIGIN : frames.get(name)!;
    named[name] = { meshes: meshesOf(name), images, position, rotation };
  }

  const model = {
    meshes: placing
      ? [...shared.values()].map(({ material, pieces }) => meshOf(material, pieces))
      : meshesOf(undefined),
    images,
    parts: named as ModelData<Name>['parts'],
    ...(skeleton && { skin: skeleton.skin }),
    clips: clipsOf(json, file, nodes, turned),
  };
  return placing ? { ...model, placements: placed } : model;
}

type Posed = [node: NodeJson, world: Float64Array, index: number, parent: number];

/** The semantics `part` reads, which it asks Draco to decode. */
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

/** Reads an accessor of a primitive, from its buffer view or what Draco decoded. */
type Take = <T extends Float32Array | Uint32Array>(
  index: unknown,
  semantic: Semantic,
  Kind: new (length: number) => T,
) => T;

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
          `Node ${nameOf(node, at)}'s mesh isn't skinned and hangs from no joint of the skin, so it can't move with it. Read it as a part of its own.`,
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

      const path = PATHS.get(target.path);
      if (!path) {
        throw new Error(
          `${what}'s channel ${c} moves ${JSON.stringify(target.path)}, not a node's translation, rotation or scale.`,
        );
      }
      const node = item(json.nodes, target.node, 'node');
      const sampler: unknown = Number.isInteger(channel?.sampler)
        ? samplers[channel!.sampler!]
        : undefined;
      if (typeof sampler !== 'object' || sampler === null) {
        throw new Error(
          `${what}'s channel ${c} has no sampler ${JSON.stringify(channel?.sampler)}.`,
        );
      }
      const { input, output, interpolation = 'LINEAR' } = sampler as (typeof samplers)[number];
      const keyed = INTERPOLATIONS.get(interpolation);
      if (!keyed) {
        throw new Error(
          `${what}'s sampler ${channel!.sampler} interpolates ${JSON.stringify(interpolation)}, not LINEAR, STEP or CUBICSPLINE.`,
        );
      }

      const times = file.accessorOf(input, 'times', Float32Array);
      for (let i = 0; i < times.length; i++) {
        if (!Number.isFinite(times[i]) || (i > 0 && !(times[i] > times[i - 1]))) {
          throw new Error(`${what}'s sampler ${channel!.sampler} has times that don't increase.`);
        }
      }
      const values = file.accessorOf(output, target.path as 'translation', Float32Array);
      const width = path === 'rotation' ? 4 : 3;
      const keys = values.length / width / (keyed === 'cubic' ? 3 : 1);
      if (keys !== times.length) {
        throw new Error(
          `${what}'s sampler ${channel!.sampler} has ${times.length} times and ${keys} keys.`,
        );
      }
      if (!values.every(Number.isFinite)) {
        throw new Error(`${what}'s sampler ${channel!.sampler} has keys that aren't numbers.`);
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
    rotation: [...quatd.fromMat(basis)] as [number, number, number, number],
    scale: scale as [number, number, number],
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

/**
 * Budget 32 bytes per vertex for position, normal and UV coordinates, 16 more for colours, 24
 * more for skin data and 4 per index. Flat normals need a vertex per index. Placement mode
 * counts each non-part mesh once. Invalid counts contribute zero here and throw during reading.
 */
function placedBytes(json: Json, everyUvs1: boolean, placed?: ReadonlySet<string>) {
  const skinned = Array.isArray(json.skins) && json.skins.length > 0;
  const countOf = (index: unknown) => {
    const count = typeof index === 'number' ? json.accessors?.[index]?.count : undefined;
    return Number.isSafeInteger(count) && count! > 0 ? count! : 0;
  };

  let total = 0;
  const counted = new Set<number>();
  const inPart = new Set<number>();
  for (const [node, , index, parent] of posed(json)) {
    if (inPart.has(parent) || (typeof node.name === 'string' && placed?.has(node.name))) {
      inPart.add(index);
    }
    if (node.mesh === undefined) continue;
    if (placed && !inPart.has(index)) {
      if (counted.has(node.mesh)) continue;
      counted.add(node.mesh);
    }

    const mesh = item(json.meshes, node.mesh, 'mesh');
    for (const primitive of listOf(mesh.primitives, `Mesh ${node.mesh}'s primitives`)) {
      const vertices = countOf(primitive?.attributes?.POSITION);
      const listed = primitive?.indices === undefined ? vertices : countOf(primitive.indices);
      const strip = primitive?.mode === STRIP || primitive?.mode === FAN;
      const indices = strip ? Math.max(0, listed - 2) * 3 : listed;
      const kept = primitive?.attributes?.NORMAL === undefined ? indices : vertices;
      const colored = primitive?.attributes?.COLOR_0 !== undefined;
      const uvs1 =
        primitive?.attributes?.TEXCOORD_1 !== undefined &&
        (everyUvs1 || readsSecondSet(json, primitive.material));

      total += kept * ((colored ? 48 : 32) + (uvs1 ? 8 : 0) + (skinned ? 24 : 0)) + indices * 4;
    }
  }
  return total;
}

/** Tells whether a material's textures read `TEXCOORD_1`, before `gltf.read` checks them. */
function readsSecondSet(json: Json, index: unknown) {
  const material = typeof index === 'number' ? json.materials?.[index] : undefined;
  const pbr = material?.pbrMetallicRoughness;
  return [
    pbr?.baseColorTexture,
    pbr?.metallicRoughnessTexture,
    material?.normalTexture,
    material?.emissiveTexture,
    material?.occlusionTexture,
  ].some((info) => (info?.extensions?.KHR_texture_transform?.texCoord ?? info?.texCoord) === 1);
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

  let json: unknown;
  let bin: Uint8Array | undefined;
  for (let at = 12; at + 8 <= length;) {
    const size = view.getUint32(at, true);
    const type = view.getUint32(at + 4, true);
    if (at + 8 + size > length) throw new Error('A chunk of the binary glTF runs past its end.');
    const data = bytes.subarray(at + 8, at + 8 + size);

    if (type === JSON_CHUNK) {
      try {
        json = JSON.parse(utf8(data));
      } catch (error) {
        throw new Error(`The binary glTF contains invalid JSON. ${(error as Error).message}`, {
          cause: error,
        });
      }
    } else if (type === BIN_CHUNK) {
      bin ??= data;
    }
    at += 8 + size;
  }

  if (json === undefined) throw new Error('The binary glTF has no JSON chunk.');
  if (typeof json !== 'object' || json === null || Array.isArray(json)) {
    throw new Error("The binary glTF's JSON isn't an object.");
  }
  return { json: json as Json, bin };
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

  let json: unknown;
  try {
    json = JSON.parse(utf8(bytes.subarray(at)));
  } catch (error) {
    throw new Error(`${url} contains invalid JSON. ${(error as Error).message}`, { cause: error });
  }
  const version = (json as Json).asset?.version;
  if (typeof version === 'string' && !version.startsWith('2.')) {
    throw new Error(`gltf.load reads glTF 2, got version ${version}.`);
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

function mimeTypeOf(uri: string): string {
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

interface Piece {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs: Float32Array;
  readonly uvs1?: Float32Array | undefined;
  readonly colors?: Float32Array | undefined;
  readonly indices: Uint32Array;
  readonly joints?: Uint16Array | undefined;
  readonly weights?: Float32Array | undefined;
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

  function part(primitive: PrimitiveJson, world: Float64Array, binding?: Binding): Piece {
    const mode = primitive.mode ?? TRIANGLES;
    if (mode !== TRIANGLES && mode !== STRIP && mode !== FAN) {
      throw new Error(`gltf.read reads triangles, got a primitive of mode ${mode}.`);
    }

    const { POSITION, NORMAL, TEXCOORD_0, TEXCOORD_1, COLOR_0 } = primitive.attributes ?? {};
    if (POSITION === undefined) throw new Error('A primitive has no POSITION.');

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
    const unpacked = compressed && draco ? decodedOf(primitive, draco) : undefined;
    const take: Take = (index, semantic, Kind) =>
      accessorOf(index, semantic, Kind, unpacked?.values.get(index));

    const positions = take(POSITION, 'POSITION', Float32Array);
    const count = positions.length / 3;
    const normals = NORMAL === undefined ? undefined : take(NORMAL, 'NORMAL', Float32Array);
    const uvs =
      TEXCOORD_0 === undefined
        ? new Float32Array(count * 2)
        : take(TEXCOORD_0, 'TEXCOORD_0', Float32Array);

    if (normals && normals.length !== count * 3) {
      throw new Error(`A primitive has ${count} positions and ${normals.length / 3} normals.`);
    }
    if (uvs.length !== count * 2) {
      throw new Error(`A primitive has ${count} positions and ${uvs.length / 2} uvs.`);
    }

    let uvs1: Float32Array | undefined;
    const sampled = readsUvs1(primitive.material);
    if (sampled && TEXCOORD_1 === undefined) {
      throw new Error(
        `Material ${primitive.material} reads TEXCOORD_1, and a primitive drawn with it has none.`,
      );
    }
    if (TEXCOORD_1 !== undefined && (sampled || everyUvs1)) {
      uvs1 = take(TEXCOORD_1, 'TEXCOORD_1', Float32Array);
      if (uvs1.length !== count * 2) {
        throw new Error(
          `A primitive has ${count} positions and ${uvs1.length / 2} TEXCOORD_1 uvs.`,
        );
      }
    }

    const colors =
      COLOR_0 === undefined
        ? undefined
        : rgba(
            take(COLOR_0, 'COLOR_0', Float32Array),
            item(json.accessors, COLOR_0, 'accessor').type === 'VEC3' ? 3 : 4,
          );
    if (colors && colors.length !== count * 4) {
      throw new Error(`A primitive has ${count} positions and ${colors.length / 4} colours.`);
    }

    // Draco decodes a list of triangles, whatever the primitive's mode
    const listed = unpacked
      ? unpacked.indices.slice()
      : primitive.indices !== undefined
        ? take(primitive.indices, 'indices', Uint32Array)
        : Uint32Array.from({ length: count }, (_, i) => i);
    const shape = unpacked ? TRIANGLES : mode;
    const outside = listed.find((index) => index >= count);

    if (outside !== undefined) {
      throw new Error(`A primitive's indices reach vertex ${outside}, and it has ${count}.`);
    }
    if (shape === TRIANGLES && listed.length % 3 !== 0) {
      throw new Error(
        `A primitive of triangles lists ${listed.length} vertices, which isn't a multiple of three.`,
      );
    }
    const indices = triangles(listed, shape);
    const { joints, weights } = binding ? skinning(primitive, count, binding, take) : {};

    const flipped = pose(positions, normals, world);
    if (flipped) {
      for (let i = 0; i < indices.length; i += 3) {
        const second = indices[i + 1];
        indices[i + 1] = indices[i + 2];
        indices[i + 2] = second;
      }
    }

    const piece = { positions, uvs, uvs1, colors, indices, joints, weights };
    return normals ? { ...piece, normals } : flat(piece);
  }

  function skinning(primitive: PrimitiveJson, count: number, binding: Binding, take: Take) {
    const joints = new Uint16Array(count * 4);
    const weights = new Float32Array(count * 4);
    if ('joint' in binding) {
      for (let i = 0; i < count; i++) {
        joints[i * 4] = binding.joint;
        weights[i * 4] = 1;
      }
      return { joints, weights };
    }

    const { JOINTS_0, WEIGHTS_0, JOINTS_1 } = primitive.attributes ?? {};
    if (JOINTS_1 !== undefined) {
      throw new Error('A primitive has JOINTS_1, and gltf.read reads four joints a vertex.');
    }
    if (JOINTS_0 === undefined || WEIGHTS_0 === undefined) {
      throw new Error('A primitive of a skinned mesh has no JOINTS_0 or no WEIGHTS_0.');
    }
    const listed = take(JOINTS_0, 'JOINTS_0', Uint32Array);
    const shares = take(WEIGHTS_0, 'WEIGHTS_0', Float32Array);
    if (listed.length !== count * 4 || shares.length !== count * 4) {
      throw new Error(
        `A primitive has ${count} positions, ${listed.length / 4} joints and ${shares.length / 4} weights.`,
      );
    }

    const { remap } = binding;
    for (let i = 0; i < count; i++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        const weight = shares[i * 4 + k];
        if (!(weight >= 0 && weight <= 1)) {
          throw new Error(`Vertex ${i} of a skinned primitive has a weight of ${weight}.`);
        }
        sum += weight;
      }
      if (!(sum > 0)) throw new Error(`Vertex ${i} of a skinned primitive has no weight.`);

      // Normalize weights so they sum to 1 despite file precision
      for (let k = 0; k < 4; k++) {
        const at = i * 4 + k;
        const joint = listed[at];
        if (joint >= remap.length) {
          throw new Error(
            `Vertex ${i} of a skinned primitive is bound to joint ${joint}, and the skin has ${remap.length}.`,
          );
        }
        joints[at] = remap[joint];
        weights[at] = shares[at] / sum;
      }
    }
    return { joints, weights };
  }

  /**
   * Decode a compressed primitive's attributes that `part` reads, keyed by accessor, and its
   * indices.
   */
  function decodedOf(primitive: PrimitiveJson, decode: Draco): Decoded {
    const cached = decodedPrimitives.get(primitive);
    if (cached) return cached;

    const extension = primitive.extensions![DRACO]!;
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
      return [{ index, id, componentType }];
    });

    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    const mesh = decode(bytes, wanted);
    if (!(mesh.indices instanceof Uint32Array)) {
      throw new Error('The Draco decoder returned indices that are not a Uint32Array.');
    }
    if (mesh.attributes.length !== wanted.length) {
      throw new Error(
        `The Draco decoder returned ${mesh.attributes.length} attributes for ${wanted.length} asked.`,
      );
    }

    const values = new Map<unknown, DecodedValues>();
    wanted.forEach(({ index, componentType }, k) => {
      const array = mesh.attributes[k];
      if (!(array instanceof ARRAYS.get(componentType)!)) {
        throw new Error(
          `The Draco decoder returned a ${array.constructor.name} for accessor ${index}, of component type ${componentType}.`,
        );
      }
      values.set(index, { array, componentType });
    });

    const unpacked = { values, indices: mesh.indices };
    decodedPrimitives.set(primitive, unpacked);
    return unpacked;
  }

  /**
   * Check accessor type, component type and buffer bounds before allocating arrays. Map
   * normalized integers to [-1, 1] or [0, 1]. Read `unpacked` values in place of a buffer view for
   * an accessor Draco compressed.
   */
  function accessorOf<T extends Float32Array | Uint32Array>(
    index: unknown,
    semantic: Semantic,
    Kind: new (length: number) => T,
    unpacked?: DecodedValues,
  ): T {
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
    const out = new Kind(count * width);

    if (unpacked) {
      const { array } = unpacked;
      if (array.length !== out.length) {
        throw new Error(`${name} holds ${out.length} values, and Draco decoded ${array.length}.`);
      }
      const view = new DataView(array.buffer, array.byteOffset, array.byteLength);
      const get = getter(view, unpacked.componentType, normalized);
      for (let i = 0; i < out.length; i++) out[i] = get(i * array.BYTES_PER_ELEMENT);
    } else {
      const { view, stride = element } = viewOf(accessor.bufferView);
      const start = whole(accessor.byteOffset ?? 0, `${name}'s byteOffset`);
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

      const get = getter(view, componentType, normalized);
      for (let i = 0; i < count; i++) {
        for (let c = 0; c < width; c++) out[i * width + c] = get(start + i * stride + c * size);
      }
    }

    const { sparse } = accessor;
    if (sparse) {
      const replaced = whole(sparse.count, `${name}'s sparse count`, 1);
      const slotType = sparse.indices?.componentType;

      if (replaced > count) {
        throw new Error(`${name} replaces ${replaced} of its ${count} elements.`);
      }
      if (slotType === undefined || !SEMANTICS.indices.components.includes(slotType)) {
        throw new Error(
          `${name}'s sparse indices have component type ${slotType}, which indices can't have.`,
        );
      }

      const slotSize = SIZES.get(slotType)!;
      const slots = packed(sparse.indices, replaced * slotSize, `${name}'s sparse indices`);
      const values = packed(sparse.values, replaced * element, `${name}'s sparse values`);
      const slotAt = getter(slots, slotType, false);
      const value = getter(values, componentType, normalized);

      for (let k = 0; k < replaced; k++) {
        const i = slotAt(k * slotSize);
        if (i >= count) {
          throw new Error(`${name}'s sparse indices reach element ${i}, and it has ${count}.`);
        }
        for (let c = 0; c < width; c++) out[i * width + c] = value((k * width + c) * size);
      }
    }
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
      mimeType: image.mimeType ?? 'image/png',
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
  return { part, accessorOf, materialOf, imageOf };
}

function item<T>(list: readonly T[] | undefined, index: unknown, what: string): NonNullable<T> {
  const found: unknown =
    Array.isArray(list) && Number.isInteger(index) && (index as number) >= 0
      ? list[index as number]
      : undefined;

  if (typeof found !== 'object' || found === null) {
    throw new Error(`The model has no ${what} ${index}.`);
  }
  return found as NonNullable<T>;
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
  const numbers =
    Array.isArray(value) && value.length === length && value.every((v) => Number.isFinite(v));

  if (value !== undefined && !numbers) throw new Error(`${what} isn't ${length} numbers.`);
  return value;
}

function whole(value: unknown, what: string, least: 0 | 1 = 0): number {
  if (!Number.isSafeInteger(value) || (value as number) < least) {
    const kind = least ? 'positive whole number' : 'whole number';
    throw new Error(`${what} is ${JSON.stringify(value)}, not a ${kind}.`);
  }
  return value as number;
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

function triangles(indices: Uint32Array, mode: number): Uint32Array {
  if (mode === TRIANGLES) return indices;

  const count = Math.max(indices.length - 2, 0);
  const out = new Uint32Array(count * 3);
  for (let i = 0; i < count; i++) {
    // Swap the first two indices of odd strip triangles to preserve winding
    const first = mode === FAN ? 0 : i % 2 === 0 ? i : i + 1;
    const second = mode === FAN ? i + 1 : i % 2 === 0 ? i + 1 : i;

    out[i * 3] = indices[first];
    out[i * 3 + 1] = indices[second];
    out[i * 3 + 2] = indices[i + 2];
  }
  return out;
}

// Store 4×4 matrices in column-major order, as glTF requires

const IDENTITY = mat4d.identity();
const AXES = [vec3d.create(1, 0, 0), vec3d.create(0, 1, 0), vec3d.create(0, 0, 1)];

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

const triple = (v: ArrayLike<number>): Vec3 => [v[0], v[1], v[2]];
const quad = (v: ArrayLike<number>): Vec4 => [v[0], v[1], v[2], v[3]];

function invert(m: Float64Array, what: string) {
  if (!(Math.abs(mat4d.determinant(m)) > 1e-30)) throw new Error(`${what} can't be inverted.`);
  return mat4d.inverse(m);
}

function rgba(colors: Float32Array, components: number) {
  if (components === 4) return colors;

  const out = new Float32Array((colors.length / 3) * 4).fill(1);
  for (let i = 0; i < colors.length / 3; i++) out.set(colors.subarray(i * 3, i * 3 + 3), i * 4);
  return out;
}

function flat(piece: Omit<Piece, 'normals'>): Piece {
  const { positions, uvs, uvs1, colors, indices, joints, weights } = piece;
  const out = {
    positions: new Float32Array(indices.length * 3),
    normals: new Float32Array(indices.length * 3),
    uvs: new Float32Array(indices.length * 2),
    uvs1: uvs1 && new Float32Array(indices.length * 2),
    colors: colors && new Float32Array(indices.length * 4),
    indices: Uint32Array.from({ length: indices.length }, (_, i) => i),
    joints: joints && new Uint16Array(indices.length * 4),
    weights: weights && new Float32Array(indices.length * 4),
  };

  const along = vec3d.create();
  const across = vec3d.create();
  const n = vec3d.create();
  for (let t = 0; t < indices.length; t += 3) {
    const first = indices[t] * 3;
    for (let c = 0; c < 3; c++) {
      along[c] = positions[indices[t + 1] * 3 + c] - positions[first + c];
      across[c] = positions[indices[t + 2] * 3 + c] - positions[first + c];
    }
    vec3d.cross(along, across, n);
    vec3d.scale(n, 1 / (vec3d.length(n) || 1), n);

    for (let k = 0; k < 3; k++) {
      const v = indices[t + k];
      out.positions.set(positions.subarray(v * 3, v * 3 + 3), (t + k) * 3);
      out.normals.set(n, (t + k) * 3);
      out.uvs.set(uvs.subarray(v * 2, v * 2 + 2), (t + k) * 2);
      out.uvs1?.set(uvs1!.subarray(v * 2, v * 2 + 2), (t + k) * 2);
      out.colors?.set(colors!.subarray(v * 4, v * 4 + 4), (t + k) * 4);
      out.joints?.set(joints!.subarray(v * 4, v * 4 + 4), (t + k) * 4);
      out.weights?.set(weights!.subarray(v * 4, v * 4 + 4), (t + k) * 4);
    }
  }
  return out;
}

function joined(pieces: readonly Piece[]): Piece {
  if (pieces.length === 1) return pieces[0];

  const total = (key: 'positions' | 'normals' | 'uvs' | 'indices') =>
    pieces.reduce((sum, piece) => sum + piece[key].length, 0);
  const vertexCount = total('positions') / 3;
  // All pieces in a model must either use its skin or have no skin
  const skinned = pieces[0].joints !== undefined;
  const out = {
    positions: new Float32Array(total('positions')),
    normals: new Float32Array(total('normals')),
    uvs: new Float32Array(total('uvs')),
    // Pieces without TEXCOORD_1 read zero, as meshes without uvs1 do in a pool
    uvs1: pieces.some((piece) => piece.uvs1) ? new Float32Array(total('uvs')) : undefined,
    colors: pieces.some((piece) => piece.colors)
      ? new Float32Array(vertexCount * 4).fill(1)
      : undefined,
    indices: new Uint32Array(total('indices')),
    joints: skinned ? new Uint16Array(vertexCount * 4) : undefined,
    weights: skinned ? new Float32Array(vertexCount * 4) : undefined,
  };

  let vertices = 0;
  let indices = 0;
  for (const piece of pieces) {
    out.positions.set(piece.positions, vertices * 3);
    out.normals.set(piece.normals, vertices * 3);
    out.uvs.set(piece.uvs, vertices * 2);
    if (piece.uvs1) out.uvs1?.set(piece.uvs1, vertices * 2);
    if (piece.colors) out.colors?.set(piece.colors, vertices * 4);
    if (piece.joints) out.joints?.set(piece.joints, vertices * 4);
    if (piece.weights) out.weights?.set(piece.weights, vertices * 4);
    for (let i = 0; i < piece.indices.length; i++)
      out.indices[indices + i] = piece.indices[i] + vertices;

    vertices += piece.positions.length / 3;
    indices += piece.indices.length;
  }
  return out;
}
