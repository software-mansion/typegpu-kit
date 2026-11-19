// Packs the night market's models from CC0 packs into its models/ folder:
//
//   node packages/typegpu-kit-examples/scripts/pack-night-market.ts <kaykit> <kenney>
//
// <kaykit> holds KayKit_Medieval_Hexagon_Pack_1.0_FREE, KayKit_Adventurers_2.0_FREE and EXTRA, and
// KayKit_Character_Animations_1.1, unzipped. <kenney> is the "3D assets" folder of Kenney's Game
// Assets All-in-1. market.glb holds every prop and building, each under a root node named for it,
// with images shared and glowing triangles split into a material ending in /glow. The characters
// are copied as they are, and clips.glb keeps the rig's nodes and the clips the scene plays.

import { createHash } from 'node:crypto';
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const [kaykit, kenney] = process.argv.slice(2);
if (!kaykit || !kenney) throw new Error('Usage: pack-night-market.ts <kaykit> <kenney>');
const out = path.resolve(import.meta.dirname, '../src/night-market/models');

const hex = path.join(kaykit, 'KayKit_Medieval_Hexagon_Pack_1.0_FREE/Assets/gltf');
const goods = path.join(kaykit, 'KayKit_Adventurers_2.0_EXTRA/Assets/gltf');
const people = path.join(kaykit, 'KayKit_Adventurers_2.0_FREE/Characters/gltf');
const rig = path.join(kaykit, 'KayKit_Character_Animations_1.1/Animations/gltf/Rig_Medium');
const kit = (name: string) => path.join(kenney, name, 'Models/GLB format');
const town = kit('Fantasy Town Kit');
const graves = kit('Graveyard Kit');
const holiday = kit('Holiday Kit');
const survival = kit('Survival Kit');

type Rgb = readonly [number, number, number];
/** Which texels glow: KayKit's window-pane cell, Kenney's lamp glass, bulbs and flames. */
const GLOWS = {
  pane: (u: number, v: number) => Math.floor(u * 8) === 6 && Math.floor(v * 4) === 1,
  yellow: (_u: number, _v: number, [r, g, b]: Rgb) => r > 240 && g > 150 && b < 110,
  bulbs: (_u: number, _v: number, [r, g, b]: Rgb) => Math.max(r, g, b) - Math.min(r, g, b) > 80,
  flame: (_u: number, _v: number, [r, g, b]: Rgb) => r > 195 && r < 225 && g < 140 && b < 100,
};
type Glow = keyof typeof GLOWS;

/** Each model of market.glb: its file, and which of its texels glow. */
const MARKET: Record<string, readonly [file: string, glow?: Glow]> = {
  'market-red': [`${hex}/buildings/red/building_market_red.gltf`],
  'market-blue': [`${hex}/buildings/blue/building_market_blue.gltf`],
  'market-yellow': [`${hex}/buildings/yellow/building_market_yellow.gltf`],
  tavern: [`${hex}/buildings/red/building_tavern_red.gltf`, 'pane'],
  well: [`${hex}/buildings/blue/building_well_blue.gltf`],
  'home-a': [`${hex}/buildings/yellow/building_home_A_yellow.gltf`, 'pane'],
  'home-b': [`${hex}/buildings/green/building_home_B_green.gltf`, 'pane'],
  barrel: [`${hex}/decoration/props/barrel.gltf`],
  bucket: [`${hex}/decoration/props/bucket_water.gltf`],
  'crate-a': [`${hex}/decoration/props/crate_A_big.gltf`],
  'crate-open': [`${hex}/decoration/props/crate_open.gltf`],
  'crate-fruit': [`${hex}/decoration/props/crate_long_A.gltf`],
  'crate-goods': [`${hex}/decoration/props/crate_long_B.gltf`],
  sack: [`${hex}/decoration/props/sack.gltf`],
  'flag-red': [`${hex}/decoration/props/flag_red.gltf`],
  'flag-blue': [`${hex}/decoration/props/flag_blue.gltf`],
  tree: [`${hex}/decoration/nature/tree_single_A.gltf`],
  'potion-huge-blue': [`${goods}/potion_huge_blue.gltf`],
  'potion-large-red': [`${goods}/potion_large_red.gltf`],
  'potion-medium-green': [`${goods}/potion_medium_green.gltf`],
  'potion-small-orange': [`${goods}/potion_small_orange.gltf`],
  'potion-large-green': [`${goods}/potion_large_green.gltf`],
  'potion-medium-red': [`${goods}/potion_medium_red.gltf`],
  'potion-small-blue': [`${goods}/potion_small_blue.gltf`],
  'stall-red': [`${town}/stall-red.glb`],
  post: [`${town}/pillar-wood.glb`],
  table: [`${town}/stall.glb`],
  'banner-green': [`${town}/banner-green.glb`],
  lights: [`${holiday}/lights-colored.glb`, 'bulbs'],
  'street-lamp': [`${holiday}/lantern.glb`, 'yellow'],
  'wall-lamp': [`${holiday}/lantern-hanging.glb`, 'yellow'],
  lantern: [`${graves}/lantern-glass.glb`, 'yellow'],
  candles: [`${graves}/candle-multiple.glb`, 'flame'],
  logs: [`${survival}/campfire-pit.glb`],
};

const CHARACTERS = { mage: 'Mage.glb', rogue: 'Rogue_Hooded.glb' };
const CLIPS = {
  'Rig_Medium_General.glb': ['Idle_A', 'Idle_B'],
  'Rig_Medium_MovementBasic.glb': ['Walking_A', 'Running_A'],
  'Rig_Medium_Tools.glb': ['Holding_A'],
  'Rig_Medium_Simulation.glb': ['Waving'],
};

interface Node {
  name?: string;
  mesh?: number;
  skin?: number;
  camera?: number;
  children?: number[];
  [key: string]: unknown;
}
interface TextureInfo {
  index: number;
  [key: string]: unknown;
}
interface Material {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorTexture?: TextureInfo;
    metallicRoughnessTexture?: TextureInfo;
    [key: string]: unknown;
  };
  normalTexture?: TextureInfo;
  emissiveTexture?: TextureInfo;
  occlusionTexture?: TextureInfo;
  emissiveFactor?: number[];
  [key: string]: unknown;
}
interface Primitive {
  attributes: Record<string, number>;
  indices?: number;
  material?: number;
  [key: string]: unknown;
}
interface Accessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  [key: string]: unknown;
}
interface View {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number;
}
interface Animation {
  name?: string;
  samplers: { input: number; output: number; [key: string]: unknown }[];
  channels: { sampler: number; target: { node?: number; path: string } }[];
}
interface Gltf {
  asset: { version: string; generator?: string };
  scene?: number;
  scenes: { nodes: number[] }[];
  nodes: Node[];
  meshes?: { name?: string; primitives: Primitive[] }[];
  accessors: Accessor[];
  bufferViews: View[];
  buffers?: { uri?: string; byteLength: number }[];
  materials?: Material[];
  textures?: { sampler?: number; source: number }[];
  images?: { uri?: string; name?: string; mimeType?: string; bufferView?: number }[];
  samplers?: object[];
  animations?: Animation[];
  extensionsUsed?: string[];
}
interface Loaded {
  json: Gltf;
  buffers: Buffer[];
  folder: string;
}

const padded = (length: number) => Math.ceil(length / 4) * 4;

function load(file: string): Loaded {
  const raw = readFileSync(file);
  const folder = path.dirname(file);
  const external = (uri: string) => readFileSync(path.join(folder, decodeURIComponent(uri)));
  if (!file.endsWith('.glb')) {
    const json = JSON.parse(raw.toString()) as Gltf;
    return { json, buffers: (json.buffers ?? []).map(({ uri }) => external(uri!)), folder };
  }
  const jsonLength = raw.readUInt32LE(12);
  const json = JSON.parse(raw.subarray(20, 20 + jsonLength).toString()) as Gltf;
  const binAt = 20 + jsonLength;
  const bin =
    binAt < raw.length
      ? raw.subarray(binAt + 8, binAt + 8 + raw.readUInt32LE(binAt))
      : Buffer.alloc(0);
  return {
    json,
    buffers: (json.buffers ?? []).map(({ uri }) => (uri ? external(uri) : bin)),
    folder,
  };
}

const viewBytes = ({ json, buffers }: Loaded, index: number) => {
  const { buffer, byteOffset = 0, byteLength } = json.bufferViews[index]!;
  return buffers[buffer]!.subarray(byteOffset, byteOffset + byteLength);
};

const imageBytes = (loaded: Loaded, index: number) => {
  const image = loaded.json.images![index]!;
  return image.uri
    ? readFileSync(path.join(loaded.folder, decodeURIComponent(image.uri)))
    : viewBytes(loaded, image.bufferView!);
};

/** A GLB being written: one binary buffer that grows as views are added. */
function writer(generator: string) {
  const doc: Gltf = {
    asset: { version: '2.0', generator },
    scene: 0,
    scenes: [{ nodes: [] }],
    nodes: [],
    accessors: [],
    bufferViews: [],
  };
  const chunks: Buffer[] = [];
  let length = 0;
  return {
    doc,
    /** Appends bytes as a view and returns its index. */
    view(bytes: Buffer, extra: Omit<View, 'buffer' | 'byteOffset' | 'byteLength'> = {}) {
      const pad = padded(length) - length;
      if (pad) chunks.push(Buffer.alloc(pad));
      doc.bufferViews.push({
        buffer: 0,
        byteOffset: length + pad,
        byteLength: bytes.length,
        ...extra,
      });
      chunks.push(bytes);
      length += pad + bytes.length;
      return doc.bufferViews.length - 1;
    },
    write(file: string) {
      const bin = Buffer.concat([...chunks, Buffer.alloc(padded(length) - length)]);
      doc.buffers = [{ byteLength: bin.length }];
      const text = Buffer.from(JSON.stringify(doc));
      const json = Buffer.concat([text, Buffer.alloc(padded(text.length) - text.length, 0x20)]);
      const header = Buffer.alloc(20);
      header.writeUInt32LE(0x46546c67, 0);
      header.writeUInt32LE(2, 4);
      header.writeUInt32LE(28 + json.length + bin.length, 8);
      header.writeUInt32LE(json.length, 12);
      header.writeUInt32LE(0x4e4f534a, 16);
      const binHeader = Buffer.alloc(8);
      binHeader.writeUInt32LE(bin.length, 0);
      binHeader.writeUInt32LE(0x004e4942, 4);
      const glb = Buffer.concat([header, json, binHeader, bin]);
      writeFileSync(file, glb);
      process.stdout.write(`${path.basename(file)}: ${glb.length} bytes\n`);
    },
  };
}

const ARRAYS = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const WIDTHS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function read(loaded: Loaded, index: number) {
  const accessor = loaded.json.accessors[index]!;
  const Typed = ARRAYS[accessor.componentType as keyof typeof ARRAYS];
  const bytes = viewBytes(loaded, accessor.bufferView!).subarray(accessor.byteOffset ?? 0);
  const count = accessor.count * WIDTHS[accessor.type]!;
  const end = bytes.byteOffset + count * Typed.BYTES_PER_ELEMENT;
  return new Typed((bytes.buffer as ArrayBuffer).slice(bytes.byteOffset, end));
}

/**
 * Moves the triangles whose middle samples a glowing texel into primitives of their own, with a
 * material named `glow`. Lamp glass glows with its texture's colour; panes glow by a factor alone.
 */
async function splitGlow(loaded: Loaded, glow: Glow) {
  const { json, buffers } = loaded;
  const { data, info } = await sharp(imageBytes(loaded, 0))
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const glows = GLOWS[glow];
  const indices = (list: number[]) => {
    const bytes = Buffer.from(new Uint32Array(list).buffer);
    buffers.push(bytes);
    json.bufferViews.push({ buffer: buffers.length - 1, byteLength: bytes.length, target: 34963 });
    json.accessors.push({
      bufferView: json.bufferViews.length - 1,
      componentType: 5125,
      count: list.length,
      type: 'SCALAR',
    });
    return json.accessors.length - 1;
  };
  for (const mesh of json.meshes ?? []) {
    const kept: Primitive[] = [];
    const lit: Primitive[] = [];
    for (const primitive of mesh.primitives) {
      const uv = read(loaded, primitive.attributes.TEXCOORD_0!);
      const index = read(loaded, primitive.indices!);
      const bright: number[] = [];
      const dark: number[] = [];
      for (let t = 0; t < index.length; t += 3) {
        let u = 0;
        let v = 0;
        for (let k = 0; k < 3; k++) {
          u += uv[index[t + k]! * 2]! / 3;
          v += uv[index[t + k]! * 2 + 1]! / 3;
        }
        u -= Math.floor(u);
        v -= Math.floor(v);
        const x = Math.min(Math.floor(u * info.width), info.width - 1);
        const y = Math.min(Math.floor(v * info.height), info.height - 1);
        const at = (y * info.width + x) * 4;
        const texel: Rgb = [data[at]!, data[at + 1]!, data[at + 2]!];
        (glows(u, v, texel) ? bright : dark).push(index[t]!, index[t + 1]!, index[t + 2]!);
      }
      if (bright.length === 0) {
        kept.push(primitive);
        continue;
      }
      if (dark.length > 0) kept.push({ ...primitive, indices: indices(dark) });
      const base = json.materials![primitive.material!]!;
      const texture = base.pbrMetallicRoughness?.baseColorTexture;
      json.materials!.push({
        ...base,
        name: 'glow',
        emissiveFactor: [1, 1, 1],
        ...(glow !== 'pane' && texture ? { emissiveTexture: { ...texture } } : {}),
      });
      lit.push({ ...primitive, indices: indices(bright), material: json.materials!.length - 1 });
    }
    mesh.primitives = [...kept, ...lit];
  }
}

async function packMarket() {
  const glb = writer('night market pack');
  const { doc } = glb;
  const images = new Map<string, number>();
  const samplers = new Map<string, number>();
  for (const [name, [file, glow]] of Object.entries(MARKET)) {
    const loaded = load(file);
    const { json } = loaded;
    if (glow) await splitGlow(loaded, glow);

    const views = json.bufferViews.map(({ byteStride, target }, i) =>
      glb.view(Buffer.from(viewBytes(loaded, i)), {
        ...(byteStride ? { byteStride } : {}),
        ...(target ? { target } : {}),
      }),
    );
    const accessors = json.accessors.map((accessor) => {
      doc.accessors.push({ ...accessor, bufferView: views[accessor.bufferView!]! });
      return doc.accessors.length - 1;
    });
    const imageIds = (json.images ?? []).map((image, i) => {
      const bytes = imageBytes(loaded, i);
      const hash = createHash('md5').update(bytes).digest('hex');
      if (!images.has(hash)) {
        (doc.images ??= []).push({
          name: path.basename(image.uri ?? image.name ?? 'image'),
          mimeType: image.mimeType ?? 'image/png',
          bufferView: glb.view(Buffer.from(bytes)),
        });
        images.set(hash, doc.images.length - 1);
      }
      return images.get(hash)!;
    });
    const samplerIds = (json.samplers ?? []).map((sampler) => {
      const key = JSON.stringify(sampler);
      if (!samplers.has(key)) {
        (doc.samplers ??= []).push(sampler);
        samplers.set(key, doc.samplers.length - 1);
      }
      return samplers.get(key)!;
    });
    const textureIds = (json.textures ?? []).map(({ sampler, source }) => {
      (doc.textures ??= []).push({
        ...(sampler === undefined ? {} : { sampler: samplerIds[sampler]! }),
        source: imageIds[source]!,
      });
      return doc.textures.length - 1;
    });
    const retexture = (info: TextureInfo | undefined) =>
      info && { ...info, index: textureIds[info.index]! };
    const materialIds = (json.materials ?? []).map((material) => {
      const {
        pbrMetallicRoughness: pbr = {},
        normalTexture,
        emissiveTexture,
        occlusionTexture,
      } = material;
      (doc.materials ??= []).push({
        ...material,
        name: `${name}/${material.name ?? 'material'}`,
        pbrMetallicRoughness: {
          ...pbr,
          ...(pbr.baseColorTexture ? { baseColorTexture: retexture(pbr.baseColorTexture)! } : {}),
          ...(pbr.metallicRoughnessTexture
            ? { metallicRoughnessTexture: retexture(pbr.metallicRoughnessTexture)! }
            : {}),
        },
        ...(normalTexture ? { normalTexture: retexture(normalTexture)! } : {}),
        ...(emissiveTexture ? { emissiveTexture: retexture(emissiveTexture)! } : {}),
        ...(occlusionTexture ? { occlusionTexture: retexture(occlusionTexture)! } : {}),
      });
      return doc.materials.length - 1;
    });
    const meshIds = (json.meshes ?? []).map((mesh) => {
      (doc.meshes ??= []).push({
        name: `${name}/${mesh.name ?? 'mesh'}`,
        primitives: mesh.primitives.map(({ attributes, indices, material, ...rest }) => ({
          ...rest,
          // JSON leaves an undefined field out, which drops the morph targets
          targets: undefined,
          attributes: Object.fromEntries(
            Object.entries(attributes).map(([key, value]) => [key, accessors[value]!]),
          ),
          ...(indices === undefined ? {} : { indices: accessors[indices]! }),
          ...(material === undefined ? {} : { material: materialIds[material]! }),
        })),
      });
      return doc.meshes.length - 1;
    });

    const first = doc.nodes.length;
    for (const { mesh, children, skin: _skin, camera: _camera, ...node } of json.nodes) {
      doc.nodes.push({
        ...node,
        name: `${name}/${node.name ?? 'node'}`,
        ...(mesh === undefined ? {} : { mesh: meshIds[mesh]! }),
        ...(children ? { children: children.map((child) => child + first) } : {}),
      });
    }
    const roots = json.scenes[json.scene ?? 0]!.nodes.map((node) => node + first);
    doc.nodes.push({ name, children: roots });
    doc.scenes[0]!.nodes.push(doc.nodes.length - 1);
  }
  glb.write(path.join(out, 'market.glb'));
}

/** The rig's nodes and the chosen clips, which address joints by name. */
function packClips() {
  const glb = writer('night market clips');
  const { doc } = glb;
  let names: string[] = [];
  for (const [file, wanted] of Object.entries(CLIPS)) {
    const loaded = load(path.join(rig, file));
    const { json } = loaded;
    if (doc.nodes.length === 0) {
      // Every rig file lists the same nodes; keep the joints and drop the meshes
      doc.nodes = json.nodes.map(({ mesh: _mesh, skin: _skin, ...node }) => node);
      doc.scenes[0]!.nodes = [...json.scenes[json.scene ?? 0]!.nodes];
      names = json.nodes.map((node) => node.name ?? '');
    }
    // Samplers share their key times, so each accessor is copied once
    const copied = new Map<number, number>();
    const copy = (index: number) => {
      if (!copied.has(index)) {
        const bytes = Buffer.from(read(loaded, index).buffer as ArrayBuffer);
        const accessor = json.accessors[index]!;
        doc.accessors.push({ ...accessor, byteOffset: 0, bufferView: glb.view(bytes) });
        copied.set(index, doc.accessors.length - 1);
      }
      return copied.get(index)!;
    };
    for (const clip of wanted) {
      const animation = json.animations?.find((each) => each.name === clip);
      if (!animation) throw new Error(`${file} has no clip ${clip}.`);
      (doc.animations ??= []).push({
        name: clip,
        samplers: animation.samplers.map((sampler) => ({
          ...sampler,
          input: copy(sampler.input),
          output: copy(sampler.output),
        })),
        channels: animation.channels.map(({ sampler, target }) => ({
          sampler,
          target: { ...target, node: names.indexOf(json.nodes[target.node!]!.name ?? '') },
        })),
      });
    }
  }
  glb.write(path.join(out, 'clips.glb'));
}

await packMarket();
packClips();
for (const [name, file] of Object.entries(CHARACTERS)) {
  copyFileSync(path.join(people, file), path.join(out, `${name}.glb`));
}
