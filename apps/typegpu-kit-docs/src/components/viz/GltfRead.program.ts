import { gltf, type EncodedImage, type meshPool } from '@typegpu/kit';
import { linearToSrgb } from '@typegpu/color';
import { d } from 'typegpu';
import { clamp } from '../home/geometry.ts';

/** A node of the default scene, in the order a walk from its roots meets it. */
export interface Node {
  /** As `gltf.read` names it: its own name, or `node 12` without one. */
  readonly name: string;
  /** Whether a part can name it: only a node with a name of its own. */
  readonly named: boolean;
  readonly depth: number;
  readonly mesh: boolean;
  readonly joint: boolean;
}

export interface Mesh {
  readonly material: string;
  readonly vertices: number;
  readonly triangles: number;
  /** The colour factor as CSS, sRGB. */
  readonly swatch: string;
  readonly metallic: number;
  readonly roughness: number;
  readonly alpha: NonNullable<meshPool.MaterialData['alpha']>;
  readonly doubleSided: boolean;
  readonly skinned: boolean;
  /** Each texture's slot and image. */
  readonly textures: readonly (readonly [slot: string, image: number])[];
}

export interface Part {
  readonly name: string;
  readonly position: readonly number[];
  readonly meshes: readonly Mesh[];
}

export interface Model {
  readonly meshes: readonly Mesh[];
  readonly parts: readonly Part[];
  readonly images: readonly EncodedImage[];
  readonly joints: readonly string[];
  readonly clips: readonly { readonly name: string; readonly duration: number }[];
}

export interface Reading {
  /** The file's nodes, or none when its JSON can't be found. */
  readonly nodes: readonly Node[];
  readonly model?: Model;
  /** What `gltf.read` threw. */
  readonly error?: string;
  /** Milliseconds the read took. */
  readonly ms: number;
}

/** The sRGB byte of a linear channel, clamped. */
const byte = (linear: number) => Math.round(clamp(linear) * 255);

function meshOf(mesh: meshPool.MeshData): Mesh {
  const { material } = mesh;
  const [r, g, b, a] = material.color;
  const shown = linearToSrgb(d.vec3f(r, g, b));

  return {
    material: material.name || 'the default material',
    vertices: mesh.positions.length / 3,
    triangles: mesh.indices.length / 3,
    swatch: `rgb(${byte(shown.x)} ${byte(shown.y)} ${byte(shown.z)} / ${a})`,
    metallic: material.metallic,
    roughness: material.roughness,
    alpha: material.alpha ?? 'opaque',
    doubleSided: material.doubleSided === true,
    skinned: mesh.joints !== undefined,
    textures: Object.entries(material.textures).map(([slot, texture]) => [slot, texture.image]),
  };
}

interface NodeJson {
  name?: unknown;
  mesh?: unknown;
  children?: unknown;
}

/** The default scene's nodes from a binary glTF's JSON chunk, or none where there is no such chunk. */
function nodesOf(bytes: Uint8Array): Node[] {
  try {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.byteLength < 20 || view.getUint32(16, true) !== 0x4e4f534a) return [];
    const json = JSON.parse(
      new TextDecoder().decode(bytes.subarray(20, 20 + view.getUint32(12, true))),
    );

    const nodes: NodeJson[] = Array.isArray(json.nodes) ? json.nodes : [];
    const joints = new Set<unknown>(
      (Array.isArray(json.skins) ? json.skins : []).flatMap((skin: { joints?: unknown }) =>
        Array.isArray(skin.joints) ? skin.joints : [],
      ),
    );
    const scene = json.scenes?.[json.scene ?? 0];
    const roots: unknown[] = Array.isArray(scene?.nodes) ? scene.nodes : [];

    const found: Node[] = [];
    const seen = new Set<unknown>();
    const walk = (index: unknown, depth: number) => {
      const node = nodes[index as number];
      if (!node || seen.has(index)) return;
      seen.add(index);

      const named = typeof node.name === 'string';
      found.push({
        name: named ? (node.name as string) : `node ${index as number}`,
        named,
        depth,
        mesh: node.mesh !== undefined,
        joint: joints.has(index),
      });
      for (const child of Array.isArray(node.children) ? node.children : []) walk(child, depth + 1);
    };
    for (const index of roots) walk(index, 0);
    return found;
  } catch {
    return [];
  }
}

/** Reads `bytes` with `gltf.read`, asking for `parts`, and sums up what came back or why not. */
export function read(bytes: Uint8Array, parts: readonly string[] = []): Reading {
  const nodes = nodesOf(bytes);
  const start = performance.now();

  try {
    const model = gltf.read(bytes, { parts });
    return {
      nodes,
      ms: performance.now() - start,
      model: {
        meshes: model.meshes.map(meshOf),
        parts: Object.entries<gltf.PartData>(model.parts).map(([name, part]) => ({
          name,
          position: part.position,
          meshes: part.meshes.map(meshOf),
        })),
        images: model.images,
        joints: model.skin?.joints.map((joint) => joint.name) ?? [],
        clips: model.clips.map(({ name, duration }) => ({ name, duration })),
      },
    };
  } catch (error) {
    return { nodes, ms: performance.now() - start, error: (error as Error).message };
  }
}

/** Bytes that aren't a model: a web server's page for a file it doesn't have, and a `.gltf`. */
export const broken = {
  page: () => new TextEncoder().encode('<!doctype html><title>404 Not Found</title>'),
  json: () => new TextEncoder().encode('{"asset":{"version":"2.0"}}'),
};

export function createProgram() {
  return { read, broken };
}
