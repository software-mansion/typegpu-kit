import { gltf, type meshPool, type poses } from '@typegpu/kit';
import { mat4, quat, vec3 } from 'wgpu-matrix';
import { clamp } from '../home/geometry.ts';

/** The arm's joints, a model unit apart along +x, each the parent of the next. */
const NAMES = ['shoulder', 'elbow', 'wrist'];
/** How far the strip reaches to either side of its bones. */
export const HALF = 0.375;
/** The strip's vertices are this far apart along it. */
const STEP = 0.25;
/** A joint takes a vertex over from its parent across this distance to either side of it. */
const BLEND = 0.5;

const COMPONENTS = new Map<Function, number>([
  [Uint8Array, 5121],
  [Uint16Array, 5123],
  [Float32Array, 5126],
]);
const WIDTHS = { SCALAR: 1, VEC3: 3, VEC4: 4, MAT4: 16 };

/** A binary glTF of an arm: three joints in a chain, and a strip of two rows of vertices skinned to them. */
function arm(): Uint8Array {
  const columns = Math.round(NAMES.length / STEP) + 1;
  const positions: number[] = [];
  const normals: number[] = [];
  const joints: number[] = [];
  const weights: number[] = [];
  const indices: number[] = [];

  for (let column = 0; column < columns; column++) {
    const x = column * STEP;
    const elbow = clamp((x - (1 - BLEND)) / (2 * BLEND));
    const wrist = clamp((x - (2 - BLEND)) / (2 * BLEND));
    // A vertex names the joints that move it first, as exporters write them
    const shares = [1 - elbow, elbow - wrist, wrist].flatMap((share, joint) =>
      share > 0 ? [[joint, share]] : [],
    );

    for (const y of [-HALF, HALF]) {
      positions.push(x, y, 0);
      normals.push(0, 0, 1);
      for (let k = 0; k < 4; k++) {
        joints.push(shares[k]?.[0] ?? 0);
        weights.push(shares[k]?.[1] ?? 0);
      }
    }
    if (column > 0) {
      const at = column * 2;
      indices.push(at - 2, at, at - 1, at - 1, at, at + 1);
    }
  }

  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0, NAMES.length] }],
    nodes: [
      ...NAMES.map((name, joint) => ({
        name,
        ...(joint > 0 && { translation: [1, 0, 0] }),
        ...(joint < NAMES.length - 1 && { children: [joint + 1] }),
      })),
      { name: 'arm', mesh: 0, skin: 0 },
    ],
    meshes: [
      {
        primitives: [
          { attributes: { POSITION: 0, NORMAL: 1, JOINTS_0: 2, WEIGHTS_0: 3 }, indices: 4 },
        ],
      },
    ],
    // A joint's inverse bind moves model space to its own: the arm lies along +x, so back by its x
    skins: [{ joints: NAMES.map((_, joint) => joint), inverseBindMatrices: 5 }],
    accessors: [] as object[],
    bufferViews: [] as object[],
    buffers: [] as object[],
  };

  const chunks: Uint8Array[] = [];
  let length = 0;
  const accessor = (data: Uint8Array | Uint16Array | Float32Array, type: keyof typeof WIDTHS) => {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    json.bufferViews.push({ buffer: 0, byteOffset: length, byteLength: bytes.byteLength });
    json.accessors.push({
      bufferView: json.accessors.length,
      componentType: COMPONENTS.get(data.constructor),
      count: data.length / WIDTHS[type],
      type,
    });
    // Views start on four bytes
    const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
    padded.set(bytes);
    chunks.push(padded);
    length += padded.byteLength;
  };

  accessor(new Float32Array(positions), 'VEC3');
  accessor(new Float32Array(normals), 'VEC3');
  accessor(new Uint8Array(joints), 'VEC4');
  accessor(new Float32Array(weights), 'VEC4');
  accessor(new Uint16Array(indices), 'SCALAR');
  accessor(
    new Float32Array(NAMES.flatMap((_, joint) => [...mat4.translation([-joint, 0, 0])])),
    'MAT4',
  );
  json.buffers.push({ byteLength: length });

  const text = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = Math.ceil(text.byteLength / 4) * 4;
  const out = new Uint8Array(28 + jsonLength + length);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true); // glTF
  view.setUint32(4, 2, true);
  view.setUint32(8, out.byteLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true); // JSON
  out.fill(0x20, 20, 20 + jsonLength);
  out.set(text, 20);
  view.setUint32(20 + jsonLength, length, true);
  view.setUint32(24 + jsonLength, 0x004e4942, true); // BIN
  let at = 28 + jsonLength;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.byteLength;
  }
  return out;
}

export interface Posed {
  /** Where each vertex goes, three floats a vertex: its joints' places for it, mixed by its weights. */
  readonly positions: Float32Array;
  /** Where `joint` carries `point` of its own space, in model space. */
  place(joint: number, point?: readonly number[]): Float32Array;
  /** Where `joint` alone would carry `vertex`: its pose times its inverse bind, on the stored position. */
  carried(vertex: number, joint: number): Float32Array;
}

/**
 * Reads the arm with `gltf.read` and poses what comes back on the CPU, as `poses` does on the GPU:
 * a joint's matrix is its chain of parents times its `inverseBind`, and a vertex mixes the
 * matrices of its four `joints` by its `weights`.
 */
export function createProgram() {
  const model = gltf.read(arm());
  const skin: poses.SkinData = model.skin!;
  const mesh: meshPool.MeshData = model.meshes[0];
  const stored = (vertex: number) => mesh.positions.subarray(vertex * 3, vertex * 3 + 3);

  /** Poses the skin with each joint turned by `turns`, radians about +z, from its rest rotation. */
  function pose(turns: readonly number[]): Posed {
    const worlds: Float32Array[] = [];
    skin.joints.forEach(({ parent, position, rotation, scale }, joint) => {
      const turned = quat.multiply(rotation, quat.fromAxisAngle([0, 0, 1], turns[joint] ?? 0));
      const local = mat4.scale(
        mat4.multiply(mat4.translation(position), mat4.fromQuat(turned)),
        scale,
      );
      worlds.push(parent === -1 ? local : mat4.multiply(worlds[parent], local));
    });
    const skinning = worlds.map((world, joint) =>
      mat4.multiply(world, Float32Array.from(skin.joints[joint].inverseBind)),
    );

    const carried = (vertex: number, joint: number) =>
      vec3.transformMat4(stored(vertex), skinning[joint]);

    const positions = new Float32Array(mesh.positions.length);
    for (let vertex = 0; vertex < positions.length / 3; vertex++) {
      for (let k = 0; k < 4; k++) {
        const weight = mesh.weights![vertex * 4 + k];
        const place = carried(vertex, mesh.joints![vertex * 4 + k]);
        for (let axis = 0; axis < 3; axis++) positions[vertex * 3 + axis] += weight * place[axis];
      }
    }

    return {
      positions,
      place: (joint, point = [0, 0, 0]) => vec3.transformMat4(point, worlds[joint]),
      carried,
    };
  }

  return { skin, mesh, pose };
}
