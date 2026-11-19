import { describe, expect, it } from 'vitest';
import { gltf, type meshPool, type poses } from '../../src/index.ts';
import { builder } from './glb.ts';

/** The values with -0 as 0, which `toEqual` tells apart. */
const plain = (values: ArrayLike<number>) => Array.from(values, (v) => v + 0);

const TRIANGLE = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
const QUARTER_Z = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
const QUARTER_Y = [0, Math.SQRT1_2, 0, Math.SQRT1_2] as const;

/** A column-major translation. */
const moving = (x: number, y: number, z: number) => [
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  x,
  y,
  z,
  1,
];

/** A column-major matrix from a position, a unit quaternion and a scale. */
function matrixOf(
  [px, py, pz]: readonly number[],
  [x, y, z, w]: readonly number[],
  [sx, sy, sz]: readonly number[],
) {
  return [
    (1 - 2 * (y! * y! + z! * z!)) * sx!,
    2 * (x! * y! + w! * z!) * sx!,
    2 * (x! * z! - w! * y!) * sx!,
    0,
    2 * (x! * y! - w! * z!) * sy!,
    (1 - 2 * (x! * x! + z! * z!)) * sy!,
    2 * (y! * z! + w! * x!) * sy!,
    0,
    2 * (x! * z! + w! * y!) * sz!,
    2 * (y! * z! - w! * x!) * sz!,
    (1 - 2 * (x! * x! + y! * y!)) * sz!,
    0,
    px!,
    py!,
    pz!,
    1,
  ];
}

function multiply(a: readonly number[], b: readonly number[]) {
  return Array.from({ length: 16 }, (_, i) => {
    const column = Math.floor(i / 4);
    const row = i % 4;
    let sum = 0;
    for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
    return sum;
  });
}

/** Where each vertex of `mesh` goes under `skin` posed at rest: its joints' chains times their binds. */
function rested(skin: poses.SkinData, mesh: meshPool.MeshData) {
  const worlds: number[][] = [];
  for (const joint of skin.joints) {
    const local = matrixOf(joint.position, joint.rotation, joint.scale);
    worlds.push(joint.parent === -1 ? local : multiply(worlds[joint.parent], local));
  }
  const skinning = skin.joints.map((joint, j) =>
    multiply(worlds[j], Array.from(joint.inverseBind)),
  );

  const out = new Float32Array(mesh.positions.length);
  for (let v = 0; v < mesh.positions.length / 3; v++) {
    const [x, y, z] = mesh.positions.subarray(v * 3, v * 3 + 3);
    for (let k = 0; k < 4; k++) {
      const m = skinning[mesh.joints![v * 4 + k]];
      const weight = mesh.weights![v * 4 + k];
      for (let r = 0; r < 3; r++) {
        out[v * 3 + r] += weight * (m[r] * x! + m[4 + r] * y! + m[8 + r] * z! + m[12 + r]);
      }
    }
  }
  return out;
}

/**
 * An armature at x = 1 holding a hip a metre up, and a hand a metre above that holding a sword:
 * a triangle skinned to the hip and the hand, the sword's triangle hanging rigid from the hand.
 * Skin 0 lists the hip and the hand. `json` replaces what it names.
 */
function rigged(
  json: Record<string, unknown> = {},
  weights = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 0.25, 0.25, 0, 0]),
) {
  const file = builder();
  const position = file.accessor(TRIANGLE, 'VEC3');
  const joints = file.accessor(new Uint8Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]), 'VEC4');
  const shares = file.accessor(weights, 'VEC4');
  const binds = file.accessor(
    new Float32Array([...moving(-1, -1, 0), ...moving(-1, -2, 0)]),
    'MAT4',
  );

  const times = file.accessor(new Float32Array([0, 1]), 'SCALAR');
  const turns = file.accessor(new Float32Array([0, 0, 0, 1, ...QUARTER_Z]), 'VEC4');
  const steps = file.accessor(new Float32Array([0, 0.5]), 'SCALAR');
  const heights = file.accessor(new Float32Array([0, 1, 0, 0, 2, 0]), 'VEC3');
  const shifts = file.accessor(new Float32Array([1, 0, 0, 1, 0, 0]), 'VEC3');
  // In-tangent, value and out-tangent at each of two keys.
  const curve = file.accessor(
    new Float32Array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0]),
    'VEC3',
  );

  Object.assign(file.json, {
    scenes: [{ nodes: [0] }],
    nodes: [
      { name: 'armature', translation: [1, 0, 0], children: [1, 3] },
      { name: 'hip', translation: [0, 1, 0], children: [2] },
      { name: 'hand', translation: [0, 1, 0], children: [4] },
      { name: 'body', mesh: 0, skin: 0, translation: [5, 5, 5] },
      { name: 'sword', mesh: 1, translation: [0, 0.5, 0] },
    ],
    meshes: [
      {
        primitives: [{ attributes: { POSITION: position, JOINTS_0: joints, WEIGHTS_0: shares } }],
      },
      { primitives: [{ attributes: { POSITION: position } }] },
    ],
    skins: [{ joints: [1, 2], inverseBindMatrices: binds }],
    animations: [
      {
        name: 'wave',
        samplers: [
          { input: times, output: turns },
          { input: steps, output: heights, interpolation: 'STEP' },
          { input: times, output: shifts },
        ],
        channels: [
          { sampler: 0, target: { node: 2, path: 'rotation' } },
          { sampler: 1, target: { node: 1, path: 'translation' } },
          { sampler: 2, target: { node: 0, path: 'translation' } },
          { sampler: 0, target: { node: 2, path: 'weights' } },
        ],
      },
      {
        samplers: [{ input: times, output: curve, interpolation: 'CUBICSPLINE' }],
        channels: [{ sampler: 0, target: { node: 1, path: 'translation' } }],
      },
    ],
    ...json,
  });
  return file;
}

/** Whether `a` is `b` within a millionth. */
const near = (a: ArrayLike<number>, b: readonly number[]) =>
  Array.from(a).every((value, i) => Math.abs(value - b[i]) < 1e-6);

/** Reads `rigged()` with its first sampler changed by `change`, given the file to add to. */
function changed(change: (file: ReturnType<typeof rigged>) => object) {
  const file = rigged();
  const animations = file.json.animations as { samplers: object[] }[];
  animations[0].samplers[0] = { ...animations[0].samplers[0], ...change(file) };
  return () => gltf.read(file.build());
}

describe('gltf.read with a skin', () => {
  it('reads the joints and every node above them, parents first, as they rest and were bound', () => {
    const { skin } = gltf.read(rigged().build());
    expect(skin!.joints.map(({ name, parent }) => [name, parent])).toEqual([
      ['armature', -1],
      ['hip', 0],
      ['hand', 1],
    ]);
    expect(skin!.joints.map(({ position }) => plain(position))).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
    ]);
    // The armature isn't in the file's skin, so it binds where it rests.
    expect(skin!.joints.map(({ inverseBind }) => plain(inverseBind))).toEqual([
      moving(-1, 0, 0),
      moving(-1, -1, 0),
      moving(-1, -2, 0),
    ]);
  });

  it('binds skinned vertices to the joints they list, and a mesh hanging from a joint to it', () => {
    const [mesh] = gltf.read(rigged().build()).meshes;
    // The sword, first in the tree, rests where its node puts it; the skinned mesh ignores its
    // node's transform.
    expect(plain(mesh!.positions)).toEqual([1, 2.5, 0, 2, 2.5, 0, 1, 3.5, 0, ...TRIANGLE]);
    const held = [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0];
    const listed = [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1];
    expect(Array.from(mesh!.joints!)).toEqual([...held, ...listed]);
    const whole = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
    const shared = [1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 0, 0];
    expect(Array.from(mesh!.weights!)).toEqual([...whole, ...shared]);
  });

  it('rests every vertex where it was read, turned or not', () => {
    for (const options of [{}, { rotation: QUARTER_Y }]) {
      const model = gltf.read(rigged().build(), options);
      const [mesh] = model.meshes;
      const at = rested(model.skin!, mesh!);
      expect(at.every((value, i) => Math.abs(value - mesh!.positions[i]) < 1e-6)).toBe(true);
    }
  });

  it('reads clips by the names of the nodes they move, skipping morph weights', () => {
    const [wave, unnamed] = gltf.read(rigged().build()).clips;
    expect(wave!.name).toBe('wave');
    expect(wave!.duration).toBe(1);
    expect(
      wave!.channels.map(({ joint, path, interpolation }) => [joint, path, interpolation]),
    ).toEqual([
      ['hand', 'rotation', 'linear'],
      ['hip', 'position', 'step'],
      ['armature', 'position', 'linear'],
    ]);
    expect(plain(wave!.channels[1].values)).toEqual([0, 1, 0, 0, 2, 0]);

    expect(unnamed!.name).toBe('animation 1');
    expect(unnamed!.channels[0].interpolation).toBe('cubic');
    expect(unnamed!.channels[0].values).toHaveLength(18);
  });

  it('turns the top joints and their keys with the model', () => {
    const { skin, clips } = gltf.read(rigged().build(), { rotation: QUARTER_Y });
    const [armature, hip] = skin!.joints;

    expect(near(armature!.position, [0, 0, -1])).toBe(true);
    expect(near(armature!.rotation, QUARTER_Y)).toBe(true);
    expect(near(hip!.position, [0, 1, 0])).toBe(true);
    const shift = clips[0].channels.find(({ joint }) => joint === 'armature')!;
    expect(near(shift.values, [0, 0, -1, 0, 0, -1])).toBe(true);
    const lift = clips[0].channels.find(({ joint }) => joint === 'hip')!;
    expect(near(lift.values, [0, 1, 0, 0, 2, 0])).toBe(true);
  });

  it('reads no skin or clips from a file without them', () => {
    const file = builder();
    const position = file.accessor(TRIANGLE, 'VEC3');
    Object.assign(file.json, {
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: position } }] }],
    });
    const model = gltf.read(file.build());
    expect(model.skin).toBeUndefined();
    expect(model.clips).toEqual([]);
    expect(model.meshes[0].joints).toBeUndefined();
  });

  it('keeps a part holding a skinned mesh bound to the skin, at the model origin', () => {
    const bytes = rigged().build();
    const { parts, meshes, skin } = gltf.read(bytes, { parts: ['body'] });
    const [body] = parts.body.meshes;

    expect(parts.body.position).toEqual([0, 0, 0]);
    expect(parts.body.rotation).toEqual([0, 0, 0, 1]);
    expect(plain(body!.positions)).toEqual(Array.from(TRIANGLE));
    expect(Array.from(body!.joints!)).toEqual([1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1]);
    expect(Array.from(body!.weights!)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 0, 0]);
    expect(near(rested(skin!, body!), Array.from(body!.positions))).toBe(true);
    // The sword stays in `meshes`, still hanging from the hand
    expect(Array.from(meshes[0].joints!)).toEqual([2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0]);

    // A part around the whole rig reads as the model does
    const whole = gltf.read(bytes, { parts: ['armature'] }).parts.armature.meshes;
    expect(whole).toEqual(gltf.read(bytes).meshes);
  });

  it('refuses a mesh that hangs from no joint, and asks for it as a part', () => {
    const loose = rigged();
    const nodes = loose.json.nodes as object[];
    nodes.push({ name: 'rock', mesh: 1 });
    loose.json.scenes = [{ nodes: [0, 5] }];
    const bytes = loose.build();
    expect(() => gltf.read(bytes)).toThrow(/Node rock's mesh .* hangs from no joint/);
    expect(gltf.read(bytes, { parts: ['rock'] }).parts.rock.meshes).toHaveLength(1);
  });

  it('refuses weights and joints it cannot bind', () => {
    const none = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
    expect(() => gltf.read(rigged({}, none).build())).toThrow(/Vertex 1 .* no weight/);
    const heavy = new Float32Array([1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]);
    expect(() => gltf.read(rigged({}, heavy).build())).toThrow(/Vertex 1 .* weight of 2/);

    const far = rigged();
    const listing = new Uint8Array([0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    const meshes = far.json.meshes as { primitives: { attributes: Record<string, number> }[] }[];
    meshes[0].primitives[0].attributes.JOINTS_0 = far.accessor(listing, 'VEC4');
    expect(() => gltf.read(far.build())).toThrow(/bound to joint 2, and the skin has 2/);

    const second = rigged();
    const doubled = second.json.meshes as typeof meshes;
    doubled[0].primitives[0].attributes.JOINTS_1 = doubled[0].primitives[0].attributes.JOINTS_0!;
    expect(() => gltf.read(second.build())).toThrow(/JOINTS_1/);
  });

  it('refuses keys out of order or of the wrong count, and interpolations glTF lacks', () => {
    const backwards = changed((file) => ({
      input: file.accessor(new Float32Array([1, 0]), 'SCALAR'),
    }));
    expect(backwards).toThrow(/times that don't increase/);
    const extra = changed((file) => ({ output: file.accessor(new Float32Array(12), 'VEC4') }));
    expect(extra).toThrow(/2 times and 3 keys/);
    expect(changed(() => ({ interpolation: 'BEZIER' }))).toThrow(/not LINEAR, STEP or CUBICSPLINE/);
  });
});
