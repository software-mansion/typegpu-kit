import type { meshPool, poses } from '../../src/index.ts';

/** A quarter turn about +z, `[x, y, z, w]`. */
export const QUARTER_Z = [0, 0, Math.SQRT1_2, Math.SQRT1_2] as const;

/** The identity's first three columns. */
const AXES = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

/** A column-major translation. */
export const moving = (x: number, y: number, z: number) => [...AXES, x, y, z, 1];

const still = { rotation: [0, 0, 0, 1], scale: [1, 1, 1] } as const;

/** A hip at the origin and a knee a metre above it, bound where they rest. */
export const leg: poses.SkinData = {
  joints: [
    { name: 'hip', parent: -1, position: [0, 0, 0], ...still, inverseBind: moving(0, 0, 0) },
    { name: 'knee', parent: 0, position: [0, 1, 0], ...still, inverseBind: moving(0, -1, 0) },
  ],
};

/** The knee turning a quarter about +z over a second. */
export const bend: poses.ClipData = {
  name: 'bend',
  duration: 1,
  channels: [
    {
      joint: 'knee',
      path: 'rotation',
      interpolation: 'linear',
      times: [0, 1],
      values: [0, 0, 0, 1, ...QUARTER_Z],
    },
  ],
};

/** No channels: every joint rests. */
export const rest: poses.ClipData = { name: 'rest', duration: 0, channels: [] };

const material = {
  color: [1, 1, 1, 1],
  roughness: 1,
  metallic: 0,
  emission: [0, 0, 0],
  textures: {},
} as const;

/**
 * A shin: a vertex on the hip, at y = 0.5, one on the knee, at y = 2, and one between them at
 * y = 1.5, bound to both.
 */
export const shin: meshPool.MeshData = {
  positions: new Float32Array([0, 0.5, 0, 0, 2, 0, 0, 1.5, 0]),
  normals: new Float32Array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
  uvs: new Float32Array(6),
  indices: new Uint32Array([0, 1, 2]),
  material,
  joints: new Uint16Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
  weights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 0, 0]),
};

/** A triangle with no joints, at y = 1 in whatever space it is read in. */
export const blade: meshPool.MeshData = {
  positions: new Float32Array([0, 1, 0, 1, 1, 0, 0, 1, 1]),
  normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
  uvs: new Float32Array(6),
  indices: new Uint32Array([0, 1, 2]),
  material,
};

/** The knee turning half a turn about +z over two seconds. */
export const slow: poses.ClipData = {
  name: 'slow',
  duration: 2,
  channels: [
    {
      joint: 'knee',
      path: 'rotation',
      interpolation: 'linear',
      times: [0, 2],
      values: [0, 0, 0, 1, 0, 0, 1, 0],
    },
  ],
};

/** The hip turning a quarter about +z over a second, and the knee a quarter back. */
export const sway: poses.ClipData = {
  name: 'sway',
  duration: 1,
  channels: [
    {
      joint: 'hip',
      path: 'rotation',
      interpolation: 'linear',
      times: [0, 1],
      values: [0, 0, 0, 1, ...QUARTER_Z],
    },
    {
      joint: 'knee',
      path: 'rotation',
      interpolation: 'linear',
      times: [0, 1],
      values: [0, 0, 0, 1, 0, 0, -Math.SQRT1_2, Math.SQRT1_2],
    },
  ],
};
