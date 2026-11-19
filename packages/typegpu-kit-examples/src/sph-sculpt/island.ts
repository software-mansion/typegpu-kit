import { d } from 'typegpu';

/** A CSG primitive applied after the terrain. Rock primitives can't be dug. */
export const Prim = d.struct({
  /** Sphere: centre and radius. Box: centre and edge rounding. Capsule: start and radius. */
  a: d.vec4f,
  /** Box: half extents and blend. Capsule: end and blend. Sphere: blend in w. */
  b: d.vec4f,
  /** 0 sphere, 1 box, 2 capsule. */
  shape: d.u32,
  /** 0 union, 1 subtract. */
  op: d.u32,
  /** 1 for rock, which stays solid under the dig brush and shades grey. */
  rock: d.u32,
  pad: d.u32,
});

export const Terrain = d.struct({
  /** Ground height in metres before noise. */
  base: d.f32,
  /** Noise amplitude in metres. */
  amp: d.f32,
  /** Noise frequency per metre. */
  freq: d.f32,
  /** Island radius in metres, where the ground falls away into the sea. */
  island: d.f32,
  /** Noise offset, which picks a different landscape. */
  seed: d.vec2f,
  /** Top of the rock plinth under the clay, which no dig goes through. */
  bedrock: d.f32,
  prims: d.u32,
});

type PrimInput = d.InferInput<typeof Prim>;

const sphere = (at: [number, number, number], r: number, op: number, blend: number): PrimInput => ({
  a: [...at, r],
  b: [0, 0, 0, blend],
  shape: 0,
  op,
  rock: 0,
  pad: 0,
});

const rockBox = (at: [number, number, number], half: [number, number, number]): PrimInput => ({
  a: [...at, 0.3],
  b: [...half, 0.2],
  shape: 1,
  op: 0,
  rock: 1,
  pad: 0,
});

const DIG = 1;

/** The world box is 25.6 × 12.8 × 25.6 m, with x and z from 0 to 25.6. */
export const TERRAIN: Omit<d.InferInput<typeof Terrain>, 'prims'> = {
  base: 3.2,
  amp: 0.9,
  freq: 0.11,
  island: 11.5,
  seed: [3.1, 7.7],
  bedrock: 0.6,
};

/** A hill with a pond for the spring, a bowl beyond a rock ridge, and a gap in the ridge. */
export const PRIMS: PrimInput[] = [
  sphere([7, -2.5, 12.8], 8, 0, 1.5),
  sphere([7, 6.4, 12.8], 1.5, DIG, 0.4),
  sphere([19.5, 3.4, 12.8], 2.6, DIG, 0.6),
  rockBox([13, 4.5, 8.5], [0.6, 1.6, 4.2]),
  rockBox([13, 4.5, 18.6], [0.6, 1.6, 3.4]),
];

export const MAX_PRIMS = 8;

/**
 * Where the spring wells up, just above the pond's rim, its radius in metres and particles a
 * second. It flows while fewer than `limit` particles are alive, so water lost off the island's
 * edge comes back. A spring higher up pours a stream that splashes droplets over the hill.
 */
export const SPRING = {
  at: [7, 6, 12.8] as [number, number, number],
  radius: 0.28,
  rate: 900,
  limit: 30_000,
};
