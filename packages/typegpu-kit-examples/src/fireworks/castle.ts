import { instances, meshPool } from '@typegpu/kit';
import { quat } from 'math';
import type { TgpuRoot } from 'typegpu';
import { heightAt, PLATEAU_METRES } from './land.ts';

/** Binary glTF models from Kenney's Castle Kit and Nature Kit (CC0). */
export const MODELS = [
  'wall',
  'tower-hexagon-base',
  'tower-hexagon-mid',
  'tower-hexagon-roof',
  'tower-square-base',
  'tower-square-mid',
  'tower-square-mid-windows',
  'tower-square-top-roof-high',
  'tower-square-arch',
  'tower-square-top',
  'pine-default',
  'pine-round',
  'pine-tall',
] as const;

export type ModelName = (typeof MODELS)[number];

type Vec3 = [number, number, number];

/** Metres a Castle Kit tile spans. */
const TILE_METRES = 10;
/** Tiles from the castle's middle to its walls. */
const REACH_TILES = 2;
/** Heights of the kit's pieces, in tiles. */
const HEXAGON_BASE_TILES = 1.31;
const HEXAGON_MID_TILES = 0.46;
const SQUARE_STOREY_TILES = 1.01;
const KEEP = [
  'tower-square-base',
  'tower-square-mid-windows',
  'tower-square-mid',
  'tower-square-mid-windows',
  'tower-square-top-roof-high',
] as const satisfies readonly ModelName[];
const GATEHOUSE = ['tower-square-arch', 'tower-square-top'] as const satisfies readonly ModelName[];
/** Corner-tower middle-storey counts. Rear towers rise above the keep. */
const CORNERS = [
  { x: -1, z: -1, storeys: 5 },
  { x: 1, z: -1, storeys: 4 },
  { x: -1, z: 1, storeys: 2 },
  { x: 1, z: 1, storeys: 3 },
];
const PINES = ['pine-default', 'pine-round', 'pine-tall'] as const satisfies readonly ModelName[];
const ISLAND_PINES = 14;
/** Island pines stand in four rings, this far out and this far apart. */
const ISLAND_PINE_RADIUS_METRES = 42;
const ISLAND_PINE_RING_METRES = 4;
const ISLAND_PINE_SCALE = [7, 11] as const;
const HILL_PINES = 900;
const HILL_PINE_RADIUS_METRES = [470, 1150] as const;
const HILL_PINE_SCALE = [14, 26] as const;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

interface Piece {
  readonly model: ModelName;
  readonly position: Vec3;
  /** Radians around +y. */
  readonly yaw: number;
  readonly scale: number;
}

/** A place on the plateau, in tiles. */
const at = (x: number, y: number, z: number): Vec3 => [
  x * TILE_METRES,
  PLATEAU_METRES + y * TILE_METRES,
  z * TILE_METRES,
];

/** Evenly distributed scale between `low` and `high` for index `i`. */
const scaleOf = (i: number, [low, high]: readonly [number, number]) =>
  low + (high - low) * ((i * GOLDEN_ANGLE) % 1);

/** The castle's walls round the plateau, its corner towers, the gatehouse and the keep. */
function castlePieces(): Piece[] {
  const pieces: Piece[] = [];
  const wall = (x: number, z: number, yaw: number) => {
    pieces.push({ model: 'wall', position: at(x, 0, z), yaw, scale: TILE_METRES });
  };
  /** Stacks `models` at `x`, `z`, each piece `storey` tiles above the one before. */
  const stack = (
    x: number,
    z: number,
    models: readonly ModelName[],
    storey: (i: number) => number,
  ) => {
    let y = 0;
    models.forEach((model, i) => {
      pieces.push({ model, position: at(x, y, z), yaw: 0, scale: TILE_METRES });
      y += storey(i);
    });
  };

  // The front wall leaves a gap in its middle for the gatehouse
  for (let i = -REACH_TILES + 1; i < REACH_TILES; i++) {
    if (i !== 0) wall(i, REACH_TILES, 0);
    wall(i, -REACH_TILES, Math.PI);
    wall(REACH_TILES, i, Math.PI / 2);
    wall(-REACH_TILES, i, -Math.PI / 2);
  }

  for (const { x, z, storeys } of CORNERS) {
    const mids = Array<ModelName>(storeys).fill('tower-hexagon-mid');
    const tower = ['tower-hexagon-base', ...mids, 'tower-hexagon-roof'] as const;
    stack(x * REACH_TILES, z * REACH_TILES, tower, (i) =>
      i === 0 ? HEXAGON_BASE_TILES : HEXAGON_MID_TILES,
    );
  }

  stack(0, REACH_TILES, GATEHOUSE, () => SQUARE_STOREY_TILES);
  stack(0, 0, KEEP, () => SQUARE_STOREY_TILES);
  return pieces;
}

/** Pines on the island round the castle and over the hills round the lake. */
function naturePieces(): Piece[] {
  const pieces: Piece[] = [];
  /** Plants pine `i` at `angle` and `radius` unless the land there lies under `lowest`. */
  const pine = (
    i: number,
    angle: number,
    radius: number,
    scales: readonly [number, number],
    lowest: number,
  ) => {
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = heightAt(x, z);
    if (y < lowest) return;
    pieces.push({
      model: PINES[i % 3],
      position: [x, y, z],
      yaw: angle,
      scale: scaleOf(i, scales),
    });
  };

  for (let i = 0; i < ISLAND_PINES; i++) {
    const radius = ISLAND_PINE_RADIUS_METRES + (i % 4) * ISLAND_PINE_RING_METRES;
    pine(i, i * GOLDEN_ANGLE * 3, radius, ISLAND_PINE_SCALE, 0.5);
  }

  const [inner, outer] = HILL_PINE_RADIUS_METRES;
  for (let i = 0; i < HILL_PINES; i++) {
    const radius = inner + (outer - inner) * Math.sqrt((i + 0.5) / HILL_PINES);
    pine(i, i * GOLDEN_ANGLE, radius, HILL_PINE_SCALE, 1);
  }
  return pieces;
}

/** Stores castle and pine meshes in one pool, with instances lit by the scene's lighting. */
export function createCastle(
  root: TgpuRoot,
  models: Readonly<Record<ModelName, meshPool.ModelData>>,
) {
  const pool = meshPool.create(root, meshPool.roomFor(MODELS.map((name) => models[name])));
  const loaded = new Map(MODELS.map((name) => [name, pool.add(models[name])] as const));
  const pieces = [...castlePieces(), ...naturePieces()];
  const scenery = instances.create(root, pool, {
    capacity: pieces.reduce((sum, { model }) => sum + models[model].meshes.length, 0),
  });
  for (const { model, position, yaw, scale } of pieces) {
    const rotation = quat.setAxisAngle(quat.create(), [0, 1, 0], yaw);
    scenery.add(loaded.get(model)!, { position, rotation, scale });
  }

  return {
    renderer: scenery.renderer(),
    flush: scenery.flush,
    ready: Promise.all([...loaded.values()].map((model) => model.ready)),
    destroy() {
      scenery.destroy();
      pool.destroy();
    },
  };
}
