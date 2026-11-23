import { instances, meshPool, type MeshPool } from '@typegpu/kit';
import { quat } from 'wgpu-matrix';
import type { TgpuRoot } from 'typegpu';
import { heightAt, LOOP_METRES, loopAt } from './land.ts';

/** The nodes of `models/scenery.glb`, from Kay Lousberg's KayKit Forest Nature Pack (CC0). */
export const SCENERY = [
  'Tree_1_A',
  'Tree_2_A',
  'Tree_4_A',
  'Tree_Bare_1_A',
  'Rock_1_J',
  'Rock_3_A',
  'Rock_3_F',
  'Bush_2_A',
] as const;
export type SceneryName = (typeof SCENERY)[number];

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const UP = [0, 1, 0] as const;
/** Track obstacles that split the herd. */
const ON_TRACK = 9;
const OFF_TRACK_TREES = 90;
/** Grove count, trees per grove and grove width in metres. */
const GROVES = 70;
const GROVE_TREES = 9;
const GROVE_METRES = 60;
const KNOLL_ROCKS = 26;
const BUSHES = 320;
const TREES = ['Tree_1_A', 'Tree_2_A', 'Tree_4_A', 'Tree_Bare_1_A'] as const;
const TREE_SCALE = [2.2, 3.2] as const;
const GROVE_SCALE = [2.6, 4.2] as const;
const CRAG_SCALE = [3, 4.2] as const;
/** Crag avoidance radius at scale 1. */
const CRAG_RADIUS = 1.9;
const TREE_RADIUS = 0.5;
const BOULDER_SCALE = [3.5, 7] as const;
const BUSH_SCALE = [4, 7] as const;
/** Tint multiplier shifting bright model greens toward dry-grass olive. */
const DRY_LEAVES = [0.9, 0.72, 0.5] as const;

export interface Placed {
  readonly model: SceneryName;
  readonly position: [number, number, number];
  readonly yaw: number;
  readonly scale: number;
}

/** Evenly distributed value between `low` and `high` for index `i`. */
const between = (i: number, [low, high]: readonly [number, number], salt = 0) =>
  low + (high - low) * (((i + salt) * GOLDEN_ANGLE) % 1);

/** The point `around` radians round an oval `loop` times the track's. */
const ovalAt = (around: number, loop: number) => [
  Math.cos(around) * LOOP_METRES[0] * loop,
  Math.sin(around) * LOOP_METRES[1] * loop,
];

const onGround = (
  model: SceneryName,
  x: number,
  z: number,
  yaw: number,
  scale: number,
): Placed => ({
  model,
  position: [x, heightAt(x, z) - 0.2, z],
  yaw,
  scale,
});

/** Scenery placements and herd-avoidance obstacles. */
export function layout() {
  const placed: Placed[] = [];
  const obstacles: { place: [number, number]; radius: number }[] = [];

  for (let i = 0; i < ON_TRACK; i++) {
    const around = (i / ON_TRACK) * Math.PI * 2 + 0.9;
    const [x, z] = ovalAt(around, 1 + Math.sin(i * 2.3) * 0.06);
    const crag = i % 3 !== 1;
    const scale = between(i, crag ? CRAG_SCALE : TREE_SCALE);
    placed.push(onGround(crag ? 'Rock_1_J' : 'Tree_1_A', x, z, i * 2.1, scale));
    obstacles.push({ place: [x, z], radius: (crag ? CRAG_RADIUS : TREE_RADIUS) * scale + 1.5 });
  }

  for (let i = 0; i < OFF_TRACK_TREES; i++) {
    const around = i * GOLDEN_ANGLE * 7;
    const loop = i % 4 === 0 ? between(i, [0.35, 0.72]) : between(i, [1.32, 2.2], 3);
    const [x, z] = ovalAt(around, loop);
    placed.push(onGround(TREES[i % TREES.length], x, z, around * 3, between(i, TREE_SCALE, 5)));
  }

  for (let i = 0; i < GROVES; i++) {
    const around = i * GOLDEN_ANGLE * 5 + 0.3;
    const [middleX, middleZ] = ovalAt(around, between(i, [1.5, 3.2], 17));
    for (let j = 0; j < GROVE_TREES; j++) {
      const angle = j * GOLDEN_ANGLE;
      const r = (GROVE_METRES / 2) * Math.sqrt((j + 0.5) / GROVE_TREES);
      const x = middleX + Math.cos(angle) * r;
      const z = middleZ + Math.sin(angle) * r;
      const tree = TREES[(i + (j % 3 === 0 ? 1 : 0)) % 3];
      placed.push(onGround(tree, x, z, angle * 3, between(i * GROVE_TREES + j, GROVE_SCALE)));
    }
  }

  for (let i = 0; i < KNOLL_ROCKS; i++) {
    const angle = i * GOLDEN_ANGLE;
    const [x, z] = ovalAt(angle, 0.06 + 0.3 * Math.sqrt((i + 0.5) / KNOLL_ROCKS));
    const model = i % 5 === 0 ? 'Rock_1_J' : i % 2 === 0 ? 'Rock_3_A' : 'Rock_3_F';
    const scale = model === 'Rock_1_J' ? between(i, CRAG_SCALE) : between(i, BOULDER_SCALE);
    placed.push(onGround(model, x, z, angle * 5, scale));
  }

  for (let i = 0; i < BUSHES; i++) {
    const around = i * GOLDEN_ANGLE * 3;
    const [x, z] = ovalAt(around, between(i, [0.2, 2.4], 11));
    // Keep bushes off the herd track
    if (Math.abs(loopAt(x, z) - 1) < 0.2) continue;
    placed.push(onGround('Bush_2_A', x, z, around, between(i, BUSH_SCALE, 7)));
  }

  return { placed, obstacles };
}

/** Draws scenery instances from the shared pool, lit by the scene's lighting. */
export function createScenery(
  root: TgpuRoot,
  pool: MeshPool,
  models: Readonly<Record<SceneryName, meshPool.ModelData>>,
  placed: readonly Placed[],
) {
  const loaded = new Map(SCENERY.map((name) => [name, pool.add(models[name])] as const));
  const things = instances.create(root, pool, {
    capacity: placed.reduce((sum, { model }) => sum + models[model].meshes.length, 0),
  });
  for (const { model, position, yaw, scale } of placed) {
    const rotation = quat.fromAxisAngle(UP, yaw);
    things.add(loaded.get(model)!, { position, rotation, scale, color: DRY_LEAVES });
  }
  return {
    ready: Promise.all([...loaded.values()].map((model) => model.ready)),
    flush: things.flush,
    destroy() {
      things.destroy();
    },
    renderer: things.renderer({ cull: true }),
  };
}
