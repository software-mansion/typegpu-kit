import { instances, lighting, materials, meshPool, type Frame, type Surface } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import type { TgpuRoot } from 'typegpu';
import { linear, litStage, SKY, turn } from './stage.ts';

export const MAX = 512;

/** World units between neighbours. */
const SPACING = 1.3;

/** Seconds that take the camera all the way to its goal, for a figure without motion. */
const SETTLED = 10;

type Cell = readonly [number, number];

const ring = ([x, z]: Cell) => Math.max(Math.abs(x), Math.abs(z));
const angle = ([x, z]: Cell) => Math.atan2(z, x);

/** The cells of a square grid, ring by ring from the centre, each ring swept by angle. */
const cells = (() => {
  const rings = Math.ceil((Math.sqrt(MAX) - 1) / 2);
  const all: Cell[] = [];
  for (let x = -rings; x <= rings; x++) for (let z = -rings; z <= rings; z++) all.push([x, z]);

  return all.toSorted((a, b) => ring(a) - ring(b) || angle(a) - angle(b)).slice(0, MAX);
})();

/** How far the camera stands to see `count` instances. */
const reach = (count: number) => Math.max(5, Math.sqrt(count) * SPACING * 1.4);

export interface Options {
  /** Bobs the instances and turns the camera; without it the camera jumps to its goal. Default true. */
  readonly motion?: boolean;
  /** Instances at first. Default 0. */
  readonly count?: number;
}

export async function createScene(root: TgpuRoot, surface: Surface, options: Options = {}) {
  const { motion = true } = options;

  const pool = meshPool.create(root, { vertices: 2048, indices: 8192, capacity: 2 });
  const shapes = [pool.add(meshes.box()), pool.add(meshes.sphere())];
  const looks = [
    materials.glossy(linear(0.5, 0.66, 0.98), 0.4),
    materials.glossy(linear(0.97, 0.71, 0.31), 0.3),
  ];

  const things = instances.create(root, pool, { capacity: MAX });
  const placed = things.fields.position;
  const light = lighting.create(root, { settings: { ...SKY, sunDirection: [0.4, 1, 0.3] } });

  const stage = litStage(root, surface, {
    camera: { target: [0, 0, 0], distance: reach(0), pitch: 0.75, yaw: 0.4 },
  });
  const { view, screen, final } = stage;
  const scene = stage.lit(light);
  const drawn = scene.add(things.renderer());

  const ids: instances.Id[] = [];

  /** Adds or removes instances from the outside of the grid until there are `count`. */
  const count = (wanted: number) => {
    while (ids.length > wanted) things.remove(ids.pop()!);

    while (ids.length < wanted) {
      const i = ids.length;
      const [x, z] = cells[i];
      const shape = i % 2;
      const position = [x * SPACING, 0, z * SPACING] as const;
      ids.push(
        things.add(shapes[shape], { position, rotation: turn([0, 1, 0], i), ...looks[shape] }),
      );
    }

    view.goal.distance = reach(wanted);
  };

  count(options.count ?? 0);
  await Promise.all([scene.initAsync(), final.initAsync()]);

  return {
    count,

    frame(frame: Frame) {
      if (motion) {
        view.goal.yaw = 0.4 + frame.time * 0.1;
        ids.forEach((id, i) => {
          const [x, z] = cells[i];
          const y = Math.sin(frame.time * 2 + (x + z) * 0.6) * 0.2;
          placed.set(id, x * SPACING, y, z * SPACING);
        });
      }

      view.update(motion ? frame.dt : SETTLED);
      screen.fit();
      things.flush();

      scene.record(frame, drawn);
      stage.present(frame);
    },

    destroy() {
      things.destroy();
      pool.destroy();
      light.destroy();
      stage.destroy();
    },
  };
}
