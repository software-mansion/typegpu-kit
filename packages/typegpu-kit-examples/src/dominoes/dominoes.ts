import { srgbToLinear } from '@typegpu/color';
import { meshes } from '@typegpu/kit/geometry';
import { instances, materials, type MeshPool } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import { firstOf, HEIGHT, layOut, NONE, THICKNESS, WIDTH } from './layout.ts';
import { layField } from './field.ts';

/** Linear colours for the domino palette. */
const PALETTE = (
  [
    [0.95, 0.92, 0.86],
    [0.96, 0.7, 0.23],
    [0.96, 0.52, 0.3],
    [0.8, 0.16, 0.16],
    [0.12, 0.6, 0.58],
    [0.14, 0.28, 0.6],
  ] as const
).map(([r, g, b]) => srgbToLinear(d.vec3f(r, g, b)));

/** Satin lacquer with a broad highlight. */
const ROUGHNESS = 0.38;

/** Layouts ordered by wave arrival. */
export const LAYOUTS = { run: layOut(), portrait: layField() };
export type Layout = keyof typeof LAYOUTS;

/**
 * Each domino pivots about its bottom front edge at `foot`, with orientation `yaw`, tilt
 * `angle` and angular speed `spin` in radians per second. `hit` records impact speed for one
 * step. `prev` and `next` hold neighbour ids, with 0 for no neighbour. `order` measures
 * progress along the run from 0 to 1. The kernels in `topple.ts` write the transform on the
 * GPU.
 *
 * The front uses the material colour. A portrait kernel paints `back` from image coordinates
 * `cell`, with -1 for no pixel. Edges blend from front to back colour as the domino falls,
 * revealing the picture.
 */
export const Domino = instances.define('Domino', {
  transform: 'device',
  host: { foot: d.vec3f, yaw: d.f32, order: d.f32, prev: d.u32, next: d.vec2u, cell: d.vec2f },
  device: { angle: d.f32, spin: d.f32, hit: d.f32, back: d.vec3f },
  material: (pixel, base) => {
    'use gpu';
    const domino = pixel.instance;
    // The back faces -z in domino space
    const turnedBack = domino.rotation * d.vec4f(-1, -1, -1, 1);
    const normal = instances.rotate(turnedBack, pixel.point.normal);
    // Standing dominoes show the image only on their flat backs. Blend the edge colour as they
    // fall so the rounded tops hide the picture until then
    const fallen = std.smoothstep(0.6, 1.2, domino.angle);
    const back = std.mix(
      std.smoothstep(0.9, 0.97, -normal.z),
      std.smoothstep(0.6, 0.8, -normal.z),
      fallen,
    );
    const front = std.smoothstep(0.6, 0.8, normal.z);
    const edges = std.mix(base.color, domino.back, fallen);
    const color = std.mix(std.mix(edges, base.color, front), domino.back, back);
    return materials.glossy(color, base.roughness);
  },
});

const shape = meshes.roundedBox({
  width: WIDTH,
  height: HEIGHT,
  depth: THICKNESS,
  radius: 0.12,
  segments: 2,
});

/** Unit quaternion rotating +z toward `yaw`. */
const turned = (yaw: number) => [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)] as const;

/**
 * Allocates enough capacity for the larger layout. `lay` stands up the first `share` of a
 * layout in wave order without creating resources.
 */
export function createDominoes(
  root: TgpuRoot,
  options: { pool: MeshPool; layout: Layout; share: number },
) {
  const { pool } = options;
  const capacity = Math.max(...Object.values(LAYOUTS).map((each) => each.length));
  const set = instances.create(root, pool, { kind: Domino, capacity });
  const mesh = pool.add(shape);
  let first: readonly [number, number, number] = [0, 0, 0];

  /** Replaces the layout with its first `share` of dominoes, standing upright. */
  const lay = (layout: Layout, share: number) => {
    set.clear();

    const all = LAYOUTS[layout];
    const run = firstOf(all, Math.round(all.length * share));
    first = [all[0].x, HEIGHT / 2, all[0].z];
    const last = Math.max(...run.map((each) => each.depth), 1);

    const ids = run.map((each) =>
      set.add(mesh, {
        position: [each.x, HEIGHT / 2, each.z],
        rotation: turned(each.yaw),
        ...materials.glossy(PALETTE[each.color], ROUGHNESS),
        foot: [each.x, 0, each.z],
        yaw: each.yaw,
        order: each.depth / last,
        cell: each.cell ?? [-1, -1],
        angle: 0,
        spin: 0,
        hit: 0,
        back: PALETTE[each.color],
      }),
    );
    const idOf = (index: number) => (index === NONE ? 0 : ids[index]);
    run.forEach((each, i) => {
      set.write(ids[i], { prev: idOf(each.prev), next: [idOf(each.next[0]), idOf(each.next[1])] });
    });
    set.flush();
  };
  lay(options.layout, options.share);

  return {
    set,
    destroy() {
      set.destroy();
    },
    /** Position of the first domino, used by the automatic tap. */
    get first() {
      return first;
    },
    lay,
  };
}

export type Dominoes = ReturnType<typeof createDominoes>;
