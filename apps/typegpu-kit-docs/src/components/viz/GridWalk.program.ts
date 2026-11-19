import * as grid from '@typegpu/kit/grid';
import { recordingRoot } from '@typegpu/kit/testing';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { clamp } from '../home/geometry.ts';
import { seeded } from './svg.ts';

/** A cell's side, and how far a walk looks. */
export const SIGHT = 1;

/** The rows a walk visits in a box one cell deep: its own and one along y each way. */
const ROWS = 3;

const WORKGROUP = 64;

const Seen = d.struct({ position: d.vec3f });

const Params = d.struct({
  /** Items `0` to `count - 1` are alive. */
  count: d.u32,
  /** Neighbours within sight a walk keeps before it stops. */
  stop: d.u32,
  /** Mixed into each item's seed, for another set of turns. */
  seed: d.f32,
});

const Walked = d.struct({
  /** The mean offset to the neighbours kept, in cells. */
  offset: d.vec3f,
  /** Items the walk met, itself included. */
  visited: d.u32,
  /** Neighbours within sight it kept, at most `stop`. */
  kept: d.u32,
});

export type Point = readonly [x: number, y: number];

export interface Options {
  /** Default a recording root, where nothing computes. */
  readonly root?: TgpuRoot;
  /** Cells across. Default 8. */
  readonly cols?: number;
  /** Cells down. Default 5. */
  readonly rows?: number;
  /** Items at most. Default 100. */
  readonly capacity?: number;
  /** Visits recorded per item, in order. Default 64. */
  readonly visits?: number;
}

export interface WalkOptions {
  /** Items alive, from slot 0. Default all. */
  readonly count?: number;
  /** Neighbours within sight a walk keeps. Default 8. */
  readonly stop?: number;
  /** Another value draws other turns. Default 0. */
  readonly seed?: number;
}

export interface Walk {
  readonly count: number;
  readonly stop: number;
  /** The slot at each sorted position, the alive items first. */
  readonly sorted: readonly number[];
  /** Each item's rows in the order it walked them: its own, then one cell along y each way. */
  readonly rows: readonly (readonly grid.Range[])[];
  /** The slots each item's walk met, in order, itself included, at most `visits`. */
  readonly visits: readonly (readonly number[])[];
  /** Neighbours within sight each walk kept. */
  readonly kept: readonly number[];
  /** The mean, over every item, of its mean offset to the neighbours its walk kept, in cells. */
  readonly mean: Point;
  /** The same for a walk from the corner over the same points, on the CPU. */
  readonly corner: Point;
}

export interface Program {
  readonly cols: number;
  readonly rows: number;
  readonly capacity: number;
  /** Every slot's item, in cells from the box's top left; alive or not. */
  readonly points: readonly Point[];
  /** Sorts the alive items into cells and walks from each. */
  walk(options?: WalkOptions): Promise<Walk>;
  destroy(): void;
}

/** The cell a point is in, clamped to the box, as the grid's `cellOf`. */
export const cellOf = ([x, y]: Point, cols: number, rows: number): Point => [
  clamp(Math.floor(x / SIGHT), 0, cols - 1),
  clamp(Math.floor(y / SIGHT), 0, rows - 1),
];

const within = (a: Point, b: Point) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= SIGHT * SIGHT;

/** The neighbours a walk kept, in the order it met them: the first `stop` within sight. */
export function keptOf(walk: Walk, points: readonly Point[], item: number): number[] {
  const kept: number[] = [];
  for (const other of walk.visits[item] ?? []) {
    if (kept.length >= walk.stop) break;
    if (other !== item && within(points[other], points[item])) kept.push(other);
  }
  return kept;
}

const mean = (offsets: readonly Point[]): Point => {
  const total = offsets.reduce<[number, number]>((sum, [x, y]) => [sum[0] + x, sum[1] + y], [0, 0]);
  return [total[0] / Math.max(offsets.length, 1), total[1] / Math.max(offsets.length, 1)];
};

/**
 * A walk from the corner on the CPU, as the grid's GPU test writes one: cells from -1 to +1 along
 * y, then x, stopping after `stop` neighbours within sight. The mean of each item's mean offset.
 */
function cornerFirst(
  points: readonly Point[],
  count: number,
  stop: number,
  cols: number,
  rows: number,
): Point {
  const cells = new Map<string, number[]>();
  for (let i = 0; i < count; i++) {
    const key = cellOf(points[i], cols, rows).join();
    cells.set(key, [...(cells.get(key) ?? []), i]);
  }

  const offsets = points.slice(0, count).map((point, i): Point => {
    const [x, y] = cellOf(point, cols, rows);
    let sum: [number, number] = [0, 0];
    let seen = 0;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        for (const j of cells.get([x + dx, y + dy].join()) ?? []) {
          if (j === i || !within(points[j], point) || seen >= stop) continue;
          seen++;
          sum = [sum[0] + points[j][0] - point[0], sum[1] + points[j][1] - point[1]];
        }
      }
    }

    return [sum[0] / Math.max(seen, 1), sum[1] / Math.max(seen, 1)];
  });

  return mean(offsets);
}

export async function createProgram({
  root = recordingRoot().root,
  cols = 8,
  rows = 5,
  capacity = 100,
  visits = 64,
}: Options = {}): Promise<Program> {
  const random = seeded(1);
  const points = Array.from({ length: capacity }, (): Point => [
    random() * cols * SIGHT,
    random() * rows * SIGHT,
  ]);

  const placed = root.createReadonly(
    d.arrayOf(d.vec3f, capacity),
    points.map(([x, y]) => d.vec3f(x, y, SIGHT / 2)),
  );
  const params = root.createUniform(Params);

  const cells = grid.create(root, {
    capacity,
    low: d.vec3f(0, 0, 0),
    size: d.vec3f(cols * SIGHT, rows * SIGHT, SIGHT),
    sight: SIGHT,
    item: Seen,
    alive: (i) => {
      'use gpu';
      return i < params.$.count;
    },
    position: (i) => {
      'use gpu';
      return d.vec3f(placed.$[i]);
    },
    itemOf: (i) => {
      'use gpu';
      return Seen({ position: placed.$[i] });
    },
  });

  const walked = root.createBuffer(d.arrayOf(Walked, capacity)).$usage('storage');
  const met = root.createBuffer(d.arrayOf(grid.Range, capacity * ROWS)).$usage('storage');
  const visited = root.createBuffer(d.arrayOf(d.u32, capacity * visits)).$usage('storage');
  const sorted = root.createBuffer(d.arrayOf(d.u32, capacity)).$usage('storage');
  const walking = walked.as('mutable');
  const meeting = met.as('mutable');
  const visiting = visited.as('mutable');
  const sorting = sorted.as('mutable');

  const probe = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= capacity || i >= params.$.count) return;

    const position = d.vec3f(placed.$[i]);
    randf.seed2(d.vec2f(d.f32(i), params.$.seed));

    let steps = d.u32(0);
    let kept = d.u32(0);
    let offsets = d.vec3f();

    const from = cells.walk(position);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(from, r);
      if (r < ROWS) meeting.$[i * ROWS + r] = grid.Range(range);

      for (let j = d.u32(0); j < range.count; j++) {
        const slot = cells.slot(range, j);
        if (steps < visits) visiting.$[i * visits + steps] = slot;
        steps++;
        if (slot === i || kept >= params.$.stop) continue;

        const offset = cells.neighbour(range, j).position - position;
        if (std.dot(offset, offset) > SIGHT * SIGHT) continue;

        kept++;
        offsets += offset;
      }
    }

    walking.$[i] = Walked({ offset: offsets / std.max(d.f32(kept), 1), visited: steps, kept });
  });

  const dump = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const k = gid.x;
    if (k >= capacity) return;

    sorting.$[k] = cells.slotAt(k);
  });

  const probing = root.pipe(cells.inject()).createComputePipeline({ compute: probe });
  const dumping = root.pipe(cells.inject()).createComputePipeline({ compute: dump });
  await Promise.all([cells.initAsync(), probing.initAsync(), dumping.initAsync()]);

  const { device } = root;
  const groups = Math.ceil(capacity / WORKGROUP);

  return {
    cols,
    rows,
    capacity,
    points,

    async walk({ count = capacity, stop = 8, seed: turns = 0 } = {}) {
      const alive = clamp(Math.round(count), 0, capacity);
      params.write({ count: alive, stop, seed: turns });

      device.pushErrorScope('validation');
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      cells.sort(pass);
      pass.setPipeline(probing);
      pass.dispatchWorkgroups(groups);
      pass.setPipeline(dumping);
      pass.dispatchWorkgroups(groups);
      pass.end();
      encoder.submit();
      const error = await device.popErrorScope();
      if (error) throw new Error(error.message);

      const [each, rowsMet, slots, order] = await Promise.all([
        walked.read(),
        met.read(),
        visited.read(),
        sorted.read(),
      ]);

      const items = each.slice(0, alive);
      return {
        count: alive,
        stop,
        sorted: order.slice(0, alive),
        rows: items.map((_, i) => rowsMet.slice(i * ROWS, (i + 1) * ROWS)),
        visits: items.map(({ visited: steps }, i) =>
          slots.slice(i * visits, i * visits + Math.min(steps, visits)),
        ),
        kept: items.map(({ kept }) => kept),
        mean: mean(items.map(({ offset }): Point => [offset.x, offset.y])),
        corner: cornerFirst(points, alive, stop, cols, rows),
      };
    },

    destroy() {
      cells.destroy();
      placed.buffer.destroy();
      params.buffer.destroy();
      walked.destroy();
      met.destroy();
      visited.destroy();
      sorted.destroy();
    },
  };
}
