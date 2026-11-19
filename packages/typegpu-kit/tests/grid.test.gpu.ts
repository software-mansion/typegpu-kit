import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { randf } from '@typegpu/noise';
import { d, std, tgpu } from 'typegpu';
import { describe, expect, it } from 'vitest';
import * as grid from '../src/grid.ts';

const Item = d.struct({ position: d.vec3f, id: d.u32 });

type Binning = NonNullable<grid.Options<d.U32>['binning']>;
const BINNINGS = [
  ['counting', grid.counting],
  ['radix', grid.radix],
] as const;

const LOW = d.vec3f(-4, 0, -3);
const SIZE = d.vec3f(8, 4, 6);
const SIGHT = 1;
const WORKGROUP = 64;

/** A point moved well outside the box. */
const away = (point: d.v3f) => d.vec3f(point.x + 30, point.y - 10, point.z + 20);

/** Every seventh slot is empty. */
const alive = (i: number) => i % 7 !== 0;

/** Points spread evenly through the box, from a seeded generator so that a failure repeats. */
function scatter(count: number, size = SIZE) {
  let state = 7;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };

  return Array.from({ length: count }, () =>
    d.vec3f(LOW.x + next() * size.x, LOW.y + next() * size.y, LOW.z + next() * size.z),
  );
}

/**
 * Sorts `count` scattered items into a grid, then walks it from each alive one, stopping after
 * `cap` neighbours within sight. Reads back how many each found, the mean offset to them, whether
 * the walk met the item itself in its first row, and how many neighbours `slot` misnamed.
 */
async function walk(count: number, cap: number, binning: Binning = grid.counting) {
  const root = await gpuRoot();
  const points = scatter(count);
  const positions = root.createBuffer(d.arrayOf(d.vec3f, count), points).$usage('storage');
  const read = positions.as('readonly');

  const cells = grid.create(root, {
    capacity: count,
    low: LOW,
    size: SIZE,
    sight: SIGHT,
    item: Item,
    binning,
    alive: (i) => {
      'use gpu';
      return i % 7 !== 0;
    },
    position: (i) => {
      'use gpu';
      return d.vec3f(read.$[i]);
    },
    itemOf: (i) => {
      'use gpu';
      return Item({ position: read.$[i], id: d.u32(i) });
    },
  });

  const found = root.createBuffer(d.arrayOf(d.u32, count)).$usage('storage');
  const home = root.createBuffer(d.arrayOf(d.u32, count)).$usage('storage');
  const drift = root.createBuffer(d.arrayOf(d.vec3f, count)).$usage('storage');
  const wrong = root.createBuffer(d.arrayOf(d.u32, count)).$usage('storage');
  const finding = found.as('mutable');
  const homing = home.as('mutable');
  const drifting = drift.as('mutable');
  const misnaming = wrong.as('mutable');

  const probe = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= count || i % 7 === 0) return;

    const position = d.vec3f(read.$[i]);
    randf.seed2(d.vec2f(d.f32(i), 0.5));

    let seen = d.u32(0);
    let offsets = d.vec3f();
    let atHome = d.u32(0);
    let misnamed = d.u32(0);

    const from = cells.walk(position);
    for (let r = d.u32(0); r < grid.RANGES && seen < cap; r++) {
      const range = cells.range(from, r);

      for (let j = d.u32(0); j < range.count && seen < cap; j++) {
        const other = cells.neighbour(range, j);
        if (cells.slot(range, j) !== other.id) misnamed++;
        if (other.id === i) {
          if (r === 0) atHome = 1;
          continue;
        }

        const offset = other.position - position;
        if (std.dot(offset, offset) > SIGHT * SIGHT) continue;

        seen++;
        offsets += offset;
      }
    }

    finding.$[i] = seen;
    homing.$[i] = atHome;
    drifting.$[i] = offsets / std.max(d.f32(seen), 1);
    misnaming.$[i] = misnamed;
  });

  const pipeline = root.pipe(cells.inject()).createComputePipeline({ compute: probe });
  await cells.initAsync();

  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  cells.sort(pass);
  pass.setPipeline(pipeline);
  pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
  pass.end();
  encoder.submit();

  const result = await Promise.all([found.read(), drift.read(), home.read(), wrong.read()]);

  root.destroy();

  const [counts, offsets, atHome, misnamed] = result;
  return { points, counts, offsets, atHome, misnamed };
}

type Offset = readonly [x: number, y: number, z: number];

/** The mean of each alive item's mean offset to the neighbours its walk found. */
function meanDrift(offsets: readonly Offset[]) {
  const alives = offsets.filter((_, i) => alive(i));
  return [0, 1, 2].map(
    (axis) => alives.reduce((sum, offset) => sum + offset[axis], 0) / alives.length,
  );
}

/**
 * A walk from the corner on the CPU: cells from -1 to +1 along z, then y, then x, stopping after
 * `cap` neighbours within sight.
 */
function cornerFirst(points: readonly d.v3f[], cap: number): Offset[] {
  const dims = [SIZE.x, SIZE.y, SIZE.z].map((side) => Math.ceil(side / SIGHT));
  const cellOf = (p: d.v3f) =>
    [p.x - LOW.x, p.y - LOW.y, p.z - LOW.z].map((offset, axis) =>
      Math.min(Math.max(Math.floor(offset / SIGHT), 0), dims[axis] - 1),
    );

  const cells = new Map<string, number[]>();
  points.forEach((point, i) => {
    if (!alive(i)) return;
    const key = cellOf(point).join();
    cells.set(key, [...(cells.get(key) ?? []), i]);
  });

  return points.map((point, i) => {
    const [x, y, z] = cellOf(point) as [number, number, number];
    const sum = [0, 0, 0];
    let seen = 0;

    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          for (const j of cells.get([x + dx, y + dy, z + dz].join()) ?? []) {
            const offset = [points[j].x - point.x, points[j].y - point.y, points[j].z - point.z];
            const distance2 = offset.reduce((total, value) => total + value * value, 0);
            if (j === i || distance2 > SIGHT * SIGHT || seen >= cap) continue;

            seen++;
            offset.forEach((value, axis) => (sum[axis] += value));
          }
        }
      }
    }

    return [sum[0] / Math.max(seen, 1), sum[1] / Math.max(seen, 1), sum[2] / Math.max(seen, 1)];
  });
}

describe.each(BINNINGS)('neighbour grid, %s', (_name, binning) => {
  it('finds every alive neighbour within sight, starts at the home row, and names its slot', async () => {
    const count = 1500;
    const { points, counts, atHome, misnamed } = await walk(count, count, binning);

    const expected = points.map((point, i) => {
      if (!alive(i)) return 0;

      return points.filter((other, j) => {
        const distance2 =
          (other.x - point.x) ** 2 + (other.y - point.y) ** 2 + (other.z - point.z) ** 2;
        return j !== i && alive(j) && distance2 <= SIGHT * SIGHT;
      }).length;
    });

    expect(counts).toEqual(expected);
    expect(atHome).toEqual(points.map((_, i) => (alive(i) ? 1 : 0)));
    expect(misnamed.every((each) => each === 0)).toBe(true);
  });

  it('finds from any point, inside the box or out, every alive item within sight, by the slots its rows name', async () => {
    const root = await gpuRoot();
    const count = 800;
    const points = scatter(count);
    const positions = root.createBuffer(d.arrayOf(d.vec3f, count), points).$usage('storage');
    const read = positions.as('readonly');
    const cells = grid.create(root, {
      capacity: count,
      low: LOW,
      size: SIZE,
      sight: SIGHT,
      binning,
      item: d.u32,
      alive: (i) => {
        'use gpu';
        return i % 7 !== 0;
      },
      position: (i) => {
        'use gpu';
        return d.vec3f(read.$[i]);
      },
      itemOf: (i) => {
        'use gpu';
        return d.u32(i);
      },
    });

    // Queries where no item stands, a few past the box's edges.
    const queries = scatter(count + 200)
      .slice(count)
      .map((point, i) =>
        i % 5 === 0 ? d.vec3f(point.x * 1.2, point.y - 0.5, point.z * 1.2) : point,
      );
    const asked = root.createBuffer(d.arrayOf(d.vec3f, queries.length), queries).$usage('storage');
    const found = root.createBuffer(d.arrayOf(d.u32, queries.length)).$usage('storage');
    const query = asked.as('readonly');
    const finding = found.as('mutable');

    const probe = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const q = gid.x;
      if (q >= queries.length) return;

      const point = d.vec3f(query.$[q]);
      let seen = d.u32(0);
      const from = cells.around(point);
      for (let r = d.u32(0); r < grid.RANGES; r++) {
        const range = cells.range(from, r);
        for (let j = d.u32(0); j < range.count; j++) {
          const offset = read.$[cells.slotAt(cells.index(range, j))] - point;
          if (std.dot(offset, offset) <= SIGHT * SIGHT) seen++;
        }
      }
      finding.$[q] = seen;
    });

    const pipeline = root.pipe(cells.inject()).createComputePipeline({ compute: probe });
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    cells.sort(pass);
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(Math.ceil(queries.length / WORKGROUP));
    pass.end();
    encoder.submit();

    const counts = await found.read();
    root.destroy();

    const expected = queries.map(
      (point) =>
        points.filter(
          (other, j) =>
            alive(j) &&
            (other.x - point.x) ** 2 + (other.y - point.y) ** 2 + (other.z - point.z) ** 2 <=
              SIGHT * SIGHT,
        ).length,
    );
    expect(counts).toEqual(expected);
    expect(expected.filter((each) => each > 0).length).toBeGreaterThan(queries.length / 2);
  });

  it('covers the box it moves to, sorted again', async () => {
    const root = await gpuRoot();
    const count = 400;
    const points = scatter(count).map(away);
    const positions = root.createBuffer(d.arrayOf(d.vec3f, count), points).$usage('storage');
    const read = positions.as('readonly');
    const cells = grid.create(root, {
      capacity: count,
      low: LOW,
      size: SIZE,
      sight: SIGHT,
      binning,
      item: d.u32,
      position: (i) => {
        'use gpu';
        return d.vec3f(read.$[i]);
      },
      itemOf: (i) => {
        'use gpu';
        return d.u32(i);
      },
    });
    expect(cells.capacity).toBe(count);

    const found = root.createBuffer(d.arrayOf(d.u32, count)).$usage('storage');
    const visited = root.createBuffer(d.arrayOf(d.u32, count)).$usage('storage');
    const finding = found.as('mutable');
    const visiting = visited.as('mutable');
    const probe = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const i = gid.x;
      if (i >= count) return;

      const point = d.vec3f(read.$[i]);
      let seen = d.u32(0);
      let rows = d.u32(0);
      const from = cells.around(point);
      for (let r = d.u32(0); r < grid.RANGES; r++) {
        const range = cells.range(from, r);
        rows += range.count;
        for (let j = d.u32(0); j < range.count; j++) {
          const offset = read.$[cells.slotAt(cells.index(range, j))] - point;
          if (std.dot(offset, offset) <= SIGHT * SIGHT) seen++;
        }
      }
      finding.$[i] = seen;
      visiting.$[i] = rows;
    });
    const pipeline = root.pipe(cells.inject()).createComputePipeline({ compute: probe });

    const counted = async () => {
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      cells.sort(pass);
      pass.setPipeline(pipeline);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
      pass.end();
      encoder.submit();
      return { seen: await found.read(), visited: await visited.read() };
    };

    const expected = points.map(
      (point) =>
        points.filter(
          (other) =>
            (other.x - point.x) ** 2 + (other.y - point.y) ** 2 + (other.z - point.z) ** 2 <=
            SIGHT * SIGHT,
        ).length,
    );
    // Outside the box every item counts as in its corner cell, so each walk visits them all.
    const outside = await counted();
    expect(outside.seen).toEqual(expected);
    expect(outside.visited.every((each) => each === count)).toBe(true);

    cells.move(away(LOW));
    const inside = await counted();
    expect(inside.seen).toEqual(expected);
    expect(Math.max(...inside.visited)).toBeLessThan(count / 4);
    root.destroy();
  });

  it('sorts again after items move and die and the span changes, emptying what the last sort placed', async () => {
    const root = await gpuRoot();
    const count = 1200;
    // Lines of 100 cells take a counting reserve more than one workgroup's width
    const long = d.vec3f(100, SIZE.y, SIZE.z);
    const first = scatter(count, long);
    // Most items move across the box; one in three dies
    const second = scatter(2 * count, long).slice(count);
    const placed = root.createBuffer(d.arrayOf(d.vec3f, count), first).$usage('storage');
    // One in `x` dies, and none from `y` on
    const lives = root.createBuffer(d.vec2u, d.vec2u(3, count)).$usage('uniform');
    let span = count;
    const read = placed.as('readonly');
    const living = lives.as('uniform');
    const cells = grid.create(root, {
      capacity: count,
      low: LOW,
      size: long,
      sight: SIGHT,
      binning,
      item: d.u32,
      alive: (i) => {
        'use gpu';
        return i % living.$.x !== 0 && i < living.$.y;
      },
      span: () => span,
      position: (i) => {
        'use gpu';
        return d.vec3f(read.$[i]);
      },
      itemOf: (i) => {
        'use gpu';
        return d.u32(i);
      },
    });

    const found = root.createBuffer(d.arrayOf(d.u32, count)).$usage('storage');
    const order = root.createBuffer(d.arrayOf(d.vec2u, count)).$usage('storage');
    const tally = root.createBuffer(d.arrayOf(d.atomic(d.u32), 2)).$usage('storage');
    const finding = found.as('mutable');
    const ordering = order.as('mutable');
    const tallying = tally.as('mutable');

    const probe = tgpu.computeFn({
      workgroupSize: [WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const i = gid.x;
      if (i >= count) return;
      const point = d.vec3f(read.$[i]);
      let seen = d.u32(0);
      const from = cells.around(point);
      for (let r = d.u32(0); r < grid.RANGES; r++) {
        const range = cells.range(from, r);
        for (let j = d.u32(0); j < range.count; j++) {
          const offset = read.$[cells.neighbour(range, j)] - point;
          if (std.dot(offset, offset) <= SIGHT * SIGHT) seen++;
        }
      }
      finding.$[i] = seen;
    });
    // Over sorted items: each index's slot and copy
    const listing = tgpu.computeFn({
      workgroupSize: [grid.WORKGROUP],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const k = gid.x;
      if (k >= cells.count()) return;
      ordering.$[k] = d.vec2u(cells.slotAt(k), cells.neighbourAt(k));
    });
    // One workgroup per block holding items: those that hold items, and the blocks
    const blocking = tgpu.computeFn({
      workgroupSize: [1],
      in: { wid: d.builtin.workgroupId },
    })(({ wid }) => {
      'use gpu';
      if (cells.holds(cells.block(wid.x))) std.atomicAdd(tallying.$[0], 1);
      std.atomicAdd(tallying.$[1], 1);
    });

    const walking = root.pipe(cells.inject());
    const probing = walking.createComputePipeline({ compute: probe });
    const lister = walking.createComputePipeline({ compute: listing });
    const blocker = walking.createComputePipeline({ compute: blocking });

    const expected = (points: readonly d.v3f[], every: number) =>
      points.map(
        (point) =>
          points.filter(
            (other, j) =>
              j % every !== 0 &&
              j < span &&
              (other.x - point.x) ** 2 + (other.y - point.y) ** 2 + (other.z - point.z) ** 2 <=
                SIGHT * SIGHT,
          ).length,
      );

    const step = async (points: readonly d.v3f[], every: number, until: number) => {
      span = until;
      placed.write(points);
      lives.write(d.vec2u(every, until));
      tally.write([0, 0]);
      const encoder = root.createCommandEncoder();
      const pass = encoder.beginComputePass();
      cells.sort(pass);
      pass.setPipeline(probing);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
      pass.setPipeline(lister);
      cells.dispatch(pass);
      pass.setPipeline(blocker);
      cells.dispatchBlocks(pass);
      pass.end();
      encoder.submit();

      const live = Array.from({ length: until }, (_slot, i) => i).filter((i) => i % every !== 0);
      const [seen, sorted, [holding, blocks]] = await Promise.all([
        found.read(),
        order.read(),
        tally.read(),
      ]);
      expect(seen).toEqual(expected(points, every));
      const slots = sorted.slice(0, live.length);
      expect(slots.every(({ x, y }) => x === y)).toBe(true);
      expect(slots.map(({ x }) => x).toSorted((a, b) => a - b)).toEqual(live);
      const occupied = new Set(
        live.map((i) => {
          const at = points[i];
          return [at.x - LOW.x, at.y - LOW.y, at.z - LOW.z]
            .map((offset) => Math.floor(Math.floor(offset / SIGHT) / grid.BLOCK))
            .join();
        }),
      );
      expect(blocks).toBe(occupied.size);
      expect(holding).toBe(occupied.size);
    };

    await step(first, 7, count);
    await step(second, 3, 500);
    await step(first, 7, count);
    await step(second, 3, 100);
    root.destroy();
  });
});

it('dispatches over every block within its margin of one holding items, once each', async () => {
  const root = await gpuRoot();
  // 16 × 4 × 12 cells are 4 × 1 × 3 blocks; two items hold blocks (0, 0, 0) and (3, 0, 2)
  const points = [d.vec3f(-3.5, 0.5, -2.5), d.vec3f(11.5, 0.5, 8.5)];
  const placed = root.createBuffer(d.arrayOf(d.vec3f, 2), points).$usage('storage');
  const read = placed.as('readonly');
  const cells = grid.create(root, {
    capacity: 2,
    low: LOW,
    size: d.vec3f(16, 4, 12),
    sight: SIGHT,
    margin: 1,
    item: d.u32,
    position: (i) => {
      'use gpu';
      return d.vec3f(read.$[i]);
    },
    itemOf: (i) => {
      'use gpu';
      return d.u32(i);
    },
  });

  const seen = root.createBuffer(d.arrayOf(d.atomic(d.u32), 12)).$usage('storage');
  const seeing = seen.as('mutable');
  const visit = tgpu.computeFn({ workgroupSize: [1], in: { wid: d.builtin.workgroupId } })(({
    wid,
  }) => {
    'use gpu';
    const block = cells.block(wid.x);
    const index = d.u32(block.x + 4 * block.z);
    std.atomicAdd(seeing.$[index], std.select(d.u32(1), d.u32(100), cells.holds(block)));
  });
  const visiting = root.pipe(cells.inject()).createComputePipeline({ compute: visit });

  for (let sort = 0; sort < 2; sort++) {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    cells.sort(pass);
    pass.setPipeline(visiting);
    cells.dispatchBlocks(pass);
    pass.end();
    encoder.submit();
  }

  // Rows of z: each block once a sort, a held one counting 100
  expect(await seen.read()).toEqual([200, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 200]);
  root.destroy();
});

describe('neighbour grid walks', () => {
  // A school packs tighter than this around food. Stopping after 16, a walk from the corner sees the
  // cells at -x, -y and -z first, so the neighbours it averages lie that way and steer it there.
  it('drifts no way when a walk stops early in a dense pack, as a corner-first walk does', async () => {
    const cap = 16;
    const { points, counts, offsets } = await walk(6000, cap);
    expect(Math.min(...counts.filter((_, i) => alive(i)))).toBeGreaterThan(cap / 2);

    const walked = offsets.map(({ x, y, z }): Offset => [x, y, z]);
    for (const axis of meanDrift(walked)) expect(Math.abs(axis)).toBeLessThan(0.02);
    for (const axis of meanDrift(cornerFirst(points, cap))) expect(axis).toBeLessThan(-0.1);
  });
});
