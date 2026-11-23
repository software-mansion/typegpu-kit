import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { createJumpFlood, NONE } from '../src/jump-flood/jump-flood.ts';
import { createMap } from '../src/jump-flood/map.ts';

interface Flooding {
  /** Sets dispatch values with immediates. */
  readonly immediates: boolean;
  /** Switches to immediates or bind groups after creation. Default no switch. */
  readonly switchTo?: boolean;
  readonly size?: number;
  readonly seeds?: number;
  /** Seconds the seeds have drifted. Default 0. */
  readonly time?: number;
  /** Flood dispatches to record. Default all. */
  readonly passes?: number;
}

/** Floods a grid once, and reads back each cell's seed and each seed's position. */
async function flooded({
  immediates,
  switchTo = immediates,
  size = 64,
  seeds = 12,
  time = 0,
  passes,
}: Flooding) {
  const root = await gpuRoot();
  const flood = createJumpFlood(root, {
    size,
    seeds,
    format: 'bgra8unorm',
    time: root.createUniform(d.f32, time),
    target: { width: 16, height: 9 },
    paint: createMap(root, { size, seeds }).paint,
    immediates,
  });
  await flood.setImmediates(switchTo);
  expect(flood.immediates).toBe(switchTo);

  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  flood.flood(pass, passes);
  pass.end();
  encoder.submit();

  const [cells, seedData] = await Promise.all([flood.cells.read(), flood.seeds.read()]);
  root.destroy();

  const cellCount = cells.length / 2;
  const half = (passes ?? flood.passes) % 2;
  return {
    cells: cells.slice(half * cellCount, (half + 1) * cellCount),
    positions: seedData.map(({ position }) => position),
  };
}

describe('jump flood', () => {
  it('finds the nearest seed of almost every cell, with immediates or with bind groups', async () => {
    const withImmediates = await flooded({ immediates: true });
    const withBindGroups = await flooded({ immediates: false });
    const switched = await flooded({ immediates: false, switchTo: true });
    expect(withBindGroups.cells).toEqual(withImmediates.cells);
    expect(switched.cells).toEqual(withImmediates.cells);

    const size = 64;
    const { cells, positions } = withImmediates;
    const distance = (seed: number, x: number, y: number) =>
      Math.hypot(positions[seed].x * size - x - 0.5, positions[seed].y * size - y - 0.5);

    let wrong = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const found = cells[y * size + x];
        expect(found).not.toBe(NONE);

        const nearest = Math.min(...positions.map((_, seed) => distance(seed, x, y)));
        if (distance(found, x, y) > nearest + 1e-4) wrong++;
      }
    }

    // Jump flooding is approximate; the extra pass of step 1 mends most of what it misses.
    expect(wrong / (size * size)).toBeLessThan(0.01);
  });

  it('places colliding seeds so the lowest index holds the cell', async () => {
    // At this time, towns 11 and 23 of the scene's 48 land in cell (309, 72) of its 512² grid
    const size = 512;
    const { cells, positions } = await flooded({
      immediates: false,
      size,
      seeds: 48,
      time: 1766 / 60,
      passes: 0,
    });
    const cellOf = ({ x, y }: { x: number; y: number }) =>
      Math.floor(y * size) * size + Math.floor(x * size);
    expect(cellOf(positions[11])).toBe(cellOf(positions[23]));
    expect(cells[cellOf(positions[11])]).toBe(11);
  });

  it('floods the way it did until the other way compiles, then the latest one chosen', async () => {
    const root = await gpuRoot();
    const flood = createJumpFlood(root, {
      size: 64,
      seeds: 12,
      format: 'bgra8unorm',
      time: root.createUniform(d.f32),
      target: { width: 16, height: 9 },
      paint: createMap(root, { size: 64, seeds: 12 }).paint,
      immediates: false,
    });
    await flood.initAsync();
    const there = flood.setImmediates(true);
    const back = flood.setImmediates(false);
    expect(flood.immediates).toBe(false);
    await Promise.all([there, back]);
    expect(flood.immediates).toBe(false);

    await flood.setImmediates(true);
    expect(flood.immediates).toBe(true);
    root.destroy();
  });
});
