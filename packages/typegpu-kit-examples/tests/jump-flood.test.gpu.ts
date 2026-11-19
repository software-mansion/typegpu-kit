import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { createJumpFlood, NONE } from '../src/jump-flood/jump-flood.ts';
import { createMap } from '../src/jump-flood/map.ts';

/** Floods a small grid once, and reads back each cell's seed and each seed's position. */
async function flooded(immediates: boolean) {
  const root = await gpuRoot();
  const flood = createJumpFlood(root, {
    size: 64,
    seeds: 12,
    format: 'bgra8unorm',
    time: root.createUniform(d.f32),
    target: { width: 16, height: 9 },
    paint: createMap(root, { size: 64, seeds: 12 }).paint,
    immediates,
  });
  expect(flood.immediates).toBe(immediates);

  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  flood.flood(pass);
  pass.end();
  encoder.submit();

  const [cells, seeds] = await Promise.all([flood.cells.read(), flood.seeds.read()]);
  root.destroy();

  const cellCount = cells.length / 2;
  const half = flood.passes % 2;
  return {
    size: Math.sqrt(cellCount),
    cells: cells.slice(half * cellCount, (half + 1) * cellCount),
    positions: seeds.map(({ position }) => position),
  };
}

describe('jump flood', () => {
  it('finds the nearest seed of almost every cell, with immediates or with bind groups', async () => {
    const withImmediates = await flooded(true);
    const withBindGroups = await flooded(false);
    expect(withBindGroups.cells).toEqual(withImmediates.cells);

    const { size, cells, positions } = withImmediates;
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
});
