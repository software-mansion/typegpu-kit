import { cameras, picker } from '@typegpu/kit';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from './compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  // Placement would move the picks to the CPU, which reads each one back: these test the form
  ({ gpu, gl } = await roots({ placement: false }));
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});

const size = { width: 200, height: 100 };

/**
 * Candidates on the screen where `places` puts them and far behind the eye otherwise, picked at
 * pixels: the found index, as a kernel later in the same pass reads it.
 */
function picking(
  root: TgpuRoot,
  count: number,
  places: Record<number, readonly [number, number, number]>,
  dead: number,
  rank: picker.Options['rank'],
) {
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 10, pitch: 0 });
  const positions = Array.from({ length: count }, (_, i) => places[i] ?? [0, 0, 50 + (i % 997)]);
  const where = root.createReadonly(
    d.arrayOf(d.vec3f, count),
    positions.map((p) => d.vec3f(...p)),
  );
  const hits = picker.create(root, view, {
    capacity: count,
    size,
    position: (i) => {
      'use gpu';
      return d.vec3f(where.$[i]);
    },
    radius: () => {
      'use gpu';
      return 0.05;
    },
    alive: (i) => {
      'use gpu';
      return i !== dead;
    },
    ...(rank ? { rank } : {}),
  });
  const seen = root.createMutable(d.u32);
  const mark = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    seen.$ = hits.found();
  });
  const marking = root.pipe(hits.inject()).createComputePipeline({ compute: mark });
  const pickAt = async (x: number, y: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    hits.pick(pass, x / size.width, y / size.height);
    pass.setPipeline(marking);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.submit();
    return seen.read();
  };
  const pixelOf = (point: readonly [number, number, number]) => {
    const ndc = [0, 0, 0];
    view.project(point, ndc);
    return [
      (((ndc[0] as number) + 1) / 2) * size.width,
      ((1 - (ndc[1] as number)) / 2) * size.height,
    ] as const;
  };
  return { pickAt, pixelOf, destroy: () => hits.destroy() };
}

/**
 * A row of candidates stacked at one screen place at three depths, one pair sharing a place and a
 * depth, single candidates, the dead index 7 in front of everything, and candidates just inside
 * and just outside the reach of a point between them.
 */
const PLACES: Record<number, readonly [number, number, number]> = {
  3: [0, 0, 2],
  4: [0, 0, 0],
  5: [0, 0, -2],
  7: [0, 0, 4],
  11: [2, 1, 0],
  12: [2, 1, 0],
  20: [-3, -1, 1],
  21: [-3.1, -1, -1],
  40: [3.5, -1.5, 0],
};

/** Pixels: on each place, between places, empty sky, and steps outward from candidate 40. */
function targets(pixelOf: (point: readonly [number, number, number]) => readonly [number, number]) {
  const on = Object.values(PLACES).map(pixelOf);
  const [ex, ey] = pixelOf([3.5, -1.5, 0]);
  const around = [0, 4, 8, 12, 16, 20, 24, 28, 32].map((dx) => [ex + dx, ey] as const);
  return [...on, [100, 50], [5, 5], [195, 95], [60, 40], ...around] as const;
}

/**
 * The places at indices counted down from the top of the capacity when `high`, so the key packs
 * the largest indices its bits hold; the dead index moves with them.
 */
function placed(count: number, high: boolean) {
  const at = (k: number) => (high ? count - 1 - k : k);
  const places = Object.fromEntries(
    Object.entries(PLACES).map(([k, place]) => [at(Number(k)), place]),
  ) as Record<number, readonly [number, number, number]>;
  return { places, dead: at(7) };
}

async function picks(root: TgpuRoot, count: number, high: boolean, rank: picker.Options['rank']) {
  const { places, dead } = placed(count, high);
  const { pickAt, pixelOf, destroy } = picking(root, count, places, dead, rank);
  const found: number[] = [];
  for (const [x, y] of targets(pixelOf)) found.push(await pickAt(x, y));
  destroy();
  return found;
}

// 300 rows leave the depth rank 23 bits beside a 9-bit index; 70 000 leave 15 beside 17
it.each([
  ['by pointer', 300, false, picker.byPointer],
  ['by eye', 300, false, picker.byEye],
  ['by pointer, at the top of 300 rows', 300, true, picker.byPointer],
  ['by eye, at the top of 300 rows', 300, true, picker.byEye],
  ['by pointer, at the top of 70 000 rows', 70_000, true, picker.byPointer],
  ['by eye, at the top of 70 000 rows', 70_000, true, picker.byEye],
] as const)('picks %s as WebGPU does, a GPU form on the fallback', async (_, count, high, rank) => {
  const actual = await picks(gl, count, high, rank);
  const { stats, gl: context } = gl.device as unknown as GLDevice;
  expect([...stats.kernels.keys()]).toContain('pickPipeline (gpu)');
  expect([...stats.kernels.keys()]).not.toContain('pickPipeline');
  expect(stats.fallbacks.get('pickPipeline')).toBeUndefined();
  expect(context.getError()).toBe(0);
  expect(actual).toEqual(await picks(gpu, count, high, rank));
  if (high)
    expect(actual.filter((each) => each !== picker.none).every((each) => each > count - 64)).toBe(
      true,
    );
  // Some picks hit, some miss, so the comparison is not of nothing
  expect(actual.some((each) => each === picker.none)).toBe(true);
  expect(new Set(actual.filter((each) => each !== picker.none)).size).toBeGreaterThan(3);
});
