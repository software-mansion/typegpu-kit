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

const COUNT = 300;
const size = { width: 200, height: 100 };

/**
 * Candidates on the screen where `places` puts them and far behind the eye otherwise, picked at
 * pixels: the found index, as a kernel later in the same pass reads it.
 */
function picking(
  root: TgpuRoot,
  places: Record<number, readonly [number, number, number]>,
  rank: picker.Options['rank'],
) {
  const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 10, pitch: 0 });
  const positions = Array.from({ length: COUNT }, (_, i) => places[i] ?? [0, 0, 50 + i]);
  const where = root.createReadonly(
    d.arrayOf(d.vec3f, COUNT),
    positions.map((p) => d.vec3f(...p)),
  );
  const hits = picker.create(root, view, {
    capacity: COUNT,
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
      return i !== 7;
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

async function picks(root: TgpuRoot, rank: picker.Options['rank']) {
  const { pickAt, pixelOf, destroy } = picking(root, PLACES, rank);
  const found: number[] = [];
  for (const [x, y] of targets(pixelOf)) found.push(await pickAt(x, y));
  destroy();
  return found;
}

it.each([
  ['by pointer', picker.byPointer],
  ['by eye', picker.byEye],
] as const)('picks %s as WebGPU does, a GPU form on the fallback', async (_, rank) => {
  const actual = await picks(gl, rank);
  const { stats, gl: context } = gl.device as unknown as GLDevice;
  expect([...stats.kernels.keys()]).toContain('pickPipeline (gpu)');
  expect([...stats.kernels.keys()]).not.toContain('pickPipeline');
  expect(stats.fallbacks.get('pickPipeline')).toBeUndefined();
  expect(context.getError()).toBe(0);
  expect(actual).toEqual(await picks(gpu, rank));
  // Some picks hit, some miss, so the comparison is not of nothing
  expect(actual.some((each) => each === picker.none)).toBe(true);
  expect(new Set(actual.filter((each) => each !== picker.none)).size).toBeGreaterThan(3);
});
