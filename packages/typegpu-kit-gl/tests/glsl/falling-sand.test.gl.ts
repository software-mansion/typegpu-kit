import { cameras, counters } from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import { createCave } from '../../../typegpu-kit-examples/src/falling-sand/cave.ts';
import {
  cellOf,
  createCells,
  gridKernel,
  Matter,
  HEIGHT,
  WIDTH,
} from '../../../typegpu-kit-examples/src/falling-sand/cells.ts';
import { createEmbers } from '../../../typegpu-kit-examples/src/falling-sand/embers.ts';
import { createRules } from '../../../typegpu-kit-examples/src/falling-sand/rules.ts';
import type { GLDevice } from '../../src/device/index.ts';
import { roots } from '../device/compare.ts';

// falling-sand's own kernels, with no branch for the fallback: blocks and pairs of cells each
// invocation owns, counts it sums and embers it takes ring slots for

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

const STEPS = 120;

/** Paints the cave and steps its rules, reading the cells every 20 steps, the counts and embers. */
async function stepped(root: TgpuRoot, flying = false) {
  const counts = counters.create(root, ['cooled', 'burnt']);
  const cells = createCells(root);
  const cave = createCave(root, cells);
  const view = cameras.flat(root, { size: { width: WIDTH, height: HEIGHT }, height: HEIGHT });
  const { sparks, spark } = createEmbers(root, { camera: view, step: 1 / 120, cells, counts });
  const rules = createRules(root, { cells, counts, spark });
  const out = root.createMutable(d.arrayOf(d.u32, WIDTH * HEIGHT));
  const copy = root.createComputePipeline({
    compute: gridKernel(({ gid }) => {
      'use gpu';
      if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
      out.$[gid.y * WIDTH + gid.x] = cells.cellAt(d.vec2i(gid.xy));
    }),
  });
  // A band of wood over a band of lava, so that wood catches from the lava and from embers
  const kindling = root
    .createComputePipeline({
      compute: gridKernel(({ gid }) => {
        'use gpu';
        const at = d.vec2i(gid.xy);
        if (at.x < 20 || at.x >= WIDTH - 20) return;
        const row = at.y - d.i32(HEIGHT / 3);
        if (row >= 0 && row < 4) cells.put(at, cellOf(Matter.wood, 0));
        if (row >= 4 && row < 8) cells.put(at, cellOf(Matter.lava, 0));
      }),
    })
    .$name('kindling');
  const paint = root.createCommandEncoder();
  cave.record(paint);
  paint.submit();
  kindling.dispatchWorkgroups(Math.ceil(WIDTH / 8), Math.ceil(HEIGHT / 8));
  const grids: number[][] = [];
  for (let s = 0; s < STEPS; s++) {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    rules.step(pass);
    // The embers' own step, which lands them on the grid: it reads a cell, then writes it
    if (flying) sparks.step(pass);
    pass.end();
    encoder.submit();
    if (s % 20 === 19) {
      copy.dispatchWorkgroups(Math.ceil(WIDTH / 8), Math.ceil(HEIGHT / 8));
      grids.push(await out.read());
    }
  }
  const totals = await counts.buffer.read();
  const embers = (await sparks.pool.read())
    .filter((p) => p.age < p.life)
    .map((p) => [p.position.x, p.position.y, p.velocity.x, p.velocity.y, p.life, p.seed])
    .toSorted((a, b) => {
      for (const [k, value] of a.entries()) if (value !== b[k]) return value - (b[k] as number);
      return 0;
    });
  return { grids, totals, embers };
}

it("steps falling-sand's own kernels on the GPU as WebGPU does, cell for cell", async () => {
  const actual = await stepped(gl);
  // Every kernel has a GPU form and ran as one; the device may still move one to the CPU where
  // the CPU reads its output back, which runs the WGSL's own semantics
  const stats = (gl.device as unknown as GLDevice).stats;
  expect(Object.fromEntries(stats.fallbacks)).toEqual({});
  const ran = [...stats.kernels.keys()];
  for (const label of ['stepping', 'ticking', 'kindling']) expect(ran).toContain(`${label} (gpu)`);
  // Unlabelled pipelines take their entry functions' names, numbered past the first
  for (const label of ['flow', 'flow 2', 'copy']) expect(ran).toContain(`${label} (gpu)`);
  expect(ran.some((label) => label.startsWith('<unnamed>'))).toBe(false);
  const expected = await stepped(gpu);
  for (const [k, grid] of expected.grids.entries()) {
    const other = actual.grids[k] as number[];
    expect(grid.filter((cell, i) => cell !== other[i]).length, `after ${(k + 1) * 20} steps`).toBe(
      0,
    );
  }
  expect(actual.totals).toEqual(expected.totals);
  expect(actual.embers).toEqual(expected.embers);
  // The cave moved
  expect(expected.grids[0]).not.toEqual(expected.grids.at(-1));
  // Lava cooled, wood caught and embers flew
  expect(expected.totals.every((total) => total > 0)).toBe(true);
  expect(expected.embers.length).toBeGreaterThan(0);
});

it("steps falling-sand's embers onto the grid on the GPU as WebGPU does, reading nothing back", async () => {
  const stats = (gl.device as unknown as GLDevice).stats;
  stats.reset();
  const actual = await stepped(gl, true);
  expect(Object.fromEntries(stats.fallbacks)).toEqual({});
  const ran = [...stats.kernels.keys()];
  for (const label of ['stepping', 'stepPipeline']) expect(ran).toContain(`${label} (gpu)`);
  // No kernel ran on the CPU, so none waited for the GPU to read the grid back
  expect(ran.filter((label) => !label.endsWith('(gpu)'))).toEqual([]);
  expect(stats.readback.textures).toBe(0);
  const expected = await stepped(gpu, true);
  for (const [k, grid] of expected.grids.entries()) {
    const other = actual.grids[k] as number[];
    expect(grid.filter((cell, i) => cell !== other[i]).length, `after ${(k + 1) * 20} steps`).toBe(
      0,
    );
  }
  expect(actual.totals).toEqual(expected.totals);
  expect(actual.embers).toEqual(expected.embers);
  expect(expected.totals.every((total) => total > 0)).toBe(true);
});
