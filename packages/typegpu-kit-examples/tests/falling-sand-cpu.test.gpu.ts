import { backends, cameras, counters } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d } from 'typegpu';
import { expect, it } from 'vitest';
import { createCave } from '../src/falling-sand/cave.ts';
import { createCells, gridKernel, HEIGHT, WIDTH } from '../src/falling-sand/cells.ts';
import { createEmbers } from '../src/falling-sand/embers.ts';
import { createRules } from '../src/falling-sand/rules.ts';

const STEPS = 120;
const STEP = 1 / 120;

/** Paints the cave, steps its rules `STEPS` times, and reads back cells, counts and embers. */
async function stepped(compute: 'gpu' | 'cpu') {
  const root = await gpuRoot();
  // A device that says its compute runs on the CPU runs a cell's block a thread, and counts aside
  if (compute === 'cpu') {
    Object.assign(root.device, {
      [backends.brand]: { ...backends.of(root), compute } satisfies backends.Backend,
    });
  }
  const counts = counters.create(root, ['cooled', 'burnt']);
  const cells = createCells(root);
  const cave = createCave(root, cells);
  const view = cameras.flat(root, { size: { width: WIDTH, height: HEIGHT }, height: HEIGHT });
  const { sparks, spark, sparked } = createEmbers(root, {
    camera: view,
    step: STEP,
    cells,
    counts,
  });
  const rules = createRules(root, { cells, counts, spark, sparked });

  const out = root.createMutable(d.arrayOf(d.u32, WIDTH * HEIGHT));
  const copy = root.createComputePipeline({
    compute: gridKernel(({ gid }) => {
      'use gpu';
      if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
      out.$[gid.y * WIDTH + gid.x] = cells.cellAt(d.vec2i(gid.xy));
    }),
  });

  const paint = root.createCommandEncoder();
  cave.record(paint);
  paint.submit();
  const grids: number[][] = [];
  for (let s = 0; s < STEPS; s++) {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    rules.step(pass);
    pass.end();
    encoder.submit();
    if (s % 20 === 19) {
      copy.dispatchWorkgroups(Math.ceil(WIDTH / 8), Math.ceil(HEIGHT / 8));
      grids.push(await out.read());
    }
  }
  const totals = await counts.buffer.read();
  const embers = (await sparks.pool.read()).filter((p) => p.age < p.life).length;
  root.destroy();
  return { grids, totals, embers };
}

it('steps the cave alike where compute runs on the CPU, a block a cell, counting and spitting aside', async () => {
  const [shared, alone] = [await stepped('gpu'), await stepped('cpu')];
  for (const [k, grid] of shared.grids.entries()) {
    const other = alone.grids[k]!;
    const differing = grid.filter((cell, i) => cell !== other[i]).length;
    expect(differing, `after ${(k + 1) * 20} steps`).toBe(0);
  }
  expect(alone.totals).toEqual(shared.totals);
  expect(alone.embers).toBe(shared.embers);
  // The cave has moved and something has counted or spat
  expect(shared.grids[0]).not.toEqual(shared.grids.at(-1));
  expect(shared.totals.some((total) => total > 0) || shared.embers > 0).toBe(true);
});
