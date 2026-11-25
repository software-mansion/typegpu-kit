import { backends } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { BAND, createClay, formatsOf } from '../src/sph-sculpt/clay.ts';
import { SPRING } from '../src/sph-sculpt/island.ts';
import { CELL, createWater } from '../src/sph-sculpt/water.ts';

const SIDE = 16;
const STEPS = 6;

/** Pours one batch of water, steps it, rebuilds the surface and samples it around the spring. */
async function sampled(compute: 'gpu' | 'cpu', live?: () => number) {
  const root = await gpuRoot();
  // A device that says its compute runs on the CPU builds the surface a voxel per thread
  if (compute === 'cpu') {
    Object.assign(root.device, {
      [backends.brand]: { ...backends.of(root), compute } satisfies backends.Backend,
    });
  }
  const formats = formatsOf(root);
  const clay = createClay(root, formats);
  const water = createWater(root, clay, formats, live);
  water.spawn.write({ count: 3000, first: 0, total: 0 });
  const encoder = root['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  clay.generate(pass);
  for (let i = 0; i < STEPS; i++) water.step(pass);
  water.surface(pass);
  pass.end();
  encoder.submit();

  const count = SIDE * SIDE * SIDE;
  const seen = root.createMutable(d.arrayOf(d.vec2f, count));
  const spring = d.vec3f(...SPRING.at);
  root
    .createComputePipeline({
      compute: tgpu.computeFn({
        workgroupSize: [SIDE],
        in: { gid: d.builtin.globalInvocationId },
      })(({ gid }) => {
        'use gpu';
        // A box of two metres under the spout, off the voxel centres
        const p = spring + (d.vec3f(gid) / SIDE - d.vec3f(0.5, 0.9, 0.5)) * 2 + 0.013;
        seen.$[(gid.z * SIDE + gid.y) * SIDE + gid.x] = d.vec2f(
          water.fluidAt(p),
          water.fluidGapAt(p),
        );
      }),
    })
    .dispatchWorkgroups(1, SIDE, SIDE);
  const values = await seen.read();
  water.destroy();
  clay.destroy();
  root.destroy();
  return values;
}

it('builds the same fluid surface where compute runs on the CPU, a voxel per thread', async () => {
  const [listed, everywhere] = [await sampled('gpu'), await sampled('cpu')];
  let inside = 0;
  let worst = 0;
  for (const [i, expected] of listed.entries()) {
    const actual = everywhere[i] as d.v2f;
    if (expected.x < 0) inside++;
    worst = Math.max(worst, Math.abs(expected.x - actual.x), Math.abs(expected.y - actual.y));
  }
  // Some samples fall inside the water, and both surfaces agree within two steps of the fine
  // level's 8-bit storage; sorts that order a cell's particles differently sum them differently
  expect(inside).toBeGreaterThan(20);
  expect(worst).toBeLessThan((2 * 2 * BAND * CELL) / 255);
});

it('sorts only the slots below a bound on the live particles where compute runs on the CPU', async () => {
  const [whole, bounded] = [await sampled('cpu'), await sampled('cpu', () => 3000)];
  let worst = 0;
  for (const [i, expected] of whole.entries()) {
    const actual = bounded[i] as d.v2f;
    worst = Math.max(worst, Math.abs(expected.x - actual.x), Math.abs(expected.y - actual.y));
  }
  // Both sorts order by key and then slot, so the steps match exactly
  expect(worst).toBe(0);
});
