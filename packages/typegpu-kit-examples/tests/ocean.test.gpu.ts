import { backends } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { CASCADES, foamingAt, MIP_LEVELS, spectrumAt } from '../src/ocean/spectrum.ts';
import { createWaves } from '../src/ocean/waves.ts';

const SAMPLES = 32;

/** Steps the waves `steps` times and samples each cascade's maps on a grid at every mip level. */
async function sampled(compute: 'gpu' | 'cpu', steps: number) {
  const root = await gpuRoot();
  // A device that says its compute runs on the CPU gets the Stockham stages, which a GPU runs too
  if (compute === 'cpu') {
    Object.assign(root.device, {
      [backends.brand]: { ...backends.of(root), compute } satisfies backends.Backend,
    });
  }
  const waves = createWaves(root);
  await waves.initAsync();
  waves.spectrum.write(spectrumAt(9, Math.PI / 2));
  waves.foaming.write(foamingAt(9));

  const encoder = root['~unstable'].createCommandEncoder();
  const pass = encoder.beginComputePass();
  waves.rebuild(pass);
  pass.end();
  encoder.submit();
  for (let i = 0; i < steps; i++) {
    waves.step.write({ time: 3 + i * 0.25, dt: 0.25 });
    const stepping = root['~unstable'].createCommandEncoder();
    const stepped = stepping.beginComputePass();
    waves.record(stepped);
    stepped.end();
    waves.recordMips(stepping);
    stepping.submit();
  }

  const count = SAMPLES * SAMPLES * CASCADES * MIP_LEVELS;
  const seen = root.createMutable(d.arrayOf(d.vec4f, count * 2));
  const probe = root.createComputePipeline({
    compute: tgpu.computeFn({
      workgroupSize: [SAMPLES],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      const lod = std.intdiv(gid.z, d.u32(CASCADES));
      const c = gid.z % CASCADES;
      // Points spread over several of each cascade's patches, off its texel centres
      const xz = (d.vec2f(gid.xy) * 37.3 + 1.7) * (d.f32(c) * 0.5 + 0.25);
      const at = (gid.z * SAMPLES + gid.y) * SAMPLES + gid.x;
      seen.$[at] = waves.displacementAt(xz, c, d.f32(lod));
      // The moments carry the foam, which each step carries over
      const footprint = d.vec2f(std.exp2(d.f32(lod)) * 0.01, 0);
      seen.$[at + count] = waves.moments(xz, c, footprint, footprint.yx);
    }),
  });
  probe.dispatchWorkgroups(1, SAMPLES, CASCADES * MIP_LEVELS);
  const values = await seen.read();
  waves.destroy();
  root.destroy();
  return values;
}

it('transforms the waves the same through Stockham stages where compute runs on the CPU', async () => {
  const [shared, staged] = await Promise.all([sampled('gpu', 3), sampled('cpu', 3)]);
  let largest = 0;
  let worst = 0;
  for (const [i, expected] of shared.entries()) {
    const actual = staged[i] as d.v4f;
    for (const k of [0, 1, 2, 3] as const) {
      largest = Math.max(largest, Math.abs(expected[k]));
      worst = Math.max(worst, Math.abs(expected[k] - actual[k]));
    }
  }
  expect(largest).toBeGreaterThan(0.1);
  // Both store half floats; the transforms differ only in the order they round
  expect(worst).toBeLessThan(largest * 2e-3);
});
