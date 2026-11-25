import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { perlin2d } from '../../../typegpu-kit-examples/node_modules/@typegpu/noise/index.mjs';
import { Terrain, TERRAIN } from '../../../typegpu-kit-examples/src/sph-sculpt/island.ts';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { readBuffer } from './example-comparison.ts';
import { recordResults } from './results.ts';

async function coordinates(root: TgpuRoot) {
  const ids = root.createReadonly(d.arrayOf(d.u32, 8), [36, 60, 86, 310, 3960, 4218, 4220, 4250]);
  const terrain = root.createUniform(Terrain, { ...TERRAIN, prims: 0 });
  const output = root.createMutable(d.arrayOf(d.vec4f, 32));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [8], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const index = ids.$[id.x];
          if (id.x < 4) {
            const cell = d.vec2u(index % 201, index / 201);
            const at = (d.vec2f(cell) / 200 - 0.5) * d.vec2f(1, -1) * 80 * 0.11;
            let height = d.f32(0);
            let amplitude = d.f32(1);
            let scale = d.f32(1);
            for (let octave = d.u32(0); octave < 4; octave++) {
              const point = at * scale;
              const noise = perlin2d.sample(point);
              height += noise * amplitude;
              output.$[id.x * 4 + octave] = d.vec4f(point, noise, height * 1.8);
              amplitude *= 0.45;
              scale *= 2;
            }
          } else {
            const cell = d.vec2u(index % 256, index / 256);
            const xz = (d.vec2f(cell) + 0.5) * 0.1;
            let at = xz * terrain.$.freq + terrain.$.seed;
            let height = terrain.$.base;
            let amp = terrain.$.amp;
            for (let octave = d.u32(0); octave < 3; octave++) {
              const noise = perlin2d.sample(at);
              height += noise * amp;
              output.$[id.x * 4 + octave] = d.vec4f(at, noise, height);
              amp *= 0.45;
              at = std.fma(at, d.vec2f(2.03), d.vec2f(17.1));
            }
            const rim = std.length(xz - d.vec2f(25.6) * 0.5) / terrain.$.island;
            output.$[id.x * 4 + 3] = d.vec4f(
              at,
              rim,
              std.mix(height, -4, std.smoothstep(0.9, 1.05, rim)),
            );
          }
        },
      ),
    })
    .dispatchWorkgroups(1);
  return new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
}

async function stages(root: TgpuRoot, coords: Float32Array) {
  const points = Array.from({ length: coords.length / 4 }, (_, i) =>
    d.vec2f(coords[i * 4]!, coords[i * 4 + 1]!),
  );
  const input = root.createReadonly(d.arrayOf(d.vec2f, points.length), points);
  const output = root.createMutable(d.arrayOf(d.vec4f, points.length * 6));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const pos = d.vec2f(input.$[id.x]);
          const i = d.vec2i(std.floor(pos));
          const f = std.fract(pos);
          const u = f * f * f * (f * (f * 6 - 15) + 10);
          const ga = perlin2d.computeJunctionGradient(i);
          const gb = perlin2d.computeJunctionGradient(i + d.vec2i(1, 0));
          const gc = perlin2d.computeJunctionGradient(i + d.vec2i(0, 1));
          const gd = perlin2d.computeJunctionGradient(i + d.vec2i(1, 1));
          const va = std.dot(ga, f - d.vec2f(0, 0));
          const vb = std.dot(gb, f - d.vec2f(1, 0));
          const vc = std.dot(gc, f - d.vec2f(0, 1));
          const vd = std.dot(gd, f - d.vec2f(1, 1));
          const noise = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
          output.$[id.x * 6] = d.vec4f(pos, f);
          output.$[id.x * 6 + 1] = d.vec4f(u, d.vec2f(i));
          output.$[id.x * 6 + 2] = d.vec4f(ga, gb);
          output.$[id.x * 6 + 3] = d.vec4f(gc, gd);
          output.$[id.x * 6 + 4] = d.vec4f(va, vb, vc, vd);
          output.$[id.x * 6 + 5] = d.vec4f(noise, perlin2d.sample(pos), 0, 0);
        },
      ),
    })
    .dispatchWorkgroups(points.length);
  return new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
}

it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'locates terrain noise differences before octave accumulation',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const a = await coordinates(cpu),
        b = await coordinates(gpu);
      const coords = Float32Array.from([...a, ...b]);
      const ac = await stages(cpu, coords),
        bc = await stages(gpu, coords);
      const names = [
        'coordinates/fraction',
        'quintic/cell',
        'gradients a/b',
        'gradients c/d',
        'corner dots',
        'noise',
      ];
      const maximum = names.map((stage) => ({ stage, maxAbsolute: 0, first: -1 }));
      for (let i = 0; i < ac.length; i++) {
        const stage = Math.floor((i % 24) / 4);
        const delta = Math.abs(ac[i]! - bc[i]!);
        maximum[stage]!.maxAbsolute = Math.max(maximum[stage]!.maxAbsolute, delta);
        if (delta && maximum[stage]!.first < 0) maximum[stage]!.first = i;
      }
      const changed = Array.from({ length: 32 }, (_, row) => ({
        row,
        cpu: Array.from(a.subarray(row * 4, row * 4 + 4)),
        dawn: Array.from(b.subarray(row * 4, row * 4 + 4)),
      }));
      recordResults('noiseStages', {
        maximum,
        changed,
        first: maximum.map((item) => ({
          stage: item.stage,
          cpu: ac[item.first],
          dawn: bc[item.first],
        })),
      });
      expect(maximum[0]!.maxAbsolute).toBe(0);
      for (const item of maximum.slice(1))
        expect(item.maxAbsolute, item.stage).toBeLessThanOrEqual(1e-5);
      for (let row = 0; row < coords.length / 4; row++) {
        expect(ac[row * 24 + 20]).toBeCloseTo(ac[row * 24 + 21]!, 6);
        expect(bc[row * 24 + 20]).toBeCloseTo(bc[row * 24 + 21]!, 6);
      }
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
  30000,
);
