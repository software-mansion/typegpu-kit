import {
  perlin2d,
  randf,
} from '../../../typegpu-kit-examples/node_modules/@typegpu/noise/index.mjs';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { dispersion, LOOP_PERIOD, N } from '../../../typegpu-kit-examples/src/ocean/spectrum.ts';
import { heightAt } from '../../../typegpu-kit-examples/src/fireworks/land.ts';
import { cpuRoot } from './device.ts';
import { recordResults } from './results.ts';
import { seeded } from './example-comparison.ts';

async function thresholds(root: TgpuRoot) {
  const indices = root.createReadonly(d.arrayOf(d.u32, 4), [4805, 4923, 298949, 299067]);
  const patchSizes = root.createReadonly(d.arrayOf(d.f32, 4), [2048, 613, 101, 12]);
  const positions = root.createReadonly(d.arrayOf(d.vec2f, 8), [
    d.vec2f(0, 0),
    d.vec2f(-0, 0),
    d.vec2f(0, -0),
    d.vec2f(-0, -0),
    d.vec2f(0.6, 0),
    d.vec2f(-0.6, 0),
    d.vec2f(0, 0.6),
    d.vec2f(0, -0.6),
  ]);
  const waves = root.createMutable(d.arrayOf(d.vec4f, 4));
  const terrain = root.createMutable(d.arrayOf(d.vec4f, 8));
  const omega0 = (2 * Math.PI) / LOOP_PERIOD;
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [8], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          if (id.x < 4) {
            const index = indices.$[id.x];
            const cascade = index / (N * N);
            const dk = (2 * Math.PI) / patchSizes.$[cascade];
            const cell = d.vec2u(index % N, (index / N) % N);
            const k = (d.vec2f(cell) - N / 2) * dk;
            const length = std.max(std.length(k), 1e-6);
            const omega = dispersion(length).x;
            const ratio = omega / omega0;
            waves.$[id.x] = d.vec4f(length, omega, ratio, std.floor(ratio) * omega0);
          }
          const at = positions.$[id.x];
          terrain.$[id.x] = d.vec4f(at, std.atan2(at.y, at.x), heightAt(at.x, at.y));
        },
      ),
    })
    .dispatchWorkgroups(1);
  return {
    waves: (await waves.read()).map((v) => [...v]),
    terrain: (await terrain.read()).map((v) => [...v]),
  };
}

it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'localizes ocean floor thresholds and fireworks atan2 domain',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const expected = await thresholds(gpu),
        actual = await thresholds(cpu);
      recordResults('numericThresholds', { cpu: actual, dawn: expected });
      for (let row = 0; row < 4; row++) {
        for (let component = 0; component < 3; component++) {
          const a = actual.waves[row]![component]!,
            b = expected.waves[row]![component]!;
          expect(
            Math.abs(a - b),
            `wave ${row} component ${component}: ${a} / ${b}`,
          ).toBeLessThanOrEqual(1e-5 * Math.max(1, Math.abs(a), Math.abs(b)));
        }
        if (actual.waves[row]![3] !== expected.waves[row]![3]) {
          const a = actual.waves[row]![2]!,
            b = expected.waves[row]![2]!;
          expect(
            Math.abs(a - Math.round(a)),
            `wave ${row} is near a quantization threshold`,
          ).toBeLessThan(1e-4);
          expect(Math.abs(b - Math.round(b))).toBeLessThan(1e-4);
        }
      }
      // The origin's atan2 argument is outside WGSL's domain; retain both results as evidence.
      for (let row = 4; row < 8; row++)
        for (let component = 2; component < 4; component++)
          expect(actual.terrain[row]![component]).toBeCloseTo(
            expected.terrain[row]![component]!,
            4,
          );
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
  30000,
);

async function flowThresholds(root: TgpuRoot) {
  const output = root.createMutable(d.arrayOf(d.vec4f, 65536 * 4));
  const seed = root.createUniform(d.f32, seeded(1)());
  const viewport = root.createUniform(d.vec2f, d.vec2f(64, 36));
  const settings = root.createUniform(d.vec2f, d.vec2f(0.2, 1.5));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          randf.seed2(d.vec2f(d.f32(id.x), seed.$));
          const initial = d.vec2f(randf.sample(), randf.sample()) * 2 - 1;
          const aspect = viewport.$.x / viewport.$.y;
          const sample = initial * d.vec2f(aspect, 1) * settings.$.y;
          const angle = perlin2d.sample(sample) * (Math.PI * 2);
          const direction = d.vec2f(std.cos(angle), std.sin(angle));
          const position = initial + direction * (settings.$.x * (1 / 60));
          randf.seed3(d.vec3f(position, d.f32(id.x)));
          const decision = randf.sample();
          const spawn = d.vec2f(randf.sample(), randf.sample()) * 2 - 1;
          output.$[id.x * 4] = d.vec4f(initial, sample);
          output.$[id.x * 4 + 1] = d.vec4f(angle, direction, 0);
          output.$[id.x * 4 + 2] = d.vec4f(
            position,
            decision,
            std.select(d.f32(0), d.f32(1), decision < 1 / 240),
          );
          output.$[id.x * 4 + 3] = d.vec4f(spawn, 0, 0);
        },
      ),
    })
    .dispatchWorkgroups(1024);
  return Float32Array.from((await output.read()).flatMap((v) => [...v]));
}

it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'separates flow-field float motion from hashed respawn decisions',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const expected = await flowThresholds(gpu),
        actual = await flowThresholds(cpu);
      const differences: string[] = [],
        amplified: object[] = [];
      let shiftedSeeds = 0,
        changedDecisions = 0;
      for (let row = 0; row < 65536; row++) {
        const offset = row * 16;
        for (let component = 0; component < 10; component++) {
          const a = actual[offset + component]!,
            b = expected[offset + component]!;
          const tolerance = component < 4 ? 0 : 1e-5 * Math.max(1, Math.abs(a), Math.abs(b));
          if (Math.abs(a - b) > tolerance)
            differences.push(`row ${row} component ${component}: CPU=${a}, dawn=${b}`);
        }
        const same =
          actual[offset + 8] === expected[offset + 8] &&
          actual[offset + 9] === expected[offset + 9];
        if (!same) shiftedSeeds++;
        if (actual[offset + 11] !== expected[offset + 11]) changedDecisions++;
        if (same && actual[offset + 10] !== expected[offset + 10])
          differences.push(`row ${row}: identical float seed produced different randf sample`);
        if (!same && actual[offset + 11] !== expected[offset + 11] && amplified.length < 20)
          amplified.push({
            row,
            cpu: Array.from(actual.subarray(offset, offset + 16)),
            dawn: Array.from(expected.subarray(offset, offset + 16)),
          });
      }
      recordResults('flowFieldThresholds', {
        shiftedSeeds,
        changedDecisions,
        amplified,
        differences: differences.slice(0, 20),
        differenceCount: differences.length,
      });
      expect(differences.length, differences.slice(0, 20).join('\n')).toBe(0);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
  30000,
);

async function atmosphericHeights(root: TgpuRoot) {
  const pixels = root.createReadonly(d.arrayOf(d.vec2u, 3), [
    d.vec2u(10, 129),
    d.vec2u(38, 129),
    d.vec2u(62, 129),
  ]);
  const output = root.createMutable(d.arrayOf(d.vec4f, 3 * 16));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const uv = (d.vec2f(pixels.$[id.x]) + 0.5) / 256;
          const direction = std.normalize(d.vec3f(1, -(uv.y * 2 - 1), -(uv.x * 2 - 1)));
          const view = std.normalize(direction);
          const r = d.f32(6360.1);
          const disc = r * r * (view.y * view.y - 1) + 6360 * 6360;
          const ground = -r * view.y - std.sqrt(disc);
          const below = (r - 6460) * (r + 6460);
          const top = -r * view.y + std.sqrt(std.max(r * r * view.y * view.y - below, 0));
          const length = std.select(top, ground, view.y < 0 && disc >= 0);
          for (let i = d.u32(0); i < 16; i++) {
            const near = d.f32(i) / 16;
            const far = d.f32(i + 1) / 16;
            const t = length * (near + far) * (near + far) * 0.25;
            const position = d.vec3f(0, r, 0) + view * t;
            const radius = std.length(position);
            output.$[id.x * 16 + i] = d.vec4f(length, t, radius, radius - 6360);
          }
        },
      ),
    })
    .dispatchWorkgroups(3);
  return (await output.read()).map((v) => [...v]);
}
it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'measures atmosphere radius cancellation before half storage',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const expected = await atmosphericHeights(gpu),
        actual = await atmosphericHeights(cpu);
      recordResults('atmosphericHeights', { cpu: actual, dawn: expected });
      for (let row = 0; row < actual.length; row++) {
        const a = actual[row]![2]!,
          b = expected[row]![2]!;
        const bits = new Uint32Array(Float32Array.of(a, b).buffer);
        expect(Math.abs(bits[0]! - bits[1]!), `radius ${row}: ${a} / ${b}`).toBeLessThanOrEqual(4);
      }
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
  30000,
);
