import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { encodeHalf, decodeHalf } from '../../src/cpu/textures.ts';
import { readBuffer } from './example-comparison.ts';
import { recordResults } from './results.ts';

async function sampledRamp(root: TgpuRoot) {
  const image = root.createTexture({ size: [2, 1], format: 'rgba16float' }).$usage('sampled');
  image.write(new Uint16Array([0, 1, 0.1160888671875, -2, 1, 0, 0.3330078125, 2].map(encodeHalf)));
  const view = image.createView(d.texture2d(d.f32));
  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const output = root.createMutable(d.arrayOf(d.vec4f, 1025));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const t = d.f32(id.x) / 1024;
          output.$[id.x] = std.textureSampleLevel(
            view.$,
            sampler.$,
            d.vec2f((0.5 + t) / 2, 0.5),
            0,
          );
        },
      ),
    })
    .dispatchWorkgroups(1025);
  return new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
}
it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'measures half-texture filter precision with exact binary coordinates',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const expected = await sampledRamp(gpu),
        actual = await sampledRamp(cpu);
      let differences = 0,
        weight8 = 0,
        weight8Half = 0,
        maximumBoundRatio = 0,
        halfAdjacent = 0,
        smallerMagnitude = 0;
      const first = [];
      const from = [0, 1, 0.1160888671875, -2],
        to = [1, 0, 0.3330078125, 2];
      for (let i = 0; i < expected.length; i++) {
        const t = Math.floor(i / 4) / 1024,
          c = i % 4;
        const exact = Math.fround(from[c]! * (1 - t) + to[c]! * t);
        expect(actual[i], `CPU sample ${i}`).toBe(exact);
        if (expected[i] === actual[i]) continue;
        differences++;
        if (Math.abs(actual[i]!) < Math.abs(expected[i]!)) smallerMagnitude++;
        const q = Math.round(t * 256) / 256;
        if (expected[i] === Math.fround(from[c]! * (1 - q) + to[c]! * q)) weight8++;
        const quantized = Math.fround(from[c]! * (1 - q) + to[c]! * q);
        if (expected[i] === decodeHalf(encodeHalf(quantized))) weight8Half++;
        const halfUlp = Math.abs(
          decodeHalf(encodeHalf(quantized) + 1) - decodeHalf(encodeHalf(quantized)),
        );
        const bound = Math.abs(to[c]! - from[c]!) / 512 + halfUlp;
        maximumBoundRatio = Math.max(maximumBoundRatio, Math.abs(expected[i]! - exact) / bound);
        const half = encodeHalf(exact),
          other = encodeHalf(expected[i]!);
        if (Math.abs(half - other) <= 1) halfAdjacent++;
        if (first.length < 20)
          first.push({
            sample: Math.floor(i / 4),
            channel: c,
            cpu: actual[i],
            dawn: expected[i],
            nearestHalf: decodeHalf(half),
            halfDistance: Math.abs(half - other),
          });
      }
      recordResults('filterPrecision', {
        differences,
        weight8,
        weight8Half,
        maximumBoundRatio,
        halfAdjacent,
        smallerMagnitude,
        first,
      });
      // Interpolation weights and intermediate filter precision are implementation-dependent.
      expect(expected.every(Number.isFinite)).toBe(true);
      expect(maximumBoundRatio).toBeLessThanOrEqual(1);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
  30000,
);

async function sampledUnormRamp(root: TgpuRoot) {
  const image = root
    .createTexture({ size: [2, 1, 1], dimension: '3d', format: 'rgba8unorm' })
    .$usage('sampled');
  image.write(new Uint8Array([0, 127, 1, 250, 255, 128, 254, 251]));
  const view = image.createView(d.texture3d(d.f32));
  const sampler = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const output = root.createMutable(d.arrayOf(d.vec4f, 1025));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const t = d.f32(id.x) / 1024;
          output.$[id.x] = std.textureSampleLevel(
            view.$,
            sampler.$,
            d.vec3f((0.5 + t) / 2, 0.5, 0.5),
            0,
          );
        },
      ),
    })
    .dispatchWorkgroups(1025);
  return new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
}
it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'measures fractional-byte precision of unorm texture filtering',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const expected = await sampledUnormRamp(gpu),
        actual = await sampledUnormRamp(cpu);
      const from = [0, 127, 1, 250],
        to = [255, 128, 254, 251];
      let quantized = 0,
        maxModelError = 0,
        maxFractionalByteDistance = 0;
      const first = [];
      for (let i = 0; i < expected.length; i++) {
        const t = Math.floor(i / 4) / 1024,
          c = i % 4,
          q = Math.round(t * 256) / 256;
        const model = Math.fround(Math.round((from[c]! * (1 - q) + to[c]! * q) * 16) / (255 * 16));
        const error = Math.abs(expected[i]! - model);
        maxModelError = Math.max(maxModelError, error);
        maxFractionalByteDistance = Math.max(
          maxFractionalByteDistance,
          Math.abs(expected[i]! * 4080 - Math.round(expected[i]! * 4080)),
        );
        if (error <= 2 ** -22) quantized++;
        if (first.length < 20 && Math.abs(expected[i]! - actual[i]!) > 1e-6)
          first.push({
            sample: Math.floor(i / 4),
            channel: c,
            cpu: actual[i],
            dawn: expected[i],
            model,
          });
      }
      recordResults('unormFilterPrecision', {
        components: expected.length,
        quantized,
        maxModelError,
        maxFractionalByteDistance,
        first,
      });
      expect(maxModelError).toBeLessThanOrEqual(1 / 4080 + 2 ** -22);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
);
