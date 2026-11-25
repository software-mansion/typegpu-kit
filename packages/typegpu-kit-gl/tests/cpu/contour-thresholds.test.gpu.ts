import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import {
  heightAt,
  levelOf,
  REACH,
} from '../../../typegpu-kit-examples/src/survey-model/terrain.ts';
import { cpuRoot } from './device.ts';
import { readBuffer } from './example-comparison.ts';
import { recordResults } from './results.ts';

const indices = [1094, 3384];
const steps = 58;
async function trace(root: TgpuRoot) {
  const input = root.createReadonly(d.arrayOf(d.u32, indices.length), indices);
  const output = root.createMutable(d.arrayOf(d.vec4f, indices.length * steps));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const index = input.$[id.x];
          const angle = (d.f32(index % 192) / 192) * Math.PI * 2;
          const direction = d.vec2f(std.cos(angle), std.sin(angle));
          const level = levelOf(std.intdiv(index, d.u32(192)));
          const stride = REACH / 48;
          let inner = d.f32(0);
          let outer = d.f32(0);
          for (let step = 0; step < 48; step++) {
            const r = REACH - d.f32(step) * stride;
            const point = direction * r;
            const height = heightAt(point);
            output.$[id.x * steps + d.u32(step)] = d.vec4f(point, height, level);
            if (height >= level) {
              inner = r;
              outer = r + stride;
              break;
            }
          }
          for (let k = 0; k < 10; k++) {
            const middle = (inner + outer) * 0.5;
            const point = direction * middle;
            const height = heightAt(point);
            output.$[id.x * steps + 48 + d.u32(k)] = d.vec4f(point, height, level);
            if (height >= level) inner = middle;
            else outer = middle;
          }
        },
      ),
    })
    .dispatchWorkgroups(indices.length);
  return new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
}
async function heights(root: TgpuRoot, values: Float32Array) {
  const input = root.createReadonly(
    d.arrayOf(d.vec2f, values.length / 4),
    Array.from({ length: values.length / 4 }, (_, row) =>
      d.vec2f(values[row * 4]!, values[row * 4 + 1]!),
    ),
  );
  const output = root.createMutable(d.arrayOf(d.f32, values.length / 4));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          output.$[id.x] = heightAt(input.$[id.x]);
        },
      ),
    })
    .dispatchWorkgroups(values.length / 4);
  return new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
}
it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'locates contour bisection decisions near the height threshold',
  async () => {
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const a = await trace(cpu),
        b = await trace(gpu);
      const inputs = Float32Array.from([...a, ...b]);
      const ah = await heights(cpu, inputs),
        bh = await heights(gpu, inputs);
      let maximumHeightError = 0;
      for (let row = 0; row < ah.length; row++)
        maximumHeightError = Math.max(maximumHeightError, Math.abs(ah[row]! - bh[row]!));
      const decisions = [];
      for (let index = 0; index < indices.length; index++) {
        for (let step = 0; step < steps; step++) {
          const offset = (index * steps + step) * 4;
          const cpuInside = a[offset + 2]! >= a[offset + 3]!;
          const dawnInside = b[offset + 2]! >= b[offset + 3]!;
          if (cpuInside === dawnInside) continue;
          decisions.push({
            index: indices[index],
            step,
            cpu: Array.from(a.subarray(offset, offset + 4)),
            dawn: Array.from(b.subarray(offset, offset + 4)),
            sameCpuPoint: [ah[offset / 4], bh[offset / 4]],
            sameDawnPoint: [ah[(a.length + offset) / 4], bh[(a.length + offset) / 4]],
          });
          break;
        }
      }
      recordResults('contourThresholds', { maximumHeightError, decisions });
      expect(maximumHeightError).toBeLessThanOrEqual(1e-5);
      for (const decision of decisions) {
        expect(Math.abs(decision.cpu[2]! - decision.cpu[3]!)).toBeLessThanOrEqual(1e-5);
        expect(Math.abs(decision.dawn[2]! - decision.dawn[3]!)).toBeLessThanOrEqual(1e-5);
      }
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
  30000,
);
