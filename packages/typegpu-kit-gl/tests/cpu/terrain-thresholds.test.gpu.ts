import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { heightAt as fireworks } from '../../../typegpu-kit-examples/src/fireworks/land.ts';
import { heightAt as stampede } from '../../../typegpu-kit-examples/src/stampede/land.ts';
import { cpuRoot } from './device.ts';
import { readBuffer } from './example-comparison.ts';
import { recordResults } from './results.ts';

async function terrain(root: TgpuRoot, example: 'fireworks' | 'stampede', points?: Float32Array) {
  const heightAt = example === 'fireworks' ? fireworks : stampede;
  const cols = example === 'fireworks' ? 288 : 320;
  const rows = example === 'fireworks' ? 168 : 200;
  const reach = example === 'fireworks' ? 3200 : 4600;
  const power = example === 'fireworks' ? 2.6 : 2.2;
  const delta = example === 'fireworks' ? 0.6 : 0.8;
  const ids = [0, 72, 2489, 23862, 40000, 48000];
  const vertices = root.createReadonly(d.arrayOf(d.u32, ids.length), ids);
  const count = points ? points.length / 12 : ids.length;
  const input = root.createReadonly(
    d.arrayOf(d.vec2f, count),
    Array.from({ length: count }, (_, i) =>
      points ? d.vec2f(points[i * 12]!, points[i * 12 + 1]!) : d.vec2f(),
    ),
  );
  const fromInput = !!points;
  const output = root.createMutable(d.arrayOf(d.vec4f, count * 3));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [1], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          let point = d.vec2f(input.$[id.x]);
          if (!fromInput) {
            const index = vertices.$[id.x];
            const u = d.f32(index % (cols + 1)) / cols;
            const v = d.f32(std.intdiv(index, d.u32(cols + 1))) / rows;
            const angle = std.select(u, d.f32(0), u >= 1) * Math.PI * 2;
            point = d.vec2f(std.cos(angle), std.sin(angle)) * (reach * std.pow(v, power));
          }
          const x0 = heightAt(point.x - delta, point.y);
          const x1 = heightAt(point.x + delta, point.y);
          const z0 = heightAt(point.x, point.y - delta);
          const z1 = heightAt(point.x, point.y + delta);
          const normal = std.normalize(d.vec3f(x0 - x1, delta * 2, z0 - z1));
          output.$[id.x * 3] = d.vec4f(
            point,
            heightAt(point.x, point.y),
            std.atan2(point.y, point.x),
          );
          output.$[id.x * 3 + 1] = d.vec4f(x0, x1, z0, z1);
          output.$[id.x * 3 + 2] = d.vec4f(normal, 0);
        },
      ),
    })
    .dispatchWorkgroups(count);
  return new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
}
for (const example of ['fireworks', 'stampede'] as const)
  it.runIf(process.env.CPU_THRESHOLDS === '1')(
    `traces ${example} height and finite-difference normals separately`,
    async () => {
      const gpu = await gpuRoot(),
        cpu = cpuRoot();
      try {
        const a = await terrain(cpu, example),
          b = await terrain(gpu, example);
        const inputs = Float32Array.from([...a, ...b]);
        const ac = await terrain(cpu, example, inputs),
          bc = await terrain(gpu, example, inputs);
        let maximumHeightError = 0,
          maximumHeightRelative = 0;
        const records = [];
        const unboundedAccuracy = [];
        for (let i = 0; i < ac.length / 12; i++) {
          for (let c = 2; c < 8; c++) {
            if (c === 3) continue;
            const x = inputs[i * 12]!,
              z = inputs[i * 12 + 1]!;
            const offset = Math.fround(example === 'fireworks' ? 0.6 : 0.8);
            const atX = c === 4 ? Math.fround(x - offset) : c === 5 ? Math.fround(x + offset) : x;
            const atZ = c === 6 ? Math.fround(z - offset) : c === 7 ? Math.fround(z + offset) : z;
            if (Math.abs(atX) < 2 ** -126 || Math.abs(atZ) < 2 ** -126) {
              unboundedAccuracy.push({
                row: i,
                component: c,
                pointBits: Array.from(new Uint32Array(Float32Array.of(atX, atZ).buffer)),
                cpu: ac[i * 12 + c],
                dawn: bc[i * 12 + c],
              });
              continue;
            }
            const delta = Math.abs(ac[i * 12 + c]! - bc[i * 12 + c]!);
            maximumHeightError = Math.max(maximumHeightError, delta);
            maximumHeightRelative = Math.max(
              maximumHeightRelative,
              delta / Math.max(1, Math.abs(ac[i * 12 + c]!), Math.abs(bc[i * 12 + c]!)),
            );
          }
          records.push({
            point: Array.from(inputs.subarray(i * 12, i * 12 + 2)),
            cpu: Array.from(ac.subarray(i * 12, i * 12 + 12)),
            dawn: Array.from(bc.subarray(i * 12, i * 12 + 12)),
          });
        }
        recordResults(`terrainThresholds:${example}`, {
          maximumHeightError,
          maximumHeightRelative,
          unboundedAccuracy,
          records,
          original: { cpu: Array.from(a), dawn: Array.from(b) },
        });
        expect(maximumHeightRelative).toBeLessThanOrEqual(1e-5);
      } finally {
        gpu.destroy();
        cpu.destroy();
      }
    },
    30000,
  );
