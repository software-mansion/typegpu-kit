import { existsSync, readFileSync } from 'node:fs';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { BOX } from '../../../typegpu-kit-examples/src/jelly-world/slider.ts';
import { sdBezier } from '../../../typegpu-kit-examples/node_modules/@typegpu/sdf/index.js';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { cpuRoot } from './device.ts';
import { recordResults } from './results.ts';
import { readBuffer } from './example-comparison.ts';

const texels = [11580, 11787, 19031, 19052, 22801, 23060, 23312, 32013];
async function curves(root: TgpuRoot, bindings: { key: string; words?: number[] }[]) {
  const load = (name: string) => {
    const words = bindings.find((binding) => binding.key.startsWith(`buffer:${name}:`))!.words!;
    const floats = new Float32Array(Uint32Array.from(words).buffer);
    return Array.from({ length: floats.length / 2 }, (_, i) =>
      d.vec2f(floats[i * 2]!, floats[i * 2 + 1]!),
    );
  };
  const points = root.createReadonly(d.arrayOf(d.vec2f, 17), load('points'));
  const controls = root.createReadonly(d.arrayOf(d.vec2f, 16), load('controls'));
  const offsets = root.createReadonly(d.arrayOf(d.vec2f, 5), [
    d.vec2f(),
    d.vec2f(0.03, 0),
    d.vec2f(-0.03, 0),
    d.vec2f(0, 0.03),
    d.vec2f(0, -0.03),
  ]);
  const indices = root.createReadonly(d.arrayOf(d.u32, texels.length), texels);
  const output = root.createMutable(d.arrayOf(d.vec4f, texels.length * 5 * 16 * 3));
  root
    .createComputePipeline({
      compute: tgpu.computeFn({ workgroupSize: [16], in: { id: d.builtin.globalInvocationId } })(
        ({ id }) => {
          'use gpu';
          const segment = id.x % 16;
          const sample = id.x / 16;
          const index = indices.$[sample / 5];
          const uv = (d.vec2f(d.vec2u(index % 256, index / 256)) + 0.5) / d.vec2f(256, 128);
          const point =
            d.vec2f(
              BOX.left + uv.x * (BOX.right - BOX.left),
              BOX.top - uv.y * (BOX.top - BOX.bottom),
            ) + offsets.$[sample % 5];
          const A = d.vec2f(points.$[segment]);
          const B = d.vec2f(controls.$[segment]);
          const C = d.vec2f(points.$[segment + 1]);
          const a = B - A;
          const b = A - B * 2 + C;
          const delta = A - point;
          const kk = 1 / std.max(std.dot(b, b), 1e-4);
          const kx = kk * std.dot(a, b);
          const ky = (kk * (d.f32(2) * std.dot(a, a) + std.dot(delta, b))) / 3;
          const kz = kk * std.dot(delta, a);
          const p = ky - kx * kx;
          const q = kx * (2 * kx * kx - 3 * ky) + kz;
          const p3 = p * p * p;
          const h = q * q + 4 * p3;
          let roots = d.vec4f();
          if (h >= 0) {
            const x = (d.vec2f(std.sqrt(h), -std.sqrt(h)) - q) * 0.5;
            const uvRoots = std.sign(x) * std.pow(std.abs(x), d.vec2f(1 / 3));
            roots = d.vec4f(x, uvRoots.x + uvRoots.y, std.saturate(uvRoots.x + uvRoots.y - kx));
          } else {
            const z = std.sqrt(-p);
            const v = std.acos(q / (p * z * 2)) / 3;
            roots = d.vec4f(z, v, std.cos(v), std.sin(v));
          }
          output.$[id.x * 3] = d.vec4f(p, q, h, kx);
          output.$[id.x * 3 + 1] = d.vec4f(roots);
          output.$[id.x * 3 + 2] = d.vec4f(point, sdBezier(point, A, B, C), 0);
        },
      ),
    })
    .dispatchWorkgroups(texels.length * 5);
  const values = new Float32Array(await readBuffer(root.device, root.unwrap(output.buffer)));
  return Array.from({ length: values.length / 4 }, (_, row) =>
    Array.from(values.subarray(row * 4, row * 4 + 4)),
  );
}
it.runIf(!!process.env.CPU_CURVE_INPUT && existsSync(process.env.CPU_CURVE_INPUT))(
  'records cubic distance decisions from identical jelly-world inputs',
  async () => {
    const bindings = JSON.parse(readFileSync(process.env.CPU_CURVE_INPUT!, 'utf8')) as {
      key: string;
      words?: number[];
    }[];
    const gpu = await gpuRoot(),
      cpu = cpuRoot();
    try {
      const expected = await curves(gpu, bindings),
        actual = await curves(cpu, bindings);
      const changed = [];
      for (let sample = 0; sample < texels.length * 5; sample++) {
        const nearest = (values: number[][]) =>
          Array.from({ length: 16 }, (_, segment) => ({
            segment,
            distance: values[(sample * 16 + segment) * 3 + 2]![2]!,
          })).toSorted((a, b) => a.distance - b.distance)[0]!;
        const a = nearest(actual),
          b = nearest(expected);
        if (a.segment !== b.segment || Math.abs(a.distance - b.distance) > 1e-6) {
          const segments = [...new Set([a.segment, b.segment])];
          changed.push({
            texel: texels[Math.floor(sample / 5)],
            offset: sample % 5,
            cpu: a,
            dawn: b,
            trace: segments.map((segment) => ({
              segment,
              cpu: actual.slice((sample * 16 + segment) * 3, (sample * 16 + segment) * 3 + 3),
              dawn: expected.slice((sample * 16 + segment) * 3, (sample * 16 + segment) * 3 + 3),
            })),
          });
        }
      }
      recordResults('curveThresholds', { changed });
      expect(actual.length).toBe(expected.length);
    } finally {
      gpu.destroy();
      cpu.destroy();
    }
  },
  30000,
);
