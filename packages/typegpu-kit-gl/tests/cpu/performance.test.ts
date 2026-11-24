import { recordResults } from './results.ts';
import { recordingRoot } from '@typegpu/kit/testing';
import { d, tgpu } from 'typegpu';
import { it } from 'vitest';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile } from '../../src/cpu/index.ts';

it.runIf(process.env.CPU_BENCH === '1')('measures generated scalar hash throughput', () => {
  const { root } = recordingRoot();
  const count = 1_048_576;
  const data = root.createMutable(d.arrayOf(d.u32, count));
  const kernel = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
      'use gpu';
      let hash = data.$[id.x] + id.x;
      for (let i = 0; i < 8; i++) {
        hash = hash * d.u32(1664525) + d.u32(1013904223);
        hash ^= hash >>> d.u32(13);
      }
      data.$[id.x] = hash;
    },
  );
  const module = compile(tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() }));
  const bytes = new ArrayBuffer(count * 4);
  const resources = {
    buffer: () => ({ bytes, offset: 0, size: bytes.byteLength }),
    texture: (): never => {
      throw new Error('unexpected texture');
    },
    sampler: (): never => {
      throw new Error('unexpected sampler');
    },
  };
  module.dispatch(count / 64, 1, 1, resources);
  const start = performance.now();
  for (let i = 0; i < 5; i++) module.dispatch(count / 64, 1, 1, resources);
  const milliseconds = performance.now() - start;
  recordResults('scalarBenchmark', {
    kernel: 'scalar hash, eight multiply/add/xor/shift rounds per invocation',
    invocations: count * 5,
    milliseconds,
    invocationsPerSecond: ((count * 5) / milliseconds) * 1000,
  });
  root.destroy();
});
