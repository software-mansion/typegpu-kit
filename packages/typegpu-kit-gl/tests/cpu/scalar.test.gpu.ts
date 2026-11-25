import { bitcasts } from './bitcasts.ts';
import { cpuRoot } from './device.ts';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu } from 'typegpu';
import { expect, it } from 'vitest';
import { CpuGenerator } from '../../src/cpu/generator.ts';
import { compile } from '../../src/cpu/index.ts';

it('matches dawn for integer hash arithmetic and all invocation coordinates', async () => {
  const root = await gpuRoot();
  try {
    const data = root.createMutable(d.arrayOf(d.u32, 512));
    const kernel = tgpu.computeFn({
      workgroupSize: [4, 2, 2],
      in: {
        gid: d.builtin.globalInvocationId,
        lid: d.builtin.localInvocationId,
        index: d.builtin.localInvocationIndex,
        group: d.builtin.workgroupId,
        count: d.builtin.numWorkgroups,
      },
    })(({ gid, lid, index, group, count }) => {
      'use gpu';
      const i = gid.x + gid.y * d.u32(16) + gid.z * d.u32(128);
      let h = i * d.u32(1664525) + d.u32(1013904223);
      h ^= h >>> d.u32(13);
      data.$[i] = h + lid.x + index + group.y + count.z;
    });
    const code = tgpu.resolve([kernel], { unstable_shaderGenerator: new CpuGenerator() });
    const module = compile(code);
    const bytes = new ArrayBuffer(512 * 4);
    module.dispatch(4, 4, 2, {
      buffer: () => ({ bytes, offset: 0, size: bytes.byteLength }),
      texture: () => {
        throw new Error('unexpected texture');
      },
      sampler: () => {
        throw new Error('unexpected sampler');
      },
    });
    root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(4, 4, 2);
    expect([...new Uint32Array(bytes)]).toEqual(await data.read());
  } finally {
    root.destroy();
  }
});

it('matches dawn for scalar and vector bitcasts of buffer loads and computed values', async () => {
  const gpu = await gpuRoot(),
    cpu = cpuRoot();
  try {
    expect(await bitcasts(cpu)).toEqual(await bitcasts(gpu));
  } finally {
    gpu.destroy();
    cpu.destroy();
  }
});
