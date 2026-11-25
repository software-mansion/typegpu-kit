import { d, tgpu, type TgpuRoot } from 'typegpu';
import { expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot, hasWebGpu } from './compare.ts';

/** Three dispatches of one kernel that writes cells it computes, each with a bind group of its own. */
async function scatter(root: TgpuRoot) {
  const out = root.createMutable(d.arrayOf(d.u32, 40));
  const at = tgpu.bindGroupLayout({ layer: { uniform: d.u32 } });
  const groups = [0, 1, 2].map((k) =>
    root.createBindGroup(at, { layer: root.createUniform(d.u32, k).buffer }),
  );
  const kernel = tgpu.computeFn({
    workgroupSize: [8],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    const k = at.$.layer;
    let rank = d.u32(0);
    for (let u = d.u32(0); u < i; u++) rank += 1;
    const into = k * 10 + (7 - rank);
    out.$[into] = k * 100 + i;
  });
  const pipeline = root.createComputePipeline({ compute: kernel });
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const group of groups) {
    pass.setBindGroup(at, group);
    pass.setPipeline(pipeline);
    pass.dispatchWorkgroups(1);
  }
  pass.end();
  encoder.submit();
  return out.read();
}

test.skipIf(!hasWebGpu)(
  'every dispatch of a kernel that writes computed cells lands, as on WebGPU',
  { timeout: 30_000 },
  async () => {
    const gpu = await tgpu.init();
    const gl = glRoot();
    try {
      const expected = await scatter(gpu);
      expect(await scatter(gl)).toEqual(expected);
      expect([...(gl.device as GLDevice).stats.kernels.keys()]).toEqual(['pipeline (gpu)']);
    } finally {
      gpu.destroy();
      gl.destroy();
    }
  },
);
