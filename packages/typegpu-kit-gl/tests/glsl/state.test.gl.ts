import { common, d, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { compareRoots, roots } from '../device/compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});

const ADD = {
  color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
} as const;

/** Adds a quarter, runs a kernel the device runs as a GPU form, then adds a quarter again. */
function addAroundKernel(root: TgpuRoot) {
  const color = root.createTexture({ size: [4, 4], format: 'rgba8unorm' }).$usage('render');
  const quarter = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: tgpu.fragmentFn({ out: d.vec4f })(() => {
      'use gpu';
      return d.vec4f(0.25);
    }),
    targets: { format: 'rgba8unorm', blend: ADD },
  });
  const out = root.createMutable(d.arrayOf(d.f32, 64));
  const kernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    out.$[gid.x] = d.f32(gid.x);
  });
  quarter.withColorAttachment({ view: color, clearValue: [0, 0, 0, 0] }).draw(3);
  root.createComputePipeline({ compute: kernel }).dispatchWorkgroups(1);
  quarter.withColorAttachment({ view: color, loadOp: 'load' }).draw(3);
  return color;
}

it('leaves blending to the render passes after a GPU form', async () => {
  const result = await compareRoots(gpu, gl, addAroundKernel);
  expect(result.first).toBeUndefined();
  const kernels = [...(gl.device as unknown as GLDevice).stats.kernels.keys()];
  expect(kernels.some((label) => label.endsWith('(gpu)'))).toBe(true);
});
