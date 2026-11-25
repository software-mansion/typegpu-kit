import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
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

/**
 * Draws a triangle with back faces culled, as a scene's pipelines do, then takes eight values of a
 * counter in a kernel the device runs as a GPU form: the takes rank their invocations with
 * full-screen draws of their own, which culling left on would drop.
 */
async function takeAfterCulling(root: TgpuRoot) {
  const color = root.createTexture({ size: [4, 4], format: 'rgba8unorm' }).$usage('render');
  root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: tgpu.fragmentFn({ out: d.vec4f })(() => {
        'use gpu';
        return d.vec4f(1);
      }),
      targets: { format: 'rgba8unorm' },
      primitive: { cullMode: 'back', frontFace: 'ccw' },
    })
    .withColorAttachment({ view: color.createView('render'), loadOp: 'clear', storeOp: 'store' })
    .draw(3);
  const counter = root.createMutable(d.atomic(d.u32), 0);
  const out = root.createMutable(d.arrayOf(d.u32, 8));
  const kernel = tgpu.computeFn({ workgroupSize: [8], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      out.$[gid.x] = std.atomicAdd(counter.$, 1);
    },
  );
  root.createComputePipeline({ compute: kernel }).$name('taking').dispatchWorkgroups(1);
  return { out: await out.read(), counter: await counter.read() };
}

it('ranks takes after a draw that culls back faces', async () => {
  const actual = await takeAfterCulling(gl);
  const { stats, gl: context } = gl.device as unknown as GLDevice;
  expect(stats.fallbacks.get('taking')).toBeUndefined();
  expect([...stats.kernels.keys()]).toContain('taking (gpu)');
  expect(context.getError()).toBe(0);
  // The device takes in invocation order; WebGPU in any order, so compare the values taken
  expect(actual).toEqual({ out: [0, 1, 2, 3, 4, 5, 6, 7], counter: 8 });
  const expected = await takeAfterCulling(gpu);
  expect([...expected.out].toSorted((a, b) => a - b)).toEqual(actual.out);
  expect(expected.counter).toBe(actual.counter);
});
