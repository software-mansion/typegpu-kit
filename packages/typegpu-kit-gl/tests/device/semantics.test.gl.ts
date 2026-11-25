import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { compareRoots, hasWebGpu, pixel, roots } from './compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;

beforeAll(async () => {
  if (!hasWebGpu) return;
  ({ gpu, gl } = await roots());
});

afterAll(() => {
  gpu?.destroy();
  gl?.destroy();
});

const SIZE = 16;

const fullScreen = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  return { position: d.vec4f(corners[index]!, 0, 1) };
});

/** A triangle over the bottom-right corner of the target. */
const corner = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(0, -1), d.vec2f(1, -1), d.vec2f(1, 0)];
  return { position: d.vec4f(corners[index]!, 0, 1) };
});

const solid = (color: d.v4f) =>
  tgpu.fragmentFn({ out: d.vec4f })(() => {
    'use gpu';
    return color;
  });

/**
 * A pass fills the target red, a kernel the barrier keeps on the CPU stores green into its top
 * half, and a pass with the same attachments loads it and draws blue over a corner.
 */
function kernelBetweenPasses(root: TgpuRoot) {
  const target = root
    .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
    .$usage('render', 'storage');
  const storage = target.createView(d.textureStorage2d('rgba8unorm', 'write-only'));
  const store = tgpu.computeFn({ in: { id: d.builtin.globalInvocationId }, workgroupSize: [4, 4] })(
    ({ id }) => {
      'use gpu';
      std.workgroupBarrier();
      if (id.y < d.u32(SIZE / 2)) {
        std.textureStore(storage.$, d.vec2i(id.xy), d.vec4f(0, 1, 0, 1));
      }
    },
  );
  const view = target.createView();
  root
    .createRenderPipeline({
      vertex: fullScreen,
      fragment: solid(d.vec4f(1, 0, 0, 1)),
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view })
    .draw(3);
  root.createComputePipeline({ compute: store }).dispatchWorkgroups(SIZE / 4, SIZE / 4);
  root
    .createRenderPipeline({
      vertex: corner,
      fragment: solid(d.vec4f(0, 0, 1, 1)),
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view, loadOp: 'load' })
    .draw(3);
  return target;
}

describe.skipIf(!hasWebGpu)('render passes keep WebGPU semantics', () => {
  test('a pass that loads its target sees what a CPU kernel stored since the last pass', async () => {
    const result = await compareRoots(gpu, gl, kernelBetweenPasses);
    expect((gl.device as GLDevice).stats.fallbacks.get('store')).toMatch(/^CPU only/);
    expect(pixel(result.actual, 2, 2)).toEqual([0, 255, 0, 255]);
    expect(pixel(result.actual, 2, SIZE - 2)).toEqual([255, 0, 0, 255]);
    expect(pixel(result.actual, SIZE - 2, SIZE - 2)).toEqual([0, 0, 255, 255]);
    expect(result.first).toBeUndefined();
  });
});
