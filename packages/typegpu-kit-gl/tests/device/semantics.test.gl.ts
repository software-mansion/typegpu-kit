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

/** A triangle over the left half of the target at depth 0.25. */
const left = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(0, -1), d.vec2f(-1, 3)];
  return { position: d.vec4f(corners[index]!, 0.25, 1) };
});

/** A full-screen triangle at depth 0.5. */
const middle = tgpu.vertexFn({
  in: { index: d.builtin.vertexIndex },
  out: { position: d.builtin.position },
})(({ index }) => {
  'use gpu';
  const corners = [d.vec2f(-1, -1), d.vec2f(3, -1), d.vec2f(-1, 3)];
  return { position: d.vec4f(corners[index]!, 0.5, 1) };
});

/**
 * A pass writes depth and stencil 1 over the left half, discarding one aspect and storing the
 * other, and a second pass draws green where the stored aspect lets it.
 */
function oneAspectStored(keep: 'depth' | 'stencil') {
  return (root: TgpuRoot) => {
    const color = root
      .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
      .$usage('render', 'sampled');
    const depth = root
      .createTexture({ size: [SIZE, SIZE], format: 'depth24plus-stencil8' })
      .$usage('render');
    root
      .createRenderPipeline({
        vertex: left,
        fragment: solid(d.vec4f(1, 0, 0, 1)),
        targets: { format: 'rgba8unorm' },
        depthStencil: {
          format: 'depth24plus-stencil8',
          depthCompare: 'always',
          depthWriteEnabled: true,
          stencilFront: { compare: 'always', passOp: 'replace' },
          stencilBack: { compare: 'always', passOp: 'replace' },
        },
      })
      .withColorAttachment({ view: color, clearValue: [0, 0, 0, 1] })
      .withDepthStencilAttachment({
        view: depth,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: keep === 'depth' ? 'store' : 'discard',
        stencilClearValue: 0,
        stencilLoadOp: 'clear',
        stencilStoreOp: keep === 'stencil' ? 'store' : 'discard',
      })
      .withStencilReference(1)
      .draw(3);
    root
      .createRenderPipeline({
        vertex: middle,
        fragment: solid(d.vec4f(0, 1, 0, 1)),
        targets: { format: 'rgba8unorm' },
        depthStencil: {
          format: 'depth24plus-stencil8',
          depthCompare: keep === 'depth' ? 'less' : 'always',
          depthWriteEnabled: false,
          stencilFront: { compare: keep === 'stencil' ? 'equal' : 'always' },
          stencilBack: { compare: keep === 'stencil' ? 'equal' : 'always' },
        },
      })
      .withColorAttachment({ view: color, loadOp: 'load' })
      .withDepthStencilAttachment({
        view: depth,
        depthClearValue: 1,
        depthLoadOp: keep === 'depth' ? 'load' : 'clear',
        depthStoreOp: 'store',
        stencilClearValue: 0,
        stencilLoadOp: keep === 'stencil' ? 'load' : 'clear',
        stencilStoreOp: 'store',
      })
      .withStencilReference(1)
      .draw(3);
    return color;
  };
}

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
  root
    .createRenderPipeline({
      vertex: fullScreen,
      fragment: solid(d.vec4f(1, 0, 0, 1)),
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view: target })
    .draw(3);
  root.createComputePipeline({ compute: store }).dispatchWorkgroups(SIZE / 4, SIZE / 4);
  root
    .createRenderPipeline({
      vertex: corner,
      fragment: solid(d.vec4f(0, 0, 1, 1)),
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view: target, loadOp: 'load' })
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
  test.each(['depth', 'stencil'] as const)(
    'a pass that discards one aspect of a depth-stencil target keeps the %s it stored',
    async (keep) => {
      const result = await compareRoots(gpu, gl, oneAspectStored(keep));
      expect(pixel(result.expected, 2, 2)).toEqual(
        keep === 'depth' ? [255, 0, 0, 255] : [0, 255, 0, 255],
      );
      expect(pixel(result.expected, SIZE - 2, 2)).toEqual(
        keep === 'depth' ? [0, 255, 0, 255] : [0, 0, 0, 255],
      );
      expect(result.first).toBeUndefined();
    },
  );
});
