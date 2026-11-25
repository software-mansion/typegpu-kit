import { common, d, tgpu, type TgpuRoot } from 'typegpu';
import { afterAll, beforeAll, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { glRoot, hasWebGpu, readTexture } from './compare.ts';
import { EXTENSIONS, install } from './minimum.ts';

let undo: () => void;
beforeAll(() => {
  undo = install({ minimum: true, textureSize: 2048, hide: [...EXTENSIONS] });
});
afterAll(() => undo());

/** A kernel fills a buffer that a full-screen draw reads into each pixel's colour. */
async function scene(root: TgpuRoot) {
  const SIZE = 16;
  const buffer = root.createBuffer(d.arrayOf(d.f32, SIZE)).$usage('storage');
  const values = buffer.as('mutable');
  const fill = tgpu.computeFn({ workgroupSize: [SIZE], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
      'use gpu';
      values.$[gid.x] = d.f32(gid.x) / SIZE;
    },
  );
  root.createComputePipeline({ compute: fill }).dispatchWorkgroups(1);
  const read = buffer.as('readonly');
  const shade = tgpu.fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })(({ pos }) => {
    'use gpu';
    const v = read.$[d.u32(pos.x)]!;
    return d.vec4f(v, 1 - v, 0.5, 1);
  });
  const target = root
    .createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm' })
    .$usage('render', 'sampled');
  const view = target.createView('render');
  root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: shade,
      targets: { format: 'rgba8unorm' },
    })
    .withColorAttachment({ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] })
    .draw(3);
  return readTexture(root.device, root.unwrap(target));
}

test('the device reports WebGL 2 minimums and lacks every extension it asks for', () => {
  const root = glRoot();
  try {
    const { limits, features } = root.device;
    expect([limits.maxColorAttachments, limits.maxTextureDimension2D]).toEqual([4, 2048]);
    expect([limits.maxTextureDimension3D, limits.maxTextureArrayLayers]).toEqual([256, 256]);
    expect(features.has('float32-filterable')).toBe(false);
    expect(features.has('float32-blendable')).toBe(false);
  } finally {
    root.destroy();
  }
});

test.skipIf(!hasWebGpu)('a small scene draws at the minimums as on WebGPU', async () => {
  const gpu = await tgpu.init();
  const gl = glRoot();
  try {
    const expected = await scene(gpu);
    const actual = await scene(gl);
    expect([...actual.bytes]).toEqual([...expected.bytes]);
    expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
  } finally {
    gpu.destroy();
    gl.destroy();
  }
});

test('a half-float render target without its extension is an error that names it', async () => {
  const root = glRoot();
  try {
    root.device.pushErrorScope('validation');
    root.device.createTexture({
      size: [4, 4],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const error = await root.device.popErrorScope();
    expect(error?.message).toContain('EXT_color_buffer_half_float');
  } finally {
    root.destroy();
  }
});

test('a draw that blends one target and masks the other needs no indexed draw buffers', async () => {
  const root = glRoot();
  try {
    const shade = tgpu.fragmentFn({ out: { glow: d.vec4f, mark: d.vec4f } })(() => {
      'use gpu';
      return { glow: d.vec4f(0.5, 0.25, 0, 0), mark: d.vec4f(1) };
    });
    const textures = [0, 1].map(() =>
      root.createTexture({ size: [4, 4], format: 'rgba8unorm' }).$usage('render', 'sampled'),
    );
    const [glow, mark] = textures.map((texture) => texture.createView('render'));
    const add = { srcFactor: 'one', dstFactor: 'one', operation: 'add' } as const;
    root.device.pushErrorScope('validation');
    root
      .createRenderPipeline({
        vertex: common.fullScreenTriangle,
        fragment: shade,
        targets: {
          glow: { format: 'rgba8unorm', blend: { color: add, alpha: add } },
          mark: { format: 'rgba8unorm', writeMask: 0 },
        },
      })
      .withColorAttachment({
        glow: { view: glow!, loadOp: 'clear', storeOp: 'store', clearValue: [0.25, 0, 0, 1] },
        mark: { view: mark!, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0.5, 0, 1] },
      })
      .draw(3);
    expect(await root.device.popErrorScope()).toBeNull();
    const [glowed, marked] = await Promise.all(
      textures.map((texture) => readTexture(root.device, root.unwrap(texture))),
    );
    expect(Array.from(glowed!.bytes.subarray(0, 4))).toEqual([191, 64, 0, 255]);
    expect(Array.from(marked!.bytes.subarray(0, 4))).toEqual([0, 128, 0, 255]);
  } finally {
    root.destroy();
  }
});
