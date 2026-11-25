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
