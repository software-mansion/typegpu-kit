import { afterEach, describe, expect, test } from 'vitest';
import type { CpuModule } from '../../src/cpu/types.ts';
import { BACKEND, type GLDevice, shaderModuleFrom } from '../../src/device/index.ts';
import { glDevice, pixel, readTexture } from './compare.ts';
import { bufferBinding, bufferReader, module, textureBinding } from './glsl.ts';

let device: GLDevice;
afterEach(() => device?.destroy());

const FULLSCREEN = `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, 0.0, 1.0);`;

function pixels(canvas: HTMLCanvasElement) {
  const copy = new OffscreenCanvas(canvas.width, canvas.height).getContext('2d', {
    willReadFrequently: true,
  })!;
  copy.drawImage(canvas, 0, 0);
  return Array.from(copy.getImageData(0, 0, 1, 1).data);
}

function target(size = 4) {
  return device.createTexture({
    size: [size, size],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
}

/** A kernel that doubles each float of the buffer at @group(0) @binding(0). */
const doubler: CpuModule = {
  kind: 'compute',
  workgroupSize: [4, 1, 1],
  bindings: [{ group: 0, binding: 0, kind: 'buffer', reads: true, writes: true }],
  dispatch(x, _y, _z, resources) {
    const { bytes, offset, size } = resources.buffer(0, 0);
    const floats = new Float32Array(bytes, offset, size / 4);
    for (let i = 0; i < Math.min(x * 4, floats.length); i++) floats[i]! *= 2;
  },
};

describe('compute and passes', () => {
  test('runs a CPU kernel whose output a later draw reads', async () => {
    device = glDevice();
    const values = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(values, 0, new Float32Array([0.25, 0.125, 0, 0.5]));
    const kernel = await device.createComputePipelineAsync({
      layout: 'auto',
      label: 'doubler',
      compute: { module: shaderModuleFrom(device, doubler) },
    });
    const shader = module(device, {
      vertex: '',
      vertexMain: FULLSCREEN,
      fragment: `${bufferReader('values')}\nlayout(location = 0) out vec4 color;`,
      fragmentMain:
        '  color = vec4(values_float(0u), values_float(1u), values_float(2u), values_float(3u));',
      buffers: [bufferBinding('values', 0, 0)],
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const texture = target();
    const encoder = device.createCommandEncoder();
    // The draw before the dispatch reads the values as written
    const before = encoder.beginRenderPass({
      colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    before.setPipeline(pipeline);
    before.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: values } }],
      }),
    );
    before.draw(3);
    before.end();
    const compute = encoder.beginComputePass();
    compute.setPipeline(kernel);
    compute.setBindGroup(
      0,
      device.createBindGroup({
        layout: kernel.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: values } }],
      }),
    );
    compute.dispatchWorkgroups(1);
    compute.end();
    const copy = device.createTexture({
      size: [4, 4],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });
    encoder.copyTextureToTexture({ texture }, { texture: copy }, [4, 4]);
    const after = encoder.beginRenderPass({
      colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    after.setPipeline(pipeline);
    after.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: values } }],
      }),
    );
    after.draw(3);
    after.end();
    device.queue.submit([encoder.finish()]);
    expect(pixel(await readTexture(device, copy), 0, 0)).toEqual([64, 32, 0, 128]);
    expect(device.stats.kernels.get('doubler')?.dispatches).toBe(1);
    expect(pixel(await readTexture(device, texture), 0, 0)).toEqual([128, 64, 0, 255]);

    const staging = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const read = device.createCommandEncoder();
    read.copyBufferToBuffer(values, 0, staging, 0, 16);
    device.queue.submit([read.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    expect(Array.from(new Float32Array(staging.getMappedRange()))).toEqual([0.5, 0.25, 0, 1]);
    staging.unmap();
  });

  test('gives kernels the texels a draw rendered, and draws what a kernel stored', async () => {
    device = glDevice();
    const texture = device.createTexture({
      size: [2, 2],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    const inverter: CpuModule = {
      kind: 'compute',
      workgroupSize: [1, 1, 1],
      bindings: [{ group: 0, binding: 0, kind: 'storageTexture', reads: true, writes: true }],
      dispatch(_x, _y, _z, resources) {
        const level = resources.texture(0, 0).levels[0]!;
        const data = level.data as Uint8Array;
        for (let i = 0; i < data.length; i++) if (i % 4 !== 3) data[i] = 255 - data[i]!;
      },
    };
    const kernel = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModuleFrom(device, inverter) },
    });
    const encoder = device.createCommandEncoder();
    encoder
      .beginRenderPass({
        colorAttachments: [
          {
            view: texture.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: [1, 0.2, 0, 1],
          },
        ],
      })
      .end();
    const pass = encoder.beginComputePass();
    pass.setPipeline(kernel);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: kernel.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: texture.createView() }],
      }),
    );
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const shader = module(device, {
      vertex: '',
      vertexMain: FULLSCREEN,
      fragment: 'uniform sampler2D stored;\nlayout(location = 0) out vec4 color;',
      fragmentMain: '  color = texelFetch(stored, ivec2(1, 1), 0);',
      textures: [textureBinding('stored', 0, 0)],
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const out = target();
    const draw = device.createCommandEncoder();
    const render = draw.beginRenderPass({
      colorAttachments: [{ view: out.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    render.setPipeline(pipeline);
    render.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: texture.createView() }],
      }),
    );
    render.draw(3);
    render.end();
    device.queue.submit([draw.finish()]);
    expect(pixel(await readTexture(device, out), 0, 0)).toEqual([0, 204, 255, 255]);
  });

  test('replays render bundles', async () => {
    device = glDevice();
    const shader = module(device, {
      vertex: '',
      vertexMain: FULLSCREEN,
      fragment: 'layout(location = 0) out vec4 color;',
      fragmentMain: '  color = vec4(0.0, 0.0, 1.0, 1.0);',
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const bundle = device.createRenderBundleEncoder({ colorFormats: ['rgba8unorm'] });
    bundle.setPipeline(pipeline);
    bundle.draw(3);
    const finished = bundle.finish();
    const texture = target();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    pass.executeBundles([finished]);
    pass.end();
    device.queue.submit([encoder.finish()]);
    expect(pixel(await readTexture(device, texture), 0, 0)).toEqual([0, 0, 255, 255]);
  });

  test('presents to its own canvas and to other canvases', async () => {
    const own = document.createElement('canvas');
    own.width = 4;
    own.height = 2;
    device = glDevice({ canvas: own });
    const ownContext = device[BACKEND].configure(own, { format: 'rgba8unorm' });
    const clear = (context: GPUCanvasContext, color: GPUColor) => {
      const encoder = device.createCommandEncoder();
      encoder
        .beginRenderPass({
          colorAttachments: [
            {
              view: context.getCurrentTexture().createView(),
              loadOp: 'clear',
              storeOp: 'store',
              clearValue: color,
            },
          ],
        })
        .end();
      device.queue.submit([encoder.finish()]);
    };
    clear(ownContext, [0, 1, 0, 1]);
    await Promise.resolve();
    expect(pixels(own)).toEqual([0, 255, 0, 255]);
    device.destroy();

    device = glDevice();
    const first = document.createElement('canvas');
    first.width = first.height = 3;
    const second = document.createElement('canvas');
    second.width = 5;
    second.height = 2;
    const a = device[BACKEND].configure(first, {
      format: 'rgba8unorm',
      alphaMode: 'premultiplied',
    });
    const b = device[BACKEND].configure(second, { format: 'rgba8unorm' });
    clear(a, [1, 0, 0, 1]);
    clear(b, [0, 0, 1, 1]);
    await device.queue.onSubmittedWorkDone();
    expect(pixels(first)).toEqual([255, 0, 0, 255]);
    expect(pixels(second)).toEqual([0, 0, 255, 255]);
    expect(a.getCurrentTexture().width).toBe(3);
    second.width = 6;
    expect(b.getCurrentTexture().width).toBe(6);
  });

  test('samples an imported external image and reports loss on destroy', async () => {
    device = glDevice();
    const canvas = new OffscreenCanvas(2, 2);
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#00ff00';
    context.fillRect(0, 0, 2, 2);
    const external = device.importExternalTexture({
      source: canvas as unknown as HTMLVideoElement,
    });
    const shader = module(device, {
      vertex: '',
      vertexMain: FULLSCREEN,
      fragment: 'uniform sampler2D frame;\nlayout(location = 0) out vec4 color;',
      fragmentMain: '  color = texelFetch(frame, ivec2(0), 0);',
      textures: [textureBinding('frame', 0, 0)],
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const out = target();
    const encoder = device.createCommandEncoder();
    const render = encoder.beginRenderPass({
      colorAttachments: [{ view: out.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    render.setPipeline(pipeline);
    render.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: external }],
      }),
    );
    render.draw(3);
    render.end();
    device.queue.submit([encoder.finish()]);
    expect(pixel(await readTexture(device, out), 0, 0)).toEqual([0, 255, 0, 255]);

    device.destroy();
    expect((await device.lost).reason).toBe('destroyed');
  });
});
