import { afterEach, describe, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { errorsOf, glDevice, pixel, readTexture } from './compare.ts';
import { bufferBinding, bufferReader, module } from './glsl.ts';

const SIZE = 8;
let device: GLDevice;

afterEach(() => device?.destroy());

function target(format: GPUTextureFormat = 'rgba8unorm', size = SIZE, samples = 1) {
  return device.createTexture({
    size: [size, size],
    format,
    sampleCount: samples,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING,
  });
}

function buffer(data: ArrayBufferView, usage: number) {
  const created = device.createBuffer({
    size: Math.ceil(data.byteLength / 4) * 4,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(created.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  created.unmap();
  return created;
}

function pass(encoder: GPUCommandEncoder, view: GPUTexture, clear: GPUColor = [0, 0, 0, 1]) {
  return encoder.beginRenderPass({
    colorAttachments: [
      { view: view.createView(), loadOp: 'clear', storeOp: 'store', clearValue: clear },
    ],
  });
}

/** A full-screen triangle in one color, from vertex indices. */
function solid(color: string) {
  return module(device, {
    vertex: '',
    vertexMain: `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, 0.5, 1.0);`,
    fragment: 'layout(location = 0) out vec4 color;',
    fragmentMain: `  color = ${color};`,
  });
}

describe('render passes', () => {
  test('clears and reads back a color target', async () => {
    device = glDevice();
    const texture = target();
    const encoder = device.createCommandEncoder();
    pass(encoder, texture, [1, 0.5, 0, 1]).end();
    device.queue.submit([encoder.finish()]);
    const read = await readTexture(device, texture);
    expect(pixel(read, 0, 0)).toEqual([255, 128, 0, 255]);
    expect(pixel(read, 7, 7)).toEqual([255, 128, 0, 255]);
  });

  test('draws a triangle the right way up, with y down as in WebGPU', async () => {
    device = glDevice();
    const texture = target();
    const shader = module(device, {
      vertex: 'layout(location = 0) in vec2 position;',
      vertexMain: '  gl_Position = vec4(position, 0.0, 1.0);',
      fragment: 'layout(location = 0) out vec4 color;',
      fragmentMain: '  color = vec4(0.0, 1.0, 0.0, 1.0);',
    });
    // Covers the top half of clip space, so the top rows of the texture
    const vertices = buffer(new Float32Array([-1, 0, 3, 0, -1, 4]), GPUBufferUsage.VERTEX);
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shader,
        buffers: [
          { arrayStride: 8, attributes: [{ format: 'float32x2', offset: 0, shaderLocation: 0 }] },
        ],
      },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
      primitive: { cullMode: 'back' },
    });
    const encoder = device.createCommandEncoder();
    const render = pass(encoder, texture);
    render.setPipeline(pipeline);
    render.setVertexBuffer(0, vertices);
    render.draw(3);
    render.end();
    device.queue.submit([encoder.finish()]);
    const read = await readTexture(device, texture);
    expect(pixel(read, 3, 0)).toEqual([0, 255, 0, 255]);
    expect(pixel(read, 3, 7)).toEqual([0, 0, 0, 255]);
  });

  test('reads storage buffers and instance attributes with a first instance', async () => {
    device = glDevice();
    const texture = target('rgba8unorm', 4);
    // Instance i fills column i through a per-instance x offset and a color from a buffer
    const shader = module(device, {
      vertex: `layout(location = 0) in float column;
${bufferReader('colors')}
out vec4 tint;`,
      vertexMain: `  int instance = gl_InstanceID + kit_firstInstance;
  vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  vec2 p = vec2((column + corner.x) / 4.0, corner.y) * 2.0 - 1.0;
  tint = vec4(colors_float(uint(instance) * 4u), colors_float(uint(instance) * 4u + 1u), colors_float(uint(instance) * 4u + 2u), 1.0);
  gl_Position = vec4(p, 0.0, 1.0);`,
      fragment: `in vec4 tint;
layout(location = 0) out vec4 color;`,
      fragmentMain: '  color = tint;',
      buffers: [bufferBinding('colors', 0, 0)],
    });
    const columns = buffer(new Float32Array([0, 1, 2, 3]), GPUBufferUsage.VERTEX);
    const colors = buffer(
      new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0]),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shader,
        buffers: [
          {
            arrayStride: 4,
            stepMode: 'instance',
            attributes: [{ format: 'float32', offset: 0, shaderLocation: 0 }],
          },
        ],
      },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-strip' },
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: colors } }],
    });
    const encoder = device.createCommandEncoder();
    const render = pass(encoder, texture);
    render.setPipeline(pipeline);
    render.setBindGroup(0, group);
    render.setVertexBuffer(0, columns);
    render.draw(4, 2, 0, 1);
    render.draw(4, 1, 0, 0);
    render.end();
    device.queue.submit([encoder.finish()]);
    const read = await readTexture(device, texture);
    expect(pixel(read, 0, 0)).toEqual([255, 0, 0, 255]);
    expect(pixel(read, 1, 1)).toEqual([0, 255, 0, 255]);
    expect(pixel(read, 2, 2)).toEqual([0, 0, 255, 255]);
    expect(pixel(read, 3, 3)).toEqual([0, 0, 0, 255]);

    // A write after the first submit reaches the next draw
    device.queue.writeBuffer(colors, 0, new Float32Array([1, 1, 0, 0]));
    const again = device.createCommandEncoder();
    const second = pass(again, texture);
    second.setPipeline(pipeline);
    second.setBindGroup(0, group);
    second.setVertexBuffer(0, columns);
    second.draw(4, 1, 0, 0);
    second.end();
    device.queue.submit([again.finish()]);
    expect(pixel(await readTexture(device, texture), 0, 0)).toEqual([255, 255, 0, 255]);
  });

  test('draws indexed with a base vertex, and from indirect arguments', async () => {
    device = glDevice();
    const texture = target('rgba8unorm', 4);
    const shader = module(device, {
      vertex: 'layout(location = 0) in vec2 position;',
      vertexMain: '  gl_Position = vec4(position, 0.0, 1.0);',
      fragment: 'layout(location = 0) out vec4 color;',
      fragmentMain: '  color = vec4(1.0);',
    });
    // Two quads: the left half and the right half; indices address the right one via base vertex
    const vertices = buffer(
      new Float32Array([-1, -1, 0, -1, -1, 1, 0, 1, 0, -1, 1, -1, 0, 1, 1, 1]),
      GPUBufferUsage.VERTEX,
    );
    const indices = buffer(new Uint16Array([0, 1, 2, 2, 1, 3]), GPUBufferUsage.INDEX);
    const indirect = buffer(new Uint32Array([6, 1, 0, 4, 0]), GPUBufferUsage.INDIRECT);
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shader,
        buffers: [
          { arrayStride: 8, attributes: [{ format: 'float32x2', offset: 0, shaderLocation: 0 }] },
        ],
      },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const encoder = device.createCommandEncoder();
    const render = pass(encoder, texture);
    render.setPipeline(pipeline);
    render.setVertexBuffer(0, vertices);
    render.setIndexBuffer(indices, 'uint16');
    render.drawIndexedIndirect(indirect, 0);
    render.end();
    device.queue.submit([encoder.finish()]);
    const read = await readTexture(device, texture);
    expect(pixel(read, 0, 1)).toEqual([0, 0, 0, 255]);
    expect(pixel(read, 3, 1)).toEqual([255, 255, 255, 255]);
  });

  test('writes several targets, tests depth and blends', async () => {
    device = glDevice();
    const first = target('rgba8unorm', 4);
    const second = target('rgba16float', 4);
    const depth = device.createTexture({
      size: [4, 4],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const shader = (z: number, rgb: string) =>
      module(device, {
        vertex: '',
        vertexMain: `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, ${z.toFixed(2)}, 1.0);`,
        fragment: 'layout(location = 0) out vec4 a;\nlayout(location = 1) out vec4 b;',
        fragmentMain: `  a = vec4(${rgb}, 0.5);\n  b = vec4(2.0, 3.0, 4.0, 1.0);`,
      });
    const pipeline = (z: number, rgb: string) =>
      device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: shader(z, rgb) },
        fragment: {
          module: shader(z, rgb),
          targets: [
            {
              format: 'rgba8unorm',
              blend: {
                color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                alpha: { srcFactor: 'one', dstFactor: 'zero' },
              },
            },
            { format: 'rgba16float' },
          ],
        },
        depthStencil: { format: 'depth24plus', depthCompare: 'less', depthWriteEnabled: true },
      });
    const near = pipeline(0.2, '1.0, 0.0, 0.0');
    const far = pipeline(0.8, '0.0, 1.0, 0.0');
    const encoder = device.createCommandEncoder();
    const render = encoder.beginRenderPass({
      colorAttachments: [
        { view: first.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 1, 1] },
        { view: second.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: {
        view: depth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
    });
    render.setPipeline(near);
    render.draw(3);
    render.setPipeline(far);
    render.draw(3);
    render.end();
    device.queue.submit([encoder.finish()]);
    // Red at half alpha over the blue clear; the far green triangle fails the depth test
    expect(pixel(await readTexture(device, first), 1, 1)).toEqual([128, 0, 128, 128]);
    const floats = await readTexture(device, second);
    expect(Array.from(new Uint16Array(floats.bytes.buffer, 0, 4))).toEqual([
      0x4000, 0x4200, 0x4400, 0x3c00,
    ]);
  });

  test('resolves a multisampled target', async () => {
    device = glDevice();
    const msaa = target('rgba8unorm', 4, 4);
    const resolved = target('rgba8unorm', 4);
    const shader = solid('vec4(1.0, 0.0, 1.0, 1.0)');
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
      multisample: { count: 4 },
    });
    const encoder = device.createCommandEncoder();
    const render = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: msaa.createView(),
          resolveTarget: resolved.createView(),
          loadOp: 'clear',
          storeOp: 'discard',
          clearValue: [0, 0, 0, 1],
        },
      ],
    });
    render.setPipeline(pipeline);
    render.draw(3);
    render.end();
    device.queue.submit([encoder.finish()]);
    expect(pixel(await readTexture(device, resolved), 2, 2)).toEqual([255, 0, 255, 255]);
  });

  test('reports a GLSL error with the numbered source and skips the pipeline', async () => {
    device = glDevice();
    const texture = target();
    const messages = await errorsOf(device, async () => {
      const shader = solid('vec4(oops)');
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: shader },
        fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
      });
      const encoder = device.createCommandEncoder();
      const render = pass(encoder, texture);
      render.setPipeline(pipeline);
      render.draw(3);
      render.end();
      device.queue.submit([encoder.finish()]);
    });
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain("'oops'");
    expect(messages[0]).toMatch(/\d+: +color = vec4\(oops\);/);
    expect(pixel(await readTexture(device, texture), 0, 0)).toEqual([0, 0, 0, 255]);
  });

  test('binds uniform blocks at aligned and unaligned offsets', async () => {
    device = glDevice();
    const texture = target('rgba8unorm', 2);
    const shader = module(device, {
      vertex: '',
      vertexMain: `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, 0.0, 1.0);`,
      fragment: `layout(std140) uniform Tint { uvec4 words[1]; } tint;
layout(location = 0) out vec4 color;`,
      fragmentMain: '  color = uintBitsToFloat(tint.words[0]);',
      blocks: [{ group: 0, binding: 0, block: 'Tint', size: 16 }],
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const values = device.createBuffer({
      size: 512,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(values, 0, new Float32Array([1, 0, 0, 1]));
    device.queue.writeBuffer(values, 16, new Float32Array([0, 1, 0, 1]));
    device.queue.writeBuffer(values, 256, new Float32Array([0, 0, 1, 1]));
    const draw = async (offset: number) => {
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: values, offset, size: 16 } }],
      });
      const encoder = device.createCommandEncoder();
      const render = pass(encoder, texture);
      render.setPipeline(pipeline);
      render.setBindGroup(0, group);
      render.draw(3);
      render.end();
      device.queue.submit([encoder.finish()]);
      return pixel(await readTexture(device, texture), 0, 0);
    };
    expect(await draw(0)).toEqual([255, 0, 0, 255]);
    expect(await draw(16)).toEqual([0, 255, 0, 255]);
    expect(await draw(256)).toEqual([0, 0, 255, 255]);
    // A write reaches the copy an unaligned range binds
    device.queue.writeBuffer(values, 16, new Float32Array([1, 1, 0, 1]));
    expect(await draw(16)).toEqual([255, 255, 0, 255]);

    // A dynamic offset moves the bound range
    const layout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true },
        },
      ],
    });
    const dynamic = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const group = device.createBindGroup({
      layout,
      entries: [{ binding: 0, resource: { buffer: values, size: 16 } }],
    });
    const encoder = device.createCommandEncoder();
    const render = pass(encoder, texture);
    render.setPipeline(dynamic);
    render.setBindGroup(0, group, [256]);
    render.draw(3);
    render.end();
    device.queue.submit([encoder.finish()]);
    expect(pixel(await readTexture(device, texture), 0, 0)).toEqual([0, 0, 255, 255]);
  });

  test('keeps blending every target after a readback blits between additive draws', async () => {
    device = glDevice();
    const a = target('rgba16float', 2);
    const b = target('rgba16float', 2);
    const depth = device.createTexture({
      size: [2, 2],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const shader = module(device, {
      vertex: '',
      vertexMain: `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, 0.5, 1.0);`,
      fragment: 'layout(location = 0) out vec4 first;\nlayout(location = 1) out vec4 second;',
      fragmentMain: '  first = vec4(0.25);\n  second = vec4(0.25);',
    });
    const add: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one' },
      alpha: { srcFactor: 'one', dstFactor: 'one' },
    };
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: {
        module: shader,
        targets: [
          { format: 'rgba16float', blend: add },
          { format: 'rgba16float', blend: add },
        ],
      },
    });
    const draw = (loadOp: GPULoadOp) => {
      const encoder = device.createCommandEncoder();
      const render = encoder.beginRenderPass({
        colorAttachments: [a, b].map((texture) => ({
          view: texture.createView(),
          loadOp,
          storeOp: 'store' as const,
          clearValue: [0, 0, 0, 0],
        })),
        depthStencilAttachment: {
          view: depth.createView(),
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
          depthClearValue: 1,
        },
      });
      render.setPipeline(pipeline);
      render.draw(3);
      render.end();
      device.queue.submit([encoder.finish()]);
    };
    draw('clear');
    // Reading depth draws it into a float target through the device's own program
    await readTexture(device, depth);
    draw('load');
    const half = 0x3800;
    for (const texture of [a, b]) {
      const read = await readTexture(device, texture);
      expect(new Uint16Array(read.bytes.buffer)[0]).toBe(half);
    }
  });

  test('captures errors in an error scope', async () => {
    device = glDevice();
    device.pushErrorScope('validation');
    device.createShaderModule({
      code: '@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }',
    });
    const error = await device.popErrorScope();
    expect(error?.message).toContain('WGSL');
  });
});
