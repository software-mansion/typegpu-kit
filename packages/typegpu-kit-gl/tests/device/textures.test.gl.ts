import { tgpu } from 'typegpu';
import { afterEach, describe, expect, test } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { KitGenerator } from '../../src/generator.ts';
import { glDevice, pixel, readTexture } from './compare.ts';
import { module, textureBinding } from './glsl.ts';

let device: GLDevice;
afterEach(() => device?.destroy());

const RENDER = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
const ALL =
  GPUTextureUsage.RENDER_ATTACHMENT |
  GPUTextureUsage.COPY_SRC |
  GPUTextureUsage.COPY_DST |
  GPUTextureUsage.TEXTURE_BINDING;

function fill(texture: GPUTexture, rgba: number[], layer = 0, level = 0) {
  const width = Math.max(1, texture.width >> level);
  const height = Math.max(1, texture.height >> level);
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) data.set(rgba, i * 4);
  device.queue.writeTexture(
    { texture, mipLevel: level, origin: [0, 0, layer] },
    data,
    { bytesPerRow: width * 4 },
    [width, height, 1],
  );
}

const FULLSCREEN = `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  uv = vec2(p.x, -p.y) * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);`;

/** Draws `fragmentMain` over a 4x4 rgba8unorm target with `textures` bound in group 0. */
async function drawWith(
  declarations: string,
  fragmentMain: string,
  bindings: Parameters<typeof textureBinding>[],
  entries: GPUBindGroupEntry[],
  size = 4,
) {
  const shader = module(device, {
    vertex: 'out vec2 uv;',
    vertexMain: FULLSCREEN,
    fragment: `in vec2 uv;\n${declarations}\nlayout(location = 0) out vec4 color;`,
    fragmentMain,
    textures: bindings.map((args) => textureBinding(...args)),
  });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: shader },
    fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
  });
  const target = device.createTexture({ size: [size, size], format: 'rgba8unorm', usage: RENDER });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store' }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);
  return readTexture(device, target);
}

describe('textures', () => {
  test('samples the mip level a view starts at, while rendering into another level', async () => {
    device = glDevice();
    const texture = device.createTexture({
      size: [8, 8],
      format: 'rgba8unorm',
      usage: ALL,
      mipLevelCount: 3,
    });
    fill(texture, [255, 0, 0, 255], 0, 0);
    fill(texture, [0, 255, 0, 255], 0, 1);
    fill(texture, [0, 0, 255, 255], 0, 2);
    const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    const read = await drawWith(
      'uniform sampler2D source;',
      '  color = texture(source, uv);',
      [['source', 0, 0, { sampler: { group: 0, binding: 1 } }]],
      [
        { binding: 0, resource: texture.createView({ baseMipLevel: 1, mipLevelCount: 1 }) },
        { binding: 1, resource: sampler },
      ],
    );
    expect(pixel(read, 1, 1)).toEqual([0, 255, 0, 255]);

    // Render level 1 into level 2 from a view of level 1 only, as a bloom chain does
    const shader = module(device, {
      vertex: 'out vec2 uv;',
      vertexMain: FULLSCREEN,
      fragment: 'in vec2 uv;\nuniform sampler2D source;\nlayout(location = 0) out vec4 color;',
      fragmentMain: '  color = texture(source, uv) * 0.5;',
      textures: [textureBinding('source', 0, 0, { sampler: { group: 0, binding: 1 } })],
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      fragment: { module: shader, targets: [{ format: 'rgba8unorm' }] },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: texture.createView({ baseMipLevel: 2, mipLevelCount: 1 }),
          loadOp: 'load',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture.createView({ baseMipLevel: 1, mipLevelCount: 1 }) },
          { binding: 1, resource: sampler },
        ],
      }),
    );
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    expect(pixel(await readTexture(device, texture, { level: 2 }), 0, 0)).toEqual([0, 128, 0, 128]);
  });

  test('samples a cube, and a single layer of an array through a 2d view', async () => {
    device = glDevice();
    const cube = device.createTexture({ size: [2, 2, 6], format: 'rgba8unorm', usage: ALL });
    const cubeView = cube.createView({ dimension: 'cube' });
    const faces = [
      [255, 0, 0, 255],
      [0, 255, 0, 255],
      [0, 0, 255, 255],
      [255, 255, 0, 255],
      [0, 255, 255, 255],
      [255, 0, 255, 255],
    ];
    faces.forEach((rgba, face) => fill(cube, rgba, face));
    const sampler = device.createSampler();
    const read = await drawWith(
      'uniform samplerCube sky;',
      '  color = uv.x < 0.5 ? texture(sky, vec3(1.0, 0.0, 0.0)) : texture(sky, vec3(0.0, 0.0, -1.0));',
      [['sky', 0, 0, { dimension: 'cube', sampler: { group: 0, binding: 1 } }]],
      [
        { binding: 0, resource: cubeView },
        { binding: 1, resource: sampler },
      ],
    );
    expect(pixel(read, 0, 0)).toEqual(faces[0]);
    expect(pixel(read, 3, 0)).toEqual(faces[5]);

    const array = device.createTexture({ size: [2, 2, 3], format: 'rgba8unorm', usage: ALL });
    fill(array, [10, 20, 30, 255], 0);
    fill(array, [40, 50, 60, 255], 1);
    fill(array, [70, 80, 90, 255], 2);
    const layer = await drawWith(
      'uniform sampler2D layer;',
      '  color = texelFetch(layer, ivec2(0), 0);',
      [['layer', 0, 0]],
      [
        {
          binding: 0,
          resource: array.createView({ dimension: '2d', baseArrayLayer: 2, arrayLayerCount: 1 }),
        },
      ],
    );
    expect(pixel(layer, 0, 0)).toEqual([70, 80, 90, 255]);

    const whole = await drawWith(
      'uniform sampler2DArray layers;',
      '  color = texelFetch(layers, ivec3(0, 0, 1), 0);',
      [['layers', 0, 0, { dimension: '2d-array' }]],
      [{ binding: 0, resource: array.createView({ dimension: '2d-array' }) }],
    );
    expect(pixel(whole, 0, 0)).toEqual([40, 50, 60, 255]);

    // A cube written before any cube view existed samples a copy
    const late = device.createTexture({ size: [2, 2, 6], format: 'rgba8unorm', usage: ALL });
    faces.forEach((rgba, face) => fill(late, rgba, face));
    const copied = await drawWith(
      'uniform samplerCube sky;',
      '  color = texture(sky, vec3(0.0, 1.0, 0.0));',
      [['sky', 0, 0, { dimension: 'cube', sampler: { group: 0, binding: 1 } }]],
      [
        { binding: 0, resource: late.createView({ dimension: 'cube' }) },
        { binding: 1, resource: sampler },
      ],
    );
    expect(pixel(copied, 0, 0)).toEqual(faces[2]);
  });

  test('writes, copies and reads back float, packed and 3d textures', async () => {
    device = glDevice();
    const floats = device.createTexture({ size: [3, 2], format: 'r32float', usage: ALL });
    // Rows padded to 256 bytes, as copies from buffers need on WebGPU
    const padded = new Float32Array(128);
    padded.set([1, 2, 3], 0);
    padded.set([4, 5, 6], 64);
    device.queue.writeTexture({ texture: floats }, padded, { bytesPerRow: 256 }, [3, 2]);
    const read = await readTexture(device, floats);
    expect(Array.from(new Float32Array(read.bytes.buffer))).toEqual([1, 2, 3, 4, 5, 6]);

    const halves = device.createTexture({ size: [1, 1], format: 'rgba16float', usage: ALL });
    device.queue.writeTexture(
      { texture: halves },
      new Uint16Array([0x3c00, 0x4000, 0xbc00, 0x3800]),
      {},
      [1, 1],
    );
    const copy = device.createTexture({ size: [1, 1], format: 'rgba16float', usage: ALL });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture({ texture: halves }, { texture: copy }, [1, 1]);
    device.queue.submit([encoder.finish()]);
    const half = await readTexture(device, copy);
    expect(Array.from(new Uint16Array(half.bytes.buffer))).toEqual([
      0x3c00, 0x4000, 0xbc00, 0x3800,
    ]);

    const packed = device.createTexture({ size: [1, 1], format: 'rg11b10ufloat', usage: ALL });
    const clear = device.createCommandEncoder();
    clear
      .beginRenderPass({
        colorAttachments: [
          {
            view: packed.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: [1, 2, 0.5, 1],
          },
        ],
      })
      .end();
    device.queue.submit([clear.finish()]);
    const word = new Uint32Array((await readTexture(device, packed)).bytes.buffer)[0]!;
    expect(word & 0x7ff).toBe(0x3c0);
    expect((word >>> 11) & 0x7ff).toBe(0x400);
    expect(word >>> 22).toBe(0x1c0);

    const volume = device.createTexture({
      size: [2, 2, 2],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: ALL,
    });
    const voxels = new Uint8Array(32).map((_, i) => i * 4);
    device.queue.writeTexture(
      { texture: volume },
      voxels,
      { bytesPerRow: 8, rowsPerImage: 2 },
      [2, 2, 2],
    );
    const slice = await drawWith(
      'uniform sampler3D volume;',
      '  color = texelFetch(volume, ivec3(1, 0, 1), 0);',
      [['volume', 0, 0, { dimension: '3d' }]],
      [{ binding: 0, resource: volume.createView() }],
    );
    expect(pixel(slice, 0, 0)).toEqual([80, 84, 88, 92]);
  });

  test('reads depth back and compares it through a shadow sampler', async () => {
    device = glDevice();
    const depth = device.createTexture({
      size: [4, 4],
      format: 'depth32float',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    const shader = module(device, {
      vertex: '',
      vertexMain: `  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, 0.25 + 0.5 * step(0.0, p.x), 1.0);`,
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader },
      depthStencil: { format: 'depth32float', depthCompare: 'always', depthWriteEnabled: true },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: depth.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
    });
    pass.setPipeline(pipeline);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const values = new Float32Array((await readTexture(device, depth)).bytes.buffer);
    // z runs from 0.25 to 0.75 across the triangle, so the first texel center reads 0.28125
    expect(values[0]).toBeCloseTo(0.28125, 5);

    const shadow = device.createSampler({
      compare: 'less',
      minFilter: 'linear',
      magFilter: 'linear',
    });
    const read = await drawWith(
      'uniform sampler2DShadow shadow;',
      '  color = vec4(texture(shadow, vec3(0.125, 0.125, 0.5)), 0.0, 0.0, 1.0);',
      [
        [
          'shadow',
          0,
          0,
          { sampleType: 'depth', comparison: true, sampler: { group: 0, binding: 1 } },
        ],
      ],
      [
        { binding: 0, resource: depth.createView() },
        { binding: 1, resource: shadow },
      ],
    );
    // The reference 0.5 is not less than the stored 0.25
    expect(pixel(read, 0, 0)).toEqual([0, 0, 0, 255]);
  });

  test('takes images from a canvas, flipped on request', async () => {
    device = glDevice();
    const canvas = new OffscreenCanvas(2, 2);
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#ff0000';
    context.fillRect(0, 0, 2, 1);
    context.fillStyle = '#0000ff';
    context.fillRect(0, 1, 2, 1);
    const texture = device.createTexture({ size: [2, 2], format: 'rgba8unorm', usage: ALL });
    device.queue.copyExternalImageToTexture({ source: canvas }, { texture }, [2, 2]);
    const upright = await readTexture(device, texture);
    expect(pixel(upright, 0, 0)).toEqual([255, 0, 0, 255]);
    expect(pixel(upright, 0, 1)).toEqual([0, 0, 255, 255]);

    const bitmap = await createImageBitmap(canvas);
    device.queue.copyExternalImageToTexture({ source: bitmap, flipY: true }, { texture }, [2, 2]);
    const flipped = await readTexture(device, texture);
    expect(pixel(flipped, 0, 0)).toEqual([0, 0, 255, 255]);
  });

  test("runs TypeGPU's own mipmap shaders natively", async () => {
    device = glDevice();
    const root = tgpu.initFromDevice({ device, unstable_shaderGeneratorClass: KitGenerator });
    const texture = root
      .createTexture({ size: [4, 4], format: 'rgba8unorm', mipLevelCount: 3 })
      .$usage('sampled', 'render');
    const data = new Uint8Array(64);
    for (let i = 0; i < 16; i++) data.set(i % 2 === 0 ? [255, 0, 0, 255] : [0, 0, 255, 255], i * 4);
    texture.write(data);
    texture.generateMipmaps();
    const raw = root.unwrap(texture);
    const read = await readTexture(device, raw, { level: 2 });
    expect(pixel(read, 0, 0)).toEqual([128, 0, 128, 255]);

    const floats = root
      .createTexture({ size: [4, 4], format: 'r32float', mipLevelCount: 2 })
      .$usage('sampled', 'render');
    floats.write(new Float32Array(16).map((_, i) => i));
    floats.generateMipmaps();
    const level = await readTexture(device, root.unwrap(floats), { level: 1 });
    expect(Array.from(new Float32Array(level.bytes.buffer))).toEqual([2.5, 4.5, 10.5, 12.5]);
  });
});
