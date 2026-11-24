import { afterEach, describe, expect, test } from 'vitest';
import type { CpuModule } from '../../src/cpu/types.ts';
import type { ComputeModule } from '../../src/glsl/computeModule.ts';
import { type GLDevice, shaderModuleFrom } from '../../src/device/index.ts';
import { glDevice, pixel, readTexture } from './compare.ts';
import { bufferBinding, textureBinding } from './glsl.ts';

let device: GLDevice;
afterEach(() => device?.destroy());

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp sampler2D;
uniform uvec3 kit_invocations;`;

/** Doubles element `i` of the floats at @group(0) @binding(0) for each invocation `i`. */
const doublerGpu: ComputeModule = {
  kind: 'gpu-compute',
  workgroupSize: [4, 1, 1],
  grid: 1,
  passes: [
    {
      target: {
        kind: 'buffer',
        group: 0,
        binding: 0,
        source: 'values',
        offset: 0,
        stride: 4,
        size: 4,
      },
      fragment: `${HEADER}
uniform usampler2D values;
uniform uvec2 values_range;
layout(location = 0) out uvec4 cell;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy);
  uvec4 old = texelFetch(values, at, 0);
  uint texel = uint(at.y * textureSize(values, 0).x + at.x);
  uvec4 next = old;
  for (int c = 0; c < 4; c++) {
    uint w = texel * 4u + uint(c);
    if (w < values_range.x) continue;
    uint e = w - values_range.x;
    if (e * 4u < values_range.y && e < kit_invocations.x) {
      next[c] = floatBitsToUint(uintBitsToFloat(old[c]) * 2.0);
    }
  }
  cell = next;
}`,
    },
  ],
  buffers: [bufferBinding('values', 0, 0, 'mutable')],
  blocks: [],
  textures: [],
  levels: [],
  bindings: [{ group: 0, binding: 0, kind: 'buffer', writes: true }],
};

const doublerCpu: CpuModule = {
  kind: 'compute',
  workgroupSize: [4, 1, 1],
  bindings: [{ group: 0, binding: 0, kind: 'buffer', reads: true, writes: true }],
  dispatch(x, _y, _z, resources) {
    const { bytes, offset, size } = resources.buffer(0, 0);
    const floats = new Float32Array(bytes, offset, size / 4);
    for (let i = 0; i < Math.min(x * 4, floats.length); i++) floats[i]! *= 2;
  },
};

/** Inverts the colour of each texel of the rgba8unorm storage texture at @group(0) @binding(0). */
const inverterGpu: ComputeModule = {
  kind: 'gpu-compute',
  workgroupSize: [2, 2, 1],
  grid: 2,
  passes: [
    {
      target: {
        kind: 'texture',
        group: 0,
        binding: 0,
        source: 'image',
        dimension: '2d',
        format: 'rgba8unorm',
      },
      fragment: `${HEADER}
uniform sampler2D image;
layout(location = 0) out vec4 cell;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy);
  vec4 old = texelFetch(image, at, 0);
  bool inside = uint(at.x) < kit_invocations.x && uint(at.y) < kit_invocations.y;
  cell = inside ? vec4(1.0 - old.rgb, old.a) : old;
}`,
    },
  ],
  buffers: [],
  blocks: [],
  textures: [textureBinding('image', 0, 0)],
  levels: [],
  bindings: [{ group: 0, binding: 0, kind: 'texture', writes: true }],
};

const inverterCpu: CpuModule = {
  kind: 'compute',
  workgroupSize: [2, 2, 1],
  bindings: [{ group: 0, binding: 0, kind: 'storageTexture', reads: true, writes: true }],
  dispatch(x, y, _z, resources) {
    const level = resources.texture(0, 0).levels[0]!;
    const data = level.data as Uint8Array;
    for (let row = 0; row < Math.min(y * 2, level.height); row++)
      for (let column = 0; column < Math.min(x * 2, level.width); column++)
        for (let c = 0; c < 3; c++) {
          const at = (row * level.width + column) * 4 + c;
          data[at] = 255 - data[at]!;
        }
  },
};

async function readFloats(buffer: GPUBuffer, count: number) {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const values = Array.from(new Float32Array(staging.getMappedRange()));
  staging.unmap();
  return values;
}

function dispatch(
  pipeline: GPUComputePipeline,
  resource: GPUBindingResource,
  x: number,
  y = 1,
  z = 1,
) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource }],
    }),
  );
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

describe('compute as fragment passes', () => {
  test('writes a buffer on the GPU, which the CPU then reads and writes', async () => {
    device = glDevice();
    const values = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(values, 0, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      label: 'doubler',
      compute: { module: shaderModuleFrom(device, { cpu: doublerCpu, gpu: doublerGpu }) },
    });
    dispatch(pipeline, { buffer: values }, 1);
    expect(device.stats.kernels.get('doubler (gpu)')?.dispatches).toBe(1);
    expect(device.stats.kernels.has('doubler')).toBe(false);
    expect(await readFloats(values, 8)).toEqual([2, 4, 6, 8, 5, 6, 7, 8]);

    // A partial write lands on the contents the GPU wrote, and the next dispatch sees both
    device.queue.writeBuffer(values, 28, new Float32Array([10]));
    dispatch(pipeline, { buffer: values }, 2);
    expect(await readFloats(values, 8)).toEqual([4, 8, 12, 16, 10, 12, 14, 20]);

    // A bound range starts the elements at its offset
    dispatch(pipeline, { buffer: values, offset: 16, size: 16 }, 1);
    expect(await readFloats(values, 8)).toEqual([4, 8, 12, 16, 20, 24, 28, 40]);
  });

  test('writes a storage texture on the GPU, and falls back where a dispatch does not fit', async () => {
    device = glDevice();
    const image = device.createTexture({
      size: [4, 4],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });
    const texels = new Uint8Array(64);
    for (let i = 0; i < 16; i++) texels.set([200, 100, 0, 255], i * 4);
    device.queue.writeTexture({ texture: image }, texels, { bytesPerRow: 16 }, [4, 4]);
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      label: 'inverter',
      compute: { module: shaderModuleFrom(device, { cpu: inverterCpu, gpu: inverterGpu }) },
    });
    dispatch(pipeline, image.createView(), 1, 1);
    expect(device.stats.kernels.get('inverter (gpu)')?.dispatches).toBe(1);
    const read = await readTexture(device, image);
    expect(pixel(read, 0, 0)).toEqual([55, 155, 255, 255]);
    expect(pixel(read, 3, 3)).toEqual([200, 100, 0, 255]);

    // Two slices along z do not fit a 2d grid, so the CPU module runs
    dispatch(pipeline, image.createView(), 2, 2, 2);
    expect(device.stats.fallbacks.get('inverter')).toBe('the dispatch spans z');
    const after = await readTexture(device, image);
    expect(pixel(after, 0, 0)).toEqual([200, 100, 0, 255]);
    expect(pixel(after, 3, 3)).toEqual([55, 155, 255, 255]);
  });
});
