import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { encodeHalf, decodeHalf } from '../../src/cpu/textures.ts';
import { readBuffer } from './example-comparison.ts';
import { recordResults } from './results.ts';

it.runIf(process.env.CPU_THRESHOLDS === '1')(
  'measures explicit cube mip interpolation precision',
  async () => {
    const root = await gpuRoot();
    const device = root.device;
    const texture = device.createTexture({
      size: [8, 8, 6],
      mipLevelCount: 4,
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const output = device.createBuffer({
      size: 1025 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    try {
      for (let mip = 0; mip < 4; mip++) {
        const size = 8 >> mip;
        const values = new Uint16Array(size * size * 6 * 4).fill(encodeHalf(mip % 2));
        device.queue.writeTexture(
          { texture, mipLevel: mip },
          values,
          { bytesPerRow: size * 8, rowsPerImage: size },
          [size, size, 6],
        );
      }
      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({
            code: `
@group(0) @binding(0) var image:texture_cube<f32>;
@group(0) @binding(1) var filtering:sampler;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let t=f32(id.x)/1024.;output[id.x]=vec4f(textureSampleLevel(image,filtering,vec3f(0,0,1),t).x,textureSampleLevel(image,filtering,vec3f(.2,.3,1),1.+t).x,textureSampleLevel(image,filtering,vec3f(1,1,1),2.+t).x,1);}`,
          }),
        },
      });
      const encoder = device.createCommandEncoder(),
        pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texture.createView({ dimension: 'cube' }) },
            {
              binding: 1,
              resource: device.createSampler({
                minFilter: 'linear',
                magFilter: 'linear',
                mipmapFilter: 'linear',
              }),
            },
            { binding: 2, resource: { buffer: output } },
          ],
        }),
      );
      pass.dispatchWorkgroups(1025);
      pass.end();
      device.queue.submit([encoder.finish()]);
      const values = new Float32Array(await readBuffer(device, output));
      const rows = [];
      let maximumError = 0,
        modelDifferences = 0;
      for (let i = 0; i <= 1024; i++) {
        const t = i / 1024;
        const row = Array.from(values.subarray(i * 4, i * 4 + 3));
        maximumError = Math.max(
          maximumError,
          ...row.map((value, c) => Math.abs(value - (c === 1 ? 1 - t : t))),
        );
        const model = row.map((_value, c) => {
          const weight = Math.floor(decodeHalf(encodeHalf(c + t)) * 64) / 64 - c;
          return c === 1 ? 1 - weight : weight;
        });
        modelDifferences += row.filter((value, c) => value !== model[c]).length;
        rows.push({ t, values: row, model });
      }
      recordResults('cubeLodPrecision', { maximumError, modelDifferences, rows });
      expect(maximumError).toBeLessThanOrEqual(1 / 64);
      expect(modelDifferences).toBe(0);
    } finally {
      texture.destroy();
      output.destroy();
      root.destroy();
    }
  },
);
