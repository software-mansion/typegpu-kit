import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d } from 'typegpu';
import { expect, it } from 'vitest';
import * as baker from '../src/baker.ts';

const [WIDTH, HEIGHT, LAYERS] = [8, 4, 2];

it.each(['storage', 'render'] as const)(
  'writes uv at texel centres and the layer through %s usage',
  async (usage) => {
    const root = await gpuRoot();
    const texture = root
      .createTexture({ size: [WIDTH, HEIGHT, LAYERS], format: 'rgba32float' })
      .$usage(usage);
    const baking = baker.create(root, texture, {
      texel: (uv, layer) => {
        'use gpu';
        return d.vec4f(uv, d.f32(layer), 1);
      },
    });
    const encoder = root.createCommandEncoder();
    baking.record(encoder);
    encoder.submit();

    const row = 256;
    const buffer = root.device.createBuffer({
      size: row * HEIGHT * LAYERS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const copy = root.device.createCommandEncoder();
    copy.copyTextureToBuffer(
      { texture: root.unwrap(texture) },
      { buffer, bytesPerRow: row, rowsPerImage: HEIGHT },
      [WIDTH, HEIGHT, LAYERS],
    );
    root.device.queue.submit([copy.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const texels = new Float32Array(buffer.getMappedRange());

    for (let layer = 0; layer < LAYERS; layer++) {
      for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
          const at = (row * (layer * HEIGHT + y) + x * 16) / 4;
          expect([...texels.subarray(at, at + 4)]).toEqual([
            (x + 0.5) / WIDTH,
            (y + 0.5) / HEIGHT,
            layer,
            1,
          ]);
        }
      }
    }
  },
);
