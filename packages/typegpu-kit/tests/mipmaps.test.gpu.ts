import { gpuRoot, type GpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it, vi } from 'vitest';
import * as mipmaps from '../src/mipmaps.ts';

type Level = { width: number; height: number; texels: number[] };

/** Writes `r32float` texels to level 0 of one layer. */
function write(
  root: GpuRoot,
  texture: GPUTexture,
  layer: number,
  { width, height, texels }: Level,
) {
  root.device.queue.writeTexture(
    { texture, origin: [0, 0, layer] },
    new Float32Array(texels),
    { bytesPerRow: width * 4 },
    [width, height],
  );
}

/** Reads every level of one layer of an `r32float` texture. */
async function read(root: GpuRoot, texture: GPUTexture, layer: number): Promise<Level[]> {
  const sizes = Array.from({ length: texture.mipLevelCount }, (_, k) => ({
    width: Math.max(1, texture.width >> k),
    height: Math.max(1, texture.height >> k),
  }));
  const row = 256;
  const buffer = root.device.createBuffer({
    size: row * texture.height * sizes.length,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  sizes.forEach(({ width, height }, k) =>
    encoder.copyTextureToBuffer(
      { texture, mipLevel: k, origin: [0, 0, layer] },
      { buffer, offset: row * texture.height * k, bytesPerRow: row },
      [width, height],
    ),
  );
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  return sizes.map(({ width, height }, k) => ({
    width,
    height,
    texels: Array.from({ length: width * height }, (_, i) => {
      const x = i % width;
      const y = Math.floor(i / width);
      return data[(row * texture.height * k + row * y) / 4 + x];
    }),
  }));
}

/** The CPU's level below `above`, combining each 2×2 block with blocks past the edge clamped. */
function halve(
  above: Level,
  width: number,
  height: number,
  combine: (...texels: number[]) => number,
): Level {
  const at = (x: number, y: number) =>
    above.texels[Math.min(y, above.height - 1) * above.width + Math.min(x, above.width - 1)];
  const texels = Array.from({ length: width * height }, (_, i) => {
    const x = (i % width) * 2;
    const y = Math.floor(i / width) * 2;
    return combine(at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1));
  });
  return { width, height, texels };
}

const pattern = (width: number, height: number, seed: number): Level => ({
  width,
  height,
  texels: Array.from({ length: width * height }, (_, i) => ((i * 37 + seed) % 23) * 4),
});

it('fills farthest depths from a source, and after a resize into a raw pass with the same pipeline', async () => {
  const root = await gpuRoot();
  const made = vi.spyOn(root.device, 'createComputePipeline');
  let farthest: mipmaps.Mipmaps | undefined;
  for (const [depth, size, levels, raw] of [
    [pattern(13, 9, 5), 8, 4, false],
    [pattern(21, 30, 2), 16, 5, true],
  ] as const) {
    const source = root
      .createTexture({ size: [depth.width, depth.height], format: 'r32float' })
      .$usage('sampled');
    write(root, root.unwrap(source), 0, depth);
    const pyramid = root
      .createTexture({ size: [size, size], format: 'r32float', mipLevelCount: levels })
      .$usage('sampled', 'storage');
    farthest = farthest
      ? farthest.of(pyramid, source)
      : mipmaps.create(root, pyramid, { combine: mipmaps.largest, source });

    const encoder = root.device.createCommandEncoder();
    const pass = raw
      ? encoder.beginComputePass()
      : root.createCommandEncoder(encoder).beginComputePass();
    farthest.record(pass);
    pass.end();
    root.device.queue.submit([encoder.finish()]);

    let above: Level = depth;
    for (const level of await read(root, root.unwrap(pyramid), 0)) {
      above = halve(above, level.width, level.height, Math.max);
      expect(level).toEqual(above);
    }
  }
  expect(made).toHaveBeenCalledOnce();
});

it('averages every layer below level 0 in render passes', async () => {
  const root = await gpuRoot();
  const texture = root
    .createTexture({ size: [8, 8, 2], format: 'r32float', mipLevelCount: 4 })
    .$usage('sampled', 'render');
  const firsts = [pattern(8, 8, 0), pattern(8, 8, 11)];
  firsts.forEach((first, layer) => write(root, root.unwrap(texture), layer, first));

  const encoder = root.createCommandEncoder();
  mipmaps.create(root, texture, { combine: mipmaps.mean }).record(encoder);
  encoder.submit();

  for (const [layer, first] of firsts.entries()) {
    const [top, ...below] = await read(root, root.unwrap(texture), layer);
    expect(top).toEqual(first);
    let above = first;
    for (const level of below) {
      above = halve(above, level.width, level.height, (a, b, c, e) => (a + b + c + e) / 4);
      expect(level).toEqual(above);
    }
  }
});
