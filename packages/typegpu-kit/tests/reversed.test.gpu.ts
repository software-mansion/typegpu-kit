import { cameras, passes, targets } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { meshes } from '@typegpu/kit/geometry';
import { d } from 'typegpu';
import { expect, it } from 'vitest';
import { environment, font, instances, lighting, meshPool, particles, text } from '../src/index.ts';
import { inter } from './text/inter.ts';

// 64 RGBA8 pixels make a 256-byte row; 8 pixels to the world unit, the origin centred.
const SIZE = 64;
const dark = { sunIntensity: 0, skyTop: [0, 0, 0], skyBottom: [0, 0, 0] } as const;

const blue = () => {
  'use gpu';
  return d.vec3f(0, 0, 1);
};

/**
 * Draws, looking down -z: a red slab over the left half at z = 0.5, a green one over the bottom
 * half at z = -0.5, the blue background, and a white particle and letter in front of the red slab
 * and behind it. The red slab is drawn first, so the green one must lose the depth test.
 */
async function scene(reversed: boolean) {
  const root = await gpuRoot();
  const size = { width: SIZE, height: SIZE };
  const view = cameras.flat(root, { size, height: 8, reversed });
  const screen = targets.create(root, size, {
    color: { format: 'rgba8unorm' },
    depth: { format: 'depth24plus' },
  });
  const into = passes
    .create(
      root,
      screen,
      { color: passes.color, depth: passes.depth },
      reversed ? { depthClear: 0, depthCompare: 'greater' } : {},
    )
    .pipe(view.inject());

  const pool = meshPool.create(root, { vertices: 64, indices: 128, capacity: 2 });
  const things = instances.create(root, pool, { capacity: 2 });
  const slab = (width: number, height: number) =>
    pool.add(meshes.box({ width, height, depth: 0.2 }));
  things.add(slab(4, 8), { position: [-2, 0, 0.5], emission: [1, 0, 0] });
  things.add(slab(8, 4), { position: [0, -2, -0.5], emission: [0, 1, 0] });

  const sparks = particles.create(root, view, {
    capacity: 2,
    step: 1 / 60,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  for (const z of [0.8, 0]) {
    sparks.burst({ count: 1, position: [-2, z > 0 ? 2 : -2, z], size: 0.5, life: 9, variance: 0 });
  }

  const letters = text.create(root, font.create(root, inter), { capacity: 2, length: 1 });
  letters.labels.patch({ 0: { position: [-3.8, 2.6, 0.8] }, 1: { position: [-3.8, -1.4, 0] } });
  for (const label of [0, 1]) letters.write(label, 'H', { em: 1.2, align: 'left' });

  const drawn = [
    into.add(things.renderer().pipe(lighting.create(root, { settings: dark }).inject())),
    into.add(environment.background(blue)),
    into.add(sparks.drawing),
    into.add(letters.drawing),
  ];
  await into.initAsync();

  const encoder = root.createCommandEncoder();
  things.flush();
  const compute = encoder.beginComputePass();
  sparks.spawn(compute);
  sparks.compact(compute, 0);
  compute.end();
  into.record(encoder, drawn);
  encoder.submit();

  const buffer = root.device.createBuffer({
    size: SIZE * SIZE * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer(
    { texture: root.unwrap(screen.textures.color) },
    { buffer, bytesPerRow: SIZE * 4 },
    [SIZE, SIZE],
  );
  root.device.queue.submit([copy.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(buffer.getMappedRange().slice(0));
  root.destroy();

  /** The pixel at world `x, y`. */
  const at = (x: number, y: number) => {
    const i = (Math.floor(32 - 8 * y) * SIZE + Math.floor(8 * x + 32)) * 4;
    return [...pixels.subarray(i, i + 3)];
  };
  /** The brightest green over a world rectangle, where white shows over the red slab. */
  const whitest = (left: number, top: number, right: number, bottom: number) => {
    let most = 0;
    for (let y = top; y > bottom; y -= 1 / 8) {
      for (let x = left; x < right; x += 1 / 8) most = Math.max(most, at(x, y)[1]);
    }
    return most;
  };
  return { at, whitest };
}

it.each([false, true])(
  'keeps each drawing on its side of the nearer surface (reversed %s)',
  async (reversed) => {
    const { at, whitest } = await scene(reversed);

    expect(at(-1, 3)).toEqual([255, 0, 0]);
    expect(at(-0.5, -3)).toEqual([255, 0, 0]);
    expect(at(2, -2)).toEqual([0, 255, 0]);
    expect(at(2, 2)).toEqual([0, 0, 255]);
    // The particle and letter in front of the red slab show; those behind it don't.
    expect(at(-2, 2)[1]).toBeGreaterThan(64);
    expect(at(-2, -2)).toEqual([255, 0, 0]);
    expect(whitest(-3.8, 3.8, -2.6, 2.6)).toBeGreaterThan(128);
    expect(whitest(-3.8, -0.2, -2.6, -1.4)).toBe(0);
  },
);
