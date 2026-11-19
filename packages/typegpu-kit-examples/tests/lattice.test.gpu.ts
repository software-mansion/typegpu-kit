import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { describe, expect, it } from 'vitest';
import { createScene, defaults } from '../src/lattice/scene.ts';

// 64 BGRA8 pixels make a 256-byte row, the alignment texture copies need.
const WIDTH = 64;
const HEIGHT = 36;

type Values = typeof defaults;

/**
 * The lattice drawn with `values` for a few frames, its camera still, and what its last frame
 * showed: how many pixels are black, and how many are coloured by motion rather than grey.
 */
async function shown(values: Partial<Values>) {
  const root = await gpuRoot();
  const surface = offscreenSurface(root, { width: WIDTH, height: HEIGHT, format: 'bgra8unorm' });
  const scene = await createScene(root, surface);
  Object.assign(scene.settings.values, { speed: 1 }, values);
  await scene.settings.write!();

  const loop = frames.create(root, scene.frame);
  for (let i = 0; i < 6; i++) loop.run(i * 0.04);
  loop.destroy();

  const buffer = root.device.createBuffer({
    size: WIDTH * HEIGHT * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture: surface.texture }, { buffer, bytesPerRow: WIDTH * 4 }, [
    WIDTH,
    HEIGHT,
  ]);
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(buffer.getMappedRange().slice(0));
  buffer.destroy();
  scene.destroy();
  root.destroy();

  let black = 0;
  let moving = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    const [blue, green, red] = pixels.subarray(i, i + 3);
    if (red === 0 && green === 0 && blue === 0) black++;
    else if (Math.abs(red! - 127.5) > 8 || Math.abs(green! - 127.5) > 8) moving++;
  }
  return { black, moving, drawn: WIDTH * HEIGHT - black };
}

describe('the lattice, its camera still', () => {
  it('shows the beams turning in the motion view through their history, and still through a plain uniform', async () => {
    const alone = { view: 'motion', sky: false, embers: false } as const;
    const history = await shown({ ...alone, turn: 'history' });
    const uniform = await shown({ ...alone, turn: 'uniform' });

    expect(history.moving).toBeGreaterThan(20);
    expect(uniform.drawn).toBeGreaterThan(20);
    expect(uniform.moving).toBe(0);
  });

  it('shows the embers in the motion view drawn into the scene, and nothing of them left out of motion', async () => {
    const alone = { view: 'motion', sky: false, beams: false } as const;
    const into = await shown({ ...alone, into: 'scene' });
    const without = await shown({ ...alone, into: "scene.without('motion')" });

    expect(into.moving).toBeGreaterThan(5);
    expect(without.drawn).toBe(0);
  });

  it('shows only the beams in the depth view: the sky and the embers write no depth', async () => {
    const depth = { view: 'depth' } as const;
    expect((await shown({ ...depth, beams: false })).drawn).toBe(0);
    expect((await shown(depth)).drawn).toBeGreaterThan(20);
  });
});
