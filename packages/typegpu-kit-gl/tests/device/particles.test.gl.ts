import { cameras, particles, passes, targets } from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import { afterAll, afterEach, beforeAll, expect, it } from 'vitest';
import type { GLDevice } from '../../src/device/index.ts';
import { readTexture, roots } from './compare.ts';

let gpu: TgpuRoot;
let gl: TgpuRoot;
beforeAll(async () => {
  ({ gpu, gl } = await roots());
});
afterAll(() => {
  gpu.destroy();
  gl.destroy();
});
afterEach(() => {
  expect((gl.device as unknown as GLDevice).gl.getError()).toBe(0);
});

/**
 * Draws a pool where half the particles die, over three frames, and reads each frame's pixels and
 * live count. On the fallback, whose compute runs on the CPU, the draw covers every slot.
 */
async function drawnPool(root: TgpuRoot, steady?: () => void) {
  const unseeded = Math.random;
  let state = 5;
  Math.random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const size = { width: 64, height: 64 };
  const screen = targets.create(root, size, { color: { format: 'rgba8unorm' } });
  const view = cameras.flat(root, { size, height: 10 });
  const dust = particles.create(root, view, {
    capacity: 42,
    step: 0.1,
    dimensions: 2,
    shade: () => {
      'use gpu';
      return d.vec4f(0.1, 0.05, 0.02, 0.1);
    },
  });
  const drawn = passes.create(root, screen, { color: passes.color }).add(dust.drawing);
  for (let k = 0; k < 6; k++) {
    dust.burst({
      count: 7,
      position: [k - 3, k % 2 === 0 ? -2 : 2, 0],
      area: [0.5, 0.5, 0],
      velocity: [0.5, 0.3, 0],
      size: 0.6,
      life: k % 2 === 0 ? 0.15 : 10,
      variance: 0,
    });
  }
  const frame = (steps: number) => {
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dust.spawn(pass);
    for (let i = 0; i < steps; i++) dust.step(pass);
    dust.compact(pass, 0);
    pass.end();
    const render = encoder.beginRenderPass({
      colorAttachments: { view: screen.views.color, loadOp: 'clear', clearValue: [0, 0, 0, 0] },
    });
    drawn.draw(render);
    render.end();
    encoder.submit();
  };
  const seen: { pixels: number[]; live: number }[] = [];
  for (const steps of [0, 3, 1]) {
    // The last frame is steady: nothing read back since the one before
    if (steps === 1) steady?.();
    frame(steps);
    const pixels = await readTexture(root.device, root.unwrap(screen.textures.color));
    const args = (await dust.args.read()) as unknown as number[];
    seen.push({ pixels: [...pixels.bytes], live: args[1] ?? -1 });
  }
  Math.random = unseeded;
  dust.destroy();
  return seen;
}

it('draws only the live particles without listing them, as WebGPU does, and reads nothing back', async () => {
  const stats = (gl.device as unknown as GLDevice).stats;
  let steadyReads = -1;
  let kernels: string[] = [];
  const actual = await drawnPool(gl, () => stats.reset());
  steadyReads = stats.readback.textures;
  kernels = [...stats.kernels.keys()];
  const expected = await drawnPool(gpu);
  expect(actual.map(({ live }) => live)).toEqual(expected.map(({ live }) => live));
  expect(actual.map(({ live }) => live)).toEqual([42, 21, 21]);
  for (const [k, frame] of expected.entries()) {
    expect(actual[k]!.pixels, `frame ${k}`).toEqual(frame.pixels);
  }
  // Counted on the GPU, listed nowhere, and nothing read back for it
  expect(kernels.filter((label) => label.includes('live'))).toEqual([]);
  expect(steadyReads).toBe(0);
  expect(kernels).toEqual(expect.arrayContaining(['tally (gpu)', 'counting (gpu)']));
});
