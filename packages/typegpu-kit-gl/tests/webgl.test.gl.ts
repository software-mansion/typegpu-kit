import { backends } from '@typegpu/kit';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { expect, test } from 'vitest';
import { webgl } from '../src/index.ts';

const SIZE = 64;

/** Draws a shaded triangle over a cleared canvas and returns the canvas's pixels, rows top down. */
async function triangle(root: TgpuRoot): Promise<Uint8ClampedArray> {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const backend = backends.of(root);
  const format = backend.preferredCanvasFormat;
  const context = backend.configure(canvas, { format, alphaMode: 'opaque' });

  const tint = root.createUniform(d.vec4f, d.vec4f(1, 0.5, 0.25, 1));
  const corners = tgpu.const(d.arrayOf(d.vec2f, 3), [
    d.vec2f(-0.8, -0.8),
    d.vec2f(0.8, -0.8),
    d.vec2f(-0.8, 0.8),
  ]);

  const vertex = tgpu.vertexFn({
    in: { index: d.builtin.vertexIndex },
    out: { position: d.builtin.position, uv: d.vec2f },
  })(({ index }) => {
    'use gpu';
    const corner = corners.$[index]!;
    return { position: d.vec4f(corner, 0, 1), uv: corner.mul(0.5).add(0.5) };
  });

  const fragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
    'use gpu';
    return std.mul(tint.$, d.vec4f(uv.x, uv.y, 1, 1));
  });

  const pipeline = root.createRenderPipeline({ vertex, fragment, targets: { format } });
  pipeline
    .withColorAttachment({
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      storeOp: 'store',
      clearValue: [0, 0, 0.5, 1],
    })
    .draw(3);

  await root.device.queue.onSubmittedWorkDone();
  const copy = new OffscreenCanvas(SIZE, SIZE).getContext('2d', { willReadFrequently: true })!;
  copy.drawImage(canvas, 0, 0);
  return copy.getImageData(0, 0, SIZE, SIZE).data;
}

const at = (pixels: Uint8ClampedArray, x: number, y: number) => [
  ...pixels.subarray((y * SIZE + x) * 4, (y * SIZE + x) * 4 + 4),
];

test('draws a triangle on WebGL 2 as WebGPU does, the right way up', async () => {
  const gpu = await tgpu.init();
  const gl = await webgl.init();
  expect(backends.of(gl).kind).toBe('webgl');

  const expected = await triangle(gpu);
  const actual = await triangle(gl);

  // In WebGPU's y-up clip space the triangle fills the bottom-left half, so its corner at clip
  // (-0.8, 0.8) is near the top of the canvas and the top-right stays clear
  expect(at(expected, 56, 8)).toEqual([0, 0, 128, 255]);
  expect(at(expected, 12, 50)[2]).toBe(64);

  // Edges may differ by coverage; flat areas and gradients must match within rounding
  let far = 0;
  for (let i = 0; i < expected.length; i++) {
    if (Math.abs(expected[i]! - actual[i]!) > 2) far++;
  }
  expect(far / expected.length).toBeLessThan(0.01);
  expect(at(actual, 56, 8)).toEqual([0, 0, 128, 255]);
  expect(at(actual, 12, 50)[2]).toBe(64);

  gpu.destroy();
  gl.destroy();
});
