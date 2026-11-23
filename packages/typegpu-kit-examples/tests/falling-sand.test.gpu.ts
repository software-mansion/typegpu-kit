import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { srgbToLinear } from '@typegpu/color';
import { d, type TgpuRoot } from 'typegpu';
import { expect, it, onTestFinished } from 'vitest';
import { HEIGHT, WIDTH } from '../src/falling-sand/cells.ts';
import { createScene } from '../src/falling-sand/scene.ts';

const CHANNELS = 4;
const BYTES_PER_PIXEL = CHANNELS * 2;
const FRAME_SECONDS = 1 / 40;
const REGION = 2;

async function read(root: TgpuRoot, texture: GPUTexture) {
  const buffer = root.device.createBuffer({
    size: WIDTH * HEIGHT * BYTES_PER_PIXEL,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = root.device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: WIDTH * BYTES_PER_PIXEL }, [
    WIDTH,
    HEIGHT,
  ]);
  root.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new DataView(buffer.getMappedRange());
  const pixels = Float32Array.from({ length: WIDTH * HEIGHT * CHANNELS }, (_, i) =>
    bytes.getFloat16(i * 2, true),
  );
  buffer.unmap();
  buffer.destroy();
  return pixels;
}

const around = (pixels: Float32Array, x: number, y: number) => {
  let sum = 0;
  for (let dy = -REGION; dy <= REGION; dy++) {
    for (let dx = -REGION; dx <= REGION; dx++) {
      const at = ((y + dy) * WIDTH + x + dx) * CHANNELS;
      const rgb = srgbToLinear(d.vec3f(pixels[at], pixels[at + 1], pixels[at + 2]));
      sum += rgb.x + rgb.y + rgb.z;
    }
  }
  return sum / (2 * REGION + 1) ** 2;
};

async function build() {
  const root = await gpuRoot();
  const surface = offscreenSurface(root, { width: WIDTH, height: HEIGHT, format: 'rgba16float' });
  const scene = await createScene(root, surface);
  scene.settings.values.view = 'radiance';
  await scene.settings.write!('view');
  const loop = frames.create(root, scene.frame);
  onTestFinished(() => {
    loop.destroy();
    scene.destroy();
    surface.destroy();
    root.destroy();
  });
  let seconds = 0;
  return {
    scene,
    async draw(count = 1) {
      for (let i = 0; i < count; i++) {
        loop.run(seconds);
        seconds += FRAME_SECONDS;
        if (i % 40 === 39) await root.device.queue.onSubmittedWorkDone();
      }
      return read(root, surface.texture);
    },
  };
}

/** Air over the grotto's sand pile, lit by the crystal the sand buries. */
const GROTTO = [30, 90] as const;
/** Air between the grotto and the falls, lit mostly by what the walls reflect once the crystal is buried. */
const SHADE = [70, 95] as const;

it('darkens the buried crystal and lights the shade with bounced light', async () => {
  const { scene, draw } = await build();
  const lit = around(await draw(120), ...GROTTO);
  const settled = await draw(1680);
  const buried = around(settled, ...GROTTO);
  expect(lit).toBeGreaterThan(0.2);
  expect(buried).toBeLessThan(lit / 5);

  const reflected = around(settled, ...SHADE);
  scene.settings.values.bounce = false;
  await scene.settings.write!('bounce');
  const direct = around(await draw(20), ...SHADE);
  expect(reflected).toBeGreaterThan(direct * 1.6);
}, 300_000);
