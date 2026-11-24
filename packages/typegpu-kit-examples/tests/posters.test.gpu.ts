import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { frames, type images } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { mulberry32 } from 'math/random';
import sharp from 'sharp';
import type { TgpuRoot } from 'typegpu';
import { describe, it } from 'vitest';
import { examples, highlights } from '../src/index.ts';
import { create } from '../src/hosts.ts';
import { fromDisk } from './utils/examples.ts';

const decode: images.Decode = async ({ bytes }) => {
  const { data, info } = await sharp(bytes).toColourspace('srgb').ensureAlpha().raw().toBuffer({
    resolveWithObject: true,
  });
  return { width: info.width, height: info.height, pixels: data };
};

async function readPixels(root: TgpuRoot, texture: GPUTexture) {
  const { device } = root;
  const { width, height } = texture;
  const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
  const buffer = device.createBuffer({
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const pixels = Buffer.alloc(width * height * 4);
    for (let y = 0; y < height; y++) {
      pixels.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
    }
    buffer.unmap();
    return pixels;
  } finally {
    buffer.destroy();
  }
}

interface Render {
  /** Steps the scene at 60 frames per second until `seconds` into it. */
  readonly until: (seconds: number) => Promise<void>;
  /** Draws `count` more frames, each waiting for the GPU, and returns their times in milliseconds. */
  readonly time: (count: number) => Promise<number[]>;
  /** Reads the surface as RGBA bytes. */
  readonly pixels: () => Promise<Buffer>;
}

async function render(
  key: string,
  size: { width: number; height: number },
  options: object,
  use: (render: Render) => Promise<void>,
) {
  const root = await gpuRoot({
    optionalFeatures: ['indirect-first-instance', 'float32-filterable'],
  });
  const surface = offscreenSurface(root, { ...size, format: 'rgba8unorm' });
  // A seeded `Math.random` changes a render only when its example changes
  const unseeded = Math.random;
  const seed = mulberry32.create(1);
  Math.random = () => mulberry32.sample(seed);
  try {
    const scene = await create(key, root, surface, { ...fromDisk, decode }, options);
    const loop = frames.create(root, scene.frame);
    try {
      const state = { drawing: scene.ready === undefined };
      const ready = scene.ready?.then(() => (state.drawing = true));
      while (!state.drawing) {
        loop.run(0);
        await Promise.race([ready, new Promise((resolve) => setTimeout(resolve, 10))]);
      }

      let frame = 0;
      await use({
        until: async (seconds) => {
          for (; frame <= Math.round(seconds * 60); frame++) {
            loop.run(frame / 60);
            if (frame % 30 === 0) {
              await root.device.queue.onSubmittedWorkDone();
              await new Promise((resolve) => setTimeout(resolve, 0));
            }
          }
        },
        time: async (count) => {
          await root.device.queue.onSubmittedWorkDone();
          const times: number[] = [];
          for (let i = 0; i < count; i++, frame++) {
            const start = performance.now();
            loop.run(frame / 60);
            await root.device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
          }
          return times;
        },
        pixels: () => readPixels(root, surface.texture),
      });
    } finally {
      loop.destroy();
      scene.destroy?.();
    }
  } finally {
    Math.random = unseeded;
    surface.destroy();
    root.destroy();
  }
}

function chosen(list: string | undefined) {
  const only = list && list !== '1' ? list.split(',') : null;
  return examples.filter(({ key }) => !only || only.includes(key));
}

// Posters are 16:10, rendered larger than they are saved. The featured card crops its poster
// nearly square, about 1440 px tall on a 2× screen
describe.runIf(process.env.POSTERS)('gallery posters', () => {
  it.each(chosen(process.env.POSTERS))(
    'renders $key',
    ({ key, card = {}, poster = {} }) => {
      const featured = key === highlights[0]?.key;
      const width = featured ? 2560 : (poster.width ?? 1280);
      const saved = Math.min(width, featured ? 2304 : 960);
      const size = { width, height: (width * 5) / 8 };
      return render(key, size, card, async ({ until, pixels }) => {
        await until(poster.seconds ?? 4);
        const frame = await pixels();
        // Sharp for the plain gallery, and small for the dithered cells, which print 3 px each
        for (const [wide, name] of [
          [saved, 'poster.webp'],
          [320, 'poster-small.webp'],
        ] as const) {
          await sharp(frame, { raw: { ...size, channels: 4 } })
            .resize(wide, (wide * 5) / 8, { kernel: 'lanczos3' })
            .removeAlpha()
            .webp({ quality: poster.quality ?? 70, effort: 6 })
            .toFile(fileURLToPath(new URL(`../src/${key}/${name}`, import.meta.url)));
        }
      });
    },
    600_000,
  );
});

// Full-size frames for review, at FRAMES_AT seconds (ascending) or two seconds in and the poster's
// moment, then the median and 90th-percentile time of 120 frames that each wait for the GPU
describe.runIf(process.env.FRAMES)('review frames', () => {
  const [width = 2560, height = 1440] = process.env.FRAMES_SIZE?.split('x').map(Number) ?? [];
  const out = process.env.FRAMES_DIR ?? '/tmp/typegpu-kit-frames';
  const at = process.env.FRAMES_AT?.split(',').map(Number);

  it.each(chosen(process.env.FRAMES))(
    'renders $key',
    ({ key, poster = {} }) =>
      render(key, { width, height }, {}, async ({ until, time, pixels }) => {
        const dir = join(out, key);
        await mkdir(dir, { recursive: true });
        for (const seconds of at ??
          [...new Set([2, poster.seconds ?? 4])].toSorted((a, b) => a - b)) {
          await until(seconds);
          await sharp(await pixels(), { raw: { width, height, channels: 4 } })
            .removeAlpha()
            .png()
            .toFile(join(dir, `${seconds}s.png`));
        }
        const times = (await time(120)).toSorted((a, b) => a - b);
        const quantile = (q: number) =>
          Math.round((times[Math.floor(q * (times.length - 1))] ?? 0) * 100) / 100;
        const timing = { width, height, medianMs: quantile(0.5), p90Ms: quantile(0.9) };
        await writeFile(join(dir, 'timing.json'), `${JSON.stringify(timing, null, 2)}\n`);
      }),
    600_000,
  );
});
