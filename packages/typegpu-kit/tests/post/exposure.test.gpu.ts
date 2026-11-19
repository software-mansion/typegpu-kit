import { targets } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { exposure, finish } from '../../src/index.ts';

const size = { width: 64, height: 32 };

/** A grey source the exposure meters, and a probe that reads `exposure.current` back. */
async function setup() {
  const root = await gpuRoot();
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    adapted: { format: 'rgba8unorm' },
    manual: { format: 'rgba8unorm' },
  });
  const source = () => screen.textures.hdr;
  const eye = exposure.create(root, { source });

  const seen = root.createMutable(d.f32);
  const probe = root.pipe(eye.inject()).createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      seen.$ = exposure.current.$;
    }),
  });
  await eye.initAsync();

  /** Clears the source to `grey`, adapts for `dt` seconds and returns the exposure. */
  const adapt = async (grey: number, dt: number) => {
    const encoder = root.createCommandEncoder();
    encoder
      .beginRenderPass({
        colorAttachments: { view: screen.views.hdr, clearValue: [grey, grey, grey, 1] },
      })
      .end();
    eye.record(encoder, { dt });
    const pass = encoder.beginComputePass({});
    pass.setPipeline(probe);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.submit();
    return seen.read();
  };

  return { root, screen, source, eye, adapt };
}

/** The bytes of an rgba8unorm texture of `size`. */
async function pixels(root: TgpuRoot, texture: GPUTexture) {
  const buffer = root.device.createBuffer({
    size: size.width * 4 * size.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: size.width * 4 }, size);
  root.device.queue.submit([copy.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  return new Uint8Array(buffer.getMappedRange().slice(0));
}

/** One step of easing in stops, as the shader takes it. */
const eased = (from: number, target: number, dt: number, rate: number) =>
  2 ** (Math.log2(from) + (Math.log2(target) - Math.log2(from)) * (1 - Math.exp(-dt * rate)));

it('takes the target at once, eases at its rise and fall rates and stays within its range', async () => {
  const { root, eye, adapt } = await setup();
  const { key, lowest, highest, rise, fall } = exposure.defaults;

  // Grey values exact in rgba16float, whose luminance is the grey itself
  const bright = key / 2;
  const dark = key / 0.0625;

  expect(await adapt(2, 0.5)).toBeCloseTo(bright, 4);

  const rising = eased(bright, dark, 0.5, rise);
  expect(await adapt(0.0625, 0.5)).toBeCloseTo(rising, 3);

  for (let i = 0; i < 30; i++) await adapt(0.0625, 0.5);
  expect(await adapt(0.0625, 0.5)).toBeCloseTo(dark, 2);

  expect(await adapt(2, 0.5)).toBeCloseTo(eased(dark, bright, 0.5, fall), 3);

  eye.reset();
  expect(await adapt(0.001, 0.5)).toBeCloseTo(highest, 2);
  eye.reset();
  expect(await adapt(64, 0.5)).toBeCloseTo(lowest, 4);

  root.destroy();
});

it('exposes the final pass as much as a manual exposure of the same value', async () => {
  const { root, screen, source, eye, adapt } = await setup();
  const options = { format: 'rgba8unorm', source, settings: { vignette: 0, dither: 0 } } as const;
  const adapted = finish.create(root, { ...options, exposure: eye });
  const manual = finish.create(root, {
    ...options,
    settings: { ...options.settings, exposure: 0.18 / 0.0625 },
  });
  await Promise.all([adapted.initAsync(), manual.initAsync()]);

  await adapt(0.0625, 0);
  const encoder = root.createCommandEncoder();
  adapted.record(encoder, screen.views.adapted);
  manual.record(encoder, screen.views.manual);
  encoder.submit();

  const [ours, theirs] = await Promise.all(
    [screen.textures.adapted, screen.textures.manual].map((texture) =>
      pixels(root, root.unwrap(texture)),
    ),
  );
  expect(ours).toEqual(theirs);

  root.destroy();
});

it('rejects settings that leave no range to adapt within', async () => {
  const root = await gpuRoot();
  const screen = targets.create(root, size, { hdr: { format: 'rgba16float', sampled: true } });
  const source = () => screen.textures.hdr;
  expect(() => exposure.create(root, { source, settings: { lowest: 0 } })).toThrow(RangeError);
  expect(() => exposure.create(root, { source, settings: { highest: 1 / 32 } })).toThrow(
    RangeError,
  );
  root.destroy();
});
