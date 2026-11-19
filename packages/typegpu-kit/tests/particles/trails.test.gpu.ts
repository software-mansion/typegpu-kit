import { cameras, passes, targets, type CameraBase } from '@typegpu/kit';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import { expect, it } from 'vitest';
import { trails } from '../../src/index.ts';

const SIZE = 64;

/** An 8-bit texture's pixels, four bytes each, in rows of a multiple of 256 bytes. */
async function readPixels(root: TgpuRoot, texture: GPUTexture) {
  const readback = root.device.createBuffer({
    size: SIZE * 4 * SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = root.device.createCommandEncoder();
  copy.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow: SIZE * 4 }, [SIZE, SIZE]);
  root.device.queue.submit([copy.finish()]);
  await readback.mapAsync(GPUMapMode.READ);

  return new Uint8Array(readback.getMappedRange().slice(0));
}

const white = (_point: trails.Sample, _across: number) => {
  'use gpu';
  return d.vec4f(1);
};

const unit = (_point: trails.Sample) => {
  'use gpu';
  return d.f32(1);
};

/** A 64 by 64 target per device, and passes that draw into it. */
const targetsOf = new WeakMap<TgpuRoot, ReturnType<typeof described>>();
const described = (root: TgpuRoot) => {
  const screen = targets.create(
    root,
    { width: SIZE, height: SIZE },
    { color: { format: 'rgba8unorm' } },
  );
  return { screen, into: passes.create(root, screen, { color: passes.color }) };
};
const intoOf = (root: TgpuRoot) => {
  const made = targetsOf.get(root) ?? described(root);
  targetsOf.set(root, made);
  return made;
};

/** Draws `wakes` alone through `view` into a cleared 64 by 64 target and reads it back. */
async function drawn(root: TgpuRoot, wakes: trails.Trails, view: CameraBase) {
  const { screen, into } = intoOf(root);
  const seen = into.pipe(view.inject());
  const encoder = root.createCommandEncoder();
  const ribbons = seen.add(wakes.drawing);
  const pass = seen.begin(encoder);
  ribbons.draw(pass);
  pass.end();
  encoder.submit();

  const pixels = await readPixels(root, root.unwrap(screen.textures.color));
  const lit = (x: number, y: number) => pixels[(y * SIZE + x) * 4] > 128;
  const covered = pixels.filter((value, i) => i % 4 === 0 && value > 128).length;

  return { lit, covered };
}

const ribbon = (root: TgpuRoot, length: number) =>
  trails.create(root, {
    capacity: 2,
    length,
    width: unit,
    shade: white,
  });

it('draws a written trail a unit wide along its points, and nothing past its round ends', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: SIZE, height: SIZE }, height: 10 });
  view.update(0);
  const wakes = ribbon(root, 4);
  wakes.write(1, [-3, 0, 0, 0, 0, 0, 3, 0, 0]);

  // Ten world units over 64 pixels: row 32 is 0.08 below the centre, row 26 is 0.86 above it,
  // and columns 2 and 61 are 1.1 units past the ends.
  const { lit, covered } = await drawn(root, wakes, view);
  for (const x of [16, 24, 32, 40, 48]) expect(lit(x, 32)).toBe(true);
  for (const x of [16, 32, 48]) expect(lit(x, 26)).toBe(false);
  expect(lit(2, 32)).toBe(false);
  expect(lit(61, 32)).toBe(false);

  // 6 units by 1 and two half discs: about 278 pixels.
  expect(covered).toBeGreaterThan(230);
  expect(covered).toBeLessThan(330);
  root.destroy();
});

it('keeps the newest `length` points a kernel pushes, newest first, after the written ones', async () => {
  const root = await gpuRoot();
  const wakes = ribbon(root, 4);
  const step = root.createUniform(d.f32);
  const probe = root.createBuffer(d.arrayOf(d.vec4f, 5)).$usage('storage');
  const results = probe.as('mutable');

  const pushOne = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    wakes.push(1, d.vec3f(step.$, 0, 0));
    wakes.push(7, d.vec3f(step.$, 0, 0));
  });
  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    results.$[4] = d.vec4f(d.f32(wakes.held(0)), d.f32(wakes.held(1)), 0, 0);
    for (let k = d.u32(0); k < wakes.held(1); k++) {
      results.$[k] = d.vec4f(wakes.pointAt(1, k), 0);
    }
  });
  const injected = root.pipe(wakes.inject());
  const pushing = injected.createComputePipeline({ compute: pushOne });
  const reading = injected.createComputePipeline({ compute: read });

  wakes.write(1, [-1, 0, 0, -0.5, 0, 0]);
  for (let i = 0; i < 4; i++) {
    step.write(i + 1);
    const encoder = root.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pushing);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.submit();
  }

  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(reading);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();

  const [k0, k1, k2, k3, counts] = await probe.read();
  expect([counts!.x, counts!.y]).toEqual([0, 4]);
  expect([k0!.x, k1!.x, k2!.x, k3!.x]).toEqual([4, 3, 2, 1]);
  root.destroy();
});

it('empties one trail from a kernel, so a respawn starts a new line where it lands', async () => {
  const root = await gpuRoot();
  const wakes = ribbon(root, 4);
  const probe = root.createBuffer(d.arrayOf(d.vec4f, 2)).$usage('storage');
  const results = probe.as('mutable');

  const respawn = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    wakes.reset(1);
    wakes.push(1, d.vec3f(9, 0, 0));
  });
  const read = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    results.$[0] = d.vec4f(d.f32(wakes.held(0)), d.f32(wakes.held(1)), 0, 0);
    results.$[1] = d.vec4f(wakes.pointAt(1, 0), 0);
  });
  const injected = root.pipe(wakes.inject());

  wakes.write(0, [1, 0, 0, 2, 0, 0]);
  wakes.write(1, [-1, 0, 0, -0.5, 0, 0, 0, 0, 0]);
  const encoder = root.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(injected.createComputePipeline({ compute: respawn }));
  pass.dispatchWorkgroups(1);
  pass.setPipeline(injected.createComputePipeline({ compute: read }));
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.submit();

  const [counts, newest] = await probe.read();
  expect([counts!.x, counts!.y]).toEqual([2, 1]);
  expect(newest!.x).toBe(9);
  root.destroy();
});

it('draws what lies before the eye of a trail that passes behind it', async () => {
  const root = await gpuRoot();
  const view = cameras.orbit(root, {
    size: { width: SIZE, height: SIZE },
    target: [0, 0, 0],
    distance: 5,
    pitch: 0,
  });
  view.update(0);
  const wakes = ribbon(root, 3);

  // The eye is at z = 5; the newest point is behind it. The segment to it is cut at the near
  // plane, a unit wide a tenth of a unit from the eye, and widens off the right of the screen
  wakes.write(0, [-2, 0, 0, 2, 0, 0, 0, 0, 8]);

  const { lit, covered } = await drawn(root, wakes, view);
  expect(lit(32, 32)).toBe(true);
  expect(lit(63, 32)).toBe(true);
  expect(lit(32, 4)).toBe(false);
  expect(lit(0, 0)).toBe(false);
  expect(covered).toBeLessThan(SIZE * SIZE * 0.5);
  root.destroy();
});

it('tapers and fades towards the oldest point by default', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: SIZE, height: SIZE }, height: 10 });
  view.update(0);
  const wakes = trails.create(root, {
    capacity: 1,
    length: 2,
    settings: { width: 2 },
  });
  wakes.write(0, [-4, 0, 0, 4, 0, 0]);

  // Column 56 is 0.02 of the way from the newest point to the oldest, column 8 is 0.96.
  const { lit } = await drawn(root, wakes, view);
  expect(lit(56, 32)).toBe(true);
  expect(lit(56, 28)).toBe(true);
  expect(lit(8, 32)).toBe(false);
  expect(lit(32, 26)).toBe(false);
  root.destroy();
});

it('draws the newest end where `head` puts it, in place of the newest point', async () => {
  const root = await gpuRoot();
  const view = cameras.flat(root, { size: { width: SIZE, height: SIZE }, height: 10 });
  view.update(0);
  const wakes = trails.create(root, {
    capacity: 1,
    length: 3,
    width: unit,
    shade: white,
    head: () => {
      'use gpu';
      return d.vec3f(3, 0, 0);
    },
  });
  wakes.write(0, [-3, 0, 0, 0, 5, 0]);

  // The line runs from x = -3 to x = 3 along row 32, and not up to the newest point at y = 5,
  // which column 27 of row 8 would cross.
  const { lit } = await drawn(root, wakes, view);
  for (const x of [16, 32, 48]) expect(lit(x, 32)).toBe(true);
  expect(lit(27, 8)).toBe(false);
  root.destroy();
});
