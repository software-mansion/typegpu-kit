import { frames } from '@typegpu/kit';
import { offscreenSurface } from '@typegpu/kit/testing';
import { gpuRoot } from '@typegpu/kit/testing/dawn';
import { expect, it } from 'vitest';
import { createHome, type Drawn } from '../src/components/home/home.ts';
import { RULES, type RuleName } from '../src/components/home/rules.ts';
import { KIND, type Kind, type SheetState } from '../src/components/home/stack.ts';
import { MODULES } from '../src/components/home/subject.ts';

// 64 RGBA8 pixels make a 256-byte row, the alignment texture copies need.
const SIZE = 64;

/** A lockup whose every fourth pixel is ink. */
const read = (columns: number, rows: number) => {
  const data = new Uint8ClampedArray(columns * rows * 4);
  for (let i = 0; i < columns * rows; i += 4) data.set([120, 90, 220, 255], i * 4);
  return data;
};

/** Every kind of sheet, a sheet apart. */
const sheets: SheetState[] = (Object.keys(KIND) as Kind[]).map((kind, i) => ({
  kind,
  x: 0,
  y: -i * 0.7,
  z: 0,
  half: 2,
  hot: 0,
  shown: 1,
  keeps: 1,
}));

it('draws the frame at every stage of its build, and under every rule with the grid and without', async () => {
  const root = await gpuRoot({ features: ['indirect-first-instance'] });
  const surface = offscreenSurface(root, { width: SIZE, height: SIZE, format: 'rgba8unorm' });
  const home = createHome(root, { format: 'rgba8unorm', read });
  await home.initAsync();
  const { subject, ink } = home;
  const view = home.view(surface, () => 2);
  const count = view.write(
    { yaw: 0.8, pitch: 0.8, scale: 6, target: [0, 0, 0], anchor: [0.5, 0.5] },
    sheets,
    { base: 0.1, line: 0.6, thick: 0.035 },
  );
  ink.layout([4, 4, 40, 10]);
  const drawn: Drawn[] = [
    {
      view,
      count,
      before: (frame, dt) => ink.step(frame, dt, 1, surface, false),
      after: (pass) => ink.draw(pass),
    },
  ];
  const loop = frames.create(root, (frame) => home.record(frame, 1 / 60, drawn));
  let time = 0;
  const run = () => loop.run((time += 1 / 60));

  for (let built = 0; built <= MODULES.length; built++) {
    subject.built = built;
    run();
  }
  for (const rule of Object.keys(RULES) as RuleName[]) {
    subject.rule = rule;
    for (const out of [false, true]) {
      subject.remove('grid', out);
      run();
      run();
    }
  }
  await root.device.queue.onSubmittedWorkDone();

  // The stack drew something on the surface.
  const pixels = await (async () => {
    const buffer = root.device.createBuffer({
      size: SIZE * SIZE * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = root.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: surface.texture }, { buffer, bytesPerRow: SIZE * 4 }, [
      SIZE,
      SIZE,
    ]);
    root.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    return new Uint8Array(buffer.getMappedRange().slice(0));
  })();
  expect(pixels.some((value, i) => i % 4 === 3 && value > 0)).toBe(true);

  loop.destroy();
  home.destroy();
  surface.destroy();
  root.destroy();
});
