import { frames } from '@typegpu/kit';
import { offscreenSurface, recordingRoot } from '@typegpu/kit/testing';
import { expect, it } from 'vitest';
import { createFlowField } from '../src/flow-field/flow-field.ts';
import { createScene } from '../src/flow-field/scene.ts';
import { describeExample } from './utils/examples.ts';

/** A frame's events: its dispatches in one compute pass, then the draw. */
const recorded = (dispatches: number) => [
  'begin-compute',
  ...Array<string>(dispatches).fill('dispatch'),
  'end-compute',
  'begin-render',
  'draw',
  'end-render',
  'submit',
];

describeExample(
  'flow-field',
  createScene,
  (build) => {
    it('seeds in its first frame, then steps and draws', async () => {
      const { log, next } = await build();
      expect(log.shaders.find((code) => code.includes('@vertex'))).not.toContain('read_write');

      next();
      next();
      expect(log.events).toEqual([...recorded(1), ...recorded(2)]);
    });

    it('writes its settings uniform only when a setting changes, which its readout counts', async () => {
      const { log, scene, next } = await build();
      const writes = scene.settings.readouts!['settings writes'];
      next();
      log.writes.length = 0;
      next();
      next();
      expect(log.writes.map((write) => write.size)).toEqual([4, 4]); // `ahead`, every frame
      expect(writes(), 'the host writes once before the first frame').toBe(1);

      scene.settings.values.speed = 0.5;
      scene.settings.write!('speed');
      scene.settings.values.paused = true;
      scene.settings.write!('paused');
      expect(log.writes.map((write) => write.size)).toEqual([4, 4, 12, 12]);
      expect(writes()).toBe(3);
    });

    it('scatters the particles again in the frame after a reseed', async () => {
      const { log, scene, next } = await build();
      scene.settings.values.paused = true;
      scene.settings.write!('paused');
      next();

      scene.settings.actions!.reseed();
      log.events.length = 0;
      next(1);
      next(1);
      expect(log.events).toEqual([...recorded(1), ...recorded(0)]);
    });
  },
  { pipelines: 3 },
);

// The module on its own, as the Write a module tutorial tests it
it('records into the passes it is given, and a steady frame writes and creates nothing', async () => {
  const { root, log } = recordingRoot();
  const surface = offscreenSurface(root, { width: 64, height: 36, format: 'bgra8unorm' });
  const field = createFlowField(root, {
    count: 1000,
    step: 1 / 60,
    format: surface.format,
    size: surface,
  });
  await field.initAsync();
  expect(log.pipelines).toBe(3);

  let seeding = true;
  const loop = frames.create(root, (frame) => {
    const compute = frame.encoder.beginComputePass();
    if (seeding) field.seed(compute);
    seeding = false;
    field.step(compute);
    compute.end();
    const pass = frame.encoder.beginRenderPass({
      colorAttachments: { view: surface.view(frame) },
    });
    field.draw(pass);
    pass.end();
  });

  loop.run(0);
  loop.run(1 / 60);
  const counts = log.counts();
  log.events.length = 0;
  log.writes.length = 0;

  loop.run(2 / 60);
  expect(log.events).toEqual([
    'begin-compute',
    'dispatch',
    'end-compute',
    'begin-render',
    'draw',
    'end-render',
    'submit',
  ]);
  expect(log.writes).toEqual([]);
  expect(log.counts()).toEqual(counts);

  surface.setSize({ width: 128, height: 72 });
  loop.run(3 / 60);
  expect(log.writes.map((write) => write.size)).toEqual([8]); // the viewport

  field.destroy();
});
