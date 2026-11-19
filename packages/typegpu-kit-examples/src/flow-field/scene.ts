import { clock, type Frame, type Surface } from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { createFlowField, FLOW } from './flow-field.ts';

const PARTICLES = 65_536;
const BACKDROP = [0.01, 0.015, 0.03, 1] as const;

/** Runs the flow field at a fixed step, seeding it in the first frame and after each `reseed`. */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const sim = clock.fixedStep({ step: 1 / 60 });
  const ahead = root.createUniform(d.f32);
  const values = { ...FLOW, paused: false };
  const field = createFlowField(root, {
    count: PARTICLES,
    step: sim.step,
    ahead,
    format: surface.format,
    size: surface,
    settings: values,
  });
  await field.initAsync();

  let seeding = true;
  let writes = 0;

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      speed: { min: 0, max: 1, step: 0.01 },
      scale: { min: 0.1, max: 6, step: 0.1 },
      pointSize: { label: 'point size (px)', min: 0.5, max: 8, step: 0.5 },
      paused: {},
    },

    write() {
      sim.paused = values.paused;
      field.settings.write(values);
      writes += 1;
    },

    actions: {
      /** Schedules particle reseeding for the next frame. */
      reseed() {
        seeding = true;
      },
    },

    readouts: {
      /** Counts settings-uniform writes. Frames do not write this uniform. */
      'settings writes': () => writes,
    },
  };

  return {
    settings,

    frame(frame: Frame) {
      const compute = frame.encoder.beginComputePass();
      if (seeding) {
        field.seed(compute);
        seeding = false;
      }
      for (let steps = sim.update(frame.dt); steps > 0; steps--) field.step(compute);
      ahead.write(sim.ahead);
      compute.end();

      const pass = frame.encoder.beginRenderPass({
        colorAttachments: { view: surface.view(frame), clearValue: BACKDROP },
      });
      field.draw(pass);
      pass.end();
    },

    destroy() {
      field.destroy();
    },
  };
}
