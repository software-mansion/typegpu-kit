import { cameras, clock, passes, particles, trails } from '@typegpu/kit';
import type { Fixtures } from './harness.ts';

const size = { width: 64, height: 36 };

export const fixtures: Fixtures = {
  'kit/particles': {
    setup: (root) => {
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      const sim = clock.fixedStep({ step: 1 / 60 });
      return {
        create: () => {
          const sparks = particles.create(root, view, { capacity: 64, step: sim.step });
          return {
            // No draw: the pipeline a description builds of `drawing` is the description's.
            frame: (frame) => {
              sparks.burst({ count: 8, position: [0, 0, 0], scatter: 1 });
              const pass = frame.encoder.beginComputePass();
              sparks.spawn(pass);
              const steps = sim.update(frame.dt);
              for (let i = 0; i < steps; i++) sparks.step(pass);
              sparks.compact(pass, sim.ahead);
              pass.end();
            },
            destroy: () => sparks.destroy(),
          };
        },
      };
    },
  },

  'kit/trails': {
    setup: (root, surface) => {
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      return {
        create: async (others) => {
          // The description holds the viewport the lines read.
          const canvas = others(() =>
            passes.create(root, surface, { clear: [0, 0, 0, 1] }).pipe(view.inject()),
          );
          const wakes = trails.create(root, { capacity: 4, length: 8 });
          const drawn = canvas.add(wakes.drawing);
          await others(() => drawn.initAsync());
          return {
            frame: (frame) => {
              wakes.write(1, [0, 0, 0, 1, 1, 0]);
              const pass = canvas.begin(frame);
              drawn.draw(pass);
              pass.end();
            },
            destroy: () => {
              wakes.destroy();
              canvas.destroy();
            },
          };
        },
      };
    },
  },
};
