import { cameras, clock, passes, type Frame, type Surface, particles } from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import { flatPoint } from './stage.ts';

/** World units from the bottom of the view to the top. */
const HEIGHT = 10;

export const CAPACITY = 8192;

/** Black maps to the page's ground in the print grade. */
export const CLEAR = [0, 0, 0, 1] as const;

const COLORS = [
  [0.5, 0.66, 0.98, 1],
  [0.97, 0.71, 0.31, 1],
  [0.42, 0.82, 0.65, 1],
  [0.8, 0.65, 0.95, 1],
] as const;

export interface Forces {
  gravity: number;
  drag: number;
}

export interface Options {
  /** Default CAPACITY. */
  readonly capacity?: number;
}

/**
 * Sparks for the particles page: one system on a flat camera, drawn straight into the surface.
 * Counts what it queues the way `burst` trims, so the page can show where the ring's head is.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: Options = {}) {
  const { capacity = CAPACITY } = options;
  const view = cameras.flat(root, { size: surface, height: HEIGHT });
  const sim = clock.fixedStep({ step: 1 / 60, maxSteps: 4 });
  const forces: Forces = { gravity: -7, drag: 0.7 };
  const values = () => ({ gravity: d.vec3f(0, forces.gravity, 0), drag: forces.drag });
  const canvas = passes.create(root, surface, { clear: CLEAR });

  const sparks = particles.create(root, view, {
    capacity,
    step: sim.step,
    dimensions: 2,
    blend: 'add',
    bursts: 128,
    settings: values(),
  });
  const drawn = canvas.add(sparks.drawing);
  const writeForces = () => sparks.settings.write(values());

  const stats = { born: 0, queued: 0 };
  let pick = 0;
  let clearing = false;
  await canvas.initAsync();

  const burst = (x: number, y: number, count: number, scatter = 4.5) => {
    const color = COLORS[pick++ % COLORS.length];
    const accepted = sparks.burst({
      count,
      position: [x, y, 0],
      velocity: [0, 2.5, 0],
      scatter,
      color,
      size: 0.3,
      life: 1.4,
      variance: 0.35,
    });

    if (accepted) stats.queued += Math.min(count, capacity - stats.queued);
    return accepted;
  };

  /** When a stroke last burst, in milliseconds. */
  let stroked = 0;

  return {
    capacity,
    forces,
    stats,
    writeForces,
    burst,

    /** The half-width of the view, in world units. */
    get halfWidth() {
      return (HEIGHT * surface.width) / Math.max(surface.height, 1) / 2;
    },

    /** A tap bursts many sparks, and a stroke a few at most every 40 ms. */
    pointing: {
      press(u: number, v: number) {
        burst(...flatPoint(surface, HEIGHT, u, v), 160, 6);
      },

      drag(u: number, v: number, start: boolean) {
        const now = performance.now();
        if (!start && now - stroked < 40) return;
        stroked = now;
        burst(...flatPoint(surface, HEIGHT, u, v), 24, 2.5);
      },
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      if (clearing) {
        sparks.clear(frame);
        clearing = false;
      }

      const compute = frame.encoder.beginComputePass();
      sparks.spawn(compute);
      stats.born += stats.queued;
      stats.queued = 0;

      for (let steps = sim.update(frame.dt); steps > 0; steps--) sparks.step(compute);
      sparks.compact(compute, sim.ahead);
      compute.end();

      canvas.record(frame, drawn);
    },

    /** Ends every particle in the next frame's encoder. */
    clear() {
      clearing = true;
    },

    destroy() {
      sparks.destroy();
    },
  };
}

export type Sparks = Awaited<ReturnType<typeof createScene>>;
