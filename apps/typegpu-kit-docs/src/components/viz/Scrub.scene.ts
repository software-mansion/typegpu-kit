import { cameras, particles, passes, pipelines, type Frame, type Surface } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { randf } from '@typegpu/noise';
import { d, std, type TgpuRoot } from 'typegpu';

export const COUNT = 400;

/** Seconds the scrubber spans, from just before the blast until its last spark is gone. */
export const SPAN = [-0.2, 1.4] as const;

/** Black maps to the page's ground in the print grade. */
export const CLEAR = [0, 0, 0, 1] as const;

const quad = meshes.unindexed(meshes.plane({ width: 2, depth: 2 }));

/**
 * The scrubbed-effects recipe's sparks: each one's place `since` seconds after the blast, computed
 * from its index alone, and the drawing that draws them.
 */
export function createSparks(root: TgpuRoot) {
  const since = root.createUniform(d.f32);

  const sparkAt = (i: number) => {
    'use gpu';
    randf.seed(d.f32(i) + 1);
    const velocity = randf.inUnitSphere() * 5 + d.vec3f(0, 3, 0);
    const life = 0.6 + randf.sample() * 0.6;
    const age = since.$;
    const alive = age >= 0 && age < life;

    return particles.Particle({
      position: velocity * age + d.vec3f(0, -4.9, 0) * (age * age),
      age,
      velocity,
      life,
      color: d.vec4f(4, 1.6, 0.4, 1),
      size: std.select(0, 0.12, alive),
      seed: randf.sample(),
    });
  };

  const drawing: passes.Drawing<typeof particles.Corner, d.Infer<typeof particles.Seen>> = {
    label: 'replayed sparks',
    Vertex: particles.Corner,
    at: (vertexIndex: number, instanceIndex: number) => {
      'use gpu';
      const corner = quad.vertexAt(vertexIndex).position.xz;
      const world = particles.facing(sparkAt(instanceIndex), corner);
      const position = cameras.toClip(world);
      return particles.Corner({ position, world, local: corner, slot: instanceIndex });
    },
    pixelOf: (corner: d.Infer<typeof particles.Corner>) => {
      'use gpu';
      const color = particles.softDisc(sparkAt(corner.slot), corner.local);
      return particles.Seen({ world: corner.world, color });
    },
    color: passes.shaded,
    blend: pipelines.blend.add,
    depthStencil: { depthWriteEnabled: false, depthCompare: 'less' },
    draw: (pass) => pass.draw(quad.vertexCount, COUNT),
  };

  return { since, drawing };
}

/** One blast drawn at whatever moment the page sets, with nothing kept between frames. */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0.6, 0],
    distance: 8,
    pitch: 0.15,
  });
  const canvas = passes.create(root, surface, { clear: CLEAR }).pipe(view.inject());
  const { since, drawing } = createSparks(root);
  const drawn = canvas.add(drawing);
  await canvas.initAsync();

  let seconds = 0.3;
  let written: number | undefined;

  return {
    /** Seconds since the blast that the next frame draws. */
    get time() {
      return seconds;
    },
    set time(value: number) {
      seconds = value;
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      if (seconds !== written) {
        since.write(seconds);
        written = seconds;
      }
      canvas.record(frame, drawn);
    },
  };
}

export type Scrub = Awaited<ReturnType<typeof createScene>>;
