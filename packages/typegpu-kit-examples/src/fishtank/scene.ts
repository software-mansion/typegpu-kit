import {
  bloom,
  cameras,
  clock,
  finish,
  lighting,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { random } from 'math/random';
import { d, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { createSchool } from './school.ts';
import { createFood } from './food.ts';
import { createWater, DEEP, TANK } from './water.ts';
import { createFloor } from './floor.ts';
import { createBubbles } from './bubbles.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
const CLEAR = [DEEP.x, DEEP.y, DEEP.z, 1] as const;
/** Tank centre, camera target and shadow centre. */
const MIDDLE = [0, TANK.y / 2, 0] as const;
/** Pellets per feed, and x and z spread from the tank centre in world units. */
const FEED = { pellets: 12, x: 5, z: 3 };

/** Pipeline warm-up options, including loading-cover progress. */
export type SceneOptions = pipelines.CompileOptions;

/**
 * Builds a fish tank with boids, food, sand, bubbles, bloom and a final pass. Frames compile
 * pipelines before drawing. `ready` settles when compilation finishes.
 */
export function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions = {}) {
  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: MIDDLE,
    distance: 17,
    pitch: 0.12,
    pitchRange: [-0.3, 1.2],
    distanceRange: [4, 30],
  });
  const sim = clock.fixedStep({ step: 1 / 60, maxSteps: 3 });
  const time = root.createUniform(d.f32);
  const ahead = root.createUniform(d.f32);
  const water = createWater(root, time);

  const light = lighting.create(root, {
    shading: water.shading,
    settings: {
      sunDirection: [0.3, 1, 0.2],
      sunIntensity: 1.2,
      skyTop: [0.3, 0.45, 0.5],
      skyBottom: [0.16, 0.15, 0.12],
    },
    shadow: { extent: 18, center: MIDDLE },
  });
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: CLEAR },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  /** The settings bloom and the final pass share, which the panel's look folder tunes. */
  const look = {
    exposure: 1.2,
    threshold: bloom.defaults.threshold,
    glowIntensity: finish.defaults.glowIntensity,
    vignette: finish.defaults.vignette,
  };
  const glow = bloom.create(root, { source: () => screen.textures.hdr, settings: look });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    settings: look,
    glow,
  });

  const food = createFood(root, {
    time,
    camera: view,
    water,
    step: sim.step,
    capacity: 64,
  });
  /** The boids' weights, which the panel tunes. */
  const steering = { separation: 0.9, alignment: 1.2, cohesion: 0.7, wander: 1.2, appetite: 6 };
  const school = createSchool(root, {
    time,
    // Fish extrapolate by `ahead`; particles extrapolate themselves
    step: sim.step,
    ahead,
    capacity: 4096,
    food,
    settings: steering,
    water,
  });
  const floor = createFloor(root, water);
  const { bubbles, vent } = createBubbles(root, view, { step: sim.step });

  // Only the fish cast shadows, and crumbs blend over the opaque drawings before them
  const drawn = [
    scene.add(floor.renderer),
    scene.add(school.renderer),
    scene.add(food.pellets),
    scene.add(food.crumbs),
    scene.add(bubbles.drawing),
  ];
  const casters = [light.shadow.add(school.renderer)];

  const values = { ...steering, paused: false, ...look };

  school.populate();

  let compiled = false;
  const ready = pipelines
    .compile([light, scene, glow, final, food, school, bubbles], options)
    .then(() => {
      compiled = true;
    });

  let now = 0;

  /** Drops `FEED.pellets` pellets below the surface. */
  const feed = () => {
    for (let i = 0; i < FEED.pellets; i++) {
      const x = random.float(Math.random, -FEED.x, FEED.x);
      const z = random.float(Math.random, -FEED.z, FEED.z);
      food.drop(x, z, now);
    }
  };

  const lens = { folder: 'look' };
  const settings: Settings<typeof values> = {
    values,
    knobs: {
      separation: { min: 0, max: 3, step: 0.05 },
      alignment: { min: 0, max: 3, step: 0.05 },
      cohesion: { min: 0, max: 3, step: 0.05 },
      wander: { min: 0, max: 4, step: 0.05 },
      appetite: { min: 0, max: 12, step: 0.1 },
      paused: {},
      exposure: { ...lens, min: 0.2, max: 4, step: 0.05 },
      threshold: { ...lens, min: 0, max: 3, step: 0.05 },
      glowIntensity: { ...lens, min: 0, max: 1, step: 0.01 },
      vignette: { ...lens, min: 0, max: 1, step: 0.01 },
    },

    write() {
      sim.paused = values.paused;
      school.settings.write(values);
      final.settings.patch(values);
      glow.settings.patch(values);
    },
  };

  return {
    view,
    settings,
    /** Settles when frames finish compiling pipelines and begin drawing. */
    ready,
    ui: { controls: { feed: { press: feed } } },

    get pellets() {
      return food.table.count;
    },

    frame(frame: Frame) {
      now = frame.time;
      if (!compiled) return;
      food.eat();

      time.write(frame.time);
      view.update(frame.dt);
      screen.fit();

      food.table.flush();
      if (!sim.paused) vent(frame.dt);
      school.flush();
      floor.flush();

      const compute = frame.encoder.beginComputePass();
      bubbles.spawn(compute);
      for (let steps = sim.update(frame.dt); steps > 0; steps--) {
        school.step(compute);
        food.step(compute);
        bubbles.step(compute);
      }
      ahead.write(sim.ahead);
      food.compact(compute, sim.ahead);
      bubbles.compact(compute, sim.ahead);
      compute.end();
      food.capture(frame);

      light.shadow.record(frame, casters);
      scene.record(frame, drawn);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, water, light, glow, food, school, floor, bubbles]) made.destroy();
    },
  };
}
