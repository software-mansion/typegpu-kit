import {
  bloom,
  cameras,
  clock,
  finish,
  lighting,
  meshPool,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import type { Model } from '../models.ts';
import type { Settings } from '../settings.ts';
import { createHorde, type ModelName } from './horde.ts';
import { createFire } from './fire.ts';
import { HORIZON, MOON, night } from './sky.ts';
import { createYard, GATE_ANGLE } from './yard.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
/** Yard radius per square root of skeleton count. Gives each skeleton about 3π m². */
const SPREAD = 1.75;
const GATES = 4;
/** Seconds in a step of the skeletons' machine and of the fire. */
const STEP = 1 / 60;

export interface SceneOptions {
  /** Each file as `gltf.read` returns it. */
  readonly models: Readonly<Record<ModelName, Model>>;
  /** Skeletons in the yard. Default 300. */
  readonly count?: number;
}

/**
 * A walled yard of armed skeletons on one rig, whose actions a machine on the GPU starts, under a
 * low moon and between braziers. The view starts over the yard, looking
 * from one gate to the opposite one.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const { models, count = 300 } = options;
  const radius = Math.sqrt(count) * SPREAD;

  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const near = radius * 0.3;
  const view = cameras.orbit(root, {
    size: surface,
    target: [near * Math.cos(GATE_ANGLE), 1, near * Math.sin(GATE_ANGLE)],
    distance: 25,
    yaw: GATE_ANGLE,
    pitch: 0.4,
    pitchRange: [0.04, 1.4],
    distanceRange: [4, radius * 3],
  });
  const pool = meshPool.create(root, { capacity: 64 });
  const sim = clock.fixedStep({ step: STEP });
  const flames = clock.fixedStep({ step: STEP });
  const yard = createYard(root, pool, { radius, gates: GATES });
  const light = lighting.create(root, {
    settings: {
      sunDirection: MOON,
      sunColor: [0.68, 0.78, 1],
      sunIntensity: 1.7,
      skyTop: [0.025, 0.04, 0.09],
      skyBottom: [0.02, 0.016, 0.014],
      fogColor: HORIZON,
      fogDensity: 0.012,
      fogFalloff: 0.1,
    },
    points: yard.points,
    // A device whose textures stop below the map's size gets the largest it has
    shadow: {
      extent: yard.reach * 2,
      size: Math.min(4096, root.device.limits.maxTextureDimension2D),
      filter: 9,
    },
  });

  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: [...HORIZON, 1] },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const skeletons = createHorde(root, pool, { models, count, radius, step: STEP });
  const fire = createFire(root, view, light, { rims: yard.rims, step: STEP });

  const renderer = skeletons.horde.renderer({ cull: true }).pipe(skeletons.poses.inject());
  const stone = yard.stone.renderer();
  const iron = yard.iron.renderer();
  const drawn = [
    scene.add(yard.ground.renderer()),
    scene.add(stone),
    scene.add(iron),
    scene.add(renderer),
    scene.add(night),
    ...fire.drawings.map((drawing) => scene.add(drawing)),
  ];
  const casters = [light.shadow.add(renderer), light.shadow.add(stone), light.shadow.add(iron)];

  const glow = bloom.create(root, {
    source: () => screen.textures.hdr,
    settings: { threshold: 1.2 },
  });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    glow,
    settings: { exposure: 1.3, glowIntensity: 0.12, vignette: 0.3 },
  });

  await pipelines.compile([light, scene, skeletons, fire, glow, final]);

  const values = {
    /** Horde time multiplier relative to page time. */
    speed: 1,
  };

  const settings: Settings<typeof values> = {
    values,
    knobs: { speed: { min: 0, max: 2, step: 0.05 } },
  };
  return {
    view,
    settings,

    frame(frame: Frame) {
      view.update(frame.dt);

      const steps = sim.update(frame.dt * values.speed);
      const burning = flames.update(frame.dt);
      yard.burn(frame.time);

      const compute = frame.encoder.beginComputePass();
      skeletons.animate(compute, sim.time + sim.ahead, steps);
      renderer.cull(compute);
      fire.record(compute, burning, flames.ahead);
      compute.end();

      screen.fit();
      light.shadow.record(frame, casters);
      scene.record(frame, drawn);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, pool, yard, light, skeletons, fire, glow]) made.destroy();
    },
  };
}
