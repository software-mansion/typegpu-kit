import {
  bloom,
  cameras,
  finish,
  instances,
  lighting,
  meshPool,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { d, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { aerial, sky, SUN } from './air.ts';
import { createGround } from './ground.ts';
import { addShapes, drift, PARTS, plant, RADIUS, Tree } from './trees.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
/** Metres across the ground, past the forest's edge into the haze. */
const PLAIN = RADIUS * 4;
const TREES = 20_000;
/** Noise cells a second the gusts travel downwind. */
const GUSTS = 0.4;

export interface SceneOptions {
  /** Trees in the forest. Default 20 000. */
  readonly trees?: number;
}

/**
 * Draws swaying trees across a plain. GPU culling limits drawings to the camera view and shadow
 * casters to the sun view.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions = {}) {
  const { trees = TREES } = options;
  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 4, 0],
    distance: 30,
    yaw: 0.4,
    pitch: 0.24,
    pitchRange: [0.06, 1.45],
    distanceRange: [8, 160],
  });
  const light = lighting.create(root, {
    settings: {
      sunDirection: SUN,
      sunColor: [1, 0.66, 0.4],
      sunIntensity: 3.2,
      skyTop: [0.26, 0.38, 0.6],
      skyBottom: [0.09, 0.08, 0.05],
      skyIntensity: 0.85,
      fogDensity: 0.008,
      fogFalloff: 0.05,
    },
    shading: { fog: aerial },
    shadow: { extent: 80 },
  });
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' } },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const pool = meshPool.create(root, { capacity: 4 });
  const ground = createGround(root, pool, { size: PLAIN });
  const shapes = addShapes(pool);
  const things = instances.create(root, pool, { kind: Tree, capacity: trees * PARTS });
  plant(things, shapes, trees);

  // How far the gusts have travelled, read through `drift`
  const blown = root.createUniform(d.f32);
  const gusts: Inject = (cfg) => cfg.with(drift, blown);

  const values = { cull: true, wind: 1 };
  const culling = things.renderer({ cull: true }).pipe(gusts);
  const plain = things.renderer().pipe(gusts);
  // The sky fills what the ground and the trees leave
  const floor = scene.add(ground.renderer());
  const skyDrawn = scene.add(sky);
  const culled = {
    shown: [floor, scene.add(culling), skyDrawn],
    cast: [light.shadow.add(culling)],
  };
  const unculled = { shown: [floor, scene.add(plain), skyDrawn], cast: [light.shadow.add(plain)] };

  const glow = bloom.create(root, { source: () => screen.textures.hdr });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    glow,
    settings: { exposure: 1.15 },
  });
  await pipelines.compile([light, scene, glow, final]);

  const settings: Settings<typeof values> = {
    values,
    knobs: { cull: {}, wind: { min: 0, max: 3, step: 0.01 } },
  };
  let travelled = 0;

  return {
    view,
    settings,

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();

      if (values.wind !== 0) {
        travelled += frame.dt * values.wind * GUSTS;
        blown.write(travelled);
      }

      things.flush();
      const drawn = values.cull ? culled : unculled;
      if (values.cull) {
        const pass = frame.encoder.beginComputePass();
        culling.cull(pass);
        pass.end();
      }
      light.shadow.record(frame, drawn.cast);

      scene.record(frame, drawn.shown);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, light, pool, ground, things, glow]) made.destroy();
    },
  };
}
