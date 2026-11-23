import {
  cameras,
  clock,
  finish,
  lighting,
  passes,
  pipelines,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import { createCenser, STEP } from './censer.ts';
import { createMarch } from './march.ts';
import { createModel } from './model.ts';
import type { Settings } from '../settings.ts';
import { createSmoke } from './smoke.ts';

/** The smoke's box around the swing: cells, their size and lowest corner. */
const CELLS = [112, 112, 88] as const;
const CELL = 0.04;
const CORNER = [-2.24, 0.24, -1.76] as const;
/** Radians the camera turns for a drag across the whole surface. */
const ORBIT = 3;

export interface SceneOptions {
  /** The smoke the vents breathe out, 1 by default. */
  readonly emission?: number;
}

export async function createScene(
  root: TgpuRoot,
  surface: Surface,
  { emission = 1 }: SceneOptions = {},
) {
  const screen = targets.create(root, surface, {
    hdr: { format: 'rgba16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 2.3, 0],
    distance: 4.6,
    yaw: 0,
    pitch: 0.05,
    fov: Math.PI / 4,
    pitchRange: [-0.3, 0.9],
    distanceRange: [2, 9],
    near: 0.05,
    far: 30,
  });
  const light = lighting.create(root, {
    settings: {
      sunDirection: [-1, 1.2, 0.35],
      sunColor: [1, 0.97, 0.92],
      sunIntensity: 2.4,
      skyTop: [0.3, 0.34, 0.4],
      // The embers and the lit smoke warm the body's underside
      skyBottom: [0.9, 0.6, 0.38],
      skyIntensity: 0.08,
      skyReflection: 0.5,
    },
  });
  const scene = passes
    .create(
      root,
      screen,
      { hdr: passes.color, depth: passes.depth },
      { clear: [0.003, 0.003, 0.0035, 1] },
    )
    .pipe(view.inject())
    .beneath(light.inject());
  const censer = createCenser(root);
  const model = await createModel(root, censer);
  model.update();
  const smoke = createSmoke(root, {
    cells: CELLS,
    origin: CORNER,
    cell: CELL,
    solid: censer.solid,
    source: censer.source,
    force: censer.force,
    carry: censer.carry,
    sponge: 3,
    settings: {
      dt: STEP,
      buoyancy: 2.5,
      weight: 0.02,
      vorticity: 16,
      smokeDecay: 0.15,
      heatDecay: 0.4,
      drag: 0.05,
    },
  });
  const air = createMarch(root, smoke, censer, {
    view,
    light,
    size: surface,
    picture: () => screen.textures.hdr,
    depth: () => screen.textures.depth,
  });
  const picture = finish.create(root, {
    format: surface.format,
    source: () => air.output,
    tone: tone.aces,
    settings: { exposure: 0.6, vignette: 0.25, dither: 0 },
  });
  const drawn = [scene.add(model.drawing)];
  await pipelines.compile([scene, smoke, air, picture]);

  // Half steps, so a frame at 120 Hz records one half of a smoke step and one at 60 Hz both. A
  // frame that misses the display records no catch-up on top of its own half
  const sim = clock.fixedStep({ step: STEP / 2, maxSteps: 4, steady: true });
  let advecting = true;
  let time = 0;
  // A frame that records only the second half moves the chain half a step on, so both frames at
  // 120 Hz carry half its cost, but only the first half shows the censer where it is
  let ahead = 0;

  // A stroke that starts on the censer takes hold of it; any other orbits the camera
  const ray = { origin: [0, 0, 0], direction: [0, 0, 0] };
  let holding = false;
  const last = [0, 0];

  const values = { emission };
  const settings: Settings<typeof values> = {
    values,
    knobs: { emission: { label: 'smoke', min: 0.25, max: 4, step: 0.05 } },
    write() {
      censer.emit(values.emission);
    },
  };

  return {
    view,
    settings,
    pointing: {
      drag(x: number, y: number, start: boolean) {
        view.rayAt(x, y, ray);
        if (start) holding = censer.grab(ray.origin, ray.direction);
        else if (holding) censer.pull(ray.origin, ray.direction);
        else cameras.turn(view.goal, x - last[0], y - last[1], { sensitivity: ORBIT });
        last[0] = x;
        last[1] = y;
      },
      end() {
        if (holding) censer.release();
        holding = false;
      },
    },
    frame(frame: Frame) {
      const halves = sim.update(frame.dt);
      if (halves > 0) {
        const pass = frame.encoder.beginComputePass();
        let moved = false;
        for (let i = 0; i < halves; i++) {
          if (advecting) {
            time += STEP;
            censer.swing(STEP - ahead, time);
            ahead = 0;
            pass.setBindGroup(censer.pose);
            smoke.advect(pass);
            moved = true;
          } else {
            if (halves === 1) {
              censer.chain.advance(STEP / 2);
              ahead = STEP / 2;
            }
            smoke.project(pass);
          }
          advecting = !advecting;
        }
        if (moved) {
          // The draw uses the chain's final pose, still bound here. The smoke's light also updates
          // only on advection
          model.update();
          air.light(pass);
        }
        pass.end();
      }
      screen.fit();
      view.update(frame.dt);
      scene.record(frame, drawn);
      air.record(frame);
      picture.record(frame, surface.view(frame));
    },
    destroy() {
      for (const made of [screen, light, model, smoke, air]) made.destroy();
    },
  };
}
