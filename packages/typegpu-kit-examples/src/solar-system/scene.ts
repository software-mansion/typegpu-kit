import {
  type Baked,
  bloom,
  cameras,
  clock,
  finish,
  font,
  type Frame,
  passes,
  pipelines,
  type Surface,
  targets,
  tone,
} from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createBelt } from './belt.ts';
import { createPlanets } from './planets.ts';
import { createLabels } from './labels.ts';
import { createComets, STEP } from './comets.ts';
import { EARTH, PLANETS } from './data.ts';
import { space } from './sky.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
/** World units from the target to the eye on a wide view. Narrower views back away. */
const DISTANCE = 37;
/** Pitch above the ecliptic on wide screens, and on narrow ones, which look further down. */
const PITCH = { wide: 0.62, narrow: 1.1 };
/** Radians a second the view turns round the sun. */
const DRIFT = 0.025;
/**
 * Views narrower than `aspect` back away by the ratio to the power `power`, cropping the outer
 * orbits at their sides, and draw text larger by it to the power `growth`.
 */
const FIT = { aspect: 1.45, power: 0.75, growth: 0.4 };
/**
 * Until a planet is picked, a tour chooses Earth at `start` seconds, then each planet outwards in
 * turn for `stay` seconds.
 */
const TOUR = { start: 1.8, stay: 6 };

export interface SceneOptions {
  readonly font: Baked;
  /** Rocks in the belt between Mars and Jupiter. Default 4096. */
  readonly rocks?: number;
}

/**
 * A sun, eight planets wrapped in their facts, a belt of rocks and two comets, on kit instances,
 * drawings, lines, text and trails.
 */
export async function createScene(
  root: TgpuRoot,
  surface: Surface,
  { font: baked, rocks = 4096 }: SceneOptions,
) {
  const fit = () => Math.max(1, (FIT.aspect * surface.height) / surface.width);
  const narrow = () => surface.width < surface.height;
  const pitch = () => (narrow() ? PITCH.narrow : PITCH.wide);
  let fitted = fit();
  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, -1.6, 0],
    yaw: 0.35,
    fov: 0.5,
    distance: DISTANCE * fitted ** FIT.power,
    pitch: pitch(),
    near: 1,
    far: 1200,
    pitchRange: [0.08, 1.5],
    distanceRange: [6, 140],
  });
  const sim = clock.fixedStep({ step: STEP });
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' } },
    )
    .pipe(view.inject());

  const values = { speed: 1, tailWidth: 0.16, paused: false };
  const sans = font.create(root, baked);
  const planets = createPlanets(root, view, surface);
  const labels = createLabels(root, sans, planets);
  const belt = createBelt(root, { count: rocks });
  const comets = createComets(root, { tailWidth: values.tailWidth });

  const drawn = [
    scene.add(planets.renderer),
    scene.add(space),
    scene.add(belt.drawing),
    scene.add(planets.paths),
    scene.add(comets.tails.drawing),
    scene.add(labels.drawing),
  ];

  const glow = bloom.create(root, {
    source: () => screen.textures.hdr,
    settings: { threshold: 2, spread: 1.1 },
  });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: tone.aces,
    glow,
    settings: { exposure: 1.05, vignette: 0.35, glowIntensity: 0.2 },
  });
  await pipelines.compile([scene, planets, comets, glow, final]);

  let picked: number | undefined;
  let pointer: [x: number, y: number] | undefined;

  const pointing: Pointing = {
    move(x, y) {
      pointer = [x, y];
    },
  };

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      speed: { label: 'orbit speed', min: 0, max: 4, step: 0.05 },
      tailWidth: { label: 'tail width', min: 0.02, max: 0.5, step: 0.01 },
      paused: {},
    },

    write() {
      comets.tails.settings.patch({ width: values.tailWidth });
      sim.paused = values.paused;
    },
  };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      const fitting = fit();
      if (fitting !== fitted) {
        view.goal.distance *= (fitting / fitted) ** FIT.power;
        view.goal.pitch = pitch();
        fitted = fitting;
      }
      picked = planets.picks.result.take() ?? picked;
      const stop = Math.floor((frame.time - TOUR.start) / TOUR.stay);
      const touring = stop >= 0 ? (EARTH + stop) % PLANETS.length : undefined;
      const chosen = picked ?? touring;
      labels.choose(chosen, frame.time);

      view.goal.yaw += DRIFT * frame.dt;
      view.update(frame.dt);
      screen.fit();

      const steps = sim.update(frame.dt * values.speed);
      const seconds = sim.time + sim.ahead;
      const grow = fitted ** FIT.growth;
      labels.tick(seconds, frame.time, grow);
      const width = chosen === undefined ? 0 : labels.widths[chosen];
      planets.focusOn(chosen, { width, grow });
      belt.move(seconds);

      const compute = frame.encoder.beginComputePass();
      for (let step = steps; step > 0; step--) comets.step(compute);
      planets.move(compute, seconds);
      if (pointer) planets.picks.pick(compute, ...pointer);
      compute.end();
      if (pointer) planets.picks.result.capture(frame);
      pointer = undefined;

      scene.record(frame, drawn);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, sans, planets, labels, comets, glow]) made.destroy();
    },
  };
}
