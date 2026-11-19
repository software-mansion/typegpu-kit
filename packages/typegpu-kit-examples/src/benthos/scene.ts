import {
  bloom,
  cameras,
  clock,
  finish,
  passes,
  pipelines,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createAbyss, WATER } from './abyss.ts';

const HDR = 'rgba16float';

export interface SceneOptions {
  /** Flecks of marine snow. Default 24 000. */
  readonly snow?: number;
}

/** Deep water with marine snow, bloom and a final pass. */
export async function createScene(
  root: TgpuRoot,
  surface: Surface,
  { snow = 24_000 }: SceneOptions = {},
) {
  const screen = targets.create(root, surface, { hdr: { format: HDR, sampled: true } });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0, 0],
    distance: 5,
    pitch: -0.35,
    pitchRange: [-1.3, 0.6],
    distanceRange: [1, 12],
    damping: 4,
  });
  const scene = passes.create(root, screen, { hdr: passes.color }).pipe(view.inject());

  const values = {
    ...WATER,
    exposure: 1.4,
    threshold: 0.6,
    glowIntensity: 0.2,
    spread: bloom.defaults.spread,
    vignette: 0.55,
    paused: false,
  };
  const glow = bloom.create(root, { source: () => screen.textures.hdr, settings: values });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: tone.aces,
    settings: values,
    glow,
  });

  const sim = clock.fixedStep({ step: 1 / 60, maxSteps: 3 });
  const time = root.createUniform(d.f32);
  const abyss = createAbyss(root, { camera: view, time, step: sim.step, count: snow });
  const drawn = [scene.add(abyss.water), scene.add(abyss.snow.drawing)];

  await pipelines.compile([scene, glow, final]);
  abyss.fill();

  const look = { folder: 'look' };
  const water = { folder: 'water' };
  const settings: Settings<typeof values> = {
    values,
    knobs: {
      paused: {},
      exposure: { ...look, min: 0.2, max: 4, step: 0.05 },
      threshold: { ...look, min: 0, max: 3, step: 0.05 },
      glowIntensity: { ...look, min: 0, max: 1, step: 0.01 },
      spread: { ...look, min: 0.25, max: 2, step: 0.05 },
      vignette: { ...look, min: 0, max: 1, step: 0.01 },
      shafts: { ...water, min: 0, max: 2, step: 0.01 },
      glints: { ...water, min: 0, max: 10, step: 0.1 },
      snow: { ...water, min: 0, max: 2, step: 0.01 },
      motes: { ...water, min: 0, max: 4, step: 0.05 },
      fog: { ...water, min: 0, max: 0.4, step: 0.005 },
      focus: { ...water, min: 1, max: 12, step: 0.1 },
      blur: { ...water, min: 0, max: 0.15, step: 0.005 },
      stir: { ...water, min: 0, max: 2, step: 0.05 },
    },

    write() {
      sim.paused = values.paused;
      final.settings.patch(values);
      glow.settings.patch(values);
      abyss.settings.write(values);
    },
  };

  const pointing: Pointing = { move: abyss.point };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      time.write(frame.time);
      view.update(frame.dt);
      abyss.update(frame.dt);
      screen.fit();

      const compute = frame.encoder.beginComputePass();
      abyss.snow.spawn(compute);
      for (let steps = sim.update(frame.dt); steps > 0; steps--) abyss.snow.step(compute);
      abyss.snow.compact(compute, sim.ahead);
      compute.end();

      scene.record(frame, drawn);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, glow, abyss]) made.destroy();
    },
  };
}
