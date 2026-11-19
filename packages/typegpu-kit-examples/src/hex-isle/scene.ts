import {
  bloom,
  cameras,
  dof,
  finish,
  lighting,
  passes,
  pipelines,
  table,
  targets,
  type Frame,
  type meshPool,
  type Surface,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import type { ModelName } from './island.ts';
import { createIsle } from './isle.ts';
import { createDaylight, daylight } from './sky.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
/** Hours in a day, where the hour wraps. */
const DAY = 24;

/** The band a miniature keeps sharp, in picture heights on each side of its middle. */
const BAND = 0.07;
/** How fast the blur grows away from the band, in picture heights per picture height. */
const TILT = 0.06;
/** The middle of the band, a little below the island's centre, where the land is nearest. */
const MIDDLE = 0.54;

type Focus = 'miniature' | 'pointer' | 'off';

const Lens = d.struct({
  /** Where the pointer focuses, from 0 to 1 over the picture. */
  pointer: d.vec2f,
  /** 1 to focus under the pointer, 0 for a miniature's band. */
  pointing: d.u32,
});

/**
 * Keeps a band across the island sharp and blurs above and below it, as a tilted lens does, so
 * the island reads as a model on a table. Below the band is nearer the eye, so it blurs in front.
 */
const miniature = (uv: d.v2f) => {
  'use gpu';
  const away = uv.y - MIDDLE;
  return std.sign(-away) * std.max(std.abs(away) - BAND, 0) * TILT;
};

export interface SceneOptions {
  /** Each model as `gltf.read` returns it. */
  readonly models: Readonly<Record<ModelName, meshPool.ModelData>>;
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
}

export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const screen = targets.create(root, surface, {
    color: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const sky = createDaylight();
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0.3, 0],
    distance: 24,
    yaw: 0.35,
    pitch: 0.55,
    pitchRange: [0.1, 1.4],
    distanceRange: [6, 40],
  });
  const time = root.createUniform(d.f32);
  const houses = table.create(root, lighting.PointLight, {
    capacity: 16,
    defaults: { ...lighting.WHITE },
  });
  const light = lighting.create(root, { points: { table: houses }, shadow: { extent: 26 } });
  // `daylight` updates the horizon clear colour in place
  const scene = passes
    .create(
      root,
      screen,
      { color: passes.color, depth: passes.depth },
      { resolve: { color: 'hdr' }, clear: sky.horizon },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const isle = createIsle(root, houses, time, options.models);
  const shown = scene.add(isle.renderer);
  const casters = [light.shadow.add(isle.renderer)];

  /** The isle's depth again at half resolution, as the scene's depth is multisampled. */
  const early = targets.create(root, surface, {
    depth: { format: 'depth32float', scale: 0.5, sampled: true },
  });
  const prepass = passes
    .create(root, early, { depth: passes.depth }, { label: 'prepass' })
    .pipe(view.inject());
  const shapes = [prepass.add(isle.renderer)];

  const lens = root.createUniform(Lens, { pointer: d.vec2f(0.5), pointing: 0 });
  const focusing = (distance: number, uv: d.v2f) => {
    'use gpu';
    if (lens.$.pointing === 1) return dof.thinLens(distance, dof.distanceAt(lens.$.pointer));
    return miniature(uv);
  };
  const blur = dof.create(root, view, {
    source: () => screen.textures.hdr,
    depth: () => early.textures.depth,
    radius: focusing,
  });

  const values = {
    hour: 16.5,
    speed: 0.3,
    paused: false,
    /** Blurs above and below a band of the island, focuses under the pointer, or keeps it sharp. */
    focus: 'miniature' as Focus,
    /** The lens opening for focusing under the pointer, in world units. */
    aperture: 1.5,
    exposure: finish.defaults.exposure,
    threshold: bloom.defaults.threshold,
    glowIntensity: finish.defaults.glowIntensity,
    vignette: finish.defaults.vignette,
  };
  // Bloom and the final pass read the depth of field's output while it is on
  const picture = () => (values.focus === 'off' ? screen.textures.hdr : blur.output);
  const glow = bloom.create(root, { source: picture });
  const final = finish.create(root, { format: surface.format, source: picture, glow });

  const writeDay = () => {
    daylight(values.hour, sky);
    light.settings.patch(sky);
    isle.lightUp(sky.lamps);
  };
  writeDay();

  const writeLens = () => {
    lens.patch({ pointing: values.focus === 'pointer' ? 1 : 0 });
    blur.settings.patch({ aperture: values.aperture });
  };
  writeLens();

  const looks = { folder: 'look' };
  const settings: Settings<typeof values> = {
    values,
    knobs: {
      hour: { min: 0, max: DAY, step: 0.01, live: true },
      speed: { label: 'hours per second', min: 0, max: 2, step: 0.05 },
      paused: {},
      focus: { label: 'depth of field', options: ['miniature', 'pointer', 'off'] },
      aperture: { label: 'aperture under the pointer', min: 0, max: 4, step: 0.05 },
      exposure: { ...looks, min: 0.2, max: 4, step: 0.05 },
      threshold: { ...looks, min: 0, max: 3, step: 0.05 },
      glowIntensity: { ...looks, min: 0, max: 1, step: 0.01 },
      vignette: { ...looks, min: 0, max: 1, step: 0.01 },
    },

    write() {
      writeDay();
      writeLens();
      final.settings.patch(values);
      glow.settings.patch(values);
    },
  };

  /** Hovering focuses on what is under the pointer, in picture coordinates. */
  const pointing = {
    move(x, y) {
      lens.patch({ pointer: d.vec2f(x, y) });
    },
  } satisfies Pointing;

  await Promise.all([
    isle.ready,
    pipelines.compile([light, scene, prepass, blur, glow, final], options),
  ]);

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      time.write(frame.time);
      view.update(frame.dt);

      const hours = values.paused ? 0 : frame.dt * values.speed;
      if (hours > 0) {
        values.hour = (values.hour + hours) % DAY;
        writeDay();
      }

      isle.pieces.flush();
      houses.flush();
      screen.fit();
      light.shadow.record(frame, casters);

      const blurring = values.focus !== 'off';
      if (blurring) {
        early.fit();
        prepass.record(frame, shapes);
      }
      scene.record(frame, shown);
      if (blurring) blur.record(frame);

      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, isle, houses, light, early, blur, glow]) made.destroy();
    },
  };
}
