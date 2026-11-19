import {
  bloom,
  cameras,
  environment,
  finish,
  lighting,
  meshPool,
  passes,
  pipelines,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { createArcade, ROOM, type Part } from './arcade.ts';
import { createAtmosphere } from './atmosphere.ts';
import { createFeedback } from './feedback.ts';
import { createSignal, PRECISION } from './signal.ts';

/** A warm haze low down, cooling to blue overhead, for `environment.background`. */
const sky = (look: d.v3f) => {
  'use gpu';
  const up = std.saturate(look.y * 2);
  return std.mix(d.vec3f(0.82, 0.67, 0.47), d.vec3f(0.29, 0.38, 0.43), up) * 1.6;
};

export interface SceneOptions {
  /** The hall and the pendulum, as `gltf.read` returns their parts. */
  readonly model: Readonly<Record<Part, meshPool.ModelData>>;
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
}

/** How close the eye comes to the hall's walls, beams and floor, in metres. */
const MARGIN = 0.45;
const LOOK = { exposure: 0.9, vignette: 0.28, dither: 0 };

/** How far a ray from `from` may go `along` an axis and stay the margin inside `low` and `high`. */
const across = (from: number, along: number, [low, high]: readonly [number, number]) =>
  along === 0 ? Infinity : ((along > 0 ? high - MARGIN : low + MARGIN) - from) / along;
/** How far the eye may back away from `target` along a yaw and pitch and stay in the room. */
const reach = (target: ArrayLike<number>, yaw: number, pitch: number) =>
  Math.min(
    across(target[0], Math.cos(pitch) * Math.sin(yaw), ROOM.x),
    across(target[1], Math.sin(pitch), ROOM.y),
    across(target[2], Math.cos(pitch) * Math.cos(yaw), ROOM.z),
  );

export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const values = {
    compare: 'processed' as 'processed' | 'clean',
    lines: 320,
    feedback: false,
    persistence: 0.05,
    bloom: true,
    shafts: true,
    quantize: true,
    swing: true,
  };
  const size = targets.scaled(surface, () =>
    values.compare === 'clean' ? 1 : Math.min(1, values.lines / surface.height),
  );
  const screen = targets.create(root, size, {
    hdr: { format: 'rgba16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 3.4, -1],
    distance: 12,
    yaw: 0.3,
    pitch: 0.06,
    fov: Math.PI / 3.2,
    pitchRange: [-0.05, 0.75],
    distanceRange: [4, 14],
    reach,
    near: 0.1,
    far: 90,
  });
  const light = lighting.create(root, {
    settings: {
      sunDirection: [-1, 0.6, -0.4],
      sunColor: [1, 0.78, 0.5],
      sunIntensity: 3.4,
      skyTop: [0.31, 0.39, 0.43],
      skyBottom: [0.17, 0.125, 0.075],
      skyIntensity: 0.34,
      skyReflection: 1.6,
      fogColor: [0.14, 0.105, 0.065],
      fogDensity: 0.009,
      fogFalloff: 0.07,
      fogStart: 3,
    },
    shadow: { center: [-3, 4, -2.5], extent: 44, size: 2048, filter: 4, lift: 0.5 },
  });
  const scene = passes
    .create(root, screen, { hdr: passes.color, depth: passes.depth })
    .pipe(view.inject())
    .beneath(light.inject());
  const arcade = createArcade(root, options.model);
  const renderer = arcade.set.renderer();
  const drawn = [scene.add(renderer), scene.add(environment.background(sky))];
  const cast = [light.shadow.add(renderer)];
  const air = createAtmosphere(root, view, light, {
    size,
    source: () => screen.textures.hdr,
    depth: () => screen.textures.depth,
  });
  const picture = () => (values.shafts ? air.screen.textures.hdr : screen.textures.hdr);
  const glow = bloom.create(root, {
    source: picture,
    levels: 5,
    settings: { threshold: 0.9, spread: 1.1 },
  });
  const echo = createFeedback(root, size, glow, { source: picture });
  const signal = createSignal(root, size);
  const processed = finish.create(root, {
    format: surface.format,
    source: echo,
    tone: tone.aces,
    filter: 'sharp',
    grade: signal.grade,
    settings: LOOK,
  });
  const clean = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: tone.aces,
    settings: LOOK,
  });
  await Promise.all([
    arcade.ready,
    pipelines.compile([light, scene, air, glow, echo, processed, clean], options),
  ]);

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      compare: { options: ['processed', 'clean'] },
      lines: { label: 'render height', options: [240, 320, 480] },
      feedback: { label: 'image feedback' },
      persistence: { label: 'trail half-life (s)', min: 0.02, max: 0.5, step: 0.01 },
      bloom: {},
      shafts: { label: 'sun shafts' },
      quantize: { label: '5-bit colour' },
      swing: { label: 'swing pendulum' },
    },
    write(key) {
      signal.precision.write(values.quantize ? PRECISION : 0);
      if (key !== 'persistence' && key !== 'swing') echo.reset();
    },
    actions: { 'clear afterimage': echo.reset },
  };

  let time = 0;
  // The goal as the last frame left it, to tell a cut from a drag
  const last = { yaw: view.goal.yaw, pitch: view.goal.pitch, distance: view.goal.distance };
  const trail = { halfLife: 0, bloom: 0, vertical: 0.45 };

  return {
    view,
    settings,
    frame(frame: Frame) {
      if (values.swing) time += frame.dt;
      arcade.swing(Math.sin(time * 1.13 + 0.4) * 0.28);
      arcade.set.flush();

      const { goal } = view;
      const cut =
        Math.abs(goal.yaw - last.yaw) + Math.abs(goal.pitch - last.pitch) > 0.3 ||
        Math.abs(goal.distance - last.distance) > 2;
      if (cut) echo.reset();
      last.yaw = goal.yaw;
      last.pitch = goal.pitch;
      last.distance = goal.distance;
      screen.fit();
      view.update(frame.dt);
      light.shadow.record(frame, cast);
      scene.record(frame, drawn);
      if (values.compare === 'clean') {
        clean.record(frame, surface.view(frame));
        return;
      }
      if (values.shafts) air.record(frame);
      if (values.bloom) glow.record(frame);
      signal.fit();
      trail.halfLife = values.feedback ? values.persistence : 0;
      trail.bloom = values.bloom ? 0.22 : 0;
      echo.record(frame, trail);
      processed.record(frame, surface.view(frame));
    },
    destroy() {
      for (const made of [screen, light, arcade, air, glow, echo]) made.destroy();
    },
  };
}
