import {
  bloom,
  cameras,
  clock,
  coroutines,
  lighting,
  meshPool,
  ao,
  particles,
  passes,
  pipelines,
  finish,
  shading,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import type { Picture, Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createDominoes, type Layout } from './dominoes.ts';
import { createTopple, puff, STEP } from './topple.ts';
import { createFloor } from './floor.ts';
import { createPicture } from './picture.ts';
import { MIDDLE } from './field.ts';

/** Linear colour of the dim room. The floor fades into it. */
export const ROOM = [0.07, 0.05, 0.035] as const;

/** Direction toward a sun `degrees` above the floor, to the left and behind the run. */
export const sunAt = (degrees: number): [number, number, number] => {
  const up = (degrees * Math.PI) / 180;
  const around = 2.3;
  return [Math.cos(up) * Math.cos(around), Math.sin(up), -Math.cos(up) * Math.sin(around)];
};

/**
 * Each layout's camera, the point the automatic orbit turns round, and the seconds the fallen
 * dominoes rest before automatic replay stands them up.
 */
export const VIEWS = {
  run: {
    camera: { target: [0, 0, 45], distance: 185, yaw: -0.3, pitch: 0.38 },
    pivot: [0, -135],
    rest: 2,
  },
  portrait: {
    camera: { target: [MIDDLE[0], 0, MIDDLE[1] + 8], distance: 330, yaw: 0, pitch: 0.95 },
    pivot: MIDDLE,
    rest: 4,
  },
} as const;

/** Linear light: a warm low sun, blue sky, and light off the floor warming shaded sides. */
export const LIGHT = {
  sunDirection: sunAt(24),
  sunColor: [1, 0.7, 0.4],
  sunIntensity: 5.6,
  skyTop: [0.19, 0.33, 0.78],
  skyBottom: [0.2, 0.13, 0.07],
  skyIntensity: 0.75,
  skyReflection: 0.8,
  fogColor: ROOM,
  fogDensity: 0.0012,
} as const;

/** Shadow resolution of 4096 texels across 5.6 m, centred on the run. */
export const SHADOW = { size: 4096, extent: 560, filter: 9, center: [0, 0, -80] } as const;

/** How far occluders darken sky light, in centimetres: a little under the gap between dominoes. */
const CREASE = 2.5;

/** Darkens the sky light in creases, such as under each domino and between close ones. */
const occludedSky = (material: shading.Material, point: shading.Point) => {
  'use gpu';
  return shading.hemisphere(material, point) * ao.at(point.position);
};

/** Shows visibility as stored, white in the open and black in a closed crease. */
const grey = (color: d.v3f) => {
  'use gpu';
  return d.vec3f(color.x);
};

/** Seconds the ghost finger waits before its tap. */
const BEFORE_TAP = 1;
/** Seconds without a newly fallen domino before the ghost counts the run as settled. */
const SETTLED = 0.5;

/** Peak turn of the automatic orbit, in radians per second. */
const DRIFT_SPEED = 0.008;
/** How fast the automatic orbit swings back and forth, in radians of its cycle per second. */
const DRIFT_SWING = 0.1;

/**
 * Tap a domino to start a wave along the run or reveal a portrait on the backs of a fallen
 * field.
 */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const values = {
    layout: 'run' as Layout,
    /** Fraction of the layout to display, in wave order. */
    share: 1,
    /** Sun elevation in degrees. */
    sun: 24,
    /** Automatically starts the wave and stands dominoes up again. Stops when you tap. */
    auto: true,
    /** Darkens sky light under and between dominoes. */
    occlusion: true,
    /** Shows the finished picture, or the occlusion alone in grey. */
    view: 'picture' as 'picture' | 'occlusion',
  };

  const screen = targets.create(root, surface, {
    scene: { format: 'rgba16float', samples: 4 },
    depth: { format: 'depth24plus', samples: 4 },
    hdr: { format: 'rgba16float', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    ...VIEWS.run.camera,
    target: [...VIEWS.run.camera.target],
    fov: 0.6,
    near: 2,
    far: 4000,
    pitchRange: [0.08, 1.4],
    distanceRange: [40, 900],
  });

  /** Half-resolution depth the opaque drawings draw first, for the occlusion to read. */
  const early = targets.create(root, surface, {
    depth: { format: 'depth32float', scale: 0.5, sampled: true },
  });
  const prepass = passes
    .create(root, early, { depth: passes.depth }, { label: 'prepass' })
    .pipe(view.inject());
  const occlusion = ao.create(root, view, {
    depth: () => early.textures.depth,
    settings: { radius: CREASE },
    enabled: () => values.occlusion,
  });

  const light = lighting.create(root, {
    settings: { ...LIGHT, sunDirection: sunAt(values.sun) },
    shading: { ambient: occludedSky },
    // A device whose textures stop below the map's size gets the largest it has
    shadow: { ...SHADOW, size: Math.min(SHADOW.size, root.device.limits.maxTextureDimension2D) },
  });
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: [...ROOM, 1], shared: { ao: occlusion.part } },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const pool = meshPool.create(root, { capacity: 2 });
  const floor = createFloor(root, pool);
  const dominoes = createDominoes(root, { pool, layout: values.layout, share: values.share });
  const picture = createPicture(root, dominoes);

  const sim = clock.fixedStep({ step: STEP, maxSteps: 8 });
  const dust = particles.create(root, view, {
    capacity: 2048,
    step: STEP,
    blend: 'over',
    shade: tgpu.fn(puff).pipe(light.inject()),
    settings: { gravity: [0, -4, 0], drag: 3 },
  });
  const topple = createTopple(root, { view, surface, dominoes, dust });

  const boards = floor.renderer();
  const run = dominoes.set.renderer();
  const drawn = [scene.add(boards), scene.add(run), scene.add(dust.drawing)];
  const casters = [light.shadow.add(run)];
  const shapes = [prepass.add(boards), prepass.add(run)];

  const glow = bloom.create(root, {
    source: () => screen.textures.hdr,
    settings: { threshold: 2, knee: 0.8 },
  });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    glow,
    tone: tone.aces,
    settings: { exposure: 0.85, glowIntensity: 0.1, vignette: 0.45 },
  });

  const occlusionView = finish.create(root, {
    format: surface.format,
    source: occlusion,
    tone: null,
    encode: false,
    grade: grey,
    settings: { vignette: 0, dither: 0 },
  });
  /** The final pass for each `view`. */
  const outputs = { picture: final, occlusion: occlusionView };

  await pipelines.compile([
    prepass,
    occlusion,
    light,
    scene,
    picture,
    dust,
    topple,
    glow,
    final,
    occlusionView,
  ]);

  const scheduler = coroutines.create();
  /** The ghost finger while automatic replay runs. */
  let haunting: coroutines.Coroutine<void> | undefined;
  const stopGhost = () => {
    haunting?.cancel();
    haunting = undefined;
  };
  let clearing = false;

  /** Stands the dominoes up again and clears the fallen count. */
  const standUp = () => {
    topple.rise();
    clearing = true;
  };

  /** Resets the layout and camera. */
  const lay = () => {
    dominoes.lay(values.layout, values.share);
    picture.repaint();
    const { camera } = VIEWS[values.layout];
    Object.assign(view.goal, { ...camera, target: [...camera.target] });
    stopGhost();
    clearing = true;
  };

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      layout: { label: 'layout', options: ['run', 'portrait'], live: true },
      share: { label: 'layout fraction', min: 0.2, max: 1, step: 0.01 },
      sun: { label: 'sun elevation (degrees)', min: 10, max: 60, step: 1 },
      auto: { label: 'automatic replay', live: true },
      occlusion: { label: 'ambient occlusion' },
      view: { options: ['picture', 'occlusion'] },
    },
    write(key) {
      light.settings.patch({ sunDirection: sunAt(values.sun) });
      if (key === 'layout' || key === 'share') lay();
    },
    actions: { reset: standUp },
    readouts: { fallen: () => topple.counts.latest?.fallen ?? 0 },
  };

  const pointing = {
    /** Pushes the picked domino and stops automatic replay. */
    press(x, y) {
      topple.tap(x, y);
      values.auto = false;
    },
  } satisfies Pointing;

  const onScreen = [0, 0, 0];
  const beforeTap = coroutines.seconds(BEFORE_TAP);
  /** Taps the first domino, then stands the run up once no domino has fallen for a while. */
  function* ghost() {
    for (;;) {
      while (topple.rising) yield* coroutines.nextFrame;
      yield* beforeTap;
      if (view.projectAt(dominoes.first, onScreen)) topple.tap(onScreen[0], onScreen[1]);
      for (let quiet = 0, seen = 0; ;) {
        const { dt } = yield* coroutines.nextFrame;
        const fallen = topple.counts.latest?.fallen ?? 0;
        quiet = fallen === seen ? quiet + dt : 0;
        seen = fallen;
        if (fallen > 0 && quiet >= SETTLED + VIEWS[values.layout].rest) break;
      }
      standUp();
    }
  }

  /**
   * Orbits by `angle` radians around the layout pivot during automatic replay. Distant dominoes
   * move less on screen than nearby ones.
   */
  const drift = (angle: number) => {
    const { goal } = view;
    const [px, pz] = VIEWS[values.layout].pivot;
    const x = goal.target[0] - px;
    const z = goal.target[2] - pz;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    goal.target[0] = px + x * cos + z * sin;
    goal.target[2] = pz - x * sin + z * cos;
    goal.yaw += angle;
  };

  /** Crops `image` to fill the portrait field and switches to that layout. */
  const show = (image: Picture) => {
    picture.show(image);
    if (values.layout === 'portrait') return;
    values.layout = 'portrait';
    lay();
  };

  return {
    view,
    settings,
    pointing,
    ui: { portrait: show },

    frame(frame: Frame) {
      if (values.auto) drift(frame.dt * DRIFT_SPEED * Math.cos(frame.time * DRIFT_SWING));
      view.update(frame.dt);
      screen.fit();
      early.fit();
      if (values.auto) haunting ??= scheduler.start(ghost());
      else stopGhost();
      scheduler.run(frame);

      if (clearing) topple.counts.clear(frame);
      clearing = false;

      const steps = sim.update(frame.dt);
      const compute = frame.encoder.beginComputePass();
      picture.record(compute);
      topple.record(compute, steps, frame.dt);
      for (let step = 0; step < steps; step++) dust.step(compute);
      dust.compact(compute, sim.ahead);
      compute.end();
      topple.counts.capture(frame);

      light.shadow.record(frame, casters);
      if (values.occlusion) prepass.record(frame, shapes);
      occlusion.record(frame);
      scene.record(frame, drawn);

      glow.record(frame);
      outputs[values.view].record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [
        scheduler,
        screen,
        early,
        occlusion,
        light,
        floor,
        dominoes,
        picture,
        dust,
        topple,
        glow,
        pool,
      ]) {
        made.destroy();
      }
    },
  };
}
