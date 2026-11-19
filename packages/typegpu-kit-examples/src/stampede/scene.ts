import {
  bloom,
  cameras,
  clock,
  coroutines,
  environment,
  finish,
  impostors,
  lighting,
  meshPool,
  passes,
  picker,
  pipelines,
  readback,
  targets,
  tone,
  type Baked,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Model } from '../models.ts';
import type { Settings } from '../settings.ts';
import {
  CAPACITY,
  createHerd,
  LANE_SPREAD,
  STEP_SECONDS,
  trackAt,
  trackHeading,
  wrap,
} from './herd.ts';
import { createLevels, createPoses } from './horses.ts';
import { createLabel, createPicking, PICK_REACH_POINTS } from './selection.ts';
import { createLand, HAZE, heightAt, sky } from './land.ts';
import { createScenery, layout, SCENERY, type SceneryName } from './scenery.ts';
import { createDust } from './dust.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
/** Initial herd-centre angle in radians, on the near straight with the sun ahead. */
const START_AROUND = 1.85;
/** Camera-target height above the ground, or above a followed horse at its size. */
const LOOK_ABOVE_METRES = 1.6;
/** Camera-target offset outside the herd, in lane spreads. */
const FLANK = 0.7;
/** Camera yaw lead along the herd's outer flank. */
const AHEAD_RADIANS = 0.5;
const VIEW = {
  distance: 28,
  pitch: 0.07,
  fov: 0.9,
  near: 0.3,
  far: 9000,
  pitchRange: [-0.1, 1.45],
  distanceRange: [6, 600],
} as const;
/** Minimum camera height above the ground, including during orbit. */
const EYE_ABOVE_GROUND_METRES = 1.2;
const FOLLOW_HZ = 60;
/** How close the camera rides to a horse it follows. */
const FOLLOW_METRES = 10;

/** The sun, about 12 degrees above the horizon and ahead of the initial view. */
const SUN = {
  sunDirection: [-0.38, 0.21, -0.9],
  sunColor: [1, 0.72, 0.44],
  sunIntensity: 2.6,
} as const;
const SKY_FILL = {
  skyTop: [0.42, 0.52, 0.72],
  skyBottom: [0.34, 0.26, 0.16],
  skyIntensity: 0.55,
} as const;
const FOG = { fogDensity: 0.00055, fogFalloff: 0.0025 };
/** Three cascades fitted to the view, the nearest about 35 m deep and the last ending 320 m out. */
const SHADOW = { cascades: 3 as const, far: 320 };
const LOOK = { exposure: 1.05, glowIntensity: 0.12, vignette: 0.22 };
const BLOOM = { threshold: 1.6 };

const DUST_CAPACITY = 8192;
/** Simulation warm-up in seconds so dust appears in the first frame. */
const PREWARM_SECONDS = 4;
const PREWARM_CHUNK = 15;

/** How far the pointer's ray searches the hills, from a first step of a metre. */
const GROUND = { reach: 2500, step: 1 };

const LONG_AGO_SECONDS = 1e3;
/** Camera height threshold for herd avoidance, and avoidance radius. */
const EYE_LOW_METRES = 5;
const EYE_ROOM_METRES = 4;
/** Pointer fear fade duration after movement stops, in seconds. */
const FEAR_FADE_SECONDS = 0.8;
/** Pointer fear below which the herd ignores the pointer. */
const LEAST_FEAR = 0.01;

type Vec3 = [number, number, number];

export interface SceneOptions {
  /** The horse with its skin and clips, as `gltf.read` returns it. */
  readonly horse: Model;
  /** The trees, rocks and bush, each as its own model. */
  readonly scenery: Readonly<Record<SceneryName, meshPool.ModelData>>;
  readonly font: Baked;
  /** Initial horse count within the fixed capacity. Default 4096. */
  readonly count?: number;
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
  /** Random-number source. Defaults to `Math.random`. */
  readonly random?: () => number;
}

const along: [number, number] = [0, 0];
const headingAngle = (around: number) => {
  trackHeading(around, along);
  return Math.atan2(along[0], along[1]);
};

/** Writes the camera target beside the herd centred at `around` into `out`. */
const herdTarget = (around: number, out: Vec3): Vec3 => {
  trackAt(around, along);
  const x = along[0] * (1 + LANE_SPREAD * FLANK);
  const z = along[1] * (1 + LANE_SPREAD * FLANK);
  out[0] = x;
  out[1] = heightAt(x, z) + LOOK_ABOVE_METRES;
  out[2] = z;
  return out;
};

/**
 * Runs a galloping herd around a valley. Uses skins near the camera, cards farther away and GPU
 * panic propagation between neighbours.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const { horse, count = 4096, random } = options;
  const values = {
    count,
    urgency: 0.85,
    handOff: 50,
    dust: 1,
    tint: false,
    follow: 'herd' as 'herd' | 'picked' | 'none',
  };
  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });

  let heading = headingAngle(START_AROUND);
  const view = cameras.orbit(root, {
    size: surface,
    ...VIEW,
    target: herdTarget(START_AROUND, [0, 0, 0]),
    // The outside of a clockwise track lies to the right
    yaw: heading - Math.PI / 2 + AHEAD_RADIANS,
    floor: (x, z) => heightAt(x, z) + EYE_ABOVE_GROUND_METRES,
  });

  const light = lighting.create(root, {
    settings: {
      ...SUN,
      ...SKY_FILL,
      fogColor: [HAZE.x, HAZE.y, HAZE.z],
      ...FOG,
    },
    shadow: { eye: view, ...SHADOW },
  });
  // The land, the scenery, the horses and the sky read the lighting from beneath
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' } },
    )
    .pipe(view.inject())
    .beneath(light.inject());
  const land = createLand(root);
  const pool = meshPool.create(
    root,
    // The horse, the scenery and the horse's cards
    meshPool.roomFor([horse, ...SCENERY.map((name) => options.scenery[name]), impostors.CARD]),
  );
  const body = pool.add(horse);
  const cards = pool.add(impostors.CARD);
  const card = cards.meshes[0];
  const { placed, obstacles } = layout();
  const scenery = createScenery(root, pool, options.scenery, placed);

  const posed = createPoses(root, horse);
  const sim = clock.fixedStep({ step: STEP_SECONDS });
  const herd = createHerd(root, {
    count: values.count,
    middle: START_AROUND,
    handOff: values.handOff,
    obstacles,
    poses: posed.ids,
    random,
  });
  const picked = createPicking(root, view, { model: horse, herd, posed, surface });
  const levels = createLevels(root, pool, { body, card, posed, herd, picked, view, model: horse });
  const label = createLabel(root, view, { font: options.font, picked, herd });
  const dust = createDust(root, {
    camera: view,
    light,
    step: sim.step,
    herd,
    capacity: DUST_CAPACITY,
    amount: values.dust,
  });
  const following = readback.create(root, herd.watched, { hz: FOLLOW_HZ });

  // The sky fills what the land and the herd leave
  const drawn = [
    scene.add(land.renderer),
    scene.add(scenery.renderer),
    scene.add(levels.solid),
    scene.add(levels.distant),
    scene.add(environment.background(sky)),
    scene.add(dust.drawing),
    scene.add(label.drawing),
  ];
  const casters = [
    light.shadow.add(scenery.renderer),
    light.shadow.add(levels.solid),
    light.shadow.add(levels.distant),
  ];

  const glow = bloom.create(root, { source: () => screen.textures.hdr, settings: BLOOM });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: tone.aces,
    glow,
    settings: LOOK,
  });

  let now = 0;
  let clapAge = LONG_AGO_SECONDS;
  let followed: number = picker.none;
  const pointer = { x: 0.5, y: 0.5, over: false, movedAt: -LONG_AGO_SECONDS };
  /** Where this frame picks for a click. */
  let pressed: [number, number] | undefined;
  const clicks = coroutines.create();
  /** Resumes once this frame's pick is free, so a second click in a frame picks in the next. */
  const pickFree = coroutines.wait({ ready: () => !pressed });
  /** What the herd fears, written each frame. */
  const threats = {
    pointer: [0, 0, 0] as Vec3,
    fear: 0,
    clapAge,
    eye: [0, 0] as [number, number],
    eyeRadius: 0,
  };
  const ray = { origin: [0, 0, 0], direction: [0, 0, 0] };
  const eye = { origin: [0, 0, 0], direction: [0, 0, 0] };
  /** Writes the ground under `x`, `y` into `out`, or returns undefined when the ray misses. */
  const groundUnder = (x: number, y: number, out: Vec3) => {
    view.rayAt(x, y, ray);
    return cameras.hitHeight(ray, heightAt, out, GROUND) === undefined ? undefined : out;
  };

  const follow = (member: number) => {
    followed = member;
    herd.controls.patch({ followed });
    following.reset();
    view.goal.distance = member === picker.none ? VIEW.distance : FOLLOW_METRES;
  };
  const repopulate = () => {
    herd.populate(values.count);
    levels.populate();
    follow(picker.none);
    sim.reset();
  };

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      count: { label: 'herd size', min: 256, max: CAPACITY, step: 256 },
      urgency: { min: 0, max: 1, step: 0.01 },
      handOff: { label: 'impostor distance (m)', min: 10, max: 120, step: 1 },
      dust: { min: 0, max: 2, step: 0.05 },
      tint: { label: 'colour by detail level' },
      follow: { options: ['herd', 'picked', 'none'], live: true },
    },
    write(key) {
      if (key === 'count') repopulate();
      if (key === 'follow' && values.follow === 'herd') follow(picker.none);
      herd.controls.patch({ handOff: values.handOff });
      dust.settings.write({ amount: values.dust });
      if (key === undefined || key === 'tint') levels.tint(values.tint);
    },
    readouts: {
      skinned: () => levels.counts?.skinned ?? 'pending',
      impostors: () => levels.counts?.impostors ?? 'pending',
    },
  };

  const pointing: Pointing = {
    reach: PICK_REACH_POINTS,
    move(x, y) {
      pointer.x = x;
      pointer.y = y;
      pointer.over = true;
      pointer.movedAt = now;
    },
    leave() {
      pointer.over = false;
      picked.clear();
    },
    press(x, y) {
      clicks.start(click(x, y, groundUnder(x, y, [0, 0, 0])));
    },
  };

  /** Points the camera at the herd, or at the followed horse once its position reads back. */
  const steerView = (dt: number) => {
    const watched = following.latest;
    if (values.follow === 'herd' || (values.follow === 'picked' && followed === picker.none)) {
      herdTarget(herd.drawnMiddle, view.goal.target);
    } else if (values.follow === 'picked' && watched) {
      const size = herd.traits[followed]?.size ?? 1;
      const { position, velocity } = watched;
      // Move the read-back position on to where the horse is drawn this frame
      const age = herd.time + sim.ahead - watched.time;
      const x = position.x + velocity.x * age;
      const z = position.z + velocity.y * age;
      const { target } = view.goal;
      target[0] = x;
      target[1] = heightAt(x, z) + LOOK_ABOVE_METRES * size;
      target[2] = z;
    }
    const nextHeading = headingAngle(herd.drawnMiddle);
    if (values.follow === 'herd') view.goal.yaw += wrap(nextHeading - heading);
    heading = nextHeading;
    view.update(dt);
  };

  /** Follows the horse a click picks, or claps on the `ground` it hit when it finds no horse. */
  function* click(x: number, y: number, ground: Vec3 | undefined) {
    yield* pickFree;
    pressed = [x, y];
    const taken = yield* coroutines.until(picked.result.next());
    if (taken !== null) {
      follow(taken);
      values.follow = 'picked';
    } else if (ground) {
      clapAge = 0;
      herd.threat.patch({ clap: ground });
      dust.clap(ground);
    }
  }

  /** Writes the pointer's fear, the clap's age and a low camera for the herd to avoid. */
  const threaten = () => {
    const fear = pointer.over ? Math.exp(-(now - pointer.movedAt) / FEAR_FADE_SECONDS) : 0;
    const seen =
      fear > LEAST_FEAR && groundUnder(pointer.x, pointer.y, threats.pointer) !== undefined;
    if (!seen) threats.pointer.fill(0);
    threats.fear = seen ? fear : 0;
    threats.clapAge = clapAge;
    view.rayAt(0.5, 0.5, eye);
    const { origin } = eye;
    threats.eye[0] = origin[0];
    threats.eye[1] = origin[2];
    const low = origin[1] - heightAt(origin[0], origin[2]) < EYE_LOW_METRES;
    threats.eyeRadius = low ? EYE_ROOM_METRES : 0;
    herd.threat.patch(threats);
  };

  await Promise.all([
    body.ready,
    cards.ready,
    scenery.ready,
    pipelines.compile(
      [light, scene, posed.poses, herd, picked, levels, label, dust, glow, final],
      options,
    ),
  ]);

  scenery.flush();
  // Advance the herd centre per warm-up chunk so steering follows it
  for (; now < PREWARM_SECONDS; now += PREWARM_CHUNK * STEP_SECONDS) {
    herd.advance(PREWARM_CHUNK * STEP_SECONDS, values.urgency);
    const warm = root.createCommandEncoder();
    const warming = warm.beginComputePass();
    for (let i = 0; i < PREWARM_CHUNK; i++) {
      herd.step(warming);
      dust.step(warming);
    }
    dust.compact(warming, 0);
    warming.end();
    warm.submit();
  }

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      const steps = sim.update(frame.dt);
      const seconds = steps * STEP_SECONDS;
      herd.advance(seconds, values.urgency, sim.ahead);
      now += seconds;
      clapAge += seconds;

      steerView(frame.dt);
      clicks.run(frame);
      threaten();
      levels.flush();

      const compute = frame.encoder.beginComputePass();
      dust.spawn(compute);
      for (let i = 0; i < steps; i++) {
        herd.step(compute);
        dust.step(compute);
      }
      dust.compact(compute, sim.ahead);
      posed.poses.animate(compute, now);
      if (pressed) picked.pick(compute, ...pressed);
      else if (pointer.over) picked.pick(compute, pointer.x, pointer.y);
      label.write(compute);
      levels.arrange(compute);
      scenery.renderer.cull(compute);
      compute.end();

      if (pressed) {
        picked.result.capture(frame);
        pressed = undefined;
      }
      levels.capture(frame);
      if (followed !== picker.none) following.capture(frame);

      screen.fit();
      light.shadow.record(frame, casters);
      scene.record(frame, drawn);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      clicks.destroy();
      for (const made of [
        following,
        picked,
        label,
        dust,
        levels,
        herd,
        posed.poses,
        scenery,
        pool,
        land,
        screen,
        light,
        glow,
      ])
        made.destroy();
    },
  };
}
