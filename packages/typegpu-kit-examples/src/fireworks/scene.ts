import {
  bloom,
  cameras,
  clock,
  exposure,
  finish,
  type Frame,
  lighting,
  type meshPool,
  passes,
  pipelines,
  readback,
  type Surface,
  targets,
  tone,
} from '@typegpu/kit';
import { perlin2d } from '@typegpu/noise';
import { clamp } from 'math';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createFireworks, createFlashes, FIREWORKS } from './fireworks.ts';
import { createShow } from './show.ts';
import { createCastle, type ModelName } from './castle.ts';
import { createLand, heightAt } from './land.ts';
import { createLake } from './lake.ts';
import { createLowRes } from './lowRes.ts';
import { HORIZON, nightSky } from './sky.ts';

type Vec3 = [number, number, number];

const SAMPLES = 4;
const HDR = 'rgba16float';
const STEP_SECONDS = 1 / 60;
const VIEW = {
  target: [0, 22, 0],
  distance: 175,
  yaw: 0.2,
  pitch: -0.1,
  fov: 0.95,
  near: 0.5,
  far: 8000,
  pitchRange: [-0.5, 0.9],
  distanceRange: [80, 600],
} as const;
/**
 * How the opening view fits the castle to narrower views: it backs away from the castle to fit the
 * width of a view narrower than `aspect`, no farther than `farthest`, inside the far shore. It
 * raises its target by up to `lift` as the view narrows to `upright`, so on a portrait screen the
 * show above the castle fills the height.
 */
const FIT = { aspect: 1.1, upright: 0.5, farthest: 290, lift: 28 };
/** Minimum camera height above the water and the land, including during orbit. */
const EYE_ABOVE_GROUND_METRES = 2.5;

const NIGHT = {
  sunDirection: [-0.55, 0.32, -0.75],
  sunColor: [0.5, 0.62, 1],
  sunIntensity: 0.05,
  skyTop: [0.01, 0.014, 0.028],
  skyBottom: [0.002, 0.002, 0.003],
} as const;
const FOG_DENSITY_PER_METRE = 0.0011;
const FOG_FALLOFF_PER_METRE = 0.006;
const LOOK = { glowIntensity: 0.16, vignette: 0.3 };
/**
 * Holds the night near an exposure of 1.2, dims it under a volley and opens it back up over a few
 * seconds, no further than `highest`, so a quiet sky stays dark.
 */
const EYE = { key: 0.008, lowest: 0.5, highest: 1.3, rise: 0.5, fall: 2 };
const BLOOM = { threshold: 0.8 };

const CAPACITY = { shells: 48, stars: 6144, embers: 12288, smoke: 1024 };
/** Simulation warm-up in seconds, so smoke and bursts appear in the first frame. */
const PREWARM_SECONDS = 7;
/** Show-speed multiplier during loading. */
const PREWARM_PACE = 1.6;

/** Click-target plane distance behind the castle. The plane is upright and faces the camera. */
const CLICK_BEHIND_METRES = 110;
const CLICK_HEIGHT_METRES = [55, 170] as const;

const READOUT_HZ = 4;
/** Repeat period in Perlin cells for ripple and smoke noise. */
const NOISE_CELLS = 64;
/** Smoke resolution as a fraction of screen width and height. */
const SMOKE_SCALE = 0.5;

/** Sets `fit` to the opening view's distance and target height for an `aspect`-wide view. */
function framing(aspect: number, fit = { distance: 0, height: 0 }) {
  const narrow = 1 - std.smoothstep(FIT.upright, FIT.aspect, aspect);
  fit.distance = Math.min(VIEW.distance * Math.max(1, FIT.aspect / aspect), FIT.farthest);
  fit.height = VIEW.target[1] + FIT.lift * narrow;
  return fit;
}

/**
 * Where a ray from the eye meets the upright plane `CLICK_BEHIND_METRES` behind the castle,
 * facing the eye, held between the click heights. Undefined for a ray that turns away from the
 * castle.
 */
function clickedAt({ origin, direction }: { readonly origin: Vec3; readonly direction: Vec3 }) {
  const [ox, oy, oz] = origin;
  const [dx, dy, dz] = direction;
  const away = Math.hypot(ox, oz);
  const ahead = -(dx * ox + dz * oz) / Math.max(away, 1e-6);
  if (ahead < 0.05) return undefined;
  const distance = (away + CLICK_BEHIND_METRES) / ahead;
  const height = clamp(oy + dy * distance, ...CLICK_HEIGHT_METRES);
  return [ox + dx * distance, height, oz + dz * distance] satisfies Vec3;
}

export interface SceneOptions {
  /** Each model as `gltf.read` returns it. */
  readonly models: Readonly<Record<ModelName, meshPool.ModelData>>;
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
  /** Random-number source. Defaults to `Math.random`. */
  readonly random?: () => number;
}

/** Runs fireworks over a castle and lake. Each burst lights the castle and the smoke. */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const screen = targets.create(root, surface, {
    color: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const aspect = () => surface.width / surface.height;
  const fitted = framing(aspect());
  const view = cameras.orbit(root, {
    size: surface,
    ...VIEW,
    target: [VIEW.target[0], fitted.height, VIEW.target[2]],
    distance: fitted.distance,
    floor: (x, z) => Math.max(0, heightAt(x, z)) + EYE_ABOVE_GROUND_METRES,
  });
  const time = root.createUniform(d.f32);
  const sim = clock.fixedStep({ step: STEP_SECONDS });
  const noise = perlin2d.staticCache({ root, size: d.vec2u(NOISE_CELLS, NOISE_CELLS) });
  const flashes = createFlashes(root, view, { step: STEP_SECONDS });
  const show = createShow(options.random);

  const light = lighting.create(root, {
    settings: {
      ...NIGHT,
      fogColor: [HORIZON.x, HORIZON.y, HORIZON.z],
      fogDensity: FOG_DENSITY_PER_METRE,
      fogFalloff: FOG_FALLOFF_PER_METRE,
    },
    points: { count: flashes.capacity, light: flashes.light },
  });
  const fireworks = createFireworks(root, flashes, light, {
    camera: view,
    step: STEP_SECONDS,
    noise,
    capacity: CAPACITY,
  });
  const scene = passes
    .create(
      root,
      screen,
      { color: passes.color, depth: passes.depth },
      { resolve: { color: 'hdr' } },
    )
    .pipe(view.inject())
    .beneath(light.inject());
  const lake = createLake(root, { view, light, size: surface, time, noise });
  const castle = createCastle(root, options.models);
  const sky = nightSky(time);
  const land = createLand(root);
  const lowRes = createLowRes(root, view, { size: surface, scale: SMOKE_SCALE });
  const landRenderer = land.renderer;
  const castleRenderer = castle.renderer;
  const occluding = [lowRes.occluders.add(landRenderer), lowRes.occluders.add(castleRenderer)];
  const smoke = lowRes.matter.add(fireworks.smoke);

  // The sky fills what the solid drawings leave, before the light that blends over it
  const drawn = [
    scene.add(landRenderer),
    scene.add(castleRenderer),
    scene.add(lake.water),
    scene.add(sky),
    scene.add(lowRes.over),
    ...fireworks.sprites.map((drawing) => scene.add(drawing)),
    ...fireworks.streaks.map((drawing) => scene.add(drawing)),
  ];
  const reflected = [
    lake.reflection.add(landRenderer),
    lake.reflection.add(castleRenderer),
    lake.reflection.add(sky),
    lake.reflection.add(fireworks.smoke),
    ...fireworks.sprites.map((drawing) => lake.reflection.add(drawing)),
    ...fireworks.streaks.map((drawing) => lake.reflection.add(drawing)),
  ];

  const glow = bloom.create(root, { source: () => screen.textures.hdr, settings: BLOOM });
  const eye = exposure.create(root, { source: () => screen.textures.hdr, settings: EYE });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: tone.aces,
    glow,
    exposure: eye,
    settings: LOOK,
  });
  const lights = readback.create(root, flashes.particles.args, {
    hz: READOUT_HZ,
    decode: (args) => args[1] ?? 0,
  });

  const values = {
    show: 'auto' as 'auto' | 'on click',
    rate: 1,
    ...FIREWORKS,
    paused: false,
  };
  const settings: Settings<typeof values> = {
    values,
    knobs: {
      show: { options: ['auto', 'on click'] },
      rate: { label: 'show speed', min: 0.25, max: 2.5, step: 0.05 },
      wind: { label: 'wind (m/s)', min: -4, max: 4, step: 0.1 },
      light: { label: 'burst light', min: 0, max: 3, step: 0.05 },
      smoke: { min: 0, max: 2.5, step: 0.05 },
      paused: {},
    },
    write() {
      fireworks.wind.write(values.wind);
      flashes.brightness.write(values.light);
      fireworks.opacity.write(values.smoke);
      sim.paused = values.paused;
    },
    readouts: { 'lights alive': () => lights.latest ?? 0 },
  };

  const ray = { origin: [0, 0, 0] satisfies Vec3, direction: [0, 0, 0] satisfies Vec3 };
  const pointing: Pointing = {
    press(x, y) {
      view.rayAt(x, y, ray);
      const burst = clickedAt(ray);
      if (burst) show.fireAt(burst);
    },
  };

  await Promise.all([
    castle.ready,
    pipelines.compile([flashes, light, fireworks, scene, lake, lowRes, glow, eye, final], options),
  ]);

  // A spawn's bursts upload before its submission runs, so each launch gets a submission
  let prewarm = root.createCommandEncoder();
  let launched = false;
  const launch: typeof fireworks.launch = (shell) => {
    fireworks.launch(shell);
    launched = true;
  };
  for (let t = 0; t < PREWARM_SECONDS; t += STEP_SECONDS) {
    show.advance(STEP_SECONDS, values.rate * PREWARM_PACE, true, launch);
    const warming = prewarm.beginComputePass();
    fireworks.record(warming, 1, 0);
    warming.end();
    if (!launched) continue;
    prewarm.submit();
    prewarm = root.createCommandEncoder();
    launched = false;
  }
  prewarm.submit();

  const next = { distance: 0, height: 0 };
  /** Refits the opening view to a turned or resized screen, keeping how far it was zoomed. */
  const refit = () => {
    framing(aspect(), next);
    if (next.distance === fitted.distance && next.height === fitted.height) return;
    view.goal.distance *= next.distance / fitted.distance;
    view.goal.target[1] += next.height - fitted.height;
    fitted.distance = next.distance;
    fitted.height = next.height;
  };

  return {
    view,
    settings,
    pointing,
    frame(frame: Frame) {
      time.write(frame.time);
      refit();
      view.update(frame.dt);

      const steps = sim.update(frame.dt);
      if (!sim.paused) {
        show.advance(frame.dt, values.rate, values.show === 'auto', fireworks.launch);
      }
      castle.flush();

      const compute = frame.encoder.beginComputePass();
      fireworks.record(compute, steps, sim.ahead);
      compute.end();
      lights.capture(frame);

      screen.fit();

      lowRes.record(frame, occluding, smoke);
      lake.record(frame, reflected);
      scene.record(frame, drawn);
      glow.record(frame);
      eye.record(frame);
      final.record(frame, surface.view(frame));
    },
    destroy() {
      for (const made of [
        screen,
        noise,
        flashes,
        light,
        fireworks,
        lake,
        castle,
        land,
        lowRes,
        glow,
        eye,
        lights,
      ]) {
        made.destroy();
      }
    },
  };
}
