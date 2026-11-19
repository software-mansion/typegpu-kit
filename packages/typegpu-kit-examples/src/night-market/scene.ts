import {
  bloom,
  cameras,
  clock,
  dof,
  environment,
  finish,
  lighting,
  passes,
  pipelines,
  table,
  taa,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { vec3 } from 'wgpu-matrix';
import { d, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createStalls, FIRE, type Models } from './stalls.ts';
import { createPeople, type PeopleOptions } from './people.ts';
import { sky } from './sky.ts';
import { createGround } from './ground.ts';
import { createFire } from './fire.ts';
import { createRain } from './rain.ts';

type Vec3 = [number, number, number];

/** A blue moonlight over the rooftops, the sky's blue in the shade, and a blue haze down the street. */
const MOON = {
  sunDirection: [-0.45, 0.75, -0.5] as Vec3,
  sunColor: [0.5, 0.65, 1] as Vec3,
  sunIntensity: 0.25,
  skyTop: [0.03, 0.06, 0.16] as Vec3,
  skyBottom: [0.02, 0.016, 0.014] as Vec3,
  skyIntensity: 1,
  skyReflection: 1,
  fogColor: [0.03, 0.06, 0.13] as Vec3,
  fogDensity: 0.015,
  fogFalloff: 0.12,
};
/** The lights that cast shadows: the hung lantern, the fire, and the lamps over the banner and the table. */
const LANTERN = { color: [1, 0.55, 0.22] as Vec3, intensity: 8, range: 11 };
const FIRE_LIGHT = { color: [1, 0.44, 0.14] as Vec3, intensity: 9, range: 12 };
const SIGN_LAMP = {
  color: [1, 0.84, 0.6] as Vec3,
  intensity: 1.6,
  range: 5,
  cone: lighting.cone(0.3, 0.6),
};
const COUNTER_LAMP = {
  color: [1, 0.78, 0.5] as Vec3,
  intensity: 3,
  range: 6,
  cone: lighting.cone(0.45, 0.8),
};
/** Metres from the top of the fire's barrel to the middle of its light. */
const FLAME_HEIGHT = 0.35;
/** Lowest eye height over the street, so the camera never dips under the paving. */
const EYE_ABOVE_STREET_METRES = 0.3;
/** Every lit point visits each lamp: the four that cast, then the awning spots and small lamps. */
const LAMPS = 18;
const LOOK = { exposure: 1.2, glowIntensity: 0.2, vignette: 0.35 };
const BLOOM = { threshold: 1.2 };
const SIZES = [256, 512, 1024] as const;
/** How fast the focus follows a tap, per second. */
const PULL = 5;
/** The largest blur radius, as a fraction of the picture height, wide enough for discs. */
const LIMIT = 0.03;
/** The lantern's swing: gravity over the cord's length in metres, and damping per second. */
const SWING = { gravity: 9.8, cordMetres: 1.2, damping: 0.35 };

/**
 * How the lantern swings in the wind: a pendulum on its cord, pushed by gusts. `angle` holds its
 * tilt about x and z in radians.
 */
function pendulum() {
  const angle = [0.12, 0.05];
  const speed = [0, 0];
  let seconds = 0;
  const swingAxis = (k: number, push: number, h: number) => {
    const pull = -(SWING.gravity / SWING.cordMetres) * angle[k];
    speed[k] += (pull - SWING.damping * speed[k] + push) * h;
    angle[k] += speed[k] * h;
  };

  return {
    angle: angle as readonly number[],

    /** Swings on for a step of `h` seconds in `wind`. */
    step(h: number, wind: number) {
      const t = seconds;
      const gust = wind * (0.55 * Math.sin(t * 0.7) + 0.3 * Math.sin(t * 1.9 + 1.2) + 0.15);
      swingAxis(0, gust * 0.9, h);
      swingAxis(1, gust * 0.35 * Math.sin(t * 0.43), h);
      seconds += h;
    },
  };
}

/** How brightly the fire burns at `seconds`, around 0.82 of full. */
const flickerAt = (seconds: number) =>
  0.82 +
  0.1 * Math.sin(seconds * 13) +
  0.06 * Math.sin(seconds * 29 + 1) +
  0.05 * Math.sin(seconds * 7.3 + 2);

export interface SceneOptions {
  /** Each part of the market's file. */
  readonly models: Models;
  /** The seller, the customer and their clips. */
  readonly people: Pick<PeopleOptions, 'mage' | 'rogue' | 'clips'>;
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
}

/**
 * Lights a toy market at night: a lantern swinging from a string of lights, a cooking fire, a lamp
 * on a banner and a street lamp over a potion seller's table, all casting point and spot shadows
 * onto wet cobbles. TAA smooths the frame with the walkers' and the lantern's own motion, then
 * depth of field keeps the potions sharp and opens the lights behind them into discs before the
 * bloom.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const values = {
    /** Draws the point and spot shadows; without them the lights shine through every crate. */
    shadows: true,
    /** Sets each shadow layer's size in texels. */
    map: 512 as (typeof SIZES)[number],
    /** How hard the wind swings the lantern. */
    wind: 0.6,
    /** How bright the fire burns. */
    fire: 1,
    /** Smooths edges over several frames. */
    taa: true,
    /** Blurs what is nearer or farther than the focus, which a tap moves. */
    dof: true,
    /** The lens opening, in metres. */
    aperture: 0.12,
  };
  const screen = targets.create(root, surface, {
    hdr: { format: 'rgba16float', sampled: true },
    motion: { format: 'rg16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [-0.6, 1.5, 2],
    distance: 7.5,
    yaw: 0.2,
    pitch: 0.2,
    pitchRange: [-0.2, 1.2],
    distanceRange: [2, 24],
    floor: () => EYE_ABOVE_STREET_METRES,
    near: 0.1,
    far: 150,
  });
  const sim = clock.fixedStep({ step: 1 / 60 });

  const lamps = table.create(root, lighting.SpotLight, {
    capacity: LAMPS,
    defaults: { ...lighting.WHITE_SPOT, range: 8 },
  });
  const light = lighting.create(root, {
    settings: MOON,
    points: { table: lamps },
    shadow: { eye: view, cascades: 2, far: 40, size: 1024 },
    pointShadow: { count: 4, filter: 9, enabled: () => values.shadows },
  });
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    motion: () => screen.textures.motion,
    depth: () => screen.textures.depth,
    camera: view,
  });
  // Beneath a drawing's own fills, so the ground shades with the market's lighting
  const scene = passes
    .create(
      root,
      screen,
      { hdr: passes.color, motion: taa.motion, depth: passes.depth },
      {
        jitter: aa.jitter,
        shared: { shadow: light.shadow.part, pointShadow: light.pointShadow.part },
      },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const time = root.createUniform(d.f32);
  const market = createStalls(root, options.models);
  const people = createPeople(root, {
    ...options.people,
    lantern: options.models.lantern,
    seller: market.seller,
    customer: market.customer,
  });
  const rain = createRain(root, view, { step: sim.step });
  const fire = createFire(root, view, { step: sim.step, at: FIRE });

  // The lights that cast take the first slots, the lantern and fire first, then the sign lamps
  const swing = pendulum();
  const lantern = lamps.add({ position: [...market.swing(0, 0)], ...LANTERN });
  const flame: Vec3 = [FIRE[0], FIRE[1] + FLAME_HEIGHT, FIRE[2]];
  const blaze = lamps.add({ position: flame, ...FIRE_LIGHT });
  const { sign, counter } = market.spots;
  const signs = [
    lamps.add({ position: sign.lamp, aim: sign.aim, ...SIGN_LAMP }),
    lamps.add({ position: counter.lamp, aim: counter.aim, ...COUNTER_LAMP }),
  ];
  for (const { position, aim, cone, color, intensity, range } of market.bulbs) {
    const spot = aim && cone ? { aim, cone: lighting.cone(cone[0], cone[1]) } : {};
    lamps.add({ position, color, intensity, range, ...spot });
  }
  light.pointShadow.cast([lantern, blaze, ...signs]);

  // The sky fills what the market leaves, before the fire and rain blend over it
  const solid = market.solid.renderer();
  const drawn = [
    scene.add(createGround(time)),
    scene.add(solid),
    scene.add(market.glowing.renderer()),
    scene.add(people.renderer),
    scene.add(environment.background(sky)),
    scene.add(fire.drawing),
    scene.add(rain.drawing),
  ];
  const moonCast = [light.shadow.add(solid), light.shadow.add(people.renderer)];
  const lampCast = [light.pointShadow.add(solid), light.pointShadow.add(people.renderer)];

  // The potions are in focus at the start, and a tap pulls the focus to what it lands on
  const eye = cameras.eyeOf(view.goal, vec3.create());
  const ahead = vec3.normalize(vec3.subtract(view.goal.target, eye));
  const potions = vec3.dot(vec3.subtract(market.potions, eye), ahead);
  const blur = dof.create(root, view, {
    source: aa,
    depth: () => screen.textures.depth,
    pull: PULL,
    settings: { focus: potions, aperture: values.aperture, limit: LIMIT },
  });

  // Depth of field reads TAA's output, and bloom and the final pass read the blur while it is on
  const picture = () => (values.dof ? blur.output : aa.output);
  const glow = bloom.create(root, { source: picture, settings: BLOOM });
  const final = finish.create(root, {
    format: surface.format,
    source: picture,
    tone: tone.aces,
    glow,
    settings: LOOK,
  });

  await Promise.all([
    market.ready,
    people.ready,
    pipelines.compile([light, aa, scene, people, blur, glow, final], options),
  ]);

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      shadows: { label: 'point and spot shadows' },
      map: { label: 'shadow size (texels)', options: SIZES },
      wind: { min: 0, max: 2, step: 0.05 },
      fire: { min: 0, max: 2, step: 0.05 },
      taa: { label: 'TAA' },
      dof: { label: 'depth of field' },
      aperture: { label: 'aperture (m)', min: 0.02, max: 0.6, step: 0.01 },
    },
    write() {
      light.pointShadow.resize(values.map);
      aa.enable(values.taa);
      blur.settings.patch({ aperture: values.aperture });
    },
  };

  const ray = { origin: new Float32Array(3), direction: new Float32Array(3) };
  const pointing = {
    press(x, y) {
      view.rayAt(x, y, ray);
      people.tap(ray);
      blur.aim(x, y);
    },
  } satisfies Pointing;

  let seconds = 0;

  /** Moves the lantern's light with the lantern and flickers the fire. */
  const sway = () => {
    const at = market.swing(swing.angle[0], swing.angle[1]);
    lamps.fields.position.set(lantern, at[0], at[1], at[2]);

    const flicker = flickerAt(seconds);
    lamps.fields.intensity.set(blaze, FIRE_LIGHT.intensity * values.fire * flicker);
    lamps.fields.position.set(
      blaze,
      flame[0] + Math.sin(seconds * 11) * 0.03,
      flame[1] + Math.sin(seconds * 17) * 0.02,
      flame[2] + Math.cos(seconds * 9) * 0.03,
    );
  };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      seconds += frame.dt;
      view.update(frame.dt);
      const steps = sim.update(frame.dt);
      for (let step = 0; step < steps; step++) swing.step(sim.step, values.wind);

      sway();
      time.write(seconds);
      fire.burn(values.fire);

      people.step(seconds, frame.dt);
      market.solid.flush();
      market.glowing.flush(frame);
      people.flush(frame);
      lamps.flush();

      const compute = frame.encoder.beginComputePass();
      people.animate(compute, seconds);
      fire.spawn(compute);
      rain.spawn(compute);
      for (let step = 0; step < steps; step++) {
        fire.step(compute);
        rain.step(compute);
      }
      fire.compact(compute, sim.ahead);
      rain.compact(compute, sim.ahead);
      compute.end();

      screen.fit();
      light.shadow.record(frame, moonCast);
      light.pointShadow.record(frame, lampCast);
      scene.record(frame, drawn);
      aa.record(frame);
      if (values.dof) blur.record(frame);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, lamps, light, aa, market, people, rain, fire, blur, glow])
        made.destroy();
    },
  };
}
