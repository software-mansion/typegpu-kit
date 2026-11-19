import {
  bloom,
  cameras,
  clock,
  environment,
  instances,
  lighting,
  lines,
  materials,
  meshPool,
  passes,
  pipelines,
  finish,
  shading,
  targets,
  images,
  exposure,
  type Frame,
  type Mesh,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { disc, LOOKS, mobileOf, place, TOP, WIRE, type Shape } from './mobile.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';

/**
 * Whether the backdrop draws each panorama's sun disc. `environment.clip` takes each one's sun, or
 * the studio's stronger softbox, and the kit's sun gives that light back with a shadow. The
 * studio's other softbox gives as much light, but no ceiling clips one without the other, so its
 * clipped light is lost.
 */
const PANORAMAS = {
  meadow: { disc: 1 },
  venice: { disc: 1 },
  studio: { disc: 0 },
};
/** The metered exposure's range, wide enough for deep dusk. */
const EYE = { key: 0.12, lowest: 1 / 16, highest: 20 };

type Panorama = keyof typeof PANORAMAS;
const SKIES = ['daylight', 'meadow', 'venice', 'studio'] as const;
type Sky = (typeof SKIES)[number];

/** The daylight sun's intensity above the air. */
const DAY_SUN = 1.4;
/** The hours the time of day runs through, from noon to deep dusk, 3° below the horizon. */
const HOURS = [12, 20.6] as const;
/**
 * A midsummer day at 51° south, the sun 62° up at noon and setting just after 20:00. In the south
 * it crosses the sky from right to left as the first view sees it, lighting the faces of the
 * mobile through the afternoon.
 */
const LATITUDE = -0.9;
const MIDSUMMER = -0.41;
/** Turns the compass so the sun shines over the eye's shoulder after noon and sets ahead of it. */
const NORTH = 2.08;
/** Re-bakes per second while the time of day or the turn moves. */
const REBAKES = 15;
/** Radians per second the camera drifts round the mobile. */
const DRIFT = 0.02;
/** The sun's disc in the backdrop, over its light on a surface. */
const DISC = 40;

/** The wires' lacquer, lit as the pieces are, with edges as soft as the pixels they cross. */
const lacquered = (fragment: lines.Fragment) => {
  'use gpu';
  const point = shading.pointAt(fragment.position, fragment.normal);
  const lit = shading.shade(materials.glossy(fragment.color.xyz, LOOKS.wire.roughness), point);
  return d.vec4f(lit * fragment.coverage, fragment.coverage);
};

export interface SceneOptions {
  readonly skies: Readonly<Record<Panorama, images.HdrImage>>;
}

/** A hanging mobile turning under a sky that an environment bakes and lights it with. */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const screen = targets.create(root, surface, {
    color: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0.24, 1.8, -0.4],
    distance: 5.1,
    yaw: 0.45,
    pitch: -0.22,
    fov: 0.75,
    pitchRange: [-1.2, 0.8],
    distanceRange: [2.5, 14],
  });
  const sky = environment.create(root, { settings: { blur: 0.08 } });
  const light = lighting.create(root, {
    shadow: { eye: view, cascades: 2 },
    shading: { ambient: sky.ambient },
  });
  const sunShown = root.createUniform(d.f32);
  const behind = (direction: d.v3f) => {
    'use gpu';
    return environment.backdrop(direction) + environment.sunDisc(direction, sunShown.$);
  };
  // Beneath a drawing's own fills, so the backdrop reads the lighting and the environment
  const scene = passes
    .create(
      root,
      screen,
      { color: passes.color, depth: passes.depth },
      { resolve: { color: 'hdr' } },
    )
    .pipe(view.inject())
    .beneath(light.inject())
    .beneath(sky.inject());
  const pool = meshPool.create(root, { capacity: 64 });
  const mobile = mobileOf(TOP);
  const pieces = instances.create(root, pool, { capacity: mobile.parts.length });
  const renderer = pieces.renderer();
  // A wire is finer than a shadow map's texel, so it casts no shadow, which would smear
  const wires = lines.create(root, {
    capacity: mobile.wires.length,
    length: 2,
    shade: lacquered,
    label: 'wires',
  });
  // The backdrop fills what the pieces leave, and the wires blend their edges over both
  const drawn = [
    scene.add(renderer),
    scene.add(environment.background(behind)),
    scene.add(wires.drawing),
  ];
  const casters = [light.shadow.add(renderer)];

  const hdr = () => screen.textures.hdr;
  const glow = bloom.create(root, { source: hdr, settings: { threshold: 4 } });
  const eye = exposure.create(root, { source: hdr, settings: EYE });
  const final = finish.create(root, { format: surface.format, source: hdr, glow, exposure: eye });

  const shapes: Record<Shape, Mesh> = {
    disc: pool.add(disc(96, 32)),
    ball: pool.add(meshes.sphere({ radius: 1, segments: 48, rings: 24 })),
  };
  const ids = mobile.parts.map((part) => pieces.add(shapes[part.shape], part.look));
  const { x, y, z } = LOOKS.wire.color;
  const strung = { width: WIRE * 2, color: [x, y, z, 1] };

  const suns = {
    meadow: environment.clip(options.skies.meadow),
    venice: environment.clip(options.skies.venice),
    studio: environment.clip(options.skies.studio),
  };
  const panoramas = {
    meadow: images.create(root, suns.meadow.image),
    venice: images.create(root, suns.venice.image),
    studio: images.create(root, suns.studio.image),
  };
  const sources: Record<Sky, environment.Prepared> = {
    daylight: sky.add(tgpu.fn(environment.daylight).pipe(light.inject())),
    meadow: sky.add(panoramas.meadow),
    venice: sky.add(panoramas.venice),
    studio: sky.add(panoramas.studio),
  };

  await pipelines.compile([sky, light, scene, glow, eye, final]);

  const values = {
    sky: 'daylight' as Sky,
    /** Hours from noon to deep dusk, for the daylight sky. */
    hour: 14,
    /** Turns the sky, and with it the sun, about the vertical. */
    turn: 0,
    /** How soft the backdrop is. */
    blur: 0.08,
    /** Drifts the camera round the mobile. */
    orbit: true,
  };

  const rebake = clock.cadence({ hz: REBAKES });
  let dirty = true;

  /** Points the kit's sun as the sky has it. */
  const aim = () => {
    if (values.sky === 'daylight') {
      const sun = environment.turn(
        environment.sunAt(values.hour, LATITUDE, MIDSUMMER),
        values.turn + NORTH,
      );
      const color = environment.sunColor(sun);
      light.settings.patch({
        skyTop: [0, 0, 0],
        skyBottom: [0, 0, 0],
        sunDirection: sun,
        sunColor: color,
        sunIntensity: DAY_SUN,
        skyIntensity: 1,
      });
      sky.settings.patch({ rotation: 0 });
      sunShown.write(DISC);
      return;
    }
    const { disc: shown } = PANORAMAS[values.sky];
    const { direction, color, intensity } = suns[values.sky];
    light.settings.patch({
      sunDirection: environment.turn(d.vec3f(...direction), values.turn),
      sunColor: color,
      sunIntensity: intensity,
    });
    sky.settings.patch({ rotation: values.turn });
    sunShown.write(DISC * shown);
  };
  aim();

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      sky: { options: SKIES },
      hour: { label: 'time of day (h)', min: HOURS[0], max: HOURS[1], step: 0.05 },
      turn: { label: 'turn (rad)', min: 0, max: 6.28, step: 0.01 },
      blur: { label: 'backdrop blur', min: 0, max: 1, step: 0.01 },
      orbit: { label: 'drift' },
    },
    write(key) {
      sky.settings.patch({ blur: values.blur });
      aim();
      // A panorama turns at lookup; the daylight bakes its sun where the turn puts it
      if (key === 'blur' || key === 'orbit') return;
      if (key === undefined || key === 'sky' || values.sky === 'daylight') dirty = true;
    },
  };

  return {
    view,
    settings,

    frame(frame: Frame) {
      if (values.orbit) view.goal.yaw += frame.dt * DRIFT;
      view.update(frame.dt);

      place(TOP, mobile, frame.time);
      for (let i = 0; i < ids.length; i++) pieces.write(ids[i], mobile.parts[i].placed);
      for (let i = 0; i < mobile.wires.length; i++) wires.write(i, mobile.wires[i], strung);

      if (dirty && rebake.due(frame.dt)) {
        sky.bake(frame, sources[values.sky]);
        dirty = false;
      }
      screen.fit();
      pieces.flush();
      light.shadow.record(frame, casters);
      scene.record(frame, drawn);
      glow.record(frame);
      eye.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [
        screen,
        sky,
        light,
        pool,
        pieces,
        wires,
        glow,
        eye,
        ...Object.values(panoramas),
      ]) {
        made.destroy();
      }
    },
  };
}
