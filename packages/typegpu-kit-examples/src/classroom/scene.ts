import {
  bindings,
  bloom,
  cameras,
  clock,
  environment,
  finish,
  images,
  instances,
  lighting,
  meshPool,
  passes,
  pipelines,
  shading,
  taa,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Picture } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createAir } from './air.ts';

type Vec3 = [number, number, number];

/**
 * A low afternoon sun through the windows and a clear sky, as the daylight was baked under. The
 * sun's direct light and shadow stay live; the lightmap holds the sky's light and the sun's bounce.
 */
const DAY = {
  sunDirection: [0.851, 0.431, -0.3],
  sunColor: [1, 0.86, 0.68],
  sunIntensity: 2.8,
  skyTop: [0.45, 0.55, 0.75],
  skyBottom: [0.18, 0.22, 0.3],
  skyIntensity: 1,
  skyReflection: 1,
} as const;

/**
 * The light the lightmaps span. Each channel stores the logarithm of its light between these in 8
 * bits, which leaves alpha alone: React Native's decoder premultiplies by alpha and would lose any
 * light packed there.
 */
const LOWEST = 1 / 1024;
const HIGHEST = 4;

/** Unpacks a lightmap's texel, a logarithm of light per channel, into light. */
const unlogged = (texel: d.v4f) => {
  'use gpu';
  const stops = std.log2((HIGHEST + LOWEST) / LOWEST);
  return d.vec4f(std.exp2(texel.xyz * stops) * LOWEST - LOWEST, 1);
};

const LOOK = { exposure: 1, glowIntensity: 0.2, vignette: 0.25 };
const BLOOM = { threshold: 1.2 };
/** The room the camera stays inside, as its centre, its farthest reach and its eye height. */
const ROOM = { center: [0.55, 1.2, 0.5] as Vec3, reach: 2.9, eye: 0.5 };
/** The motes' simulation step in seconds. */
const STEP = 1 / 30;

const Switches = d.struct({
  /** 1 lights the room by the baked daylight, 0 by the live sky. */
  gi: d.u32,
  /** 1 darkens the light by the baked occlusion, 0 leaves it open. */
  ao: d.u32,
  /** The lamps' brightness, as a share of the brightness they were baked at. */
  lamps: d.f32,
});
const switches = tgpu.accessor(Switches);
/** The daylight and the lamps' light, baked on the second uv set and unpacked to linear light. */
const daylight = tgpu.accessor(d.texture2d(d.f32));
const lampLight = tgpu.accessor(d.texture2d(d.f32));
const lightmapFiltering = tgpu.accessor(d.sampler());

/**
 * The room's light besides the sun's: the lamps', baked with their shadows and bounces, and the
 * daylight's, baked with the sky through the windows and the sun's bounce, or else the live sky's.
 * Baked light lights the reflections too, since indoors they see the room, not the sky. The bake
 * already holds the room's shadowing, so the occlusion darkens only the reflections.
 */
const ambient: shading.AmbientHook = (material, point) => {
  'use gpu';
  const lamps =
    std.textureSampleLevel(lampLight.$, lightmapFiltering.$, point.uv1, 0).xyz * switches.$.lamps;
  if (switches.$.gi === 0) {
    return shading.hemisphere(material, point) + shading.surrounded(material, point, lamps, lamps);
  }
  const light = std.textureSampleLevel(daylight.$, lightmapFiltering.$, point.uv1, 0).xyz + lamps;
  return shading.surrounded(material, point, light, light);
};

/** The room, its furniture and the lamps' glowing globes, which burn as the lamps do. */
const Room = instances.define('Room', {
  material: (_pixel, base) => {
    'use gpu';
    const made = shading.Material(base);
    made.emission = base.emission * switches.$.lamps;
    if (switches.$.ao === 0) made.occlusion = 1;
    return made;
  },
});

export interface SceneOptions {
  /** The classroom as `gltf.read` returns it. */
  readonly model: meshPool.ModelData;
  /** The sky's light and the sun's bounce on the second uv set, as `unlogged` reads it. */
  readonly daylight: Picture;
  /** The lamps' light, direct and bounced, on the second uv set, as `unlogged` reads it. */
  readonly lamps: Picture;
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
}

/**
 * Lights a classroom by a low sun through its windows and six ceiling lamps. The sun draws its
 * light and shadow live. The sky's light, the sun's bounce and the lamps' light come baked on the
 * model's second uv set, and settings turn the baked daylight, the occlusion and the lamps off.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const values = {
    /** Lights the room by the daylight baked on its second uv set, or by the live sky. */
    gi: true,
    /** Darkens the light by the occlusion baked on the second uv set. */
    ao: true,
    /** Lights the ceiling lamps, whose light comes baked on the second uv set. */
    lamps: true,
    /** Draws the dust in the air: beams where the sun gets through the windows, and motes. */
    sunbeams: true,
  };
  const screen = targets.create(root, surface, {
    hdr: { format: 'rgba16float', sampled: true },
    motion: { format: 'rg16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: ROOM.center,
    distance: 2.8,
    yaw: -0.65,
    pitch: 0.12,
    pitchRange: [-0.2, 0.45],
    distanceRange: [1, ROOM.reach],
    floor: () => ROOM.eye,
    fov: 1.1,
    near: 0.05,
    far: 60,
  });

  // The model carries a second uv set, so the room it takes keeps one
  const pool = meshPool.create(root, meshPool.roomFor([options.model]));
  const model = pool.add(options.model);
  const room = instances.create(root, pool, { kind: Room, capacity: options.model.meshes.length });
  room.add(model);

  const light = lighting.create(root, {
    settings: DAY,
    shading: { ambient },
    shadow: { center: ROOM.center, extent: 10, size: 2048, filter: 4 },
  });
  // Unpacked to linear light once, so the lightmaps filter as light. Sampled at their first level
  // only: the room shows them magnified
  const unpacked = { packed: unlogged, mips: false };
  const dayMap = images.create(root, options.daylight, unpacked);
  const lampMap = images.create(root, options.lamps, unpacked);
  const lookValues = () => ({
    gi: values.gi ? 1 : 0,
    ao: values.ao ? 1 : 0,
    lamps: values.lamps ? 1 : 0,
  });
  const switched = root.createUniform(Switches, lookValues());
  const filtering = images.sampler(root, { wrap: 'clamp-to-edge' });

  const shared = bindings.share(root, { shadow: light.shadow.part });
  const aa = taa.create(root, {
    source: () => screen.textures.hdr,
    motion: () => screen.textures.motion,
    depth: () => screen.textures.depth,
    camera: view,
  });
  const scene = passes
    .create(
      root,
      screen,
      { hdr: passes.color, motion: taa.motion, depth: passes.depth },
      { jitter: aa.jitter, shared },
    )
    .pipe(view.inject())
    .pipe((cfg) =>
      cfg
        .with(switches, switched)
        .with(daylight, dayMap.createView(d.texture2d(d.f32)))
        .with(lampLight, lampMap.createView(d.texture2d(d.f32)))
        .with(lightmapFiltering, () => filtering.$),
    )
    .beneath(light.inject());

  const sim = clock.fixedStep({ step: STEP });
  const dust = createAir(root, {
    camera: view,
    light,
    screen,
    depth: () => screen.textures.depth,
    shared,
    step: STEP,
  });

  const furniture = room.renderer();
  // The sky behind the windows is the gradient the daylight was baked under
  const clear = [scene.add(furniture), scene.add(environment.background(environment.gradient))];
  const dusty = [...clear, scene.add(dust.drawing)];
  const cast = [light.shadow.add(furniture)];

  const glow = bloom.create(root, { source: aa, settings: BLOOM });
  const final = finish.create(root, {
    format: surface.format,
    source: aa,
    tone: tone.aces,
    glow,
    settings: LOOK,
  });

  await Promise.all([
    model.ready,
    pipelines.compile([light, aa, scene, dust, glow, final], options),
  ]);

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      gi: { label: 'baked GI' },
      ao: { label: 'baked AO' },
      lamps: { label: 'ceiling lamps' },
      sunbeams: { label: 'sunbeams' },
    },
    write() {
      switched.write(lookValues());
    },
  };

  // Nothing in the room moves, so the sun's shadow draws once and keeps its map
  const still = { redraw: () => false };

  return {
    view,
    settings,

    frame(frame: Frame) {
      view.update(frame.dt);
      room.flush();
      if (values.sunbeams) {
        const steps = sim.update(frame.dt);
        const compute = frame.encoder.beginComputePass();
        dust.simulate(compute, steps, sim.ahead);
        compute.end();
      }

      screen.fit();
      light.shadow.record(frame, cast, still);
      scene.record(frame, values.sunbeams ? dusty : clear);
      if (values.sunbeams) dust.record(frame);
      aa.record(frame);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, pool, room, light, dayMap, lampMap, aa, dust, glow]) {
        made.destroy();
      }
    },
  };
}
