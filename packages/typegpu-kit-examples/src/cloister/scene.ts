import {
  bloom,
  cameras,
  environment,
  finish,
  instances,
  lighting,
  meshPool,
  passes,
  pipelines,
  shading,
  table,
  taa,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { DUSK, sky } from './sky.ts';

type Vec3 = [number, number, number];

/** Where the model's empties `Lantern.001` to `Lantern.008` hang, in its metres: two in each walk. */
export const LANTERNS: readonly Vec3[] = [
  [-3, 2.48, -5.78],
  [3, 2.48, -5.78],
  [5.78, 2.48, -3],
  [5.78, 2.48, 3],
  [3, 2.48, 5.78],
  [-3, 2.48, 5.78],
  [-5.78, 2.48, 3],
  [-5.78, 2.48, -3],
];
const LANTERN = { color: [1, 0.5, 0.2] as Vec3, intensity: 1.8, range: 8 };
/** The flame's flicker, as a share of its brightness. */
const FLICKER = 0.12;

/** The ways to show the occlusion the model bakes into its second uv set. */
const SHOWN = ['baked', 'off', 'alone'] as const;
/** The occlusion's brightness drawn alone, white where nothing hides the sky. */
const ALONE = 0.3;

const LOOK = { exposure: 1.3, glowIntensity: 0.25, vignette: 0.3 };
const BLOOM = { threshold: 1.1 };
/** The lanterns and the cloister never move, so each lantern's shadow is drawn once and kept. */
const STILL = { redraw: () => false };
/** Lowest eye height over the paving and the garth's planting, in metres. */
const EYE = 0.6;
/** Half the side of the square the eye stays in: the column line less the capitals and a margin. */
const COURT = 3.9;
const PITCH = [-0.3, 1.2] as const;
const DISTANCE = [1.5, 7] as const;

/** How far a ray from `from` may go `along` an axis before it crosses the court's side. */
const across = (from: number, along: number) =>
  along === 0 ? Infinity : (Math.sign(along) * COURT - from) / along;
/** How far the eye may go from `target` along a yaw and pitch before it leaves the court. */
const reach = (target: ArrayLike<number>, yaw: number, pitch: number) =>
  Math.min(across(target[0], Math.sin(yaw)), across(target[2], Math.cos(yaw))) / Math.cos(pitch);

const Shown = d.struct({
  /** 1 to darken the sky's light by the baked occlusion, 0 to leave it out. */
  baked: d.f32,
  /** 1 to draw the occlusion alone, as grey. */
  alone: d.u32,
});
const shown = tgpu.accessor(Shown);

/**
 * The cloister's stone, plaster and lanterns. Their colour textures read the first uv set through
 * each texture's transform, and their occlusion the second, as the model's materials say.
 */
const Masonry = instances.define('Masonry', {
  material: (_point, _masonry, base) => {
    'use gpu';
    const made = shading.Material(base);
    made.occlusion = std.mix(1, base.occlusion, shown.$.baked);
    if (shown.$.alone === 1) {
      made.color = d.vec3f();
      made.roughness = 1;
      made.metallic = 0;
      made.emission = d.vec3f(base.occlusion * ALONE);
    }
    return made;
  },
});

/** The lanterns' iron and glass, which hold the lights and so cast no shadow from them. */
const isLantern = (mesh: meshPool.MeshData) => /glass|iron/i.test(mesh.material.name ?? '');

export interface SceneOptions {
  /** The cloister as `gltf.read` returns it. */
  readonly model: meshPool.ModelData;
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
}

/**
 * Lights a cloister at the blue hour with the sky and eight lanterns casting point shadows. Its
 * tiled paving, rotated border and atlas panels move their uvs with `KHR_texture_transform`, and
 * its ambient occlusion comes baked on a second uv set, which a setting turns off or shows alone.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const values = {
    /** Darkens the sky's light by the occlusion baked on the second uv set, or shows it alone. */
    occlusion: 'baked' as (typeof SHOWN)[number],
    /** Draws each lantern's shadow; without them, light passes through the columns. */
    shadows: true,
  };
  const screen = targets.create(root, surface, {
    hdr: { format: 'rgba16float', sampled: true },
    motion: { format: 'rg16float', sampled: true },
    depth: { format: 'depth24plus', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    // Across the garth to the north walk's door, the well to one side
    target: [-1, 2.5, -2.2],
    distance: 6.8,
    yaw: 0.55,
    pitch: -0.06,
    pitchRange: PITCH,
    distanceRange: DISTANCE,
    floor: () => EYE,
    reach,
    near: 0.05,
    far: 100,
  });

  // The model carries a second uv set, so the room it takes keeps one
  const { meshes, images } = options.model;
  const pool = meshPool.create(root, meshPool.roomFor([options.model]));
  const masonry = pool.add({ meshes: meshes.filter((mesh) => !isLantern(mesh)), images });
  const lanterns = pool.add({ meshes: meshes.filter(isLantern), images });

  const walls = instances.create(root, pool, { kind: Masonry, capacity: meshes.length });
  const glass = instances.create(root, pool, { kind: Masonry, capacity: meshes.length });
  walls.add(masonry);
  glass.add(lanterns);

  const lamps = table.create(root, lighting.PointLight, {
    capacity: LANTERNS.length,
    defaults: { ...lighting.WHITE },
  });
  const light = lighting.create(root, {
    settings: DUSK,
    points: { table: lamps },
    pointShadow: { count: LANTERNS.length, filter: 9, enabled: () => values.shadows },
  });
  const flames = LANTERNS.map((position) => lamps.add({ position, ...LANTERN }));
  light.pointShadow.cast(flames);

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
      { jitter: aa.jitter, shared: { pointShadow: light.pointShadow.part } },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const lookValues = () => ({
    baked: values.occlusion === 'off' ? 0 : 1,
    alone: values.occlusion === 'alone' ? 1 : 0,
  });
  const showing = root.createUniform(Shown, lookValues());
  const show: Inject = (cfg) => cfg.with(shown, showing);
  const stone = walls.renderer().pipe(show);
  const drawn = [
    scene.add(stone),
    scene.add(glass.renderer().pipe(show)),
    scene.add(environment.background(sky)),
  ];
  const cast = [light.pointShadow.add(stone)];

  const glow = bloom.create(root, { source: aa, settings: BLOOM });
  const final = finish.create(root, {
    format: surface.format,
    source: aa,
    tone: tone.aces,
    glow,
    settings: LOOK,
  });

  await Promise.all([
    masonry.ready,
    lanterns.ready,
    pipelines.compile([light, aa, scene, glow, final], options),
  ]);

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      occlusion: { label: 'baked occlusion (second uv set)', options: SHOWN },
      shadows: { label: 'lantern shadows' },
    },
    write() {
      showing.write(lookValues());
    },
  };

  let seconds = 0;
  /** Each flame flickers on its own, out of step with the others. */
  const flicker = () => {
    flames.forEach((flame, i) => {
      const t = seconds + i * 1.7;
      const wave = 0.6 * Math.sin(t * 9.1) + 0.3 * Math.sin(t * 23 + i) + 0.1 * Math.sin(t * 3.7);
      lamps.fields.intensity.set(flame, LANTERN.intensity * (1 + FLICKER * wave));
    });
  };

  return {
    view,
    settings,

    frame(frame: Frame) {
      seconds += frame.dt;
      view.update(frame.dt);
      flicker();
      lamps.flush();
      walls.flush();
      glass.flush();

      screen.fit();
      light.pointShadow.record(frame, cast, STILL);
      scene.record(frame, drawn);
      aa.record(frame);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, pool, walls, glass, lamps, light, aa, glow]) made.destroy();
    },
  };
}
