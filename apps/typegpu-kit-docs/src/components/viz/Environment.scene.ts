import {
  environment,
  instances,
  lighting,
  materials,
  meshPool,
  pipelines,
  images,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, tgpu, type TgpuRoot } from 'typegpu';
import type { Loading } from 'typegpu-kit-examples/contract';
import { litStage } from './stage.ts';

export const SKIES = ['daylight', 'meadow'] as const;
export type Sky = (typeof SKIES)[number];

export const HOUR = { min: 12, max: 19.9, step: 0.05 } as const;
export const BLUR = { min: 0, max: 1, step: 0.01 } as const;
export const START = { sky: 'daylight', hour: 16, blur: 0.1 } as const;

/** The roughness of each metal sphere, left to right, the last one a white matte sphere. */
export const ROUGHNESS = [0, 0.2, 0.4, 0.6, 1] as const;

const DAY_SUN = 1.4;

/** The meadow panorama, 1.4 MB, with its sun clipped out for the kit's sun to give back. */
export async function load({ read }: Pick<Loading, 'read'>) {
  const meadow = images.decodeHdr(new Uint8Array(await read('kinetic/textures/kloofendal.hdr')));
  return environment.clip(meadow);
}

export type EnvironmentOptions = Awaited<ReturnType<typeof load>>;

/** The sun at `hour`, 62° up at noon and on the horizon at 19:30, coming round as it sinks. */
const sunAt = (hour: number): [number, number, number] => {
  const up = hour <= 19.5 ? 62 * Math.cos(((hour - 12) / 7.5) * (Math.PI / 2)) : -9 * (hour - 19.5);
  const elevation = (up * Math.PI) / 180;
  const around = -1.4 + (19.5 - hour) * 0.57;
  return [
    Math.cos(elevation) * Math.cos(around),
    Math.sin(elevation),
    Math.cos(elevation) * Math.sin(around),
  ];
};

/** Spheres from a mirror to rough metal and a matte one, lit by a sky an environment bakes. */
export async function createScene(root: TgpuRoot, surface: Surface, options: EnvironmentOptions) {
  const stage = litStage(root, surface, {
    camera: { target: [0, 0, 0], distance: 6.2, pitch: -0.1, fov: 0.6 },
    samples: 4,
    clear: [0, 0, 0, 0],
    glow: true,
  });
  const { view, screen, final } = stage;
  const sky = environment.create(root);
  const light = lighting.create(root, {
    shadow: { eye: view, cascades: 2 },
    shading: { ambient: sky.ambient },
  });
  const scene = stage.lit(light);

  const pool = meshPool.create(root, { capacity: 64 });
  const spheres = instances.create(root, pool, { capacity: 8 });
  const renderer = spheres.renderer();
  // The background draws at the far plane, where the spheres leave the depth clear
  const drawn = [scene.add(renderer), scene.add(sky.background())];
  const casters = [light.shadow.add(renderer)];

  const ball = pool.add(meshes.sphere({ radius: 0.5, segments: 48, rings: 24 }));
  for (const [i, roughness] of ROUGHNESS.entries()) {
    const look =
      i === ROUGHNESS.length - 1
        ? materials.matte(d.vec3f(0.8))
        : materials.metal(d.vec3f(0.95), roughness);
    spheres.add(ball, { position: [(i - 2) * 1.25, 0, 0], ...look });
  }

  const meadow = images.create(root, options.image);
  const sources = {
    daylight: sky.add(tgpu.fn(environment.daylight).pipe(light.inject())),
    meadow: sky.add(meadow),
  };
  await Promise.all([
    pipelines.compile([scene, light.shadow, stage.glow!, final]),
    sky.initAsync(),
  ]);

  let shown: Sky = START.sky;
  let hour: number = START.hour;
  let dirty = true;

  const aim = () => {
    if (shown === 'meadow') {
      const { direction, color, intensity } = options;
      light.settings.patch({
        sunDirection: direction,
        sunColor: color,
        sunIntensity: intensity,
      });
      final.settings.patch({ exposure: 1 });
      return;
    }
    const sun = sunAt(hour);
    const color = environment.sunColor(d.vec3f(...sun));
    light.settings.patch({
      sunDirection: sun,
      sunColor: [color.x, color.y, color.z],
      sunIntensity: DAY_SUN,
      skyTop: [0, 0, 0],
      skyBottom: [0, 0, 0],
    });
    const seen = (0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z) * DAY_SUN;
    final.settings.patch({ exposure: Math.min(1.5 / Math.sqrt(Math.max(seen, 0.005)), 20) });
  };
  aim();
  sky.settings.patch({ blur: START.blur });

  return {
    view,

    /** Bakes `chosen` before the next frame. */
    sky(chosen: Sky) {
      shown = chosen;
      aim();
      dirty = true;
    },

    /** Moves the daylight's sun to `hours`, which re-bakes the daylight. */
    hour(hours: number) {
      hour = hours;
      if (shown !== 'daylight') return;
      aim();
      dirty = true;
    },

    /** Softens the backdrop, which needs no bake. */
    blur(amount: number) {
      sky.settings.patch({ blur: amount });
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      if (dirty) sky.bake(frame, sources[shown]);
      dirty = false;

      screen.fit();
      spheres.flush();
      light.shadow.record(frame, casters);
      scene.record(frame, drawn);
      stage.present(frame);
    },

    destroy() {
      scene.destroy();
      spheres.destroy();
      pool.destroy();
      light.destroy();
      stage.destroy();
      sky.destroy();
      meadow.destroy();
    },
  };
}
