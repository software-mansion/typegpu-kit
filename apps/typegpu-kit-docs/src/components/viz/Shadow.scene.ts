import { instances, lighting, materials, meshPool, type Frame, type Surface } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import type { TgpuRoot } from 'typegpu';
import { linear, litStage, SKY, turn } from './stage.ts';

/** The comparisons a shadow lookup takes, and the shadow's texels a side, that the figure offers. */
export const FILTERS = [1, 4, 9] as const;
export const SIZES = [256, 1024, 2048] as const;
export type Filter = (typeof FILTERS)[number];

/** What the figure starts with; the sun in degrees around the scene. */
export const START = { filter: 4, texels: 2048, sun: -140 } as const;

/** World units across the shadow's square, which covers every shape. */
const EXTENT = 9;

/** The sun's height above the horizon, in radians. */
const ELEVATION = 0.75;

export function createScene(root: TgpuRoot, surface: Surface) {
  const pool = meshPool.create(root, { vertices: 2048, indices: 10240, capacity: 3 });
  const box = pool.add(meshes.box());
  const ball = pool.add(meshes.sphere());
  const ring = pool.add(meshes.torus());

  const things = instances.create(root, pool, { capacity: 8 });
  const stone = materials.matte(linear(0.55, 0.55, 0.52));
  things.add(box, { position: [0, -0.1, 0], scale: [40, 0.2, 40], ...stone });
  things.add(box, { position: [0, 1.2, -1.3], scale: [0.3, 2.4, 0.3], ...stone });
  things.add(box, {
    position: [-1.6, 0.5, 0],
    rotation: turn([0, 1, 0], 0.5),
    ...materials.glossy(linear(0.75, 0.32, 0.22), 0.5),
  });
  things.add(ball, {
    position: [1.5, 0.6, 0.1],
    scale: 1.2,
    ...materials.glossy(linear(0.25, 0.55, 0.6), 0.3),
  });
  things.add(ring, {
    position: [0, 0.8, 1.2],
    rotation: turn([0, 0, 1], Math.PI / 2),
    scale: 1.6,
    ...materials.metal(linear(0.95, 0.75, 0.4), 0.35),
  });

  const stage = litStage(root, surface, {
    camera: { target: [0.1, 0.4, 0.8], distance: 8, pitch: 0.7, yaw: 0.3 },
  });
  const { view, screen } = stage;

  let sunDirection: [number, number, number] = [0.4, 1, 0.3];
  // A description holds what is added to it, so a new lighting gets new passes
  const lit = (texels: number, filter: Filter) => {
    const light = lighting.create(root, {
      settings: { ...SKY, sunDirection },
      shadow: { size: texels, extent: EXTENT, filter },
    });
    const scene = stage.lit(light);
    const renderer = things.renderer();
    return { light, scene, shown: scene.add(renderer), cast: light.shadow.add(renderer) };
  };

  let { light, scene, shown, cast } = lit(START.texels, START.filter);

  /** Puts the sun `angle` radians around the scene. */
  const aim = (angle: number) => {
    sunDirection = [Math.sin(angle), Math.tan(ELEVATION), Math.cos(angle)];
    light.settings.patch({ sunDirection });
  };
  aim((START.sun * Math.PI) / 180);

  return {
    aim,

    /** Replaces the lighting with one whose shadow has `texels` a side and `filter` comparisons. */
    remake(texels: number, filter: Filter) {
      light.destroy();
      ({ light, scene, shown, cast } = lit(texels, filter));
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();

      things.flush();

      light.shadow.record(frame, cast);

      scene.record(frame, shown);
      stage.present(frame);
    },

    destroy() {
      light.destroy();
      things.destroy();
      pool.destroy();
      stage.destroy();
    },
  };
}
