import {
  cameras,
  instances,
  lighting,
  meshPool,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import type { TgpuRoot } from 'typegpu';

export async function createScene(root: TgpuRoot, surface: Surface) {
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0, 0],
    distance: 6,
    yaw: 0.6,
    pitch: 0.5,
  });
  const light = lighting.create(root, {
    shadow: { extent: 12 },
    settings: { sunDirection: [-0.6, 0.7, 0.5] },
  });
  const pool = meshPool.create(root, { capacity: 1 });
  const box = pool.add(meshes.box());
  const things = instances.create(root, pool, { capacity: 2 });
  things.add(box, { position: [0, 0.5, 0], color: [0.95, 0.45, 0.2] });
  things.add(box, { position: [0, -0.1, 0], scale: [8, 0.2, 8], color: [0.14, 0.15, 0.18] });
  const renderer = things.renderer();
  const screen = targets.create(root, surface, {
    color: { format: surface.format, samples: 4 },
    depth: { format: 'depth24plus', samples: 4 },
  });
  const scene = passes
    .create(
      root,
      screen,
      { color: passes.color, depth: passes.depth },
      { resolve: { color: surface } },
    )
    .pipe(view.inject())
    .beneath(light.inject());
  const shown = scene.add(renderer);
  const cast = light.shadow.add(renderer);
  await pipelines.compile([light, scene]);
  return {
    view,
    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();
      things.flush();
      light.shadow.record(frame, cast);
      scene.record(frame, shown);
    },
    destroy() {
      light.destroy();
      things.destroy();
      pool.destroy();
      screen.destroy();
    },
  };
}
