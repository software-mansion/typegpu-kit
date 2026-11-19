import {
  cameras,
  instances,
  lighting,
  materials,
  meshPool,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, type TgpuRoot } from 'typegpu';
import { mat4 } from 'wgpu-matrix';
import { linear, litStage, SKY } from './stage.ts';

/** Trees along each side of the field. */
const SIDE = 21;

/** Metres between neighbours. */
const SPACING = 1.5;

/** Metres from the eye where the spheres hand over to the boxes. */
export const REACH = 7;

/** The fades the figure offers, in metres. */
export const FADES = [0, 2, 5] as const;
export type Fade = (typeof FADES)[number];

/** The radius of the eye's walk, and its height. */
const WALK = 6;
const EYE_HEIGHT = 1.5;

export interface Options {
  /** Walks the eye; without it the eye stands still. Default true. */
  readonly motion?: boolean;
  /** Metres across which the renderers hand over. Default 2. */
  readonly fade?: Fade;
}

export async function createScene(root: TgpuRoot, surface: Surface, options: Options = {}) {
  const { motion = true } = options;

  const pool = meshPool.create(root, { vertices: 2048, indices: 8192, capacity: 4 });
  const encoder = root.createCommandEncoder();
  const bake = (shape: meshPool.Shape) => pool.add(shape, encoder);
  const sphere = bake(meshes.icosphere({ radius: 0.55, segments: 3 }));
  const box = bake(meshes.box({ width: 0.8, height: 0.8, depth: 0.8 }));
  const marker = bake(meshes.sphere({ radius: 0.35, segments: 16, rings: 8 }));
  const ground = bake(meshes.plane({ width: SIDE * SPACING + 2, depth: SIDE * SPACING + 2 }));
  encoder.submit();

  const capacity = SIDE * SIDE;
  const trees = instances.create(root, pool, { capacity });
  const cards = instances.create(root, pool, { capacity });
  const props = instances.create(root, pool, { capacity: 2 });

  const leaf = materials.matte(linear(0.35, 0.62, 0.3));
  const card = materials.matte(linear(0.9, 0.62, 0.3));
  for (let x = 0; x < SIDE; x++) {
    for (let z = 0; z < SIDE; z++) {
      const position = [
        (x - (SIDE - 1) / 2) * SPACING,
        0.5,
        (z - (SIDE - 1) / 2) * SPACING,
      ] as const;
      trees.add(sphere, { position, ...leaf });
      cards.add(box, { position, ...card });
    }
  }
  const walker = props.add(marker, { ...materials.glow(d.vec3f(1, 0.95, 0.85), 3) });
  props.add(ground, { ...materials.matte(linear(0.12, 0.11, 0.1)) });

  const light = lighting.create(root, { settings: { ...SKY, sunDirection: [0.4, 1, 0.3] } });

  const stage = litStage(root, surface, {
    camera: { target: [0, 0, 0], distance: 38, pitch: 1.1, yaw: 0.3 },
  });
  const { view, screen, final } = stage;
  const eye = cameras.custom(root);
  const lookAt = mat4.create();
  const lens = mat4.perspective(0.8, 1, 0.1, 100);

  const scene = stage.lit(light);

  const marked = scene.add(props.renderer());

  /** The two renderers, near spheres and far boxes, handing over across `fade` metres. */
  const rangesOf = (fade: number) => {
    const near = trees.renderer({
      cull: true,
      to: REACH,
      fade,
      eye,
    });
    const far = cards.renderer({
      cull: true,
      from: REACH,
      fade,
      eye,
    });
    return { renderers: [near, far], drawn: [scene.add(near), scene.add(far)] };
  };

  let ranges = rangesOf(options.fade ?? 2);
  let angle = 0;

  /** Places the eye on its walk, and its marker with it. */
  const walk = (dt: number) => {
    if (motion) angle += dt * 0.35;
    const position = [Math.cos(angle) * WALK, EYE_HEIGHT, Math.sin(angle) * WALK] as const;
    const ahead = [-Math.sin(angle) * WALK, EYE_HEIGHT, Math.cos(angle) * WALK] as const;

    mat4.lookAt(position, ahead, [0, 1, 0], lookAt);
    eye.write(lookAt, lens);
    props.write(walker, { position });
  };

  await Promise.all([scene.initAsync(), final.initAsync()]);

  return {
    /** Hands over across `fade` metres from now on, through renderers made for it. */
    fade(fade: Fade) {
      for (const drawn of ranges.drawn) drawn.destroy();
      ranges = rangesOf(fade);
    },

    frame(frame: Frame) {
      walk(frame.dt);
      view.update(frame.dt);
      screen.fit();
      trees.flush();
      cards.flush();
      props.flush();

      const culling = frame.encoder.beginComputePass();
      for (const renderer of ranges.renderers) renderer.cull(culling);
      culling.end();

      const pass = scene.begin(frame);
      for (const drawn of ranges.drawn) drawn.draw(pass);
      marked.draw(pass);
      pass.end();
      stage.present(frame);
    },

    destroy() {
      for (const each of [trees, cards, props]) each.destroy();
      pool.destroy();
      light.destroy();
      stage.destroy();
    },
  };
}
