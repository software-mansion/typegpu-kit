import {
  cameras,
  passes,
  targets,
  gltf,
  environment,
  impostors,
  instances,
  lighting,
  lines,
  meshPool,
  ao,
  poses,
  images,
  table,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { decoded } from 'typegpu-kit-examples/models.ts';
import { readFromDisk, white } from 'typegpu-kit-examples/testing';
import type { Fixtures } from './harness.ts';

const size = { width: 64, height: 36 };
const FIRST_INSTANCE = ['indirect-first-instance'];

/** A model of the skeletons example, its images decoded to white. */
const model = async (name: string) =>
  decoded(gltf.read(await readFromDisk(`skeletons/models/${name}.glb`)), white);

/** The warrior, skinned and textured, and the clips its rig plays. */
const warrior = async () => {
  const [body, clips] = await Promise.all([model('warrior'), model('skeleton-clips')]);
  return { ...body, skin: body.skin!, clips: clips.clips };
};

export const fixtures: Fixtures = {
  'kit/environment': {
    setup: (root) => {
      const image = { width: 8, height: 4, pixels: new Uint8Array(128) };
      const face = { width: 4, height: 4, pixels: new Uint8Array(64) };
      const panorama = images.create(root, image, { mips: false });
      const cube = images.cube(
        root,
        Array.from({ length: 6 }, () => face),
      );
      return {
        create: () => {
          const sky = environment.create(root, { size: 16 });
          const sources = [sky.add(panorama), sky.add(cube)];
          // Its settings are made when a lookup, such as `sky.ambient`, first binds them
          root.unwrap(sky.settings.buffer);
          return {
            frame: (frame) => {
              for (const source of sources) sky.bake(frame.encoder, source);
            },
            destroy: () => sky.destroy(),
          };
        },
      };
    },
  },

  'kit/lines': {
    setup: (root, surface) => {
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      return {
        create: async (others) => {
          // The description holds the viewport the lines read.
          const canvas = others(() =>
            passes.create(root, surface, { clear: [0, 0, 0, 1] }).pipe(view.inject()),
          );
          const paths = lines.create(root, { capacity: 2, length: 4 });
          const drawn = canvas.add(paths.drawing);
          await others(() => drawn.initAsync());
          paths.write(0, [0, 0, 0, 1, 1, 0, 2, 0, 0], { width: 0.1 });
          return {
            frame: (frame) => {
              const pass = canvas.begin(frame);
              drawn.draw(pass);
              pass.end();
            },
            destroy: () => {
              paths.destroy();
              canvas.destroy();
            },
          };
        },
      };
    },
  },

  'kit/impostors': {
    features: FIRST_INSTANCE,
    setup: (root) => {
      const pool = meshPool.create(root, { vertices: 200, indices: 400, capacity: 4 });
      const box = pool.add(meshes.box());
      const plane = pool.add(meshes.plane());
      const card = pool.add(impostors.CARD).meshes[0];
      return {
        create: () => {
          // Captured in the first frame; instances draw the atlas through their own pipelines.
          const atlas = impostors.create(
            root,
            pool,
            [{ mesh: box }, { mesh: plane, position: [0, 1, 0] }],
            { card, views: 2, size: 16 },
          );
          let captured = false;
          return {
            frame: (frame) => {
              if (captured) return;
              captured = true;
              atlas.record(frame);
            },
            destroy: () => atlas.destroy(),
          };
        },
      };
    },
  },

  'kit/instances': {
    features: FIRST_INSTANCE,
    setup: (root) => {
      const pool = meshPool.create(root, { vertices: 200, indices: 400, capacity: 4 });
      const box = pool.add(meshes.box());
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      const light = lighting.create(root);
      const screen = targets.create(root, size, {
        hdr: { format: 'rgba16float' },
        depth: { format: 'depth24plus' },
      });
      const scene = passes
        .create(root, screen, { hdr: passes.color, depth: passes.depth })
        .pipe(view.inject())
        .beneath(light.inject());
      // What the renderer's pipelines bind of the lighting and the pool, theirs.
      root.unwrap(light.settings.buffer);
      root.unwrap(pool.texturesOf(box.index));
      return {
        create: () => {
          const things = instances.create(root, pool, { capacity: 8 });
          const renderer = things.renderer({ cull: true });
          const drawn = scene.add(renderer);
          const id = things.add(box, { position: [0, 0, 0] });
          let x = 0;
          return {
            frame: (frame) => {
              things.write(id, { position: [(x += 0.01), 0, 0] });
              things.flush();
              const compute = frame.encoder.beginComputePass();
              renderer.cull(compute);
              compute.end();
              const pass = scene.begin(frame);
              drawn.draw(pass);
              pass.end();
            },
            destroy: () => things.destroy(),
          };
        },
      };
    },
  },

  'kit/lighting': {
    setup: (root) => {
      // The table is the caller's, as a lighting reads it and never frees it.
      const lamps = table.create(root, lighting.PointLight, { capacity: 4 });
      const lamp = lamps.add({ position: [0, 1, 0] });
      return {
        create: () => {
          const light = lighting.create(root, { shadow: { size: 64 }, points: { table: lamps } });
          // What a pipeline that shades binds of it.
          root.unwrap(light.settings.buffer);
          return {
            // Its shadow draws only what a renderer added to it, in pipelines the shadow's passes own.
            frame: ({ encoder }) => {
              lamps.write(lamp, { intensity: 2 });
              lamps.flush();
              light.shadow.record(encoder, []);
            },
            destroy: () => light.destroy(),
          };
        },
      };
    },
  },

  'kit/meshPool': {
    setup: async (root) => {
      const body = await warrior();
      return {
        create: async () => {
          const pool = meshPool.create(root, { vertices: 1 << 13, indices: 1 << 15, capacity: 8 });
          pool.add(meshes.box());
          await pool.add(body).ready;
          return {
            // A renderer binds each mesh's textures as it draws it.
            frame: () => {
              for (const mesh of pool.meshes) root.unwrap(pool.texturesOf(mesh.index));
            },
            destroy: () => pool.destroy(),
          };
        },
      };
    },
  },

  'kit/ao': {
    setup: (root) => {
      const view = cameras.orbit(root, { size, target: [0, 0, 0], distance: 5 });
      const early = targets.create(root, size, {
        normal: { format: 'rg16float', sampled: true },
        depth: { format: 'depth32float', sampled: true },
      });
      return {
        create: () => {
          const occlusion = ao.create(root, view, {
            depth: () => early.textures.depth,
            normal: () => early.textures.normal,
            temporal: true,
          });
          return {
            frame: ({ encoder }) => {
              early.fit();
              occlusion.record(encoder);
            },
            destroy: () => occlusion.destroy(),
          };
        },
      };
    },
  },

  'kit/poses': {
    setup: async (root) => {
      const { skin, clips } = await warrior();
      const named = Object.fromEntries(clips.slice(0, 2).map((clip) => [clip.name, clip]));
      const [first, second] = Object.keys(named) as [string, string];
      return {
        create: () => {
          const figures = poses.create(root, skin, named, { capacity: 4 });
          const pose = figures.add(first, 0);
          figures.add(second, 0);
          return {
            frame: (frame) => {
              if (frame.index === 2) figures.play(pose, second, frame.time);
              const pass = frame.encoder.beginComputePass();
              figures.animate(pass, frame.time);
              pass.end();
            },
            destroy: () => figures.destroy(),
          };
        },
      };
    },
  },
};
