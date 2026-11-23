import {
  instances,
  lighting,
  materials,
  meshPool,
  shading,
  table,
  type Frame,
  type RowId,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import * as hrc from '@typegpu/radiance-cascades/holographic';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { linear, litStage } from './stage.ts';

/** World units across the square of ground the field covers. */
export const SIDE = 12;

/** The field's texels a side that the figure offers, and the size it starts with. */
export const SIZES = [64, 128, 256] as const;
export type Size = (typeof SIZES)[number];
export const START: Size = 128;

export const MAX = 24;

/** Extinction per texel of a glow or a pillar: light dies within a few texels. */
const OPAQUE = 4;

/** Half the width of a pillar, in world units. */
const HALF = 0.35;

const PILLARS = [
  [-2.2, -1.4],
  [2, -1.8],
  [-1.2, 2.2],
  [2.4, 1.6],
] as const;

const COLORS = [
  [2, 2.64, 3.92],
  [3.88, 2.84, 1.24],
  [1.68, 3.28, 2.6],
  [3.2, 2.6, 3.8],
] as const;

const SKY = {
  skyTop: [0.02, 0.025, 0.04],
  skyBottom: [0.008, 0.008, 0.01],
  skyIntensity: 1,
} as const;

const Glow = table.define('Glow', {
  host: { position: d.vec2f, radius: d.f32, color: d.vec3f },
});

/** Where glow `i` is at `time`: a slow loop of its own around the pillars. */
const wander = (i: number, time: number) => {
  const phase = i * 2.39996;
  const speed = 0.25 + (i % 5) * 0.04;
  const reach = 1.2 + (i % 4) * 1.1;
  const x = Math.cos(time * speed + phase) * reach;
  const z = Math.sin(time * speed * 1.3 + phase) * reach * 0.9;
  return [x, z] as const;
};

export interface Options {
  /** Moves the glows. Default true. */
  readonly motion?: boolean;
  /** Glows at first. Default 0. */
  readonly count?: number;
}

export async function createScene(root: TgpuRoot, surface: Surface, options: Options = {}) {
  const { motion = true } = options;

  const glows = table.create(root, Glow, { capacity: MAX });
  const pillars = root.createReadonly(
    d.arrayOf(d.vec2f, PILLARS.length),
    PILLARS.map(([x, z]) => d.vec2f(x, z)),
  );

  const pool = meshPool.create(root, { vertices: 512, indices: 1024, capacity: 1 });
  const box = pool.add(meshes.box());
  const things = instances.create(root, pool, { capacity: 1 + PILLARS.length });
  const stone = materials.matte(linear(0.5, 0.5, 0.48));
  things.add(box, { position: [0, -0.1, 0], scale: [SIDE * 1.5, 0.2, SIDE * 1.5], ...stone });
  for (const [x, z] of PILLARS) {
    things.add(box, { position: [x, 0.6, z], scale: [HALF * 2, 1.2, HALF * 2], ...stone });
  }

  const stage = litStage(root, surface, {
    camera: { target: [0, 0, 0.2], distance: 7.2, pitch: 1.1, yaw: 0.1 },
  });
  const { view, screen, final } = stage;

  /** The field at `texels` a side, the lighting that reads it, and the passes it lights. */
  const fieldOf = (texels: Size) => {
    const medium = root
      .createTexture({ size: [texels, texels], format: 'rgba16float' })
      .$usage('storage', 'sampled');
    const writing = medium.createView(d.textureStorage2d('rgba16float', 'write-only'));
    const reading = medium.createView(d.texture2d(d.f32));

    const draw = tgpu.computeFn({
      workgroupSize: [8, 8],
      in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
      'use gpu';
      if (gid.x >= texels || gid.y >= texels) return;

      const ground = ((d.vec2f(gid.xy) + 0.5) / texels - 0.5) * SIDE;
      let emission = d.vec3f();
      let extinction = d.f32(0);

      for (let i = d.u32(0); i < Glow.span(); i++) {
        if (Glow.alive(i) && std.distance(ground, Glow.position(i)) < Glow.radius(i)) {
          emission = Glow.color(i) * OPAQUE;
          extinction = OPAQUE;
        }
      }

      for (let i = 0; i < PILLARS.length; i++) {
        const offset = std.abs(ground - pillars.$[i]);
        if (std.max(offset.x, offset.y) < HALF) {
          emission = d.vec3f();
          extinction = OPAQUE;
        }
      }

      std.textureStore(writing.$, gid.xy, d.vec4f(emission, extinction));
    });
    const drawing = root.pipe(glows.inject()).createComputePipeline({ compute: draw });

    const field = hrc.create({
      root,
      size: { width: texels, height: texels },
      medium: (pixel) => {
        'use gpu';
        const texel = std.textureLoad(reading.$, pixel, 0);
        return { emission: texel.rgb, extinction: texel.a };
      },
    });

    const radiance = field.output.createView(d.texture2d(d.f32));
    const smooth = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const lift = (2 * SIDE) / texels;

    const light = lighting.create(root, {
      settings: {
        ...SKY,
        sunDirection: [-0.3, 1, 0.5],
        sunColor: [0.6, 0.7, 1],
        sunIntensity: 0.12,
      },
      shading: {
        ambient: (material, point) => {
          'use gpu';
          const ground = point.position.xz + point.normal.xz * lift;
          const glow = std.textureSampleLevel(radiance.$, smooth.$, ground / SIDE + 0.5, 0).rgb;
          return shading.hemisphere(material, point) + material.color * glow;
        },
      },
    });
    // A description holds what is added to it, so a new lighting gets new passes
    const scene = stage.lit(light);
    const shown = scene.add(things.renderer());

    return {
      texels,
      field,
      drawing,
      scene,
      shown,

      initAsync: () => Promise.all([drawing.initAsync(), field.initAsync(), scene.initAsync()]),

      destroy() {
        field.destroy();
        light.destroy();
        medium.destroy();
      },
    };
  };

  let current = fieldOf(START);
  const ids: RowId<typeof Glow>[] = [];
  let time = 0;

  const place = () => {
    const position = glows.fields.position;
    ids.forEach((id, i) => position.set(id, ...wander(i, time)));
  };

  const count = (wanted: number) => {
    while (ids.length > wanted) glows.remove(ids.pop()!);

    while (ids.length < wanted) {
      const i = ids.length;
      ids.push(glows.add({ radius: 0.2 + (i % 3) * 0.06, color: COLORS[i % COLORS.length] }));
    }

    place();
  };

  count(options.count ?? 0);
  await Promise.all([current.initAsync(), final.initAsync()]);

  return {
    /** The field's size in bytes, its medium included. */
    get byteSize() {
      return current.field.byteSize + current.texels * current.texels * 8;
    },

    count,

    /** Replaces the field with one of `texels` a side. */
    resize(texels: Size) {
      current.destroy();
      current = fieldOf(texels);
    },

    frame(frame: Frame) {
      if (motion) {
        time = frame.time;
        place();
      }

      glows.flush();
      view.update(frame.dt);
      screen.fit();

      const { texels, drawing, field, scene, shown } = current;
      things.flush();
      const compute = frame.encoder.beginComputePass();
      compute.setPipeline(drawing);
      compute.dispatchWorkgroups(texels / 8, texels / 8);
      field.run({ pass: compute });
      compute.end();

      scene.record(frame, shown);
      stage.present(frame);
    },

    destroy() {
      current.destroy();
      glows.destroy();
      pillars.buffer.destroy();
      things.destroy();
      pool.destroy();
      stage.destroy();
    },
  };
}
