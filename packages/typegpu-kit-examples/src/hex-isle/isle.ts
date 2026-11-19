import { srgbToLinear } from '@typegpu/color';
import { instances, meshPool, shading, type lighting } from '@typegpu/kit';
import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { layout, LOOP, MODELS, type ModelName } from './island.ts';

/** Seconds since the start, for the fan and the clouds. */
const clock = tgpu.accessor(d.f32);
/** Night-light factor, 0 by day and 1 at night. */
const lamps = tgpu.accessor(d.f32);

const WARM = srgbToLinear(d.vec3f(1, 0.6, 0.28));
/** Window brightness relative to white at full glow. Values above white produce bloom. */
const BLAZE = 4;

/** House lamp colour and range at night. */
const LAMP = { range: 2.2, intensity: 0.4 };

/**
 * Clouds take on the colour of the air, most on their undersides and on the side facing the sun
 * or moon, which still lights them while it sits low. They carry a share of the sunlight through
 * to their shaded sides, so they stay soft against the sky at every hour.
 */
const CLOUD = { air: 0.3, under: 0.6, lit: 0.8, through: 0.15 };

/**
 * KayKit's atlas is 8 cells across and 4 down. The window panes are the cell at column 6, row 1,
 * and the water, of the sea and of the beaches alike, the cell at column 1, row 1.
 */
const ATLAS = d.vec2f(8, 4);
const PANE = d.vec2f(6, 1);
const WATER = d.vec2f(1, 1);
/** Glossier water roughness, to catch sunlight. */
const SEA_ROUGHNESS = 0.2;

/** Whether a KayKit atlas uv falls on `cell`. */
const onCell = (uv: d.v2f, cell: d.v2f) => {
  'use gpu';
  return std.all(std.eq(std.floor(uv * ATLAS), cell));
};

/**
 * Deforms island instances for windmill rotation and cloud drift, lights windows, and lights
 * clouds from the open sky around them.
 */
const Piece = instances.define('Piece', {
  host: { spin: d.f32, drift: d.f32, glow: d.f32 },

  deform: (vertex, piece) => {
    'use gpu';
    const angle = clock.$ * piece.spin;
    const c = std.cos(angle);
    const s = std.sin(angle);
    const at = vertex.position;
    const spun = d.vec3f(at.x * c - at.y * s, at.x * s + at.y * c, at.z);
    if (piece.drift === 0) return spun;

    // Divide world displacement by cloud scale because deformation uses unrotated local
    // coordinates
    const travelled = std.fract((piece.position.x + clock.$ * piece.drift) / LOOP + 0.5) - 0.5;
    return spun + d.vec3f((travelled * LOOP - piece.position.x) / piece.scale.x, 0, 0);
  },

  material: (point, piece, base) => {
    'use gpu';
    const pane = onCell(point.uv, PANE);
    const burning = std.select(d.f32(0), piece.glow * lamps.$ * BLAZE, pane);

    // The lighting's fog colour holds the colour of the air, which the sky shows behind the clouds
    const env = shading.settings();
    const under = 0.5 - point.normal.y * 0.5;
    const lit = std.saturate(std.dot(point.normal, shading.towardsSun()));
    const air = env.fogColor * (CLOUD.air + CLOUD.under * under + CLOUD.lit * lit);
    const through = env.sunColor * (env.sunIntensity * CLOUD.through);
    const cloud = std.select(d.vec3f(), air + through, piece.drift !== 0);

    const shaded = shading.Material(base);
    shaded.emission = base.emission + WARM * burning + base.color * cloud;
    shaded.roughness = std.select(base.roughness, SEA_ROUGHNESS, onCell(point.uv, WATER));
    return shaded;
  },
});

/**
 * Stores all island models in one mesh pool and all pieces in one instance set, with one lamp
 * per house.
 */
export function createIsle(
  root: TgpuRoot,
  lights: lighting.PointLights,
  time: TgpuUniform<d.F32>,
  data: Readonly<Record<ModelName, meshPool.ModelData>>,
) {
  const pool = meshPool.create(root, meshPool.roomFor(MODELS.map((name) => data[name])));
  const models = new Map(MODELS.map((name) => [name, pool.add(data[name])]));
  const modelOf = (name: ModelName) => models.get(name)!;

  const island = layout();
  const rows = island.pieces.reduce((sum, piece) => sum + modelOf(piece.model).meshes.length, 0);
  const pieces = instances.create(root, pool, { kind: Piece, capacity: rows });

  for (const piece of island.pieces) {
    const { position, yaw, scale = 1, spin = 0, drift = 0, glow = 0 } = piece;
    const rotation = [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)] as const;

    pieces.add(modelOf(piece.model), { position, rotation, scale, spin, drift, glow });
  }

  const color = [WARM.x, WARM.y, WARM.z] as const;
  const houses = island.lamps.map((position) =>
    lights.add({ position, color, range: LAMP.range, intensity: 0 }),
  );
  const intensity = lights.fields.intensity;
  const burning = root.createUniform(d.f32);

  const renderer = pieces.renderer().pipe((cfg) => cfg.with(clock, time).with(lamps, burning));

  let lit = 0;

  return {
    pieces,
    /** Settles when all model textures reach the GPU. */
    ready: Promise.all([...models.values()].map((model) => model.ready)),

    /** Sets window and lamp brightness from 0, dark, to 1, fully lit. */
    lightUp(level: number) {
      if (level === lit) return;
      lit = level;
      burning.write(level);
      for (const house of houses) intensity.set(house, level * LAMP.intensity);
    },

    renderer,

    destroy() {
      for (const house of houses) lights.remove(house);
      pieces.destroy();
      pool.destroy();
    },
  };
}
