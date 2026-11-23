import { instances, materials, shading, type meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { perlin2d } from '@typegpu/noise';
import { perlin2d as cpuPerlin } from 'math/noise';
import { mulberry32 } from 'math/random';
import { d, std, tgpu } from 'typegpu';
import { SUN } from './air.ts';

const TAU = Math.PI * 2;

/** Forest radius in metres. */
export const RADIUS = 220;
/**
 * The glade around the centre: its mean radius, how far its edge wanders as a fraction of that,
 * the edge noise frequency per metre and the width over which trees thin out towards it.
 */
const GLADE = { radius: 15, wander: 0.5, frequency: 0.06, fringe: 10 };
/** Turning trees grow only beyond this distance from the centre. */
const NEAR = 70;
const SEED = 7;
/** Instances per tree, one trunk and one crown. */
export const PARTS = 2;

const TRUNK = { radius: 0.14, height: 1.8 };
/** Spruce crown dimensions. `seat` is its base height as a fraction of trunk height. */
const SPRUCE = { radius: 1.5, height: 5, seat: 0.6, roughness: 0.9 };
/**
 * Oak crown proportions. `squash` is height divided by width, and `lift` offsets its centre
 * above the trunk in crown heights.
 */
const OAK = { radius: 1.6, squash: 0.8, lift: 0.7, roughness: 0.8 };
/** Spruce crown tier count, and its mesh's columns around and rows up. */
const TIERS = 3;
const COLUMNS = 14;
const ROWS = 24;

/** The share of trees that are spruces; the rest are oaks. */
const SPRUCES = 0.7;
/** Minimum and maximum tree scale. */
const SIZES = [0.7, 1.4] as const;
/** Spruce crown height variation over its width. */
const STRETCH = [0.85, 1.3] as const;
/** Crown brightness variation around the picked colour. */
const SHADES = [0.7, 1.2] as const;
/** The share of broadleaf trees already turning. */
const TURNING = 0.025;

/** Which way the wind blows, along the ground. */
const WIND = d.vec3f(0.8, 0, 0.6);
/** Bend per square metre of height before applying gust strength. */
const SWAY = 0.004;
/** Gust spatial frequency in noise cells per metre. */
const GUST = 0.05;

/** Warm light passing through sunlit crown edges, before crown colour. */
const THROUGH = d.vec3f(3.6, 2.4, 1.2);

/** How far the gusts have travelled downwind, in noise cells. */
export const drift = tgpu.accessor(d.f32);

/**
 * Deforms trunks and crowns by squared height above ground, modulated by gusts. Trees have no
 * rotation, so scaling local deformation gives the world-space bend.
 */
export const Tree = instances.define('Tree', {
  deform: (vertex, tree) => {
    'use gpu';
    const height = tree.position.y + vertex.position.y * tree.scale.y;
    const gust = std.clamp(perlin2d.sample(tree.position.xz * GUST - WIND.xz * drift.$), -1, 1);
    const bend = WIND * (SWAY * height * height * (1 + gust));
    return vertex.position + bend / tree.scale;
  },

  // The sphere top bounds every vertex height and therefore the maximum bend
  bounds: (_tree, base) => {
    'use gpu';
    const top = base.center.y + base.radius;
    return instances.Sphere({ center: base.center, radius: base.radius + 2 * SWAY * top * top });
  },

  material: (pixel, base) => {
    'use gpu';
    const point = pixel.point;
    // Sunlight through the edges of crowns between the eye and the sun
    const lit = shading.Material(base);
    const rim = 1 - std.saturate(std.dot(point.normal, point.view));
    const behind = std.saturate(-std.dot(point.view, SUN));
    const through = rim * rim * behind * behind * shading.sunShare(point);
    lit.emission = base.emission + base.color * THROUGH * through;
    return lit;
  },
});

const BARK = materials.matte(srgbToLinear(d.vec3f(0.36, 0.29, 0.23)));
/** Crown colours, each tree mixing a random pair of its species' palette. */
const NEEDLES = [d.vec3f(0.07, 0.2, 0.12), d.vec3f(0.14, 0.27, 0.13), d.vec3f(0.2, 0.3, 0.12)];
const LEAVES = [d.vec3f(0.3, 0.4, 0.13), d.vec3f(0.38, 0.47, 0.15), d.vec3f(0.24, 0.34, 0.12)];
const AUTUMN = [d.vec3f(0.5, 0.32, 0.15), d.vec3f(0.52, 0.45, 0.2), d.vec3f(0.42, 0.25, 0.14)];
const linear = (palette: d.v3f[]) => palette.map((color) => srgbToLinear(color));
const PALETTES = { needles: linear(NEEDLES), leaves: linear(LEAVES), autumn: linear(AUTUMN) };

/** Adds the tree meshes to `pool`: a trunk centred vertically, and crowns with their bases at 0. */
export function addShapes(pool: meshPool.MeshPool) {
  const spruce = meshes.parametric(
    {
      at: (u, v) => {
        'use gpu';
        // Clockwise winding from +z toward +x makes the crown face outward
        const tier = 0.5 + 0.5 * (1 - std.fract(v * TIERS + 0.02));
        const radius = (1 - v) * tier * std.min(v * ROWS, 1) * SPRUCE.radius;
        const angle = u * TAU;
        return d.vec3f(std.sin(angle) * radius, v * SPRUCE.height, std.cos(angle) * radius);
      },
    },
    { cols: COLUMNS, rows: ROWS },
  );

  return {
    trunk: pool.add(meshes.cylinder({ ...TRUNK, radialSegments: 8 })),
    spruce: pool.add(spruce),
    oak: pool.add(meshes.icosphere({ radius: OAK.radius, segments: 2 })),
  };
}

/** Creates trunk and crown instances at seeded positions between the glade and forest edge. */
export function plant(
  things: instances.Instances<typeof Tree>,
  shapes: ReturnType<typeof addShapes>,
  count: number,
) {
  const state = mulberry32.create(SEED);
  const next = () => mulberry32.sample(state);
  const between = ([low, high]: readonly [number, number]) => low + next() * (high - low);
  const pick = <T>(list: readonly T[]) => list[Math.floor(next() * list.length)];
  const edge = cpuPerlin.create(SEED);
  const color: [number, number, number] = [0, 0, 0];

  for (let planted = 0; planted < count;) {
    const out = Math.sqrt(next()) * RADIUS;
    const angle = next() * TAU;
    const x = Math.cos(angle) * out;
    const z = Math.sin(angle) * out;
    const wander = cpuPerlin.sample(edge, x * GLADE.frequency, z * GLADE.frequency);
    const glade = GLADE.radius * (1 + GLADE.wander * wander);
    if (next() > (out - glade) / GLADE.fringe) continue;
    planted++;
    const size = between(SIZES);

    const spruce = next() < SPRUCES;
    const turning = !spruce && out > NEAR && next() < TURNING;
    const palette = spruce ? PALETTES.needles : turning ? PALETTES.autumn : PALETTES.leaves;
    const a = pick(palette);
    const b = pick(palette);
    const mix = next();
    const shade = between(SHADES);
    color[0] = (a.x + (b.x - a.x) * mix) * shade;
    color[1] = (a.y + (b.y - a.y) * mix) * shade;
    color[2] = (a.z + (b.z - a.z) * mix) * shade;

    things.add(shapes.trunk, { position: [x, (TRUNK.height / 2) * size, z], scale: size, ...BARK });
    if (spruce) {
      things.add(shapes.spruce, {
        position: [x, TRUNK.height * SPRUCE.seat * size, z],
        scale: [size, size * between(STRETCH), size],
        color,
        roughness: SPRUCE.roughness,
      });
    } else {
      const crown = OAK.radius * OAK.squash * size;
      things.add(shapes.oak, {
        position: [x, TRUNK.height * size + crown * OAK.lift, z],
        scale: [size, size * OAK.squash, size],
        color,
        roughness: OAK.roughness,
      });
    }
  }
}
