import { cameras, materials, passes, shading } from '@typegpu/kit';
import { srgbToLinear } from '@typegpu/color';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuUniform } from 'typegpu';

/** Metres across the paved street, and how far down the street its middle lies. */
const SIDE = 80;
const MIDDLE = -6;
/** Metres along a paving slab, which run in courses offset by half a slab. */
const SLAB = 0.62;
/** Metres across a cell that holds one raindrop's ripple. */
const DROP = 0.45;
/** Ripples a second each cell's drop makes. */
const DROPS = 1.4;
const RIPPLE = 0.2;

const STONE = srgbToLinear(d.vec3f(0.2, 0.19, 0.18));
const GROUT = srgbToLinear(d.vec3f(0.07, 0.065, 0.06));

const Corner = d.struct({ position: d.vec4f, world: d.vec3f });
const Seen = d.struct({ color: d.vec4f });

const corners = tgpu.const(
  d.arrayOf(d.vec2f, 6),
  [
    [-1, -1],
    [1, 1],
    [1, -1],
    [-1, -1],
    [-1, 1],
    [1, 1],
  ].map(([x, z]) => d.vec2f(x!, z!)),
);

/** How deep the water stands, 0 on dry stone to 1 in a puddle. */
const waterAt = (place: d.v2f) => {
  'use gpu';
  const pools = perlin2d.sample(place * 0.32) + perlin2d.sample(place * 1.1 + 7.3) * 0.3;
  return std.smoothstep(-0.05, 0.15, pools);
};

/** A ring spreading from a drop that falls somewhere in each cell, as a tilt of the water. */
const rippleAt = (place: d.v2f, seconds: number) => {
  'use gpu';
  const cell = std.floor(place / DROP);
  randf.seed2(cell);
  const center = (cell + 0.25 + d.vec2f(randf.sample(), randf.sample()) * 0.5) * DROP;
  const age = std.fract(seconds * DROPS + randf.sample());
  const away = place - center;
  const distance = std.length(away);
  const off = (distance - age * RIPPLE) / 0.012;
  const ring = std.exp(-off * off) * (1 - age);
  return (away / std.max(distance, 1e-4)) * (ring * std.sin(off * 2) * 0.35);
};

/**
 * Draws the wet paving: stone slabs in offset courses, dark grout holding water, and puddles that
 * mirror the lights, rippled by the rain.
 */
export function createGround(time: TgpuUniform<d.F32>) {
  const drawing: passes.Drawing<typeof Corner, d.Infer<typeof Seen>> = {
    label: 'ground',
    Vertex: Corner,
    at: (vertexIndex: number, _instanceIndex: number) => {
      'use gpu';
      const corner = corners.$[vertexIndex] * (SIDE / 2);
      const world = d.vec3f(corner.x, 0, corner.y + MIDDLE);
      return Corner({ position: cameras.toClip(world), world });
    },
    pixelOf: (corner) => {
      'use gpu';
      const place = corner.world.xz;
      const course = std.floor(place.y / SLAB);
      const along = place.x / SLAB + course * 0.5;
      const slab = d.vec2f(std.floor(along), course);
      const inside = d.vec2f(std.fract(along), std.fract(place.y / SLAB));
      const edge = std.min(std.min(inside.x, 1 - inside.x), std.min(inside.y, 1 - inside.y));
      const grout = 1 - std.smoothstep(0.015, 0.045, edge);
      const tone = perlin2d.sample(slab * 0.83 + 0.5);
      const grain = perlin2d.sample(place * 9);

      const water = std.max(waterAt(place), grout * 0.35);
      const stone = STONE * (0.85 + tone * 0.22 + grain * 0.1);
      // Water darkens what it wets and smooths it towards a mirror
      const color = std.mix(stone, GROUT, grout) * std.mix(1, 0.4, water);
      const roughness = std.mix(0.55 - grain * 0.1, 0.05, water);

      // Slabs bevel at their edges, and drops ripple where the water stands
      const bevel = (inside - 0.5) * (grout * 0.4) * (1 - water);
      const ripple = rippleAt(place, time.$) * water;
      const normal = std.normalize(d.vec3f(bevel.x + ripple.x, 1, bevel.y + ripple.y));

      const lit = shading.shade(
        materials.glossy(color, roughness),
        shading.pointAt(corner.world, normal),
      );
      return Seen({ color: d.vec4f(lit, 1) });
    },
    color: passes.shaded,
    draw: (pass) => pass.draw(6),
  };
  return drawing;
}
