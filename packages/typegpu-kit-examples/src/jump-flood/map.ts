import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { Cell, NONE } from './jump-flood.ts';

/** Display colours. */
const PAPER = d.vec3f(0.96, 0.91, 0.8);
const INK = d.vec3f(0.22, 0.17, 0.14);
/** Washes: sage, ochre, rose, slate, lilac and teal. */
const WASHES = tgpu.const(d.arrayOf(d.vec3f, 6), [
  d.vec3f(0.46, 0.66, 0.38),
  d.vec3f(0.94, 0.66, 0.26),
  d.vec3f(0.9, 0.5, 0.42),
  d.vec3f(0.36, 0.62, 0.88),
  d.vec3f(0.64, 0.52, 0.78),
  d.vec3f(0.34, 0.66, 0.62),
]);
/** Distances in cells: border half-width and the colour band inside it. */
const BORDER = 0.26;
const BAND = 16;
/** Noise frequencies per cell for the paper and the washes, and how far apart each wash samples. */
const MOTTLE = 0.008;
const UNEVEN = { every: 0.02, apart: 7.31 };
/** Town marks in cells. Capitals carry a ring around their dot. */
const TOWN = { dot: 1.15, ring: 2.5, stroke: 0.24, clear: 0.8 };

/** How much of a pixel `distance` cells from a shape lies within `radius` of it. */
const cover = (distance: number, radius: number, footprint: number) => {
  'use gpu';
  return std.saturate((radius - distance) / footprint + 0.5);
};

/** 1 for a capital, 0 for a town. */
const capital = (seed: number) => {
  'use gpu';
  randf.seed(d.f32(seed) + 0.5);
  randf.sample();
  return std.select(d.f32(0), d.f32(1), randf.sample() < 0.3);
};

/** Ink coverage and clearing of the mark of a town `toSite` cells away. */
const mark = (toSite: number, seed: number, footprint: number) => {
  'use gpu';
  if (toSite > TOWN.ring + TOWN.clear + footprint) return d.vec2f();
  const big = capital(seed);
  const ring = TOWN.ring * big;
  const dot = cover(toSite, TOWN.dot, footprint);
  const circle = cover(std.abs(toSite - ring), TOWN.stroke, footprint) * big;
  const clear = cover(toSite, std.max(ring, TOWN.dot) + TOWN.clear, footprint);
  return d.vec2f(std.max(dot, circle), clear);
};

/** Paints a pixel of the map, for a grid of `size` cells. */
const mapOf = (size: number) => (cell: d.Infer<typeof Cell>) => {
  'use gpu';
  const f = cell.footprint;
  // Paper with a broad mottle, lit from the top left and darker towards its edges
  const uv = cell.point / size;
  const mottle = perlin2d.sample(cell.point * MOTTLE) * 0.035;
  const fromMiddle = std.length((uv - 0.5) * d.vec2f(1, 1.4));
  const light = 1 + 0.04 * (0.5 - uv.x) + 0.04 * (0.5 - uv.y);
  const paper = PAPER * (1 + mottle) * light * (1 - fromMiddle * fromMiddle * 0.28);
  if (cell.seed === NONE) return paper;

  randf.seed(d.f32(cell.seed) + 0.5);
  const wash = WASHES.$[d.u32(randf.sample() * 6) % 6];
  // A pale fill, deepening into a band along the border, unevenly as a wash dries
  const band = std.pow(1 - std.saturate(cell.toEdge / BAND), 3);
  const uneven =
    perlin2d.sample(cell.point * UNEVEN.every + d.f32(cell.seed) * UNEVEN.apart) * 0.07;
  const strength = std.saturate(0.3 + band * 0.42 + uneven);
  let color = paper * std.mix(d.vec3f(1), wash, strength);

  color = std.mix(color, INK, cover(cell.toEdge, std.max(BORDER, f * 0.5), f) * 0.9);

  // Clear the paper around towns, then ink their marks, the second-nearest's too where marks meet
  const own = mark(cell.toSite, cell.seed, f);
  let ink = own.x;
  let clear = own.y;
  if (cell.second !== NONE) {
    const other = mark(cell.toSecond, cell.second, f);
    ink = std.max(ink, other.x);
    clear = std.max(clear, other.y);
  }
  color = std.mix(color, paper * std.mix(d.vec3f(1), wash, 0.12), clear * 0.8);
  return std.mix(color, INK, ink * 0.95);
};

/**
 * A territory map on paper, drawn as an atlas colours its countries, for a grid of `size` cells
 * and `seeds` towns, with the noise gradients its paint reads.
 */
export function createMap(root: TgpuRoot, options: { size: number; seeds: number }) {
  const { size, seeds } = options;
  // Gradients for every lattice point the paper and the washes sample, so none wraps
  const reach = Math.ceil(size * UNEVEN.every + (seeds - 1) * UNEVEN.apart) + 2;
  const gradients = perlin2d.staticCache({ root, size: d.vec2u(reach, reach) });
  const paint = tgpu
    .fn(
      [Cell],
      d.vec3f,
    )(mapOf(size))
    .with(perlin2d.getJunctionGradientSlot, gradients.getJunctionGradient);
  return {
    paint,
    destroy() {
      gradients.destroy();
    },
  };
}
