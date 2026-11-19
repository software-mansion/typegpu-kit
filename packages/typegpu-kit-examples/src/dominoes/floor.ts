import { srgbToLinear } from '@typegpu/color';
import { meshes } from '@typegpu/kit/geometry';
import { perlin2d, randf } from '@typegpu/noise';
import { instances, materials, type MeshPool } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';

/** A board's width and length, in centimetres. The boards run along z. */
const BOARD = d.vec2f(16, 180);

const DARK = srgbToLinear(d.vec3f(0.2, 0.13, 0.085));
const LIGHT = srgbToLinear(d.vec3f(0.36, 0.24, 0.155));
const SEAM = srgbToLinear(d.vec3f(0.08, 0.05, 0.035));
/** The point the floor fades out from, and how far out it fades into the dim room. */
const CENTER = d.vec2f(0, -80);
const FADE = d.vec2f(350, 1200);

/**
 * Returns pixel coverage of lines around integer coordinates, with half-width `width`. Fades
 * lines at a spacing of a few pixels to prevent distant shimmer.
 */
const seam = (at: number, width: number) => {
  'use gpu';
  const off = std.abs(std.fract(at + 0.5) - 0.5);
  const pixel = std.max(std.fwidth(at), 1e-5);
  const covered = 1 - std.smoothstep(width - pixel, width + pixel, off);
  return covered * std.saturate(1.5 - pixel * 6);
};

/**
 * Draws walnut boards with per-board shades, broad patterns, fine grain and dark seams.
 * Averages grain smaller than a pixel.
 */
const woodAt = (at: d.v2f) => {
  'use gpu';
  const across = at.x / BOARD.x;
  const row = std.floor(across);
  randf.seed(row);
  // Offset joints between rows
  const along = at.y / BOARD.y + randf.sample();
  randf.seed2(d.vec2f(row, std.floor(along)));
  const shade = randf.sample();
  const phase = randf.sample() * 50;

  const figure = perlin2d.sample(d.vec2f(at.x * 0.3, at.y * 0.008) + phase);
  const lines = perlin2d.sample(d.vec2f(at.x * 1.1 + figure * 2, at.y * 0.02) + phase * 2);
  const fine = std.pow(1 - std.abs(lines), 5);
  const sharp = std.saturate(2 - std.fwidth(at.x) * 8);
  const grain = std.mix(0.18, fine, sharp);

  const tone = std.saturate(0.25 + shade * 0.45 + figure * 0.3);
  const wood = std.mix(DARK, LIGHT, tone) * (1 - grain * 0.45);
  const seams = std.max(seam(across, 0.012), seam(along, 0.003));
  const far = std.smoothstep(FADE.x, FADE.y, std.distance(at, CENTER));
  return std.mix(wood, SEAM, seams * 0.8) * (1 - far * 0.85);
};

const Floor = instances.define('Floor', {
  material: (point) => {
    'use gpu';
    return materials.glossy(woodAt(point.position.xz), 0.78);
  },
});

/** Lays a walnut floor 40 m square under the run. */
export function createFloor(root: TgpuRoot, pool: MeshPool) {
  const boards = instances.create(root, pool, { kind: Floor, capacity: 1 });
  boards.add(pool.add(meshes.plane({ width: 4000, depth: 4000 })), { position: [0, 0, -400] });
  boards.flush();
  return boards;
}
