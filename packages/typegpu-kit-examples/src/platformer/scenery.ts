import { cameras, passes } from '@typegpu/kit';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { BRICKS, EMPTY, GROUND, type Level } from './level.ts';
import { CARDS, onCard, Seen, TEXELS } from './sprites.ts';

const SKY = d.vec3f(0.36, 0.6, 0.88);
const HAZE = d.vec3f(0.8, 0.88, 0.92);
const CLOUD = d.vec3f(0.98, 0.97, 0.94);
const CLOUD_BELLY = d.vec3f(0.83, 0.87, 0.93);
const FAR = d.vec3f(0.56, 0.71, 0.8);
const NEAR = d.vec3f(0.36, 0.62, 0.47);
const NEAR_CREST = d.vec3f(0.5, 0.74, 0.5);
const FAR_CREST = d.vec3f(0.68, 0.8, 0.86);
const GRASS = d.vec3f(0.38, 0.76, 0.3);
const GRASS_TIP = d.vec3f(0.62, 0.88, 0.4);
const GRASS_ROOT = d.vec3f(0.24, 0.5, 0.2);
const DIRT = d.vec3f(0.58, 0.38, 0.23);
const PEBBLE = d.vec3f(0.44, 0.27, 0.16);
const BRICK = d.vec3f(0.8, 0.42, 0.28);
const BRICK_EDGE = d.vec3f(0.93, 0.6, 0.42);
const MORTAR = d.vec3f(0.45, 0.23, 0.18);
const POLE = d.vec3f(0.9, 0.9, 0.85);
const PENNANT = d.vec3f(0.95, 0.3, 0.35);

/** Cloud, far-hill and near-hill parallax factors relative to camera movement. */
const PARALLAX = { clouds: 0.9, far: 0.7, near: 0.45 };
/**
 * Cumulus: the range of their flat bases above the horizon, the tiles between banks, the share of
 * clear sky between banks, and the smallest and largest puff in tiles.
 */
const CLOUDS = { base: [3.2, 5.2], spacing: 8, clear: 0.2, size: [0.8, 1.6] };
/** Tiles a second the clouds drift. */
const DRIFT = 0.3;
/** Radians a second the pennant waves. */
const FLUTTER = 6;
const TEXEL = 1 / TEXELS;

/** Moves a layer by whole texels, so its art keeps its pixels as the camera pans. */
const shift = (offset: d.v2f) => {
  'use gpu';
  return std.round(offset * TEXELS) / TEXELS;
};

/** Dirt and pebbles, under a lumpy grass edge where `open` says the tile above is empty. */
const ground = (world: d.v2f, open: boolean) => {
  'use gpu';
  const texel = std.floor(world * TEXELS);
  const depth = std.floor((std.ceil(world.y) - world.y) * TEXELS);
  const lumps = perlin2d.sample(texel * 0.21);

  if (open) {
    const grass = 3 + std.step(0, lumps) * 2;
    if (depth < 1) return GRASS_TIP;
    if (depth < grass - 1) return GRASS;
    if (depth < grass) return GRASS_ROOT;
  }
  return std.mix(DIRT, PEBBLE, std.step(0.35, lumps));
};

/** Bricks in a running bond, two courses a tile, mortar along their bottoms and left ends. */
const bricks = (world: d.v2f) => {
  'use gpu';
  const row = std.floor(world.y * 2);
  const run = world.x + (row % 2) * 0.5;
  const local = std.fract(d.vec2f(run, world.y * 2));
  if (local.x < TEXEL || local.y < 2 * TEXEL) return MORTAR;
  // A lit texel along each brick's top and left, as if each one were bevelled
  if (local.y > 1 - 2 * TEXEL || local.x < 2 * TEXEL) return BRICK_EDGE;
  return BRICK * (0.9 + 0.1 * perlin2d.sample(d.vec2f(std.floor(run), row) * 0.71));
};

/** Whether `at` lies in a cumulus: puffs of a size each bank picks, on a flat base. */
const cloudAt = (at: d.v2f) => {
  'use gpu';
  if (at.y < CLOUDS.base[0]) return false;
  const cell = std.floor(at.x / CLOUDS.spacing);
  // A bank may reach into the next cell, so test this cell's and the one before
  for (let k = -1; k <= 0; k++) {
    const bank = cell + d.f32(k);
    randf.seed(bank * 0.37 + 11);
    if (randf.sample() < CLOUDS.clear) continue;
    const size = std.mix(CLOUDS.size[0], CLOUDS.size[1], randf.sample());
    const middle = (bank + 0.3 + randf.sample() * 0.3) * CLOUDS.spacing;
    const base = std.mix(CLOUDS.base[0], CLOUDS.base[1], randf.sample());
    if (at.y < base) continue;
    for (let puff = 0; puff < 4; puff++) {
      const across = (d.f32(puff) - 1.3 + randf.sample() * 0.4) * size * 0.9;
      const radius =
        size * std.mix(0.45, 0.85, randf.sample()) * (1 - std.abs(across) / (size * 3));
      const centre = d.vec2f(middle + across, base + radius * 0.25);
      const offset = (at - centre) * d.vec2f(0.8, 1);
      if (std.dot(offset, offset) < radius * radius) return true;
    }
  }
  return false;
};

/**
 * Shades sky, clouds, hills, tiles and flag per texel in one full-screen pass, each on its card.
 */
export function createScenery(root: TgpuRoot, time: TgpuUniform<d.F32>, level: Level) {
  const { width, height } = level;
  const flag = d.vec2f(level.flag[0] - 0.5, level.flag[1] - 0.5);

  const tiles = root.createReadonly(d.arrayOf(d.u32, width * height), [...level.tiles]);

  const tileAt = (cell: d.v2i) => {
    'use gpu';
    if (cell.x < 0 || cell.x >= width) return d.u32(GROUND);
    if (cell.y < 0 || cell.y >= height) return d.u32(EMPTY);
    return tiles.$[cell.y * width + cell.x];
  };

  /** The cards behind the level, which scroll slower than it. */
  const backdrop = (world: d.v2f, eye: d.v2f) => {
    'use gpu';
    const near = world - shift(eye * PARALLAX.near);
    const rise = 1 + perlin2d.sample(d.vec2f(near.x * 0.2, 7.5)) * 1.5;
    if (near.y < rise) {
      // Darker toward the foot of the hills, in whole texel rows
      const below = std.floor((rise - near.y) * TEXELS) * TEXEL;
      if (below < 2 * TEXEL) return d.vec4f(NEAR_CREST, onCard(CARDS.near));
      return d.vec4f(NEAR * (1 - std.min(below, 3) * 0.05), onCard(CARDS.near));
    }

    const far = world - shift(eye * PARALLAX.far);
    const crest = 2.5 + perlin2d.sample(d.vec2f(far.x * 0.12, 1.5)) * 2;
    if (far.y < crest) {
      const below = std.floor((crest - far.y) * TEXELS) * TEXEL;
      if (below < 2 * TEXEL) return d.vec4f(FAR_CREST, onCard(CARDS.far));
      return d.vec4f(std.mix(FAR, HAZE, std.min(below, 4) * 0.06), onCard(CARDS.far));
    }

    const drift = world - shift(eye * PARALLAX.clouds - d.vec2f(time.$ * DRIFT, 0));
    if (cloudAt(drift)) {
      if (!cloudAt(drift - d.vec2f(0, 2 * TEXEL)))
        return d.vec4f(CLOUD_BELLY, onCard(CARDS.clouds));
      return d.vec4f(CLOUD, onCard(CARDS.clouds));
    }

    const sky = std.mix(HAZE, SKY, std.saturate((world.y - eye.y) / 6 + 0.5));
    return d.vec4f(sky, onCard(CARDS.sky));
  };

  /** The flag's pole and waving pennant, or zero alpha beside them. */
  const flagAt = (world: d.v2f) => {
    'use gpu';
    const p = world - flag;
    if (std.abs(p.x - 0.5) < TEXEL && p.y < 3.2 && p.y > 0) return d.vec4f(POLE, 1);

    const along = p.x - 0.56;
    const wave = std.sin(time.$ * FLUTTER - along * 5) * 0.06 * along;
    if (along > 0 && std.abs(p.y - 2.75 - wave) < 0.32 * (1 - along / 0.9)) {
      return d.vec4f(PENNANT, 1);
    }
    return d.vec4f();
  };

  /** The tile, or the cards behind, under each texel the camera sees. */
  const pixelOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
    'use gpu';
    const ndc = d.vec2f(screen.uv.x * 2 - 1, 1 - screen.uv.y * 2);
    const world = cameras.rayOf(cameras.current.$, ndc).origin.xy;
    const eye = cameras.current.$.position.xy;
    const front = onCard(CARDS.level);

    const cell = d.vec2i(std.floor(world));
    const kind = tileAt(cell);

    if (kind === GROUND) {
      const open = tileAt(cell + d.vec2i(0, 1)) === EMPTY;
      return Seen({ color: d.vec4f(ground(world, open), front) });
    }
    if (kind === BRICKS) return Seen({ color: d.vec4f(bricks(world), front) });
    const pole = flagAt(world);
    if (pole.w > 0) return Seen({ color: d.vec4f(pole.xyz, front) });
    return Seen({ color: backdrop(world, eye) });
  };

  const drawing: passes.Drawing<typeof passes.ScreenVertex, Seen> = {
    label: 'scenery',
    ...passes.fullScreen,
    pixelOf,
    color: passes.shaded,
    draw: (pass) => pass.draw(3),
  };

  return {
    drawing,

    destroy() {
      tiles.buffer.destroy();
    },
  };
}
