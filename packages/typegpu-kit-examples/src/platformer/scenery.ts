import { passes } from '@typegpu/kit';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { LAYERS, type Layers } from './layers.ts';
import { BRICKS, EMPTY, GROUND, type Level } from './level.ts';
import { CARDS, onCard, Seen, TEXELS } from './sprites.ts';

const HAZE = d.vec3f(0.8, 0.88, 0.92);
const CLOUD = d.vec3f(0.98, 0.97, 0.94);
const CLOUD_BELLY = d.vec3f(0.83, 0.87, 0.93);
const FAR = d.vec3f(0.56, 0.71, 0.8);
const NEAR = d.vec3f(0.36, 0.62, 0.47);
const NEAR_CREST = d.vec3f(0.5, 0.74, 0.5);
const FAR_CREST = d.vec3f(0.68, 0.8, 0.86);
const GRASS_TIP = d.vec3f(0.62, 0.88, 0.4);
/** The ground's greens from the surface down, each deeper and colder, as palette colours. */
const GREENS = [
  d.vec3f(0.38, 0.761, 0.29),
  d.vec3f(0.275, 0.533, 0.302),
  d.vec3f(0.18, 0.412, 0.243),
  d.vec3f(0.133, 0.29, 0.263),
];
const greens = tgpu.const(d.arrayOf(d.vec3f, GREENS.length), GREENS);
/** Ground tiles below the surface that the greens count down, the last green reaching on. */
const DEEPEST = GREENS.length - 1;
/**
 * The texel row from a tile's top where the next green takes over, on the tile's left half; the
 * right half's lies a texel lower, so the change waves across each tile.
 */
const STEP = 8;
const BRICK = d.vec3f(0.8, 0.42, 0.28);
const BRICK_EDGE = d.vec3f(0.93, 0.6, 0.42);
const MORTAR = d.vec3f(0.45, 0.23, 0.18);
const POLE = d.vec3f(0.9, 0.9, 0.85);
const PENNANT = d.vec3f(0.95, 0.3, 0.35);

/**
 * Cumulus: the range of their flat bases above the horizon, the tiles between banks, the share of
 * clear sky between banks, and the smallest and largest puff in tiles.
 */
const CLOUDS = { base: [3.2, 5.2], spacing: 8, clear: 0.2, size: [0.8, 1.6] };
/** Radians a second the pennant waves. */
const FLUTTER = 6;
const TEXEL = 1 / TEXELS;

/**
 * Grass all the way down, `below` tiles under the surface, a deeper green each tile. Each change
 * of green is ordered-dithered across the two texel rows above it and waves a texel between the
 * tile's halves. A lumpy, pale edge tops the tile where `below` is 0.
 */
const ground = (world: d.v2f, below: number) => {
  'use gpu';
  const texel = std.floor(world * TEXELS);
  const column = d.i32(texel.x) - d.i32(std.floor(world.x)) * TEXELS;
  const row = d.i32(std.floor((std.ceil(world.y) - world.y) * TEXELS));
  if (below === 0 && row < 1 + d.i32(std.step(0.3, perlin2d.sample(texel * 0.21)))) {
    return GRASS_TIP;
  }

  // Rows from the change of green, negative above it
  const from = row - (STEP + std.select(0, 1, column >= TEXELS >> 1));
  let deeper = from >= 0;
  // Every fourth texel two rows above the change, every other one in the row above
  if (from === -2) deeper = column % 4 === 1;
  if (from === -1) deeper = column % 2 === 0;
  return d.vec3f(greens.$[std.min(below + std.select(0, 1, deeper), GREENS.length - 1)]);
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

/** A backdrop layer's card at `at`, in its own tiles, or zero alpha where it shows the sky. */
const backdropAt = (layer: number, at: d.v2f) => {
  'use gpu';
  if (layer === LAYERS.near) {
    const rise = 1 + perlin2d.sample(d.vec2f(at.x * 0.2, 7.5)) * 1.5;
    if (at.y >= rise) return d.vec4f();
    // Darker toward the foot of the hills, in whole texel rows
    const below = std.floor((rise - at.y) * TEXELS) * TEXEL;
    if (below < 2 * TEXEL) return d.vec4f(NEAR_CREST, onCard(CARDS.near));
    return d.vec4f(NEAR * (1 - std.min(below, 3) * 0.05), onCard(CARDS.near));
  }

  if (layer === LAYERS.far) {
    const crest = 2.5 + perlin2d.sample(d.vec2f(at.x * 0.12, 1.5)) * 2;
    if (at.y >= crest) return d.vec4f();
    const below = std.floor((crest - at.y) * TEXELS) * TEXEL;
    if (below < 2 * TEXEL) return d.vec4f(FAR_CREST, onCard(CARDS.far));
    return d.vec4f(std.mix(FAR, HAZE, std.min(below, 4) * 0.06), onCard(CARDS.far));
  }

  if (!cloudAt(at)) return d.vec4f();
  if (!cloudAt(at - d.vec2f(0, 2 * TEXEL))) return d.vec4f(CLOUD_BELLY, onCard(CARDS.clouds));
  return d.vec4f(CLOUD, onCard(CARDS.clouds));
};

/**
 * Shades clouds, hills, tiles and flag per texel, each on its card: `backdrop` into whichever
 * backdrop layer its pass draws, `drawing` into the level layer, transparent around the tiles.
 */
export function createScenery(
  root: TgpuRoot,
  time: TgpuUniform<d.F32>,
  level: Level,
  layers: Layers,
) {
  const { width, height } = level;
  const flag = d.vec2f(level.flag[0] - 0.5, level.flag[1] - 0.5);

  const tiles = root.createReadonly(d.arrayOf(d.u32, width * height), [...level.tiles]);

  const tileAt = (cell: d.v2i) => {
    'use gpu';
    if (cell.x < 0 || cell.x >= width) return d.u32(GROUND);
    if (cell.y < 0 || cell.y >= height) return d.u32(EMPTY);
    return tiles.$[cell.y * width + cell.x];
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

  /** The tile or the flag under each texel of the level layer. */
  const pixelOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
    'use gpu';
    const world = layers.pointOf(LAYERS.level, screen.uv);
    const front = onCard(CARDS.level);

    const cell = d.vec2i(std.floor(world));
    const kind = tileAt(cell);

    if (kind === GROUND) {
      // Ground tiles above this one, up to the deepest the greens count
      let below = 0;
      for (let k = 1; k <= DEEPEST; k++) {
        if (below === k - 1 && tileAt(cell + d.vec2i(0, k)) === GROUND) below = k;
      }
      return Seen({ color: d.vec4f(ground(world, below), front) });
    }
    if (kind === BRICKS) return Seen({ color: d.vec4f(bricks(world), front) });
    const pole = flagAt(world);
    if (pole.w > 0) return Seen({ color: d.vec4f(pole.xyz, front) });
    return Seen({ color: d.vec4f() });
  };

  /** The backdrop layer that the pass draws into, in its own tiles. */
  const backdropOf = (screen: d.Infer<typeof passes.ScreenVertex>) => {
    'use gpu';
    const layer = passes.layer.$;
    return Seen({ color: backdropAt(layer, layers.pointOf(layer, screen.uv)) });
  };

  const drawing: passes.Drawing<typeof passes.ScreenVertex, Seen> = {
    label: 'scenery',
    ...passes.fullScreen,
    pixelOf,
    color: passes.shaded,
    draw: (pass) => pass.draw(3),
  };
  const backdrop: passes.Drawing<typeof passes.ScreenVertex, Seen> = {
    label: 'backdrop',
    ...passes.fullScreen,
    pixelOf: backdropOf,
    color: passes.shaded,
    draw: (pass) => pass.draw(3),
  };

  return {
    /** Add it to the level layer's pass. */
    drawing,
    /** Add it to a pass recorded into each backdrop layer. */
    backdrop,

    destroy() {
      tiles.buffer.destroy();
    },
  };
}
