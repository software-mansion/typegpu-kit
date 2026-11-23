import { baker, cameras, pipelines, type CameraBase } from '@typegpu/kit';
import type { Injectable } from '@typegpu/kit/inject';
import { perlin2d, randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { CRITTERS } from './body.ts';
import { jelly, spanOf } from './candy.ts';
import { dropAt, flowerAt, stoneAt } from './decals.ts';
import { Critter, glimpse, marchCritter, shadeCritter, underCritters } from './figures.ts';
import { Floor, Found, type Look, type Moment, type Scenery } from './jelly.ts';
import { boundOf, castAt, marchProp, Prop, shadeProp, shownProp } from './props.ts';
import { GROUND } from './tray.ts';

/**
 * The lawn's colours, how far its edge reaches with noise, and farther than which from the front
 * the noise changes nothing, how much its ripples and swells tilt it, its gloss and glow, and how
 * far behind the front its fresh glow fades.
 */
const LAWN = {
  color: d.vec3f(0.58, 0.88, 0.56),
  meadow: d.vec3f(0.86, 0.94, 0.55),
  slot: d.vec3f(0.98, 0.66, 0.78),
  ragged: 0.22,
  settled: 1.2,
  wobble: 0.035,
  swell: 0.6,
  finish: d.vec2f(0.15, 0.25),
  fresh: 0.15,
  glow: d.vec3f(0.85, 1, 0.62),
};
/**
 * The jelly lip at the front: how far in it rises, how steeply its edge falls to the paper, how
 * clear its thin edge is, and how far out it focuses a bright line onto the paper.
 */
const LIP = { width: 0.08, tilt: 2.6, clear: 0.6, caustic: 0.02 };
/** Bubbles in the young jelly behind the front, and sparkles on its lip. */
const BUBBLE = { cell: 0.05, share: 0.3, size: 0.006, reach: 0.2 };
const SPARKLE = { cell: 0.08, size: 0.022 };
/** The floor that props' shadows are baked over, in world x and z, and its texels. */
const SHADED = { left: -3, right: 3, back: -2.2, front: 1.6, width: 512, height: 320 };
/**
 * Pixels on a side of the screen tiles that list the props and critters they may show before the
 * march, and the tiles the list holds, enough for a march 5120 by 2880 pixels. Pixels of tiles past
 * them try every prop and critter.
 */
const TILE = 8;
const TILES = (5120 / TILE) * (2880 / TILE);
const CULL_GROUP = 8;

/** Indices `find` returns for critters start here, after the props. */
const FIRST_CRITTER = 32;

/**
 * Marches the shapes whose bits are set in `passed`, nearest entry first, until the next enters
 * past the hit in `before`. `bound` gives shape `i`'s bounding sphere and `march` where a ray
 * first meets it; a hit's index is `first + i`.
 */
const nearestOf = (
  bound: (i: number) => d.v4f,
  march: (eye: d.v3f, direction: d.v3f, i: number, start: number, stop: number) => number,
  first: number,
) => {
  const nearestHit = (
    eye: d.v3f,
    direction: d.v3f,
    passed: number,
    before: d.Infer<typeof Found>,
  ) => {
    'use gpu';
    let mask = d.u32(passed);
    let found = Found({ t: before.t, index: before.index });
    while (mask !== 0) {
      let next = d.i32(0);
      let span = d.vec2f(1e9, 0);
      let left = d.u32(mask);
      while (left !== 0) {
        const i = d.i32(std.countTrailingZeros(left));
        left = left & (left - 1);
        const at = spanOf(eye, direction, bound(i));
        if (at.x < span.x) {
          span = d.vec2f(at);
          next = i;
        }
      }
      if (span.x >= found.t) break;
      mask = mask ^ (d.u32(1) << d.u32(next));
      const t = march(eye, direction, next, span.x, std.min(span.y, found.t));
      if (t >= 0) found = Found({ t, index: first + next });
    }
    return found;
  };
  return nearestHit;
};

const nearestProp = nearestOf(boundOf, marchProp, 0);
const nearestCritter = nearestOf(Critter.bound, marchCritter, FIRST_CRITTER);

/** The march's size in pixels and in tiles. */
const Grid = d.struct({ size: d.vec2f, tiles: d.vec2u });

/** Whether a sphere may show in the cone round `axis` from `eye`, `spread` radians wide. */
const inCone = (sphere: d.v4f, eye: d.v3f, axis: d.v3f, spread: number) => {
  'use gpu';
  const to = sphere.xyz - eye;
  const distance = std.length(to);
  if (distance <= sphere.w) return true;
  const angle = std.acos(std.clamp(std.dot(to, axis) / distance, -1, 1));
  return angle <= spread + std.asin(sphere.w / distance);
};

/** The direction of the ray through `pixel` of a march `size` pixels wide. */
const rayAt = (pixel: d.v2f, size: d.v2f) => {
  'use gpu';
  const uv = pixel / size;
  return cameras.rayOf(cameras.current.$, d.vec2f(uv.x, 1 - uv.y) * 2 - 1).direction;
};

/**
 * The nearest of the props and critters set in `masks`, x for props and y for critters, along a
 * ray before `limit`, or index -1. Each one the ray passes marches alone within its bounding
 * sphere, nearest entry first, until the next enters past a hit.
 */
const nearestIn = (eye: d.v3f, direction: d.v3f, limit: number, masks: d.v2u) => {
  'use gpu';
  const count = d.i32(Prop.span());
  let shown = d.u32(masks.x);
  let props = d.u32(0);
  while (shown !== 0) {
    const k = d.i32(std.countTrailingZeros(shown));
    shown = shown & (shown - 1);
    if (k >= count || !shownProp(k)) continue;
    const span = spanOf(eye, direction, boundOf(k));
    if (span.y <= 0 || span.x > limit) continue;
    props = props | (d.u32(1) << d.u32(k));
  }
  let near = d.u32(masks.y);
  let crowd = d.u32(0);
  while (near !== 0) {
    const c = d.i32(std.countTrailingZeros(near));
    near = near & (near - 1);
    if (c >= CRITTERS) continue;
    const bound = Critter.bound(c);
    if (bound.w < 0.001) continue;
    const span = spanOf(eye, direction, bound);
    if (span.y <= 0 || span.x > limit) continue;
    crowd = crowd | (d.u32(1) << d.u32(c));
  }
  const nearest = nearestProp(eye, direction, props, Found({ t: limit, index: -1 }));
  return nearestCritter(eye, direction, crowd, nearest);
};

/**
 * Bubbles in the young jelly `depth` behind the front and stars twinkling on its `fresh` lip, as
 * brightness at `p`, with `soft` a pixel's width on the floor.
 */
const glintsAt = (p: d.v3f, depth: number, soft: number, fresh: number, time: number) => {
  'use gpu';
  if (depth >= BUBBLE.reach) return d.f32(0);
  let sparkle = d.f32(0);
  const fade = 1 - depth / BUBBLE.reach;
  const cell = std.floor(p.xz / BUBBLE.cell);
  randf.seed2(cell + 3.3);
  if (randf.sample() < BUBBLE.share) {
    const jitter = d.vec2f(randf.sample(), randf.sample()) - 0.5;
    const local = p.xz - (cell + 0.5 + jitter * 0.5) * BUBBLE.cell;
    const size = BUBBLE.size * (0.5 + randf.sample());
    const ring = std.smoothstep(soft, -soft, std.abs(std.length(local) - size) - size * 0.2);
    const glint = std.smoothstep(size * 0.45, 0, std.length(local + size * 0.4));
    sparkle += (ring * 0.3 + glint * 0.7) * fade;
  }
  // Sparkles twinkle on the fresh jelly, each in its own time
  const spot = std.floor(p.xz / SPARKLE.cell);
  randf.seed2(spot + 5.9);
  const phase = randf.sample() * 6.28;
  const jitter = d.vec2f(randf.sample(), randf.sample()) - 0.5;
  const local = std.abs(p.xz - (spot + 0.5 + jitter * 0.6) * SPARKLE.cell);
  const twinkle = std.max(0, std.sin(time * 2.2 + phase)) ** 6 * fresh;
  const size = SPARKLE.size * twinkle + 1e-5;
  const star = std.max(
    1 - local.x / size - local.y / (size * 0.25),
    1 - local.y / size - local.x / (size * 0.25),
  );
  sparkle += std.saturate(star) * 1.2 + std.smoothstep(size * 0.3, 0, std.length(local));
  // Later random draws in this pixel stay different every frame
  randf.seed3(d.vec3f(p.xz, time));
  return sparkle;
};

/** How far back whatever `find` found was when the last frame was drawn. Props stand still. */
const moved = (index: number) => {
  'use gpu';
  if (index >= FIRST_CRITTER) return Critter.moved(index - FIRST_CRITTER).xyz;
  return d.vec3f();
};

/**
 * The jelly world on the tray: finding and shading its props and critters, and the lawn they stand
 * on, as the jelly's scenery. Pipelines that draw it fill perlin2d's cache.
 */
export function createLand(
  root: TgpuRoot,
  options: {
    /** The camera the march sees through, whose tiles list what they may show. */
    readonly camera: CameraBase;
    /** The jelly world's props, from `createWorld`, whose shadows the land bakes. */
    readonly world: Injectable;
    /** The critters, from `createCritters`, which stand on it. */
    readonly critters: Injectable;
    readonly now: TgpuUniform<typeof Moment>;
    /** The light, whose direction the props' shadows are baked along. */
    readonly look: TgpuUniform<typeof Look>;
  },
) {
  const { camera, world, critters, now, look } = options;
  const shadows = root
    .createTexture({ size: [SHADED.width, SHADED.height], format: 'rgba8unorm' })
    .$usage('storage', 'sampled');
  const shadowsRead = shadows.createView(d.texture2d(d.f32));
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const tiles = root.createBuffer(d.arrayOf(d.vec2u, TILES)).$usage('storage');
  const tilesWritten = tiles.as('mutable');
  const tilesRead = tiles.as('readonly');
  const grid = root.createUniform(Grid);
  const measured = { width: 0, height: 0 };

  /** The nearest prop or critter along a ray through `pixel` before `limit`, or index -1. */
  const find = (eye: d.v3f, direction: d.v3f, limit: number, pixel: d.v2f) => {
    'use gpu';
    const tile = d.vec2u(pixel) / TILE;
    const index = tile.y * grid.$.tiles.x + tile.x;
    let masks = d.vec2u(0xffffffff);
    if (tile.x < grid.$.tiles.x && index < TILES) masks = d.vec2u(tilesRead.$[index]);
    return nearestIn(eye, direction, limit, masks);
  };

  /** The colour of whatever `find` found. */
  const shade = (p: d.v3f, direction: d.v3f, index: number, light: d.v3f) => {
    'use gpu';
    if (index >= FIRST_CRITTER) {
      return shadeCritter(p, direction, index - FIRST_CRITTER, light, now.$.time);
    }
    return shadeProp(p, direction, index, light);
  };

  /** How far into the jelly world a point is: positive behind the front. */
  const behind = (p: d.v3f) => {
    'use gpu';
    const ahead = now.$.front - p.x;
    if (std.abs(ahead) > LAWN.settled) return ahead;
    const at = p.xz * 1.8 + d.vec2f(0, now.$.time * 0.15);
    const ragged = perlin2d.sample(at) + perlin2d.sample(at * 2.3 + 5.1) * 0.3;
    return ahead + ragged * LAWN.ragged;
  };

  /** The props' baked shadows on the floor under `p`, as `castAt` gives them. */
  const shadowAt = (p: d.v3f) => {
    'use gpu';
    const uv = d.vec2f(
      (p.x - SHADED.left) / (SHADED.right - SHADED.left),
      (p.z - SHADED.back) / (SHADED.front - SHADED.back),
    );
    if (std.any(std.lt(uv, d.vec2f())) || std.any(std.gt(uv, d.vec2f(1)))) return d.vec4f(1);
    return std.textureSampleLevel(shadowsRead.$, sampler.$, uv, 0);
  };

  /**
   * The floor at a point near or behind the front: the jelly lawn with its flowers and gumdrops,
   * the lip where it meets the paper with bubbles and sparkles in it, and the bright line it
   * focuses onto `printed`. `soft` is a pixel's width on the floor, `ink` the label's cover and
   * `sunlit` the slider's shadow.
   */
  const lawn = (
    p: d.v3f,
    normal: d.v3f,
    direction: d.v3f,
    light: d.v3f,
    depth: number,
    printed: d.v3f,
    ink: number,
    soft: number,
    sunlit: d.v3f,
  ) => {
    'use gpu';
    const time = now.$.time;
    let outward = d.vec2f(-1, 0);
    if (std.abs(depth) < LIP.width) {
      const slope =
        d.vec2f(behind(p + d.vec3f(0.01, 0, 0)), behind(p + d.vec3f(0, 0, 0.01))) - depth;
      outward = std.normalize(slope * -1 + d.vec2f(-1e-6, 0));
    }
    const lens = std.saturate(std.dot(outward, std.normalize(light.xz)) * 0.6 + 0.4);
    // `**` compiles to `pow`, which is NaN for a negative base on most GPUs
    const off = (depth + LIP.caustic) / (LIP.caustic * 0.55);
    const line = std.exp(-off * off);
    const near = depth / 0.007;
    const pinch = std.exp(-near * near);
    const paper =
      printed * std.mix(d.vec3f(1), LAWN.color * 0.7, pinch * 0.45) +
      LAWN.color * LAWN.color * (line * lens * 0.9);
    if (depth < -soft) return paper;

    const at = p.xz * 5 + d.vec2f(time * 0.4, time * 0.3);
    const here = perlin2d.sample(at);
    const wobble = d.vec2f(
      here - perlin2d.sample(at + d.vec2f(0.15, 0)),
      here - perlin2d.sample(at + d.vec2f(0, 0.15)),
    );
    const lip = (1 - std.saturate(depth / LIP.width)) ** 2;
    // Light bends away inside the edge, which shows as a deeper band
    const inside = (depth - LIP.width * 0.3) / (LIP.width * 0.22);
    const band = std.exp(-inside * inside);
    // Broad swells that roll the light across the lawn, and lighter meadows on their tops
    const broad = p.xz * 1.1 + 11.7;
    const swell = perlin2d.sample(broad);
    const roll = d.vec2f(
      swell - perlin2d.sample(broad + d.vec2f(0.2, 0)),
      swell - perlin2d.sample(broad + d.vec2f(0, 0.2)),
    );
    let base = std.mix(LAWN.color, LAWN.meadow, std.smoothstep(0.35, 0.85, swell * 0.5 + 0.5));
    // Fresh jelly just behind the front is brighter, and fades to the lawn's own colour
    const fresh = std.exp(-depth / LAWN.fresh);
    base = std.mix(base, LAWN.glow, fresh * 0.5);
    let tilt = wobble * LAWN.wobble + roll * LAWN.swell + outward * (LIP.tilt * lip);
    let dimmed = d.f32(1);
    if (p.y < 0) {
      base = d.vec3f(LAWN.slot);
    } else if (normal.y > 0.95) {
      const flower = flowerAt(p.xz, soft, depth);
      const drop = dropAt(p.xz, soft, depth);
      const stone = stoneAt(p.xz, soft, depth);
      base = std.mix(base, flower.color, flower.cover);
      base = std.mix(base, drop.color, drop.cover);
      base = std.mix(base, stone.color, stone.cover);
      tilt = std.mix(tilt + flower.tilt + drop.tilt, stone.tilt, stone.cover);
      dimmed = flower.shade * drop.shade * stone.shade;
    }

    const sparkle = glintsAt(p, depth, soft, fresh, time);

    const bent = std.normalize(normal + d.vec3f(tilt.x, 0, tilt.y));
    const cover = shadowAt(p);
    const top = p.y > 0;
    const thick = std.select(0.25, 0.2 + band * 0.6, top);
    // The slot's floor sits in the slider's shade already, so only the tray's top takes it
    const sun = cover.xyz * std.select(d.vec3f(1), std.mix(d.vec3f(1), sunlit, 0.5), top);
    // The lip is fresh and wet, glossier than the lawn behind it
    const finish = std.mix(LAWN.finish, d.vec2f(1, LAWN.finish.y), lip);
    const lit = jelly(base, bent, direction, light, thick, sun, finish) * (cover.w * dimmed);
    const printedOn = std.mix(lit, lit * 0.45, ink) + d.vec3f(1, 0.97, 0.9) * sparkle;
    // Paper shows through where the young jelly is thinnest
    const seen = printed * std.mix(d.vec3f(1), base * 1.1, 0.6);
    const thin = 1 - std.saturate(depth / (LIP.width * 0.2));
    const grown = std.mix(printedOn, seen, LIP.clear * thin);
    return std.mix(paper, grown, std.smoothstep(-soft, soft, depth));
  };

  /** The lawn behind the front, and the critters' shadows on the floor. */
  const ground = (point: d.Infer<typeof Floor>, light: d.v3f) => {
    'use gpu';
    const depth = behind(point.at);
    const under = underCritters(point.at);
    if (depth < -0.04) {
      // The bare tray takes the shadows of what stands on it, such as the lever
      const cover = shadowAt(point.at);
      return point.color * cover.xyz * (cover.w * under);
    }
    const covered = lawn(
      point.shaken,
      point.normal,
      point.direction,
      light,
      depth,
      point.color,
      point.ink,
      point.soft,
      point.shadow,
    );
    const sheltered = std.select(1, std.mix(1, point.open, 0.6), point.shaken.y > 0);
    return (covered * sheltered + point.bounce * (0.5 * std.smoothstep(0, 0.02, depth))) * under;
  };

  const scenery: Scenery = {
    find: tgpu.fn(find).pipe(world.inject()).pipe(critters.inject()),
    shade: tgpu.fn(shade).pipe(world.inject()).pipe(critters.inject()),
    moved: tgpu.fn(moved).pipe(critters.inject()),
    glimpse: tgpu.fn(glimpse).pipe(critters.inject()),
    ground: tgpu.fn(ground).pipe(critters.inject()),
  };

  const cast = (uv: d.v2f) => {
    'use gpu';
    const p = d.vec3f(
      std.mix(SHADED.left, SHADED.right, uv.x),
      GROUND,
      std.mix(SHADED.back, SHADED.front, uv.y),
    );
    return castAt(p, look.$.light);
  };
  const baking = baker.create(root, shadows, {
    texel: tgpu.fn(cast).pipe(world.inject()),
  });

  const cullTiles = tgpu.computeFn({
    workgroupSize: [CULL_GROUP, CULL_GROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const across = grid.$.tiles;
    if (gid.x >= across.x || gid.y >= across.y) return;
    const index = gid.y * across.x + gid.x;
    if (index >= TILES) return;
    const size = grid.$.size;
    // A pixel's margin round the tile covers TAA's jitter
    const low = d.vec2f(gid.xy) * TILE - 1;
    const high = d.vec2f(gid.xy + 1) * TILE + 1;
    const a = rayAt(low, size);
    const b = rayAt(d.vec2f(high.x, low.y), size);
    const c = rayAt(d.vec2f(low.x, high.y), size);
    const e = rayAt(high, size);
    const axis = std.normalize(a + b + c + e);
    const nearest = std.min(
      std.min(std.dot(axis, a), std.dot(axis, b)),
      std.min(std.dot(axis, c), std.dot(axis, e)),
    );
    const spread = std.acos(std.clamp(nearest, -1, 1));
    const eye = cameras.rayOf(cameras.current.$, d.vec2f()).origin;

    let props = d.u32(0);
    const count = d.i32(Prop.span());
    for (let k = 0; k < count; k++) {
      if (!shownProp(k)) continue;
      if (inCone(boundOf(k), eye, axis, spread)) props = props | (d.u32(1) << d.u32(k));
    }
    let crowd = d.u32(0);
    for (let k = 0; k < CRITTERS; k++) {
      const bound = Critter.bound(k);
      if (bound.w < 0.001) continue;
      if (inCone(bound, eye, axis, spread)) crowd = crowd | (d.u32(1) << d.u32(k));
    }
    tilesWritten.$[index] = d.vec2u(props, crowd);
  });
  const culling = root
    .pipe(camera.inject())
    .pipe(world.inject())
    .pipe(critters.inject())
    .createComputePipeline({ compute: cullTiles });

  return {
    scenery,
    bakeShadows: baking.record,
    initAsync: () => pipelines.compile([baking, culling]),
    /**
     * Lists what each tile of a march `width` by `height` pixels may show, after the props and
     * critters move.
     */
    cull(pass: TgpuComputePass, width: number, height: number) {
      if (width !== measured.width || height !== measured.height) {
        measured.width = width;
        measured.height = height;
        grid.write({
          size: d.vec2f(width, height),
          tiles: d.vec2u(Math.ceil(width / TILE), Math.ceil(height / TILE)),
        });
      }
      pass.setPipeline(culling);
      pass.dispatchWorkgroups(
        Math.ceil(width / TILE / CULL_GROUP),
        Math.ceil(height / TILE / CULL_GROUP),
      );
    },
    destroy() {
      for (const made of [shadows, tiles]) made.destroy();
    },
  };
}
