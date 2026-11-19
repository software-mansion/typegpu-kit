import { perlin2d, randf } from '@typegpu/noise';
import * as sdf from '@typegpu/sdf';
import { d, std } from 'typegpu';
import { CANDY, CREAM } from './candy.ts';

/** Flowers in meadows and gumdrop pebbles over the lawn: cell size, share of cells and size. */
const FLOWER = { cell: 0.12, share: 0.75, size: 0.02, meadow: 0.9 };
const DROP = { cell: 0.11, share: 0.08, size: 0.011 };
/** Tufts of grass drawn as three strokes in some of the pebbles' cells: share and size. */
const TUFT = { share: 0.3, size: 0.026, color: d.vec3f(0.4, 0.76, 0.5) };
/** How far behind the front marks wait to pop up, how long they take, and how far they overshoot. */
const BLOOM = { start: 0.03, stagger: 0.3, span: 0.12, overshoot: 0.35 };
/**
 * Candy stepping stones from the lever across the front of the tray: where the path starts and
 * ends in x, its middle in z and how it winds, and the stones' spacing, size and stagger.
 */
const PATH = {
  from: -2.8,
  to: 0.98,
  line: 0.52,
  sway: 0.1,
  wave: 2.1,
  phase: -2.27,
  spacing: 0.1,
  size: 0.03,
  stagger: 0.016,
};

/** A mark on the lawn: its colour and coverage, how it tilts the lawn and the shade around it. */
export const Decal = d.struct({ color: d.vec3f, cover: d.f32, tilt: d.vec2f, shade: d.f32 });

const clear = () => {
  'use gpu';
  return Decal({ color: d.vec3f(), cover: 0, tilt: d.vec2f(), shade: 1 });
};

/** How far a mark has popped up `depth` behind the front, overshooting a little on the way. */
const bloomOf = (depth: number, delay: number) => {
  'use gpu';
  const grown = std.saturate((depth - BLOOM.start - delay * BLOOM.stagger) / BLOOM.span);
  return grown + std.sin(grown * Math.PI) * BLOOM.overshoot;
};

/** A five-petal flower in some cells of the lawn, raised a little off it. */
export const flowerAt = (at: d.v2f, soft: number, depth: number) => {
  'use gpu';
  const cell = std.floor(at / FLOWER.cell);
  // Flowers gather in meadows rather than spread evenly
  const meadow = perlin2d.sample(cell * (FLOWER.cell * FLOWER.meadow) + 4.2);
  randf.seed2(cell + 0.37);
  if (randf.sample() > FLOWER.share * std.smoothstep(0.05, 0.45, meadow)) return clear();
  const jitter = d.vec2f(randf.sample(), randf.sample()) - 0.5;
  const centre = (cell + 0.5 + jitter * 0.45) * FLOWER.cell;
  const scale = FLOWER.size * (0.75 + randf.sample() * 0.5);
  const pick = randf.sample();
  const spin = randf.sample() * 6.28;
  const bloom = bloomOf(depth, randf.sample());
  if (bloom < 0.02) return clear();
  const size = scale * bloom;
  const local = at - centre;
  const r = std.length(local);
  const angle = std.atan2(local.y, local.x) + spin + (1 - bloom) * 2;
  const petals = r - size * (0.55 + 0.45 * std.abs(std.cos(angle * 2.5)));
  const cover = std.smoothstep(soft, -soft, petals);
  const middle = std.smoothstep(soft, -soft, r - size * 0.3);
  let petal = d.vec3f(CANDY.white);
  if (pick < 0.3) petal = CANDY.pink;
  else if (pick < 0.5) petal = CANDY.lilac;
  else if (pick < 0.65) petal = CANDY.butter;
  return Decal({
    color: std.mix(petal, CANDY.middle, middle),
    cover,
    tilt: (local / size) * (0.5 * cover),
    shade: 1 - std.smoothstep(size * 1.4, size * 0.6, r) * (1 - cover) * 0.3,
  });
};

/** A tuft of grass, three strokes fanning up the view from `local`'s origin. */
const tuftAt = (local: d.v2f, size: number, soft: number) => {
  'use gpu';
  const q = local / size + d.vec2f(0, 0.45);
  const blades = std.min(
    std.min(
      sdf.sdBezierApprox(q, d.vec2f(-0.1, 0), d.vec2f(-0.2, -0.55), d.vec2f(-0.7, -0.8)),
      sdf.sdBezierApprox(q, d.vec2f(0, 0), d.vec2f(0, -0.6), d.vec2f(0.1, -1.15)),
    ),
    sdf.sdBezierApprox(q, d.vec2f(0.1, 0), d.vec2f(0.25, -0.5), d.vec2f(0.65, -0.7)),
  );
  return Decal({
    color: TUFT.color,
    cover: std.smoothstep(soft, -soft, (blades - 0.13) * size) * 0.8,
    tilt: d.vec2f(),
    shade: 1,
  });
};

/**
 * A gumdrop pebble, a glossy dome with a soft shadow round it, or a tuft of grass in some cells
 * of the lawn.
 */
export const dropAt = (at: d.v2f, soft: number, depth: number) => {
  'use gpu';
  const cell = std.floor(at / DROP.cell);
  randf.seed2(cell + 7.1);
  const chance = randf.sample();
  if (chance > DROP.share + TUFT.share) return clear();
  const jitter = d.vec2f(randf.sample(), randf.sample()) - 0.5;
  const scale = 0.7 + randf.sample() * 0.6;
  const bloom = bloomOf(depth, randf.sample());
  if (bloom < 0.02) return clear();
  if (chance > DROP.share) {
    const size = TUFT.size * scale * bloom;
    return tuftAt(at - (cell + 0.5 + jitter * 0.4) * DROP.cell, size, soft);
  }
  const centre = (cell + 0.5 + jitter * 0.5) * DROP.cell;
  const size = DROP.size * scale * bloom;
  const pick = randf.sample();
  const local = at - centre;
  const r = std.length(local);
  const cover = std.smoothstep(soft, -soft, r - size);
  let color = d.vec3f(CANDY.pink);
  if (pick < 0.25) color = CANDY.sky;
  else if (pick < 0.5) color = CANDY.butter;
  else if (pick < 0.7) color = CANDY.lilac;
  return Decal({
    color,
    cover,
    tilt: (local / size) * (1.3 * cover),
    shade: 1 - std.smoothstep(size * 1.8, size, r) * (1 - cover) * 0.4,
  });
};

/** A stepping stone of the path where one lies, glossy and domed with a sugary pink rim. */
export const stoneAt = (at: d.v2f, soft: number, depth: number) => {
  'use gpu';
  const i = std.floor(at.x / PATH.spacing);
  const x = (i + 0.5) * PATH.spacing;
  if (x < PATH.from || x > PATH.to) return clear();
  const side = std.fract(i * 0.5) * 4 - 1;
  const z = PATH.line + PATH.sway * std.sin(x * PATH.wave + PATH.phase) + side * PATH.stagger;
  const local = (at - d.vec2f(x, z)) / d.vec2f(1.15, 1);
  randf.seed(i + 2.3);
  const bloom = bloomOf(depth, randf.sample());
  if (bloom < 0.02) return clear();
  const size = PATH.size * bloom * (0.85 + randf.sample() * 0.3);
  const r = std.length(local);
  const cover = std.smoothstep(soft, -soft, r - size);
  const rim = std.smoothstep(size * 0.62, size * 0.72, r);
  return Decal({
    color: std.mix(CREAM, CANDY.pink, rim * 0.7),
    cover,
    tilt: (local / size) * (0.9 * cover),
    shade: 1 - std.smoothstep(size * 1.5, size, r) * (1 - cover) * 0.4,
  });
};
