import { perlin2d, randf } from '@typegpu/noise';
import { common, d, std, tgpu, type TgpuRoot } from 'typegpu';
import { Finish, reachOf, warmed } from './card.ts';
import { bayerCell } from './dither.ts';

/** Most cells a poster has across and down: a 360 px card has 120. */
export const MAX_CELLS = 128;

/**
 * One card face, drawn in cells. Lengths are points of the face; the canvas starts at `origin`, so
 * its cells line up with the art window's.
 */
const Face = d.struct({
  /** The card's ground, sRGB. */
  ground: d.vec4f,
  /** The face's size, points a cell, and pixels of the canvas a cell: 1 for a texel a cell. */
  size: d.vec2f,
  cell: d.f32,
  texel: d.f32,
  /** Where the canvas's first cell starts. */
  origin: d.vec2f,
  /** The art window's left, top, right and bottom edges. */
  art: d.vec4f,
  /** Where the card leans, -1 to 1 across and down. */
  tilt: d.vec2f,
  /** How much the foil catches the light, 0 to 1. */
  sheen: d.f32,
  /** The flare's sweep across the face, -0.3 to 1.4, and its brightness. */
  flare: d.vec2f,
  /** The finish the card's rarity gives the face. */
  finish: d.align(16, Finish),
});

/** The face's uniform and its poster's cells, which `faceAt` reads. */
const faceLayout = tgpu.bindGroupLayout({
  face: { uniform: Face },
  /** The poster in cells, as its host shrinks it, from the top-left texel. */
  cells: { texture: d.texture2d(d.f32) },
});

/** Interference colours along `phase`. */
const rainbow = (phase: number) => {
  'use gpu';
  return std.cos((d.vec3f(phase) + d.vec3f(0, 0.33, 0.67)) * 6.2831853) * 0.5 + 0.5;
};

/**
 * Thin-film interference: a cosine palette over a film whose thickness swirls across the card and
 * shifts with the angle it is seen at, so the colours slide as the card leans.
 */
export const film = (uv: d.v2f, tilt: d.v2f) => {
  'use gpu';
  const swirl = perlin2d.sample(uv * d.vec2f(2.3, 3.2) + tilt * 0.4);
  const thickness = uv.x * 0.75 + uv.y * 1.1 + swirl * 0.55 + tilt.x * 0.9 - tilt.y * 0.7;

  return rainbow(thickness * 1.4);
};

/** The diagonal bands a holo foil is pressed with, sliding as the card leans. */
export const bands = (uv: d.v2f, tilt: d.v2f) => {
  'use gpu';
  const wave = std.sin((uv.x * 0.7 + uv.y) * 10 + tilt.x * 5 - tilt.y * 3);

  return std.smoothstep(-0.2, 1, wave);
};

/**
 * One cell's glint. About one cell in ten is a facet that faces its own direction, and flashes
 * when the card leans that way, or when the flare passes.
 */
export const glint = (cell: d.v2u, tilt: d.v2f, flare: number) => {
  'use gpu';
  randf.seed2(d.vec2f(cell));
  const facet = d.vec2f(randf.sample(), randf.sample()) * 2.6 - 1.3;
  const kept = std.step(0.9, randf.sample());
  const off = tilt - facet;

  return kept * (std.exp(-std.dot(off, off) * 26) + flare * 0.9);
};

/** `color` in six levels a channel through `threshold`, as the veil prints. */
export const print = (color: d.v3f, threshold: number) => {
  'use gpu';
  const level = std.saturate(color) * 6;
  const low = std.floor(level);

  return (low + std.step(d.vec3f(threshold), level - low)) / 6;
};

/** The face at cell `cell` from the canvas's first, opaque: its ground, foil and poster. */
export const faceAt = (cell: d.v2u) => {
  'use gpu';
  const f = faceLayout.$.face;
  const at = f.origin + (d.vec2f(cell) + 0.5) * f.cell;
  const art = f.art;
  const inArt = at.x >= art.x && at.y >= art.y && at.x < art.z && at.y < art.w;

  // In the window, cells count from its corner, as the poster's print counts them
  const artCell = d.vec2u(std.max(std.floor((at - art.xy) / f.cell), d.vec2f()));
  const own = std.select(cell, artCell, inArt);
  const threshold = bayerCell(own);
  const uv = at / f.size;

  const tilt = f.tilt;
  const colors = film(uv, tilt);
  const off = uv - (d.vec2f(0.5) + tilt * 0.6);
  const glare = std.exp(-std.dot(off, off) * 5);
  const swept = uv.x * 0.6 + uv.y * 0.4 - f.flare.x;
  const flare = std.exp(-swept * swept * 36) * f.flare.y;

  // The finish decides which parts are foil: the poster, or the whole face
  const reach = reachOf(f.finish, inArt);
  const foil = (f.sheen * (0.2 + 0.8 * bands(uv, tilt)) * (0.35 + 0.65 * glare) + flare) * reach;
  const sparkle = glint(own, tilt, flare) * (0.25 + f.sheen) * reach * f.finish.sparkle;
  const ground = f.ground.xyz;

  if (inArt) {
    // Foil shows most through the poster's darks, as through thin ink
    const poster = std.textureLoad(faceLayout.$.cells, d.vec2i(artCell), 0).xyz;
    const amount = foil * (1.1 - 0.8 * std.dot(poster, d.vec3f(0.299, 0.587, 0.114))) * 0.62;
    const lit =
      poster * (1 - 0.3 * std.saturate(amount)) +
      warmed(colors, f.finish.warm) * amount +
      d.vec3f(sparkle);
    return d.vec4f(std.max(ground, print(lit, threshold)), 1);
  }

  // The faintest foil would scatter lone cells over a card at rest, so it starts a little above zero
  const pressed = std.max(
    warmed(colors, f.finish.warm) * (foil * 0.22) - d.vec3f(0.025),
    d.vec3f(),
  );
  const light = pressed + d.vec3f(sparkle * 0.35);
  return d.vec4f(ground + print(light, threshold), 1);
};

const faceFragment = tgpu.fragmentFn({
  in: { position: d.builtin.position },
  out: d.vec4f,
})(({ position }) => {
  'use gpu';
  return faceAt(d.vec2u(position.xy / faceLayout.$.face.texel));
});

/** Draws a card face in cells: the poster's print with the holo foil over it. */
export function createFoil(root: TgpuRoot, format: GPUTextureFormat) {
  const face = root.createUniform(Face);
  const cells = root
    .createTexture({ size: [MAX_CELLS, MAX_CELLS], format: 'rgba8unorm' })
    .$usage('sampled', 'render');
  const group = root.createBindGroup(faceLayout, { face, cells });
  const pipeline = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: faceFragment,
      targets: { format },
    })
    .with(group);

  return {
    face,
    /** The face's group of `faceLayout`, for a pipeline of your own that calls `faceAt`. */
    group,
    /** The poster in cells, from the top-left texel, which `print` fills or a pass renders into. */
    cells,
    /** Draws three vertices over the whole face canvas. */
    pipeline,

    /** Puts a poster's cells, at most `MAX_CELLS` each way, in the corner of the texture. */
    print(pixels: Uint8ClampedArray<ArrayBuffer>, columns: number, rows: number) {
      const bytes = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
      cells.write(bytes, { size: [columns, rows, 1] });
    },
  };
}
