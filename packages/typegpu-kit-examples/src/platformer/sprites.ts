import { meshes } from '@typegpu/kit/geometry';
import { sdDisk } from '@typegpu/sdf';
import { d, std, tgpu, type TgpuUniform } from 'typegpu';

/** Sprite quad spans -1 to 1 on xz, with two triangles. */
export const quad = meshes.unindexed(meshes.plane({ width: 2, depth: 2 }));

/** Texels per tile. Each layer draws on its own grid of them, so its art keeps its pixels. */
export const TEXELS = 16;

/**
 * The paper cards, back to front. Each pixel of the art stores its card in alpha, so the paper
 * pass knows which card it lies on and which cards stand in front of it.
 */
export const CARDS = { sky: 0, clouds: 1, far: 2, near: 3, level: 4, sprites: 5 };
const TOP = CARDS.sprites;

/** The alpha that marks a pixel of the art as lying on `card`. */
export const onCard = (card: number) => {
  'use gpu';
  return card / TOP;
};
/** The card a pixel's alpha marks, fractional between filtered texels. */
export const cardOf = (alpha: number) => {
  'use gpu';
  return alpha * TOP;
};

/** What every drawing of the game reads besides the camera, which the scene's description fills. */
export interface ArtOptions {
  /** Frame-clock time in seconds. */
  readonly time: TgpuUniform<d.F32>;
}

/** A pixel's colour and card, which every drawing of the game writes unchanged. */
export const Seen = d.struct({ color: d.vec4f });
export type Seen = d.Infer<typeof Seen>;

/** A sprite's colour on the front card. */
export const sprite = (color: d.v3f) => {
  'use gpu';
  return Seen({ color: d.vec4f(color, onCard(CARDS.sprites)) });
};

/**
 * Pixels of what sprites are drawn into per tile along each axis. The sprites' pass fills it with
 * the surface's, and it defaults to the art's texels.
 */
export const grid = tgpu.accessor(d.vec2f, d.vec2f(TEXELS));

/** Rounds a world position to the pixels sprites are drawn into, so sprites glide without blur. */
export const snapped = (world: d.v2f) => {
  'use gpu';
  return std.round(world * grid.$) / grid.$;
};

/**
 * Moves a sprite's `local` coordinate to the centre of its texel, so the sprite keeps square
 * texels of its own as it moves by pixels. `texel` is the same point's offset in texels from the
 * sprite's snapped position, and both vary linearly across the sprite's quad.
 */
export const pixelated = (local: d.v2f, texel: d.v2f) => {
  'use gpu';
  const perTexel = d.vec2f(
    std.dpdx(local).x / std.dpdx(texel).x,
    std.dpdy(local).y / std.dpdy(texel).y,
  );
  return local + (std.floor(texel) + 0.5 - texel) * perTexel;
};

/** Times a second a shaking sprite moves from side to side. */
const SHAKE = 12;

/**
 * Tiles a shaking sprite is moved sideways at `time` seconds: a texel each way, so its texels stay
 * on the grid.
 */
export const shakenAt = (time: number) => {
  'use gpu';
  return std.select(-1, d.f32(1), std.fract(time * SHAKE) < 0.5) / TEXELS;
};

/** Moves a particle's `local` coordinate to the centre of its texel, for a particle `size` tiles wide. */
export const pixelatedMote = (local: d.v2f, size: number) => {
  'use gpu';
  const texels = size * TEXELS * 0.5;
  return (std.floor(local * texels) + 0.5) / texels;
};

const WHITE = d.vec3f(0.98, 0.96, 0.9);
const PUPIL = d.vec3f(0.08, 0.06, 0.1);

/** Eye dimensions in quad units. */
export interface Eyes {
  /** Offset from the pair midpoint to each eye centre. */
  readonly apart: number;
  readonly radius: number;
  readonly pupil: number;
  /** Forward pupil offset from each eye centre. */
  readonly look: number;
}

/**
 * Returns eye colour at a point relative to the pair midpoint, using the sprite's x facing.
 * Returns zero alpha outside the eyes.
 */
export function eyes({ apart, radius, pupil, look }: Eyes) {
  const eyesAt = (at: d.v2f, facing: number) => {
    'use gpu';
    const pupils = at - d.vec2f(facing * look, 0);
    if (sdDisk(d.vec2f(std.abs(pupils.x) - apart, pupils.y), pupil) < 0) return d.vec4f(PUPIL, 1);
    if (sdDisk(d.vec2f(std.abs(at.x) - apart, at.y), radius) < 0) return d.vec4f(WHITE, 1);
    return d.vec4f();
  };
  return eyesAt;
}
