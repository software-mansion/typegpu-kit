import { images, passes } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Picture } from '../contract.ts';
import { quad, Seen } from './sprites.ts';

/** Texels along each side of a raspberry's frame. */
const FRAME = 16;
/** The column from which a half raspberry shows an empty one. */
const HALF = FRAME / 2;
/** The sheet's frames: a whole raspberry and an empty one. */
const WHOLE = 0;
const EMPTY = 1;
/** Texels from the screen's top-left corner to the first raspberry, and from one to the next. */
const MARGIN = 6;
const STRIDE = 16;

const Shown = d.struct({
  /** The surface's size in pixels. */
  size: d.vec2f,
  /** Surface pixels per texel of the raspberries, a whole number so their texels stay square. */
  scale: d.f32,
  /** Halves of raspberries left. */
  health: d.f32,
});

/** A corner of a raspberry's quad: its texel in the frame from the top left, and which it is. */
const Corner = d.struct({ position: d.vec4f, texel: d.vec2f, heart: d.u32 });

export interface HudOptions {
  /** The raspberries' sheet: a whole one and an empty one, `FRAME` texels square, in a row. */
  readonly sheet: Picture;
  /** Raspberries the hero can have. */
  readonly hearts: number;
}

/**
 * Shows the hero's health as a row of raspberries in the screen's top-left corner, drawn over the
 * print in the sheet's own colours. A half raspberry is a whole one's left half beside an empty
 * one's right half.
 */
export function createHud(root: TgpuRoot, options: HudOptions) {
  const { hearts } = options;
  // The sheet holds display colours, as the print shows them, so it loads unconverted
  const texture = images.create(root, options.sheet, { color: 'linear', mips: false });
  const sheet = texture.createView(d.texture2d(d.f32));

  const shown = { size: d.vec2f(), scale: 0, health: 0 };
  const hud = root.createUniform(Shown);

  /** The raspberries' texel at `texel` of raspberry `heart`, or zero alpha around it. */
  const raspberryAt = (texel: d.v2f, heart: number) => {
    'use gpu';
    const t = std.clamp(d.vec2i(std.floor(texel)), d.vec2i(0), d.vec2i(FRAME - 1));
    const left = hud.$.health - 2 * d.f32(heart);
    const whole = left >= 2 || (left >= 1 && t.x < HALF);
    const frame = std.select(d.i32(EMPTY), d.i32(WHOLE), whole);
    return std.textureLoad(sheet.$, d.vec2i(frame * FRAME + t.x, t.y), 0);
  };

  const drawing: passes.Drawing<typeof Corner, Seen> = {
    label: 'hud',
    Vertex: Corner,
    flat: ['heart'],

    at: (vertexIndex, instanceIndex) => {
      'use gpu';
      // From the quad's top left, 0 to 1 across and down
      const corner = quad.vertexAt(vertexIndex).position.xz * d.vec2f(0.5, -0.5) + 0.5;
      const texel = corner * FRAME;
      const origin = d.vec2f(MARGIN + d.f32(instanceIndex) * STRIDE, MARGIN);
      const pixel = (origin + texel) * hud.$.scale;
      const clip = d.vec2f(pixel.x / hud.$.size.x, 1 - pixel.y / hud.$.size.y) * 2 - d.vec2f(1, 1);
      return Corner({ position: d.vec4f(clip, 0, 1), texel, heart: instanceIndex });
    },

    cutout: (corner) => {
      'use gpu';
      return raspberryAt(corner.texel, corner.heart).w === 0;
    },

    pixelOf: (corner) => {
      'use gpu';
      return Seen({ color: d.vec4f(raspberryAt(corner.texel, corner.heart).xyz, 1) });
    },

    color: passes.shaded,
    draw: (pass) => pass.draw(quad.vertexCount, hearts),
  };

  return {
    drawing,

    /**
     * Shows `health` halves of raspberries on a surface `width` by `height` pixels, `scale` pixels
     * a texel. Writes only when they change.
     */
    update(health: number, width: number, height: number, scale: number) {
      const { size } = shown;
      if (
        health === shown.health &&
        width === size.x &&
        height === size.y &&
        scale === shown.scale
      ) {
        return;
      }
      shown.health = health;
      shown.scale = scale;
      size.x = width;
      size.y = height;
      hud.write(shown);
    },

    destroy() {
      texture.destroy();
    },
  };
}

export type Hud = ReturnType<typeof createHud>;
