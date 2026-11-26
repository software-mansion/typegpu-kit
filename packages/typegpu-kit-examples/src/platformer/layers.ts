import { cameras, targets, type SizeSource } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import { mat4 } from 'wgpu-matrix';
import { TEXELS } from './sprites.ts';

/** The art's layers in the layered target, back to front. Each backdrop layer is one card. */
export const LAYERS = { clouds: 0, far: 1, near: 2, level: 3 } as const;
const COUNT = 4;
/** How far each layer moves per tile the camera moves: the backdrop lags behind the level. */
const SCROLL = [0.1, 0.3, 0.55, 1] as const;
/** Tiles a second the clouds drift. */
const DRIFT = 0.3;

/**
 * Where each layer lies. Per layer, `xy` is the bottom left corner of its texture in the layer's
 * own tiles, and `zw` the texels its texture lies left of and above the view, rounded to whole
 * pixels of the surface. `size` is each layer's size in texels.
 */
export const Placement = d.struct({ layers: d.arrayOf(d.vec4f, COUNT), size: d.vec2f });

export interface LayersOptions {
  /** The view in texels, which each layer covers with a texel to spare on the right and bottom. */
  readonly size: SizeSource;
  /** The surface, whose pixels each layer moves by. */
  readonly surface: SizeSource;
}

/**
 * Holds the art in layers that scroll at their own speeds. Each layer is drawn on its own texel
 * grid, a texel larger than the view, and moved by whole pixels of the surface when it is printed.
 * The level layer is drawn through `camera`. Sprites are drawn at the surface's resolution
 * through `screen`, which shows the view as the print places the level layer.
 */
export function createLayers(root: TgpuRoot, options: LayersOptions) {
  const { size, surface } = options;
  const padded = {
    get width() {
      return size.width + 1;
    },
    get height() {
      return size.height + 1;
    },
  };
  const target = targets.create(root, padded, {
    color: { format: 'rgba8unorm', sampled: true, layers: COUNT },
  });
  const camera = cameras.custom(root);
  const screen = cameras.custom(root);
  const placement = root.createUniform(Placement);
  const grid = root.createUniform(d.vec2f);
  const pixels = d.vec2f();

  const floats = new Float32Array(d.sizeOf(Placement) / 4);
  const next = new Float32Array(floats.length);
  const view = mat4.identity();
  const projection = mat4.create();
  let written = false;

  /** The point in a layer's tiles at `uv` across its texture, such as a texel it draws. */
  const pointOf = (layer: number, uv: d.v2f) => {
    'use gpu';
    const flipped = d.vec2f(uv.x, 1 - uv.y);
    return placement.$.layers[layer].xy + (flipped * placement.$.size) / TEXELS;
  };

  /** Where the view's texel position `at`, from its top left, lies in a layer's texture. */
  const inLayer = (layer: number, at: d.v2f) => {
    'use gpu';
    return at + placement.$.layers[layer].zw;
  };

  /**
   * The texel at `texel` of a layer's texture, counted from the origin of the layer's own tiles,
   * so it names the same texel of the layer's art wherever the texture lies.
   */
  const cellOf = (layer: number, texel: d.v2f) => {
    'use gpu';
    const corner = placement.$.layers[layer].xy * TEXELS;
    const up = placement.$.size.y - 1 - texel.y;
    return d.vec2i(std.round(d.vec2f(corner.x + texel.x, corner.y + up)));
  };

  return {
    /** The layered target, `textures.color` holding a layer per entry of `LAYERS`. */
    target,
    /** Draws the level layer, in world units. */
    camera,
    /** Draws the view at the surface's resolution, in world units, as the level layer is printed. */
    screen,
    placement,
    /** Surface pixels per tile along each axis, which fills `sprites.grid`. */
    grid,
    pointOf,
    inLayer,
    cellOf,

    /**
     * Places every layer around `eye`, the view's centre in tiles, with the clouds drifted by
     * `time` seconds, and writes the placement and the cameras when they move.
     */
    place(eye: { readonly x: number; readonly y: number }, time: number) {
      const { width, height } = size;
      // Surface pixels per texel along each axis
      const across = surface.width / width;
      const down = surface.height / height;

      for (let k = 0; k < COUNT; k++) {
        const x = eye.x * SCROLL[k] + (k === LAYERS.clouds ? time * DRIFT : 0);
        const y = eye.y * SCROLL[k];
        // The view's top left in the layer's texels, x right and y down, on the surface's pixels
        const left = Math.round((x * TEXELS - width / 2) * across) / across;
        const top = Math.round((-y * TEXELS - height / 2) * down) / down;
        const column = Math.floor(left);
        const row = Math.floor(top);
        next[4 * k] = column / TEXELS;
        next[4 * k + 1] = (-row - height - 1) / TEXELS;
        next[4 * k + 2] = left - column;
        next[4 * k + 3] = top - row;
      }
      next[4 * COUNT] = width + 1;
      next[4 * COUNT + 1] = height + 1;

      let same = written;
      for (let i = 0; same && i < next.length; i++) same = next[i] === floats[i];
      if (same) return;
      floats.set(next);
      placement.buffer.write(floats.buffer);
      written = true;

      if (pixels.x !== across * TEXELS || pixels.y !== down * TEXELS) {
        pixels.x = across * TEXELS;
        pixels.y = down * TEXELS;
        grid.write(pixels);
      }

      const at = 4 * LAYERS.level;
      const left = floats[at];
      const bottom = floats[at + 1];
      const right = left + (width + 1) / TEXELS;
      const top = bottom + (height + 1) / TEXELS;
      camera.write(view, mat4.ortho(left, right, bottom, top, -1, 1, projection));

      // The view lies its shift into the level layer, right of and below the layer's top left
      const viewLeft = left + floats[at + 2] / TEXELS;
      const viewTop = top - floats[at + 3] / TEXELS;
      const viewRight = viewLeft + width / TEXELS;
      const viewBottom = viewTop - height / TEXELS;
      screen.write(view, mat4.ortho(viewLeft, viewRight, viewBottom, viewTop, -1, 1, projection));
    },

    destroy() {
      target.destroy();
    },
  };
}

export type Layers = ReturnType<typeof createLayers>;
