import { linearRgbToOklab, srgbToLinear } from '@typegpu/color';
import { clamp } from 'math';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot, type TgpuTexture } from 'typegpu';
import { images, pipelines } from '@typegpu/kit';
import type { Picture } from '../contract.ts';
import type { Dominoes } from './dominoes.ts';
import { COLUMNS, ROWS } from './field.ts';
import { createSunset, SIZE } from './sunset.ts';

/**
 * Sixteen sRGB colours available for domino art. Each image pixel uses the nearest palette
 * colour.
 */
const COLORS = [
  [0.07, 0.07, 0.08],
  [0.95, 0.94, 0.9],
  [0.55, 0.55, 0.56],
  [0.82, 0.12, 0.14],
  [0.48, 0.08, 0.1],
  [0.97, 0.5, 0.1],
  [0.98, 0.84, 0.18],
  [0.5, 0.78, 0.28],
  [0.1, 0.42, 0.22],
  [0.45, 0.72, 0.95],
  [0.2, 0.45, 0.82],
  [0.12, 0.22, 0.6],
  [0.46, 0.22, 0.62],
  [0.96, 0.55, 0.7],
  [0.45, 0.27, 0.14],
  [0.9, 0.72, 0.55],
] as const;
const PALETTE = COLORS.map(([r, g, b]) => srgbToLinear(d.vec3f(r, g, b)));

/** Image crop coordinates from 0 to 1 and the mip level matching the cell sample spacing. */
const Framing = d.struct({ offset: d.vec2f, size: d.vec2f, level: d.f32 });

/** Swappable image binding for the portrait. */
const source = tgpu.bindGroupLayout({ image: { texture: d.texture2d(d.f32) } });

/** Samples per cell side, averaged to one colour. */
const SAMPLES = 4;
const WORKGROUP = 64;

/** Crops a `width × height` image with `levels` mips to fill the `COLUMNS × ROWS` field. */
const framingOf = (width: number, height: number, levels: number) => {
  const image = width / height;
  const field = COLUMNS / ROWS;
  const size = image > field ? d.vec2f(field / image, 1) : d.vec2f(1, image / field);
  const offset = d.vec2f((1 - size.x) / 2, (1 - size.y) / 2);
  const texels = (size.x * width) / COLUMNS / SAMPLES;
  const level = clamp(Math.log2(texels), 0, levels - 1);
  return { offset, size, level };
};

/**
 * Samples an image over each domino cell, averages it and paints the back with the nearest
 * palette colour. Starts with a sunset. `show` changes the image.
 */
export function createPicture(root: TgpuRoot, { set }: Dominoes) {
  const palette = root.createReadonly(d.arrayOf(d.vec3f, PALETTE.length), PALETTE);
  const framing = root.createUniform(Framing, framingOf(SIZE[0], SIZE[1], 1));
  const smooth = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });
  const sunset = createSunset(root);

  /** Average linear colour over the image cell. */
  const averageOver = (cell: d.v2f) => {
    'use gpu';
    let sum = d.vec3f();
    for (let y = 0; y < SAMPLES; y++) {
      for (let x = 0; x < SAMPLES; x++) {
        const inField = (cell + (d.vec2f(x, y) + 0.5) / SAMPLES) / d.vec2f(COLUMNS, ROWS);
        const uv = framing.$.offset + inField * framing.$.size;
        sum += std.textureSampleLevel(source.$.image, smooth.$, uv, framing.$.level).rgb;
      }
    }
    return sum / (SAMPLES * SAMPLES);
  };

  /** Finds the nearest palette colour in Oklab. */
  const nearest = (color: d.v3f) => {
    'use gpu';
    const wanted = linearRgbToOklab(color);
    let best = d.vec3f(palette.$[0]);
    let least = d.f32(1e9);
    for (let k = d.u32(0); k < PALETTE.length; k++) {
      const each = d.vec3f(palette.$[k]);
      const gap = std.distance(linearRgbToOklab(each), wanted);
      if (gap < least) {
        least = gap;
        best = d.vec3f(each);
      }
    }
    return best;
  };

  const domino = set.gpu;
  const paint = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= domino.span() || !domino.alive(i)) return;
    const cell = domino.cell(i);
    if (cell.x < 0) return;
    domino.device.$[i].back = nearest(averageOver(cell));
  });

  const painting = root.createComputePipeline({ compute: paint });
  let image = root.createBindGroup(source, { image: sunset.texture });
  /** The texture of the last image shown, which replaces the sunset. */
  let shown: TgpuTexture | undefined;
  let sunsetDrawn = false;
  let changed = true;

  return {
    initAsync: () => pipelines.compile([sunset.drawing, painting]),

    /**
     * Crops `picture` to fill the field. Creates its texture and mips when the page receives a
     * dropped file.
     */
    show(picture: Picture) {
      const texture = images.create(root, picture);
      const [width, height] = texture.props.size;
      shown?.destroy();
      shown = texture;
      image = root.createBindGroup(source, { image: texture });
      framing.write(framingOf(width, height, root.unwrap(texture).mipLevelCount));
      changed = true;
    },

    /** Requests repainting after a layout change. */
    repaint() {
      changed = true;
    },

    /** Repaints only when the image or layout has changed. */
    record(pass: TgpuComputePass) {
      if (!changed) return;
      changed = false;
      if (!sunsetDrawn) {
        sunset.drawing.record(pass);
        sunsetDrawn = true;
      }
      pass.setPipeline(painting);
      pass.setBindGroup(image);
      pass.dispatchWorkgroups(Math.ceil(set.span / WORKGROUP));
    },

    destroy() {
      sunset.texture.destroy();
      palette.buffer.destroy();
      shown?.destroy();
    },
  };
}
