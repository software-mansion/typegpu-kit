import * as cameras from '../cameras.ts';
import * as passes from '../passes.ts';
import * as pipelines from '../pipelines.ts';
import { byStage, type Inject } from '../inject.ts';
import { type CameraBase } from '../cameras.ts';
import { d, std, tgpu, type StorageFlag, type TgpuBuffer, type TgpuRoot } from 'typegpu';
import type { Font } from './font.ts';
import * as typeset from './typeset.ts';

/**
 * This schema holds a label anchor and colour. The CPU and kernels can change both without
 * laying out the text again.
 */
export const Label = d.struct({
  /**
   * Holds the world-space anchor. For screen text, x and y use pixels from the top-left of the
   * region the pass draws, and z is depth from 0 to 1.
   */
  position: d.vec3f,
  /** Holds linear colour and opacity. */
  color: d.vec4f,
});

/**
 * This schema holds a glyph pen position, font index plus 1 and label units per em. A glyph
 * index of 0 draws nothing.
 */
export const Quad = d.struct({ pen: d.vec2f, glyph: d.u32, em: d.f32 });

/** This schema describes a glyph fragment passed to `shade`. */
export const Fragment = d.struct({
  label: d.u32,
  /** Gives the zero-based glyph index within the label. Spaces use no index. */
  glyph: d.u32,
  /** Holds the pixel offset from the label anchor in em, with y increasing upwards. */
  local: d.vec2f,
  /** Holds the label colour and opacity. */
  color: d.vec4f,
  /**
   * Gives signed distance to the glyph edge in screen pixels, positive inside. It uses the
   * median of the first three atlas channels.
   */
  distance: d.f32,
  /**
   * Gives signed distance from the fourth atlas channel in screen pixels. It rounds corners and
   * extends up to `reach`, for outlines and glows.
   */
  round: d.f32,
  /**
   * Gives the maximum range of `distance` and `round` in screen pixels. It is half the font
   * `distanceRange` in texels, scaled to the drawn size. Values stop at `-reach` outside the
   * glyph, so an outline wider than `reach` fills the entire quad.
   */
  reach: d.f32,
});

export type Fragment = d.InferGPU<typeof Fragment>;

/** This schema describes the glyph passed to `place` for each corner. */
export const Placed = d.struct({
  label: d.u32,
  /** Gives the zero-based glyph index within the label. Spaces use no index. */
  glyph: d.u32,
  /**
   * Holds the pen offset from the label anchor, with y increasing upwards, in the units of `em`.
   */
  pen: d.vec2f,
  /** Holds the label units per em. */
  em: d.f32,
});

export type Placed = d.InferGPU<typeof Placed>;

/**
 * Returns a glyph corner in clip space. `position` is the label anchor. `offset` is the corner
 * offset in the units of label `em`, with y increasing upwards. `glyph` identifies the placed
 * glyph.
 */
export type Place = (position: d.v3f, offset: d.v2f, glyph: Placed) => d.v4f;

/** Places camera-facing text in world space, with `em` in world units. */
export const facing = (position: d.v3f, offset: d.v2f): d.v4f => {
  'use gpu';
  const view = cameras.current.$;
  const corner = position + view.right * offset.x + view.up * offset.y;
  return cameras.toClip(corner);
};

/** Places text on the ground, reading along +x with its top towards -z. `em` uses world units. */
export const ground = (position: d.v3f, offset: d.v2f): d.v4f => {
  'use gpu';
  return cameras.toClip(position + d.vec3f(offset.x, 0, -offset.y));
};

/**
 * Places text on screen with `position` in pixels from the top-left of the region the pass draws.
 * `em` uses points, the pass's `passes.viewport` pixels per point, to preserve label size across
 * pixel densities.
 */
export const screen = (position: d.v3f, offset: d.v2f): d.v4f => {
  'use gpu';
  const view = passes.viewport.$;
  const pixel = position.xy + d.vec2f(offset.x, -offset.y) * view.pixelRatio;
  return d.vec4f((pixel / view.size) * d.vec2f(2, -2) + d.vec2f(-1, 1), position.z, 1);
};

/** Places text at a world point with constant screen size. `em` uses points. */
export const pinned = (position: d.v3f, offset: d.v2f): d.v4f => {
  'use gpu';
  const view = passes.viewport.$;
  const clip = cameras.toClip(position);
  return d.vec4f(clip.xy + ((offset * 2 * view.pixelRatio) / view.size) * clip.w, clip.z, clip.w);
};

const fill = (fragment: Fragment) => {
  'use gpu';
  const alpha = fragment.color.w * std.saturate(fragment.distance + 0.5);
  return d.vec4f(fragment.color.xyz * alpha, alpha);
};

/** This schema provides the label index and shaded colour to pass outputs. */
export const Seen = d.struct({
  label: d.u32,
  /** Holds the premultiplied colour returned by `shade`. */
  color: d.vec4f,
});
/** This schema describes a glyph quad corner for outputs that read the vertex. */
export const Corner = d.struct({
  /** Holds the clip position returned by `place`. */
  position: d.vec4f,
  /** Holds atlas coordinates from 0 to 1. */
  uv: d.vec2f,
  /**
   * Holds the glyph cell bounds as left, top, right and bottom coordinates from 0 to 1 across
   * the atlas.
   */
  bounds: d.vec4f,
  /** Holds the label colour and opacity. */
  color: d.vec4f,
  label: d.u32,
  /** Gives the glyph index within its label. */
  glyph: d.u32,
  /** Holds the corner offset from the label anchor in em, with y increasing upwards. */
  local: d.vec2f,
});

export interface Options {
  /** Sets the number of labels to an integer of at least 1. */
  readonly capacity: number;
  /**
   * Limits glyphs drawn per label to an integer of at least 1. Spaces use no glyph slot; glyphs
   * beyond the limit are omitted.
   */
  readonly length: number;
  /**
   * Places glyph corners with `facing`, `ground`, `pinned`, `screen` or a custom function. Default
   * `facing`.
   * @gpu
   */
  readonly place?: Place;
  /**
   * Supplies the camera for `facing`, `ground` and `pinned`, unless the pass description injects
   * its own. Screen text needs no camera. Default none, which requires camera-dependent text to
   * use a description that injects a camera.
   */
  readonly camera?: CameraBase;
  /**
   * Returns premultiplied colour for a fragment. Default the label colour over the glyph with
   * one pixel of antialiasing at its edge.
   * @gpu
   */
  readonly shade?: (fragment: Fragment) => d.v4f;
  /**
   * Selects `over` for text that covers the background, `add` for glowing text, `translucent` for
   * the passes' translucency targets, or a custom blend state. Default `over`.
   */
  readonly blend?: 'over' | 'add' | 'translucent' | GPUBlendState;
  /** Sets the drawing pipeline label. Default `text`. */
  readonly label?: string;
}

export interface TextSet extends Disposable {
  /** Reports the number of labels, from the `capacity` option. */
  readonly capacity: number;
  /** Reports the most glyphs a label draws, from the `length` option. */
  readonly length: number;
  /** Fills `move`, `setLabel`, `setQuad`, `quadAt` and `erase` with this set, as their defaults do. */
  inject(): Inject;
  /** Moves a label anchor from a kernel. */
  readonly move: (label: number, position: d.v3f) => void;
  /** Sets a label anchor and colour from a kernel. */
  readonly setLabel: (label: number, value: d.Infer<typeof Label>) => void;
  /**
   * Sets glyph `k` below `length` from a kernel. `glyph` is the font index plus 1, or 0 to omit it. `pen` and `em` use the units of `place`. If you
   * also use CPU `write`, erase unused glyphs yourself.
   */
  readonly setQuad: (label: number, k: number, quad: d.Infer<typeof Quad>) => void;
  /**
   * Reads glyph `k` of a label from a kernel, such as one copying a label `write` laid out into another set. Returns an empty quad outside the set.
   */
  readonly quadAt: (label: number, k: number) => d.Infer<typeof Quad>;
  /** Erases every glyph in a label from a kernel, as `write(label, '')` does on the CPU. */
  readonly erase: (label: number) => void;
  /**
   * Holds label anchors and colours, initially at the origin and white. Use TypeGPU `patch` to
   * upload selected fields, or `move` and `setLabel` to write from a kernel.
   */
  readonly labels: TgpuBuffer<d.WgslArray<typeof Label>> & StorageFlag;
  /** Holds `length` quads per label in label order. `write` and `setQuad` fill them. */
  readonly quads: TgpuBuffer<d.WgslArray<typeof Quad>> & StorageFlag;
  /**
   * Lays out and uploads only the specified label. Use `write(label, '')` to empty it. Returns
   * the layout, valid until the next `write`. Throws for a label outside the set or an `em` value
   * that is not greater than 0.
   */
  write(label: number, text: string, options: typeset.Options): typeset.Layout;
  /**
   * Draws all labels when added to a canvas or scene pass description. It tests depth without
   * writing it.
   */
  readonly drawing: passes.Drawing<typeof Corner, d.Infer<typeof Seen>>;
  /** Destroys the label and quad buffers. */
  destroy(): void;
}

/**
 * Creates labels drawn from a baked MSDF font atlas. The CPU lays out each string, and instanced
 * glyph quads place it in world or screen space. Throws unless `capacity` and `length` are
 * integers of at least 1.
 *
 * @injects member `move`, `setLabel`, `setQuad`, `quadAt` and `erase`, which write its own
 * buffers in any pipeline
 * @owns 2 buffers, a sampler: the quads and the labels
 * @binds 3 storage, 1 uniform: the drawing, the quads, the labels and the font's glyphs, and
 * `passes.viewport` with `text.screen` or `text.pinned`
 * @binds 1 storage: `move` or `setLabel`, the labels
 * @binds 1 storage: `setQuad`, `quadAt` or `erase`, the quads
 * @records `drawing`, added to passes, after the passes that move the labels.
 * @param font the font it draws with, which it borrows
 * @param options.place a `'use gpu'` function; it and `shade` carry the fills of what they read
 * beside the camera, such as `tgpu.fn(lifted).pipe(scene.inject())`
 * @returns a set of labels you destroy before its font
 */
export function create(root: TgpuRoot, font: Font, options: Options): TextSet {
  const { capacity, length, blend = 'over', camera } = options;
  for (const [name, value] of [
    ['capacity', capacity],
    ['length', length],
  ] as const) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(
        `text.create requires \`${name}\` to be an integer of at least 1. Received ${value}.`,
      );
    }
  }

  const place = options.place ?? facing;
  const shade = options.shade ?? fill;

  const quadsBuffer = root.createBuffer(d.arrayOf(Quad, capacity * length)).$usage('storage');
  const labelsBuffer = root
    .createBuffer(
      d.arrayOf(Label, capacity),
      Array.from({ length: capacity }, () => ({ position: d.vec3f(), color: d.vec4f(1) })),
    )
    .$usage('storage');

  // Per-set accessors filled with its own buffers, so one kernel can write to several text sets
  const labelRows = byStage(labelsBuffer);
  const quadRows = byStage(quadsBuffer);
  const labels = tgpu.accessor(d.arrayOf(Label), labelRows);
  const quads = tgpu.accessor(d.arrayOf(Quad), quadRows);
  const inject: Inject = (cfg) => cfg.with(labels, labelRows).with(quads, quadRows);

  const move = (label: number, position: d.v3f) => {
    'use gpu';
    if (label >= capacity) return;
    labels.$[label].position = d.vec3f(position);
  };

  const setLabel = (label: number, value: d.Infer<typeof Label>) => {
    'use gpu';
    if (label >= capacity) return;
    labels.$[label] = Label(value);
  };

  const setQuad = (label: number, k: number, quad: d.Infer<typeof Quad>) => {
    'use gpu';
    if (label >= capacity || k >= length) return;
    quads.$[d.u32(label) * length + d.u32(k)] = Quad(quad);
  };

  const quadAt = (label: number, k: number) => {
    'use gpu';
    if (label >= capacity || k >= length) return Quad();
    return Quad(quads.$[d.u32(label) * length + d.u32(k)]);
  };

  const erase = (label: number) => {
    'use gpu';
    if (label >= capacity) return;
    for (let k = d.u32(0); k < length; k++) quads.$[d.u32(label) * length + k] = Quad();
  };

  const glyphs = font.glyphs.as('readonly');
  const atlas = font.atlas.createView(d.texture2d(d.f32));
  const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const { width, height, distanceRange } = font.data.atlas;
  const halfTexel = d.vec2f(0.5 / width, 0.5 / height);
  const range = d.vec2f(distanceRange / width, distanceRange / height);

  const at = (vertexIndex: number, instanceIndex: number) => {
    'use gpu';
    const label = std.intdiv(instanceIndex, d.u32(length));
    const k = instanceIndex - label * length;
    const quad = quads.$[instanceIndex];
    if (quad.glyph === 0) {
      // Collapse all corners beyond the far plane to produce no fragments
      return Corner({
        position: d.vec4f(0, 0, 2, 1),
        uv: d.vec2f(),
        bounds: d.vec4f(),
        color: d.vec4f(),
        label,
        glyph: k,
        local: d.vec2f(),
      });
    }

    // Order the strip bottom-left, bottom-right, top-left, top-right
    const glyph = glyphs.$[quad.glyph - 1];
    const corner = d.vec2f(d.f32(vertexIndex & 1), d.f32(vertexIndex >>> 1));
    const offset = quad.pen + std.mix(glyph.plane.xy, glyph.plane.zw, corner) * quad.em;
    const uv = d.vec2f(
      std.mix(glyph.uv.x, glyph.uv.z, corner.x),
      std.mix(glyph.uv.w, glyph.uv.y, corner.y),
    );

    const row = labels.$[label];
    const placed = Placed({ label, glyph: k, pen: quad.pen, em: quad.em });
    return Corner({
      position: place(row.position, offset, placed),
      uv,
      bounds: glyph.uv,
      color: row.color,
      label,
      glyph: k,
      local: offset / quad.em,
    });
  };

  // Use the MSDF median and screen-space antialiasing from @pmndrs/glyph's MIT-licensed TypeGPU
  // shaders by Justin Walsh
  const pixelOf = (corner: d.Infer<typeof Corner>) => {
    'use gpu';
    const uv = corner.uv;
    const bounds = corner.bounds;
    // Clamp half a texel inside the cell so filtering cannot read a neighbouring glyph
    const inside = std.clamp(uv, bounds.xy + halfTexel, bounds.zw - halfTexel);
    const texel = std.textureSampleLevel(atlas.$, sampler.$, inside, 0);
    const median = std.max(std.min(texel.x, texel.y), std.min(std.max(texel.x, texel.y), texel.z));

    const dx = std.dpdx(uv);
    const dy = std.dpdy(uv);
    const pixelsPerUv = std.inverseSqrt(std.max(dx * dx + dy * dy, d.vec2f(1e-12)));
    const span = std.max(0.5 * std.dot(range, pixelsPerUv), 1);

    const color = shade(
      Fragment({
        label: corner.label,
        glyph: corner.glyph,
        local: corner.local,
        color: corner.color,
        distance: (median - 0.5) * span,
        round: (texel.w - 0.5) * span,
        reach: span / 2,
      }),
    );
    return Seen({ label: corner.label, color });
  };

  const drawn: Inject = (cfg) => {
    const filled = cfg.pipe(inject);
    return camera ? filled.pipe(camera.inject()) : filled;
  };

  const queue = root.device.queue;
  const quadsGpu = root.unwrap(quadsBuffer);
  const laid = typeset.create(length);
  const staging = new ArrayBuffer(length * 16);
  const floats = new Float32Array(staging);
  const words = new Uint32Array(staging);
  const held = new Uint32Array(capacity);

  const destroy = () => {
    quadsBuffer.destroy();
    labelsBuffer.destroy();
  };
  return {
    capacity,
    length,

    inject: () => inject,

    move,
    setLabel,
    setQuad,
    quadAt,
    erase,
    labels: labelsBuffer,
    quads: quadsBuffer,

    write(label, text, layoutOptions) {
      if (!Number.isInteger(label) || label < 0 || label >= capacity) {
        throw new RangeError(`The text set has no label ${label}; there are ${capacity}.`);
      }

      laid.set(font, text, layoutOptions);
      const placed = laid.count;

      for (let i = 0; i < placed; i++) {
        floats[i * 4] = laid.pens[i * 2];
        floats[i * 4 + 1] = laid.pens[i * 2 + 1];
        words[i * 4 + 2] = laid.glyphs[i] + 1;
        floats[i * 4 + 3] = layoutOptions.em;
      }

      // Erase trailing glyphs when text becomes shorter
      const written = Math.max(placed, held[label]);
      words.fill(0, placed * 4, written * 4);
      held[label] = placed;

      if (written > 0) queue.writeBuffer(quadsGpu, label * length * 16, staging, 0, written * 16);
      return laid;
    },

    drawing: {
      label: options.label ?? 'text',
      Vertex: Corner,
      at,
      pixelOf,
      color: passes.shaded,
      ...(blend === 'translucent'
        ? { translucent: true }
        : { blend: typeof blend === 'string' ? pipelines.blend[blend] : blend }),
      depthStencil: { depthWriteEnabled: false },
      flat: ['bounds', 'color'],
      primitive: { topology: 'triangle-strip' },
      inject: () => drawn,
      draw: (pass) => pass.draw(4, capacity * length),
    },

    destroy,
    [Symbol.dispose]: destroy,
  };
}
