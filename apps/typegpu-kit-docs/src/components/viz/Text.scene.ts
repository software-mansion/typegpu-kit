import { cameras, passes, type Frame, type Surface, font, text, type Baked } from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Loading } from 'typegpu-kit-examples/contract';

/** Glyphs the label draws at most. */
export const LENGTH = 32;

export const SHADES = ['fill', 'outline', 'glow', 'distance', 'round'] as const;
export type Shade = (typeof SHADES)[number];

/** What the figure starts with: the size in points, the outline's width in pixels. */
export const START = {
  words: 'TypeGPU Kit',
  size: 64,
  width: 2,
  shade: 'outline',
} as const satisfies {
  words: string;
  size: number;
  width: number;
  shade: Shade;
};

const [FILL, OUTLINE, GLOW, DISTANCE] = [0, 1, 2, 3];

/** Which shade, by its index in SHADES, and the outline's or the glow's width in pixels. */
const Look = d.struct({ shade: d.u32, width: d.f32 });

/** Pixels between two bands of the field. */
const BAND = 3;

const INK = d.vec3f(0.93, 0.95, 0.97);
const EDGE = d.vec3f(0.5, 0.66, 0.98);
/** Where the field stops, past `reach`: half covering what is under it, so the quads show. */
const STOPPED = d.vec4f(0.15, 0.17, 0.2, 0.5);

export interface Options {
  readonly font: Baked;
}

/** The label's font: Inter as the kit ships it baked, read as the solar system example does. */
export async function load(loading: Pick<Loading, 'font'>): Promise<Options> {
  const { json, atlas } = await loading.font('inter');
  return { font: font.read(json, atlas) };
}

/** A label for the text page, in Inter, with its shades and the distances they read. */
export function createScene(root: TgpuRoot, surface: Surface, options: Options) {
  const sans = font.create(root, options.font);
  const view = cameras.flat(root, { size: surface, height: 2 });
  const look = root.createUniform(Look, {
    shade: SHADES.indexOf(START.shade),
    width: START.width,
  });

  const shade = (fragment: text.Fragment) => {
    'use gpu';
    const chosen = look.$.shade;
    const width = look.$.width;
    const fill = std.saturate(fragment.distance + 0.5);
    const ink = fragment.color.xyz;

    if (chosen === FILL) return d.vec4f(ink * fill, fill);

    if (chosen === OUTLINE) {
      const outline = std.saturate(fragment.round + width + 0.5);
      return d.vec4f(std.mix(EDGE, ink, fill) * outline, outline);
    }

    if (chosen === GLOW) {
      const near = std.saturate(1 + fragment.round / width);
      const glow = near * near * 0.85 * (1 - fill);
      return d.vec4f(ink * fill + EDGE * glow, fill + glow);
    }

    const value = std.select(fragment.round, fragment.distance, chosen === DISTANCE);
    const reach = fragment.reach;
    const side = std.select(EDGE, INK, value > 0);
    const fade = 0.2 + 0.5 * (1 - std.abs(value) / reach);
    const band = std.saturate(1 - std.abs(value - std.round(value / BAND) * BAND));
    const edge = std.saturate(1 - std.abs(value));
    const shaded = side * std.max(fade, band) + INK * edge;
    return std.select(d.vec4f(shaded, 1), STOPPED, std.abs(value) >= reach - 0.01);
  };

  const label = text.create(root, sans, {
    camera: view,
    capacity: 1,
    length: LENGTH,
    place: text.pinned,
    shade,
  });
  const canvas = passes.create(root, surface).pipe(view.inject());
  const drawn = canvas.add(label.drawing);
  label.labels.patch({ 0: { color: d.vec4f(INK, 1) } });

  let words: string = START.words;
  let size: number = START.size;
  const write = () => label.write(0, words, { em: size });
  write();

  const { distanceRange } = sans.data.atlas;
  const { emSize } = sans.data.source;

  return {
    /** Lays the label out again with new words. */
    write(next: string) {
      words = next;
      write();
    },

    /** Points to the em. */
    resize(points: number) {
      size = points;
      write();
    },

    shade(chosen: Shade) {
      look.patch({ shade: SHADES.indexOf(chosen) });
    },

    /** The outline's or the glow's width in pixels. */
    widen(pixels: number) {
      look.patch({ width: pixels });
    },

    /**
     * Pixels from the edge that `distance` and `round` measure at most at the label's size: half
     * the atlas's distance range, as large as the glyph is drawn, and at least half a pixel.
     */
    get reach() {
      return Math.max((distanceRange * size * surface.pixelRatio) / emSize, 1) / 2;
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      canvas.record(frame, drawn);
    },

    destroy() {
      canvas.destroy();
      label.destroy();
      look.buffer.destroy();
      sans.destroy();
    },
  };
}

export type TextFigure = ReturnType<typeof createScene>;
