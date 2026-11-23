/**
 * This module lays out strings on the CPU using baked font advances and kerning. It aligns lines
 * separated by `\n`, `\r\n` or `\r`. It runs in plain JavaScript and allocates nothing once its
 * arrays fit the input.
 *
 * @module
 */

import type { Metrics } from './font.ts';

export interface Options {
  /**
   * Sets positive units per em. World text uses world units. Text placed with `screen` or
   * `pinned` uses points, scaled by the text set `pixelRatio`.
   */
  readonly em: number;
  /** Sets the anchor position along each line. Default `center`. */
  readonly align?: 'left' | 'center' | 'right';
  /**
   * Sets the vertical anchor at the first line top, the top of its capitals, block middle, first
   * baseline or last line bottom. Default `middle`.
   */
  readonly baseline?: 'top' | 'cap' | 'middle' | 'alphabetic' | 'bottom';
  /** Adds space after each character but the last of a line, in em. Default 0. */
  readonly tracking?: number;
}

/** This type describes laid-out text: its size and each glyph's pen. */
export interface Layout {
  /** Gives the number of placed glyphs with ink, up to capacity. */
  readonly count: number;
  /** Gives the number of lines. */
  readonly lines: number;
  /** Gives the widest line in the units of `em`. */
  readonly width: number;
  /**
   * Gives the block height in the units of `em`, from the first line top to the last line bottom.
   */
  readonly height: number;
  /** Lists placed glyph indices in the font. */
  readonly glyphs: Uint32Array;
  /**
   * Holds x and y pen coordinates from the anchor, with y increasing upwards, in the units of
   * `em`.
   */
  readonly pens: Float32Array;
}

export interface Typesetter extends Layout {
  /**
   * Replaces the layout with `text` using a font or its `metrics`. Returns this layout. Throws
   * unless `em` is greater than 0.
   */
  set(font: Metrics, text: string, options: Options): Typesetter;
}

type Mutable<T> = { -readonly [K in keyof T]: T[K] };

const NEWLINE = 0xffffffff;
const SPACE = 0x20;
const LINE_FEED = 0x0a;
const RETURN = 0x0d;

const step = (font: Metrics, glyph: number, next: number, tracking: number) =>
  font.advance(glyph) + (next === NEWLINE ? 0 : font.kerning(glyph, next) + tracking);

function widthOf(font: Metrics, run: Uint32Array, from: number, to: number, tracking: number) {
  let width = 0;
  for (let i = from; i < to; i++) {
    width += step(font, run[i], i + 1 < to ? run[i + 1] : NEWLINE, tracking);
  }
  return width;
}

function grow<T extends Uint32Array | Float32Array>(array: T, length: number): T {
  const grown = new (array.constructor as new (length: number) => T)(length);
  grown.set(array);
  return grown;
}

/**
 * Creates a typesetter that places at most `capacity` glyphs and measures text of any length. Text
 * sets use one for each label. Use it separately to measure text before drawing, such as when
 * sizing a panel behind a label. Throws unless `capacity` is an integer of at least 1.
 *
 * @param capacity how many glyphs it places: a text set lays each label out into one of its `length`
 * @returns a typesetter whose `set` lays a string out into its arrays, growing them only for longer text
 */
export function create(capacity: number): Typesetter {
  if (!Number.isInteger(capacity) || capacity < 1) {
    throw new RangeError(
      `typeset.create requires \`capacity\` to be an integer of at least 1. Received ${capacity}.`,
    );
  }

  // Grow scratch arrays to the longest input so shorter layouts allocate nothing
  let run = new Uint32Array(64);
  let starts = new Uint32Array(8);
  let ends = new Uint32Array(8);
  let widths = new Float32Array(8);
  let lines = 0;

  const end = (font: Metrics, from: number, to: number, tracking: number) => {
    if (lines === starts.length) {
      starts = grow(starts, lines * 2);
      ends = grow(ends, lines * 2);
      widths = grow(widths, lines * 2);
    }

    starts[lines] = from;
    ends[lines] = to;
    widths[lines] = widthOf(font, run, from, to, tracking);
    lines++;
  };

  const layout: Mutable<Typesetter> = {
    count: 0,
    lines: 0,
    width: 0,
    height: 0,
    glyphs: new Uint32Array(capacity),
    pens: new Float32Array(capacity * 2),

    set(font, text, options) {
      const { em, align = 'center', baseline = 'middle', tracking = 0 } = options;
      if (!(em > 0))
        throw new RangeError(
          `Set \`em\` above 0 when calling the typeset's \`set\` method. Got ${em}.`,
        );

      if (run.length < text.length) run = new Uint32Array(text.length * 2);
      lines = 0;
      let start = 0;
      let n = 0;
      let trimmed = 0;
      for (let i = 0; i < text.length; i++) {
        const code = text.codePointAt(i)!;
        if (code > 0xffff) i++;

        if (code === RETURN && text.charCodeAt(i + 1) === LINE_FEED) continue;
        if (code === LINE_FEED || code === RETURN) {
          end(font, start, trimmed, tracking);
          n = start = trimmed;
        } else {
          run[n++] = font.glyphOf(code);
          if (code !== SPACE) trimmed = n;
        }
      }

      // A trailing newline adds an empty line. An empty string has no lines
      if (text.length > 0) end(font, start, trimmed, tracking);

      const { ascender, descender, capHeight, lineGap } = font.data;
      const advance = ascender - descender + lineGap;
      const bottom = descender - (lines - 1) * advance;
      const shift =
        baseline === 'top'
          ? -ascender
          : baseline === 'cap'
            ? -capHeight
            : baseline === 'bottom'
              ? -bottom
              : baseline === 'middle'
                ? -(ascender + bottom) / 2
                : 0;

      let count = 0;
      let width = 0;

      for (let l = 0; l < lines; l++) {
        const from = starts[l];
        const to = ends[l];
        width = Math.max(width, widths[l]);

        let x = align === 'center' ? -widths[l] / 2 : align === 'right' ? -widths[l] : 0;
        const y = shift - l * advance;

        for (let i = from; i < to; i++) {
          const glyph = run[i];

          if (count < capacity && font.inked(glyph)) {
            layout.glyphs[count] = glyph;
            layout.pens[count * 2] = x * em;
            layout.pens[count * 2 + 1] = y * em;
            count++;
          }

          x += step(font, glyph, i + 1 < to ? run[i + 1] : NEWLINE, tracking);
        }
      }

      layout.count = count;
      layout.lines = lines;
      layout.width = width * em;
      layout.height = lines === 0 ? 0 : (ascender - bottom) * em;
      return layout;
    },
  };

  return layout;
}
