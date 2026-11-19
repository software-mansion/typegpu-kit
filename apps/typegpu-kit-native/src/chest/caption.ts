// The pulled card's caption laid out in points of the card, as the gallery's React Native card
// lays it out: the kicker, the title in up to two lines, the example's line in up to three, and the
// foot with the rarity's gem, its name and the line count. The card's scene draws it.

import { typeset, type Metrics } from '@typegpu/kit';
import { rgbOf, type Rgb01 } from 'typegpu-kit-pull/dither.ts';
import { colors } from '../theme.ts';
import type { Entry } from './cards.ts';

/** One line of type: its text, baseline start, or end with `align: 'right'`, size and colour. */
export interface Label {
  readonly text: string;
  readonly x: number;
  readonly y: number;
  readonly em: number;
  readonly tracking: number;
  readonly align: 'left' | 'right';
  readonly color: Rgb01;
}

export interface Caption {
  /** The title's lines, in the display face. */
  readonly display: readonly Label[];
  /** The rest, in the mono face. */
  readonly mono: readonly Label[];
  /** Where the art window ends, and the foot's rule starts, across from `rule[0]` to `rule[1]`. */
  readonly artBottom: number;
  readonly footTop: number;
  readonly rule: readonly [number, number];
  /** The gem's centre and its half-diagonal. */
  readonly gem: readonly [number, number, number];
}

const ACCENT = rgbOf(colors.accent);
const TEXT = rgbOf(colors.text);
const MUTED = rgbOf(colors.muted);
const TITLE_LINES = 2;
const LINE_LINES = 3;
const ELLIPSIS = '…';

/** The height of a line of `font` at `em`, and its baseline from the line's top. */
function lineOf(font: Metrics, em: number, height?: number) {
  const { ascender, descender, lineGap } = font.data;
  const box = height ?? (ascender - descender + lineGap) * em;
  return { box, baseline: (box - (ascender - descender) * em) / 2 + ascender * em };
}

/** `text` in at most `max` lines that each `fits`, its last line cut with an ellipsis if it runs on. */
function wrap(fits: (line: string) => boolean, text: string, max: number) {
  const words = text.split(' ');
  const lines: string[] = [];
  let line = '';
  for (let i = 0; i < words.length; i++) {
    const next = line ? `${line} ${words[i]}` : words[i];
    if (!line || fits(next)) {
      line = next;
      continue;
    }
    if (lines.length === max - 1) {
      let cut = `${line} ${words.slice(i).join(' ')}`;
      while (cut.length > 1 && !fits(cut + ELLIPSIS)) cut = cut.slice(0, -1).trimEnd();
      lines.push(cut + ELLIPSIS);
      return lines;
    }
    lines.push(line);
    line = words[i];
  }
  if (line) lines.push(line);
  return lines;
}

/** `entry`'s caption on a card `width` by `height` points, `hair` a hairline, in `display` and `mono`. */
export function captionOf(
  entry: Entry,
  width: number,
  height: number,
  hair: number,
  display: Metrics,
  mono: Metrics,
): Caption {
  const u = width / 340;
  const number = String(entry.number).padStart(2, '0');
  const set = typeset.create(128);
  const measure = (font: Metrics, text: string, em: number, tracking: number) =>
    set.set(font, text, { em, tracking, align: 'left' }).width;

  // Inside the frame's point-wide border: the 4:3 art window, then the caption's padding
  const left = 1 + 16 * u;
  const right = width - 1 - 16 * u;
  const artBottom = 1 + (width - 2) * 0.75;
  const mono10 = lineOf(mono, 10 * u);
  const displayed: Label[] = [];
  const typed: Label[] = [];

  let top = artBottom + 14 * u;
  const kicker = { em: 10 * u, tracking: 0.14, align: 'left' as const };
  typed.push({
    ...kicker,
    text: number,
    x: left,
    y: top + mono10.baseline,
    color: ACCENT,
  });
  typed.push({
    ...kicker,
    text: `  ${entry.category.toUpperCase()}`,
    x: left + measure(mono, number, kicker.em, kicker.tracking) + 0.14 * kicker.em,
    y: top + mono10.baseline,
    color: MUTED,
  });
  top += mono10.box + 6 * u;

  const title = { em: 20 * u, tracking: -0.01, align: 'left' as const };
  const titleLine = lineOf(display, title.em);
  const titleFits = (line: string) =>
    measure(display, line, title.em, title.tracking) <= right - left;
  for (const text of wrap(titleFits, entry.title, TITLE_LINES)) {
    displayed.push({ ...title, text, x: left, y: top + titleLine.baseline, color: TEXT });
    top += titleLine.box;
  }
  top += 6 * u;

  const line = { em: 11 * u, tracking: 0, align: 'left' as const };
  const lineLine = lineOf(mono, line.em, 17.6 * u);
  const lineFits = (text: string) => measure(mono, text, line.em, 0) <= right - left;
  for (const text of wrap(lineFits, entry.line, LINE_LINES)) {
    typed.push({ ...line, text, x: left, y: top + lineLine.baseline, color: MUTED });
    top += lineLine.box;
  }

  // The foot: a rule, then the gem, the rarity and the line count on one row
  const footTop = height - 1 - 12 * u - mono10.box - 10 * u - hair;
  const row = footTop + hair + 10 * u;
  const foot = { em: 10 * u, tracking: 0.14, y: row + mono10.baseline };
  const tier = entry.tier.toUpperCase();
  const count = `${entry.lines.toLocaleString('en')} LINES`;
  typed.push({ ...foot, text: tier, x: left + 14 * u, align: 'left', color: entry.tint });
  typed.push({ ...foot, text: count, x: right, align: 'right', color: MUTED });

  return {
    display: displayed,
    mono: typed,
    artBottom,
    footTop,
    rule: [left, right],
    gem: [left + 3 * u, row + mono10.box / 2, 3 * Math.SQRT2 * u],
  };
}
