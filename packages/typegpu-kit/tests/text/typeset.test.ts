import { describe, expect, it } from 'vitest';
import { font, typeset } from '../../src/index.ts';
import { inter } from './inter.ts';

// Metrics alone: typeset needs no root.
const sans = font.metrics(inter.data);
const glyph = (character: string) => sans.glyphOf(character.codePointAt(0)!);
const advance = (character: string) => sans.advance(glyph(character));

/** The width of `text` on one line, kerning included, in em. */
const width = (text: string) => {
  let total = 0;
  for (let i = 0; i < text.length; i++) {
    total += advance(text[i]);
    if (i + 1 < text.length) total += sans.kerning(glyph(text[i]), glyph(text[i + 1]));
  }
  return total;
};

/** The placed glyphs as characters, and their pens. */
const placed = (laid: typeset.Typesetter) =>
  Array.from({ length: laid.count }, (_, i) => ({
    glyph: laid.glyphs[i],
    x: laid.pens[i * 2],
    y: laid.pens[i * 2 + 1],
  }));

/** The anchor at the left of the first baseline, where pens are simplest to check. */
const LEFT = { align: 'left', baseline: 'alphabetic' } as const;

/** Its lines, placed glyphs and width, laid out from the left of the first baseline. */
const lines = (text: string) => {
  const laid = typeset.create(8).set(sans, text, { em: 1, ...LEFT });
  return { lines: laid.lines, placed: placed(laid), width: laid.width };
};

const lineHeight = () => sans.data.ascender - sans.data.descender + sans.data.lineGap;

describe('typeset', () => {
  it("moves each glyph by the one before's advance and their kerning", () => {
    const laid = typeset.create(8).set(sans, 'AV', { em: 100, ...LEFT });
    const [a, v] = placed(laid);

    expect(sans.data.kerning.AV).toBe(-0.104492);
    expect(a!.x).toBe(0);
    expect(v!.x).toBeCloseTo((advance('A') - 0.104492) * 100, 3);
    expect(laid.width).toBeCloseTo(width('AV') * 100, 3);
    expect(laid.width).toBeLessThan((advance('A') + advance('V')) * 100);
  });

  it('places no quad for a space, but advances past it', () => {
    const laid = typeset.create(8).set(sans, 'a b', { em: 1, ...LEFT });
    const [a, b] = placed(laid);

    expect(laid.count).toBe(2);
    expect(a!.glyph).toBe(glyph('a'));
    expect(b!.x).toBeCloseTo(width('a '), 5);
  });

  it('centres the block on the anchor by default', () => {
    const { ascender, descender } = sans.data;
    const laid = typeset.create(8).set(sans, 'ab', { em: 2 });

    expect(laid.pens[0]).toBeCloseTo(-width('ab'), 5);
    expect(laid.pens[1]).toBeCloseTo(-(ascender + descender), 5);
  });

  it('keeps a long line whole, and breaks lines only at newlines, keeping an empty one', () => {
    expect(typeset.create(32).set(sans, 'one two three fourteen', { em: 1 }).lines).toBe(1);

    const laid = typeset.create(8).set(sans, 'a\n\nb\n', { em: 1, ...LEFT });

    expect(laid.lines).toBe(4);
    expect(placed(laid).map(({ y }) => y)).toEqual([0, -2 * lineHeight()].map(Math.fround));
  });

  it('ends a line at \\r\\n and at a lone \\r as at \\n', () => {
    expect(lines('ab\r\ncd')).toEqual(lines('ab\ncd'));
    expect(lines('ab\rcd')).toEqual(lines('ab\ncd'));
    expect(lines('ab\r\n\r\ncd')).toEqual(lines('ab\n\ncd'));
  });

  it('puts the anchor at the left, centre or right of each line', () => {
    const text = 'wide line\nshort';
    const left = placed(typeset.create(16).set(sans, text, { em: 1, align: 'left' }));
    const center = placed(typeset.create(16).set(sans, text, { em: 1 }));
    const right = placed(typeset.create(16).set(sans, text, { em: 1, align: 'right' }));

    const second = left.findIndex(({ y }) => y < 0);
    expect(left[0].x).toBe(0);
    expect(left[second].x).toBe(0);
    expect(center[0].x).toBeCloseTo(-width('wide line') / 2, 5);
    expect(center[second].x).toBeCloseTo(-width('short') / 2, 5);
    expect(right[0].x).toBeCloseTo(-width('wide line'), 5);
    expect(right[second].x).toBeCloseTo(-width('short'), 5);
  });

  it('puts the anchor at the top, capitals, middle, first baseline or bottom of the block', () => {
    const { ascender, descender, capHeight } = sans.data;
    const y = (baseline: NonNullable<typeset.Options['baseline']>) =>
      typeset.create(8).set(sans, 'a\nb', { em: 1, baseline }).pens[1];

    expect(y('alphabetic')).toBe(0);
    expect(y('top')).toBeCloseTo(-ascender, 5);
    expect(y('cap')).toBeCloseTo(-capHeight, 5);
    expect(y('bottom')).toBeCloseTo(lineHeight() - descender, 5);
    expect(y('middle')).toBeCloseTo((lineHeight() - ascender - descender) / 2, 5);

    const laid = typeset.create(8).set(sans, 'a\nb', { em: 2 });
    expect(laid.height).toBeCloseTo((ascender - descender + lineHeight()) * 2, 5);
  });

  it('tracks every character but the last of a line, spaces included, and widens by it', () => {
    const tracked = typeset.create(8).set(sans, 'AV A\nV', { em: 2, ...LEFT, tracking: 0.25 });
    const pens = placed(tracked).map(({ x }) => x);

    expect(pens[1]).toBeCloseTo((width('AV') - advance('V') + 0.25) * 2, 5);
    expect(pens[2]).toBeCloseTo((width('AV A') - advance('A') + 0.75) * 2, 5);
    expect(pens[3]).toBe(0);
    expect(tracked.width).toBeCloseTo((width('AV A') + 0.75) * 2, 5);
  });

  it('advances past a character the font lacks, drawing nothing', () => {
    const laid = typeset.create(8).set(sans, 'A一B', { em: 1, ...LEFT });
    const [a, b] = placed(laid);

    expect(laid.count).toBe(2);
    expect(a!.glyph).toBe(glyph('A'));
    expect(b!.x).toBeCloseTo(advance('A') + sans.data.missing.advance, 5);
  });

  it('keeps the advance for a missing character at the end of a line', () => {
    const glyphs = { A: inter.data.glyphs.A };
    const withoutSpace = font.metrics({ ...inter.data, glyphs });
    const laid = typeset.create(8).set(withoutSpace, 'A一  \nA ', { em: 1, ...LEFT });

    expect(laid.lines).toBe(2);
    expect(laid.count).toBe(2);
    expect(laid.width).toBeCloseTo(advance('A') + sans.data.missing.advance, 5);
  });

  it('places at most its capacity, and measures the whole text', () => {
    const laid = typeset.create(2).set(sans, 'ABC', { em: 1 });

    expect(laid.count).toBe(2);
    expect(laid.width).toBeCloseTo(width('ABC'), 5);
  });

  it('lays nothing out for an empty string', () => {
    const laid = typeset.create(2).set(sans, 'AB', { em: 1 }).set(sans, '', { em: 1 });
    expect([laid.count, laid.lines, laid.width, laid.height]).toEqual([0, 0, 0, 0]);
  });

  it('refuses an em of 0 and a capacity of 0, naming them', () => {
    expect(() => typeset.create(4).set(sans, 'A', { em: 0 })).toThrow(
      new RangeError("Set `em` above 0 when calling the typeset's `set` method. Got 0."),
    );
    expect(() => typeset.create(0)).toThrow(
      new RangeError(
        'typeset.create requires `capacity` to be an integer of at least 1. Received 0.',
      ),
    );
  });
});
