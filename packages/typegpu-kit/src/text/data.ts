/** This type describes a glyph in em, relative to the font size. */
export interface GlyphData {
  /** Gives the pen advance before kerning. */
  readonly advance: number;
  /**
   * Gives left, bottom, right and top bounds around the pen, with y increasing from the baseline
   * upwards.
   */
  readonly plane?: readonly [number, number, number, number];
  /** Gives left, top, right and bottom atlas bounds in texels, with y increasing downwards. */
  readonly atlas?: readonly [number, number, number, number];
}

/**
 * This type describes the font written by `typegpu-kit-bake-font`. Its atlas contains
 * `atlas.width` by `atlas.height` RGBA texels. Lengths use em except atlas measurements, which
 * use texels.
 */
export interface FontData {
  /** Identifies format version 1. */
  readonly version: 1;
  /** Records the source filename, glyph version and bake settings. */
  readonly source: {
    readonly file: string;
    readonly glyph: string;
    readonly emSize: number;
    readonly pixelRange: number;
  };
  /**
   * Gives the source font units per em. Other font lengths have already been divided by this
   * value.
   */
  readonly unitsPerEm: number;
  /** Gives the distance from the baseline to the top of a line. */
  readonly ascender: number;
  /** Gives the negative distance from the baseline to the bottom of a line. */
  readonly descender: number;
  /** Gives the height of flat capitals, such as H, above the baseline. */
  readonly capHeight: number;
  /** Gives the space added between lines. */
  readonly lineGap: number;
  readonly atlas: {
    readonly width: number;
    readonly height: number;
    /** Gives the distance range in texels represented by atlas values from 0 to 1. */
    readonly distanceRange: number;
  };
  /** Sets the advance for missing characters without drawing a glyph. */
  readonly missing: GlyphData;
  /** Maps characters to glyphs. Glyphs without ink, such as spaces, have no quad. */
  readonly glyphs: Readonly<Record<string, GlyphData>>;
  /** Maps character pairs to nonzero kerning offsets added to the first character advance. */
  readonly kerning: Readonly<Record<string, number>>;
}

/** This result contains font metrics, glyph data and the atlas. */
export interface Baked {
  /** Holds the glyph metrics and kerning. */
  readonly data: FontData;
  /**
   * Holds RGBA texels in rows from top to bottom, with 4 bytes per texel. The median of the
   * first three distance channels preserves corners. The fourth channel rounds corners.
   */
  readonly atlas: Uint8Array;
}
