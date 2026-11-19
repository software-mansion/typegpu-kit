import {
  d,
  type SampledFlag,
  type StorageFlag,
  type TgpuBuffer,
  type TgpuRoot,
  type TgpuTexture,
} from 'typegpu';
import type { Baked, FontData } from './data.ts';

export type { Baked, FontData, GlyphData } from './data.ts';

/** This schema describes a glyph quad around the pen in em and its matching atlas coordinates. */
export const Glyph = d.struct({
  /** Holds left, bottom, right and top bounds with y increasing upwards from the baseline. */
  plane: d.vec4f,
  /** Holds left, top, right and bottom coordinates from 0 to 1 across the atlas. */
  uv: d.vec4f,
});

/** These metrics provide advances and kerning for layout without loading the atlas. */
export interface Metrics {
  /** Holds the glyph metrics and kerning the font was made from. */
  readonly data: FontData;
  /** Returns the glyph index for a code point, or the missing glyph index if absent. */
  glyphOf(code: number): number;
  /** Returns the pen advance in em before kerning. */
  advance(glyph: number): number;
  /** Returns whether a glyph draws ink. Spaces return false. */
  inked(glyph: number): boolean;
  /** Returns the kerning offset in em to add to the first glyph advance. */
  kerning(first: number, second: number): number;
}

export interface Font extends Metrics, Disposable {
  /**
   * Holds glyph quads by index, with font characters in order followed by the missing glyph.
   */
  readonly glyphs: TgpuBuffer<d.WgslArray<typeof Glyph>> & StorageFlag;
  /** Provides the atlas in `rgba8unorm` format. */
  readonly atlas: TgpuTexture<{ size: [number, number]; format: 'rgba8unorm' }> & SampledFlag;
  /** Frees the atlas and glyph buffer. `[Symbol.dispose]` does the same. */
  destroy(): void;
}

const shapesOf = (data: FontData) => [...Object.values(data.glyphs), data.missing];

function checkVersion(from: string, version: unknown) {
  if (version !== 1) {
    throw new Error(
      `${from} requires bake version 1. Received version ${String(version)}. Bake the font again.`,
    );
  }
}

/**
 * Creates metrics for CPU text layout without uploading an atlas. Use them to measure text in
 * Node, in a worker or before `tgpu.init`.
 *
 * @param data the bake's JSON, as `read` returns it
 * @returns the font's metrics, which hold nothing on the GPU
 */
export function metrics(data: FontData): Metrics {
  checkVersion('font.metrics', data.version);

  const characters = Object.keys(data.glyphs);
  const shapes = shapesOf(data);
  const missing = characters.length;
  const indices = new Map(characters.map((character, i) => [character.codePointAt(0)!, i]));

  const advances = new Float32Array(shapes.map((shape) => shape.advance));
  const inks = shapes.map((shape) => shape.plane !== undefined && shape.atlas !== undefined);

  // Use numeric glyph-pair keys to avoid allocating strings during lookup
  const pairs = new Map<number, number>();
  for (const [pair, kern] of Object.entries(data.kerning)) {
    const [first, second] = [...pair].map((character) => indices.get(character.codePointAt(0)!));
    if (first !== undefined && second !== undefined)
      pairs.set(first * shapes.length + second, kern);
  }

  return {
    data,
    glyphOf: (code) => indices.get(code) ?? missing,
    advance: (glyph) => advances[glyph],
    inked: (glyph) => inks[glyph],
    kerning: (first, second) => pairs.get(first * shapes.length + second) ?? 0,
  };
}

function wrong(path: string, takes: string, got: unknown): never {
  throw new TypeError(
    `font.read requires ${takes} at \`${path}\`. Received ${JSON.stringify(got) ?? String(got)}. Use JSON written by typegpu-kit-bake-font.`,
  );
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return wrong(path, 'an object', value);
  }
  return value as Record<string, unknown>;
}

function number(value: unknown, path: string) {
  if (typeof value !== 'number' || !Number.isFinite(value)) wrong(path, 'a number', value);
  return value as number;
}

function whole(value: unknown, path: string) {
  if (!Number.isInteger(value) || (value as number) < 1)
    wrong(path, 'a whole number from 1', value);
  return value as number;
}

function shapeOf(value: unknown, path: string) {
  const shape = object(value, path);
  number(shape.advance, `${path}.advance`);
  if ((shape.plane === undefined) !== (shape.atlas === undefined)) {
    wrong(path, 'both `plane` and `atlas` or neither', value);
  }
  for (const key of ['plane', 'atlas']) {
    const quad = shape[key];
    if (quad !== undefined && !(Array.isArray(quad) && quad.length === 4)) {
      wrong(`${path}.${key}`, '4 numbers', quad);
    }
    if (Array.isArray(quad)) quad.forEach((each, i) => number(each, `${path}.${key}[${i}]`));
  }
}

/**
 * Validates JSON from `typegpu-kit-bake-font` and pairs it with atlas bytes. Throws with the
 * field name for invalid data or versions, or if the atlas size does not match.
 *
 * @param json the JSON, as text or parsed
 * @param atlas the atlas's bytes, 4 a texel
 * @returns the bake, as `create` takes it; it throws, naming the field, on JSON of another shape or
 * version, and on an atlas of another size
 */
export function read(json: unknown, atlas: ArrayBufferLike | ArrayBufferView): Baked {
  const data = object(typeof json === 'string' ? JSON.parse(json) : json, 'the JSON');
  if (typeof data.version !== 'number') wrong('version', '1', data.version);
  checkVersion('font.read', data.version);

  const source = object(data.source, 'source');
  for (const key of ['file', 'glyph']) {
    if (typeof source[key] !== 'string') wrong(`source.${key}`, 'a string', source[key]);
  }
  number(source.emSize, 'source.emSize');
  number(source.pixelRange, 'source.pixelRange');
  for (const key of ['unitsPerEm', 'ascender', 'descender', 'capHeight', 'lineGap']) {
    number(data[key], key);
  }

  const sheet = object(data.atlas, 'atlas');
  const width = whole(sheet.width, 'atlas.width');
  const height = whole(sheet.height, 'atlas.height');
  number(sheet.distanceRange, 'atlas.distanceRange');

  shapeOf(data.missing, 'missing');
  for (const [character, shape] of Object.entries(object(data.glyphs, 'glyphs'))) {
    if ([...character].length !== 1) wrong('glyphs', 'a character a key', character);
    shapeOf(shape, `glyphs.${character}`);
  }
  for (const [pair, kern] of Object.entries(object(data.kerning, 'kerning'))) {
    if ([...pair].length !== 2) wrong('kerning', 'two characters a key', pair);
    number(kern, `kerning.${pair}`);
  }

  const bytes = ArrayBuffer.isView(atlas)
    ? new Uint8Array(atlas.buffer, atlas.byteOffset, atlas.byteLength)
    : new Uint8Array(atlas);
  if (bytes.byteLength !== width * height * 4) {
    throw new RangeError(
      `font.read requires ${width * height * 4} bytes for a ${width} × ${height} RGBA atlas. Received ${bytes.byteLength}.`,
    );
  }

  return { data: data as unknown as FontData, atlas: bytes };
}

/**
 * Uploads a baked font atlas and glyph quads for text sets. The font also provides advances and
 * kerning for CPU layout. Throws if the bake version is not 1 or the atlas byte count does not
 * match its dimensions.
 *
 * @owns a texture, a buffer: the atlas, and each glyph's quad
 * @param baked the bake's JSON and the atlas's bytes, as `read` returns them
 * @returns a font you destroy after the text sets that draw with it
 */
export function create(root: TgpuRoot, baked: Baked): Font {
  const { data, atlas: texels } = baked;
  const { width, height } = data.atlas;

  checkVersion('font.create', data.version);

  if (texels.byteLength !== width * height * 4) {
    throw new RangeError(
      `font.create requires ${width * height * 4} bytes for a ${width} × ${height} RGBA atlas. Received ${texels.byteLength}.`,
    );
  }

  const measured = metrics(data);
  const shapes = shapesOf(data);

  const glyphs = root
    .createBuffer(
      d.arrayOf(Glyph, shapes.length),
      shapes.map(({ plane = [0, 0, 0, 0], atlas = [0, 0, 0, 0] }) => ({
        plane: d.vec4f(...plane),
        uv: d.vec4f(atlas[0] / width, atlas[1] / height, atlas[2] / width, atlas[3] / height),
      })),
    )
    .$usage('storage');

  const atlas = root
    .createTexture({ size: [width, height], format: 'rgba8unorm' })
    .$usage('sampled');
  atlas.write(texels);

  const destroy = () => {
    glyphs.destroy();
    atlas.destroy();
  };

  return {
    ...measured,
    glyphs,
    atlas,
    destroy,
    [Symbol.dispose]: destroy,
  };
}
