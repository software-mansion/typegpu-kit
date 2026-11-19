#!/usr/bin/env node
/**
 * This Node build-time tool bakes fonts for `@typegpu/kit` with `@pmndrs/glyph`. The MSDF baker
 * draws an atlas, and the shaper measures character advances and pair kerning. The runtime reads
 * the result without glyph or WebAssembly. Requires Node 24.2 or later.
 *
 * For example, `typegpu-kit-bake-font --input Font.ttf --output fonts/font --characters "…°é"`
 * writes `fonts/font.json` and `fonts/font.rgba` with printable ASCII and the supplied
 * characters.
 *
 * @module
 * @kind tool
 * @flat
 */

import { realpathSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { glyph, msdf } from '@pmndrs/glyph';
import { bakeFont } from '@pmndrs/glyph/bake';
import { msdfBaker } from '@pmndrs/glyph/bakers/msdf';
import { defineTypeGpuConfig } from '@pmndrs/glyph/typegpu';
import type { Baked, FontData, GlyphData } from './data.ts';

/** This string contains printable ASCII from space through tilde, included in every bake. */
export const ASCII = Array.from({ length: 0x7f - 0x20 }, (_, i) =>
  String.fromCharCode(0x20 + i),
).join('');

export interface BakeOptions {
  /** Selects a static TTF, OTF, TTC or OTC file. glyph cannot bake variable fonts. */
  readonly input: string;
  /**
   * Adds characters to printable ASCII, such as `…°é`. Only Latin text is supported because the
   * runtime performs no shaping. Default none.
   */
  readonly characters?: string;
  /**
   * Sets atlas texels per em. Larger values keep sharper corners on large text and make a larger
   * atlas. Default 40.
   */
  readonly emSize?: number;
  /** Sets the signed-distance range of the atlas in texels. Default 8. */
  readonly pixelRange?: number;
  /** Selects a face within a font collection. Default 0. */
  readonly fontFaceIndex?: number;
}

/** Disable ligatures and contextual forms because the runtime draws one glyph per character. */
const FEATURES = ['liga', 'clig', 'dlig', 'calt'].map((tag) => ({ tag, value: 0 }));

/** Each glyph record holds plane bounds, atlas bounds, page and flags. */
const RECORD = 20;
const ABSENT = 0xffff;

/** Glyph pages use the KTX2 format value for Vulkan R8G8B8A8_UNORM. */
const RGBA8 = 37;

const glyphFile = (name: string) => createRequire(import.meta.url).resolve(`@pmndrs/glyph/${name}`);

/**
 * Reads `sCapHeight` in font units from the OS/2 table of a face, or returns undefined for tables
 * before version 2, which lack it.
 */
function capHeightOf(font: Uint8Array, faceIndex: number) {
  const view = new DataView(font.buffer, font.byteOffset, font.byteLength);
  // A collection lists each face's table directory after its header
  const face = view.getUint32(0) === 0x74746366 ? view.getUint32(12 + faceIndex * 4) : 0;
  const tables = view.getUint16(face + 4);
  for (let i = 0; i < tables; i++) {
    const entry = face + 12 + i * 16;
    if (view.getUint32(entry) !== 0x4f532f32) continue;
    const table = view.getUint32(entry + 8);
    return view.getUint16(table) >= 2 ? view.getInt16(table + 88) : undefined;
  }
  return undefined;
}

/** Round metrics to six decimal places to limit JSON size. */
const round = (value: number) => Math.round(value * 1e6) / 1e6 + 0;

function readGlb(bytes: Uint8Array) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const jsonLength = view.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
  const bin = bytes.subarray(28 + jsonLength);

  const bufferView = (index: number) => {
    const { byteOffset = 0, byteLength } = json.bufferViews[index];
    return bin.subarray(byteOffset, byteOffset + byteLength);
  };

  return { json, bufferView };
}

function ktxPixels(ktx: Uint8Array) {
  const view = new DataView(ktx.buffer, ktx.byteOffset, ktx.byteLength);
  if (view.getUint32(12, true) !== RGBA8 || view.getUint32(44, true) !== 0) {
    throw new Error('typegpu-kit-bake-font requires an uncompressed RGBA8 atlas page from glyph.');
  }

  const offset = Number(view.getBigUint64(80, true));
  const length = Number(view.getBigUint64(88, true));
  return ktx.slice(offset, offset + length);
}

/** The glyph TypeGPU config requires a uniform, but this shaping-only path does not draw. */
const standIn = { createUniform: () => ({ write() {}, buffer: { destroy() {} } }) } as never;

/** Use a stand-in root because shaping needs text layout without GPU drawing. */
async function shaper(font: Uint8Array, unitsPerEm: number, raster: ReturnType<typeof msdf>) {
  const wasm = glyphFile('text-shaper.wasm');
  await glyph.init({ wasm: await readFile(wasm) });

  const handle = glyph.handle(
    `typegpu-kit-bake-font ${Math.random()}`,
    defineTypeGpuConfig({ root: standIn, format: 'rgba8unorm' }),
  );

  const face = glyph.fontFace(new Blob([font as Uint8Array<ArrayBuffer>]), { format: raster });
  await face.load();

  // Use one em in font units so the shaper returns positions in font units
  const style = { fontSize: unitsPerEm, features: FEATURES };

  return {
    shape(text: string) {
      const shaped = handle.createText({ font: face, text, style });
      try {
        const { glyphIds, x, glyphAdvances } = shaped.glyphs();
        return { ids: [...glyphIds], x: [...x], advances: [...glyphAdvances] };
      } finally {
        shaped.dispose();
      }
    },
    dispose() {
      face.dispose();
      handle.dispose();
    },
  };
}

/** This result contains baked font data, atlas bytes and missing characters. */
export interface Bake extends Baked {
  /** Lists requested characters that the font lacks. They use the missing glyph. */
  readonly absent: string;
}

/**
 * Bakes printable ASCII and `characters` from a font file into an atlas and metrics. Requires Node
 * 24.2 or later, a static font and a single uncompressed RGBA8 atlas page, and throws otherwise.
 *
 * @param options the font file, and what to bake beside ASCII
 * @returns the JSON `font.create` reads, the atlas's RGBA bytes, and the characters the font lacks
 */
export async function bake(options: BakeOptions): Promise<Bake> {
  const [major = 0, minor = 0] = process.versions.node.split('.').map(Number);
  if (major < 24 || (major === 24 && minor < 2)) {
    throw new Error(`Baking fonts requires Node 24.2 or later, not ${process.versions.node}.`);
  }

  const { input, emSize = 40, pixelRange = 8, fontFaceIndex = 0 } = options;
  const characters = [...new Set(ASCII + (options.characters ?? ''))];
  const codes = characters.map((character) => character.codePointAt(0)!);

  const directory = await mkdtemp(path.join(tmpdir(), 'typegpu-kit-bake-font-'));
  let glb: Uint8Array;
  try {
    const output = path.join(directory, 'font.glb');
    await bakeFont({
      input,
      output,
      font: { fontFaceIndex },
      unicodeRanges: codes.map((code) => ({ start: code, end: code })),
      rasters: [
        {
          baker: msdfBaker,
          packaging: { artifact: 'embedded' },
          options: { emSize, pixelRange },
        },
      ],
    }).catch((error: unknown) => {
      if (error instanceof Error && error.message.includes('unsupported table fvar')) {
        throw new Error(
          `typegpu-kit-bake-font requires a static font, but ${path.basename(input)} is variable. Pin its axes first, for example with \`fonttools varLib.instancer ${path.basename(input)} wght=600\`.`,
          { cause: error },
        );
      }
      throw error;
    });
    glb = await readFile(output);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }

  const { json, bufferView } = readGlb(glb);
  const metrics = json.extensions.PMNDRS_font.metrics;
  const field = json.extensions.PMNDRS_font_distance_field;
  const { unitsPerEm } = metrics;

  if (field.pages.length !== 1) {
    throw new Error(
      `typegpu-kit-bake-font produced ${field.pages.length} atlas pages, but the runtime requires one. Bake fewer characters or reduce --em-size.`,
    );
  }

  const page = field.pages[0];
  const variant = page.variants.find(
    (each: { gpuFormat: string }) => each.gpuFormat === 'rgba8unorm',
  );
  const atlas = ktxPixels(bufferView(variant.source.bufferView));
  const records = bufferView(field.recordBufferView);
  const recordView = new DataView(records.buffer, records.byteOffset, records.byteLength);

  const glyphOf = (id: number, advance: number): GlyphData => {
    const at = id * RECORD;
    const advanceEm = round(advance / unitsPerEm);
    if (recordView.getUint16(at + 16, true) === ABSENT) return { advance: advanceEm };

    const plane = (offset: number) =>
      round(recordView.getInt16(at + offset, true) / field.planeUnitsPerEm);
    const texel = (offset: number) => recordView.getUint16(at + offset, true);
    return {
      advance: advanceEm,
      plane: [plane(0), plane(2), plane(4), plane(6)],
      atlas: [texel(8), texel(10), texel(12), texel(14)],
    };
  };

  // Open the baked font with the same descriptor as the bake
  const shaping = await shaper(glb, unitsPerEm, msdf({ emSize, pixelRange }));
  const glyphs: Record<string, GlyphData> = {};
  const ids = new Map<string, number>();
  const advances = new Map<string, number>();
  const absent: string[] = [];

  try {
    for (const character of characters) {
      const shaped = shaping.shape(character);
      if (shaped.ids.length !== 1 || shaped.ids[0] === 0) {
        absent.push(character);
        continue;
      }

      ids.set(character, shaped.ids[0]);
      advances.set(character, shaped.advances[0]);
      glyphs[character] = glyphOf(shaped.ids[0], shaped.advances[0]);
    }

    // Shape all pairs for each first character in one text to avoid collisions in the 32-bit
    // hashes glyph uses to name texts
    const kerning: Record<string, number> = {};
    const present = [...ids.keys()];
    for (const first of present) {
      const shaped = shaping.shape(present.map((second) => first + second).join(''));

      present.forEach((second, i) => {
        const at = 2 * i;
        if (shaped.ids[at] !== ids.get(first) || shaped.ids[at + 1] !== ids.get(second)) return;

        const kern = round((shaped.x[at + 1] - shaped.x[at] - advances.get(first)!) / unitsPerEm);
        if (kern !== 0) kerning[first + second] = kern;
      });
    }

    const notdef = shaping.shape('￿');
    const missing = glyphOf(0, notdef.ids[0] === 0 ? notdef.advances[0] : unitsPerEm / 2);

    const version = JSON.parse(await readFile(glyphFile('package.json'), 'utf8')).version;

    // Fonts without the OS/2 cap height use the top of H, less the padding the atlas adds
    const capUnits = capHeightOf(await readFile(input), fontFaceIndex);
    const capHeight =
      capUnits === undefined
        ? (glyphs.H?.plane?.[3] ?? metrics.ascender / unitsPerEm) - pixelRange / 2 / emSize
        : capUnits / unitsPerEm;

    const data: FontData = {
      version: 1,
      source: { file: path.basename(input), glyph: version, emSize, pixelRange },
      unitsPerEm,
      ascender: round(metrics.ascender / unitsPerEm),
      descender: round(metrics.descender / unitsPerEm),
      capHeight: round(capHeight),
      lineGap: round(metrics.lineGap / unitsPerEm),
      atlas: { width: page.width, height: page.height, distanceRange: field.pixelRange },
      missing,
      glyphs,
      kerning,
    };

    return { data, atlas, absent: absent.join('') };
  } finally {
    shaping.dispose();
  }
}

const lines = (record: Readonly<Record<string, unknown>>, indent: string) =>
  Object.entries(record)
    .map(([key, value]) => `${indent}${JSON.stringify(key)}: ${JSON.stringify(value)}`)
    .join(',\n');

/**
 * Formats baked font JSON with one line per glyph and kerning pair so diffs show individual
 * changes.
 *
 * @param data the font's metrics, glyphs and kerning: what `bake` returns as `data`
 * @returns the JSON text
 */
export function format(data: FontData) {
  const { glyphs, kerning, ...rest } = data;
  return `{\n${lines(rest, '  ')},\n  "glyphs": {\n${lines(glyphs, '    ')}\n  },\n  "kerning": {\n${lines(kerning, '    ')}\n  }\n}\n`;
}

async function main() {
  const { values } = parseArgs({
    options: {
      input: { type: 'string' },
      output: { type: 'string' },
      characters: { type: 'string' },
      'em-size': { type: 'string' },
      'pixel-range': { type: 'string' },
      'font-face-index': { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.help || !values.input || !values.output) {
    console.log(
      'Usage: typegpu-kit-bake-font --input <font> --output <path without extension> [options]\n\nWrites <path>.json and <path>.rgba with printable ASCII and any additional characters.\n\nOptions:\n  --input <font>         Read a static TTF, OTF, TTC or OTC file.\n  --output <path>        Write to this path without an extension.\n  --characters <text>    Add these characters to printable ASCII.\n  --em-size <number>     Set atlas texels per em. Default 40.\n  --pixel-range <number> Set the distance range in texels. Default 8.\n  --font-face-index <n>   Select a face in a font collection. Default 0.\n  --help, -h             Show this help.',
    );
    process.exitCode = values.help ? 0 : 1;
    return;
  }

  const { data, atlas, absent } = await bake({
    input: values.input,
    ...(values.characters === undefined ? {} : { characters: values.characters }),
    ...(values['em-size'] === undefined ? {} : { emSize: Number(values['em-size']) }),
    ...(values['pixel-range'] === undefined ? {} : { pixelRange: Number(values['pixel-range']) }),
    ...(values['font-face-index'] === undefined
      ? {}
      : { fontFaceIndex: Number(values['font-face-index']) }),
  });

  if (absent.length > 0) {
    console.warn(
      `typegpu-kit-bake-font found no glyph for ${JSON.stringify(absent)}. These characters will be blank.`,
    );
  }

  await writeFile(`${values.output}.json`, format(data));
  await writeFile(`${values.output}.rgba`, atlas);

  const pairs = Object.keys(data.kerning).length;
  console.log(
    `Wrote ${values.output}.json and ${values.output}.rgba from ${values.input} with ${Object.keys(data.glyphs).length} characters, ${pairs} kerning pairs and a ${data.atlas.width} × ${data.atlas.height} atlas.`,
  );
}

// Package managers link the command to this file, so compare real paths
const command = process.argv[1];
if (command && realpathSync(command) === fileURLToPath(import.meta.url)) await main();
