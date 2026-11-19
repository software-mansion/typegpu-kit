import { describe, expect, it } from 'vitest';
import { images } from '../src/index.ts';
import { recordingRoot } from '../src/testing.ts';

const ascii = (text: string) => [...text].map((each) => each.charCodeAt(0));

/** An 8 × 2 file: a run-length row of (1, 0.5, 0.25), then a flat row repeating 3.98 in red. */
const hdrFile = (size = '-Y 2 +X 8') =>
  new Uint8Array([
    ...ascii(`#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n${size}\n`),
    // Run-length row: each channel runs alone, red and blue as runs, green as literals
    2,
    2,
    0,
    8,
    128 + 8,
    128,
    8,
    64,
    64,
    64,
    64,
    64,
    64,
    64,
    64,
    128 + 8,
    32,
    128 + 8,
    129,
    // Flat row: one pixel, then the old repeat marker for the other seven
    255,
    0,
    0,
    130,
    1,
    1,
    1,
    7,
  ]);

const blank = (width: number, height: number) => ({
  width,
  height,
  pixels: new Uint8Array(width * height * 4),
});

const raw = <S extends [number, number] | [number, number, number]>(
  texture: images.Texture<S>,
  root: ReturnType<typeof recordingRoot>['root'],
) => root.unwrap(texture);

describe('decodeHdr', () => {
  it('decodes run-length and flat RGBE rows into half floats', () => {
    const { width, height, pixels } = images.decodeHdr(hdrFile());

    expect([width, height]).toEqual([8, 2]);
    // 1, 0.5, 0.25 and an alpha of 1 in every pixel of the first row
    expect([...pixels.subarray(0, 8)]).toEqual([
      0x3c00, 0x3800, 0x3400, 0x3c00, 0x3c00, 0x3800, 0x3400, 0x3c00,
    ]);
    // 255 × 2^-6 = 3.984375, exact in a half float, repeated along the second row
    expect(new Set(pixels.subarray(32).filter((_, i) => i % 4 === 0))).toEqual(new Set([0x43f8]));
  });

  it('clamps at the largest half float', () => {
    const file = new Uint8Array([...ascii('#?RGBE\n\n-Y 1 +X 1\n'), 255, 0, 0, 255]);

    expect([...images.decodeHdr(file).pixels]).toEqual([0x7bff, 0, 0, 0x3c00]);
  });

  it('rejects other files, orientations, sizes, truncated rows and corrupt run markers', () => {
    const file = hdrFile();
    // A run-length marker for 9 pixels on a row of 8
    const marked = file.slice();
    marked[file.indexOf(2) + 3] = 9;

    expect(() => images.decodeHdr(new Uint8Array(ascii('PNG\n')))).toThrow('not a Radiance');
    expect(() => images.decodeHdr(hdrFile('+Y 2 +X 8'))).toThrow('rows from the top');
    expect(() => images.decodeHdr(hdrFile('-Y 0 +X 0'))).toThrow('1 to 16 384');
    expect(() => images.decodeHdr(hdrFile('-Y 30000 +X 30000'))).toThrow('1 to 16 384');
    expect(() => images.decodeHdr(file.subarray(0, 60))).toThrow('before its last row');
    expect(() => images.decodeHdr(marked)).toThrow('bad row');
  });

  it('is what decode gives for the Radiance MIME type, without createImageBitmap', async () => {
    const decoded = await images.decode({ bytes: hdrFile(), mimeType: images.HDR });

    expect(decoded).toEqual(images.decodeHdr(hdrFile()));
  });
});

describe('create', () => {
  it('stores colour as sRGB, data as linear and half floats as rgba16float, with every mip', () => {
    const { root } = recordingRoot();

    const color = raw(images.create(root, blank(8, 2)), root);
    const data = raw(images.create(root, blank(8, 2), { color: 'linear', mips: false }), root);
    const hdr = raw(images.create(root, images.decodeHdr(hdrFile())), root);

    expect([color.format, color.mipLevelCount]).toEqual(['rgba8unorm-srgb', 4]);
    expect([data.format, data.mipLevelCount]).toEqual(['rgba8unorm', 1]);
    expect([hdr.format, hdr.width, hdr.height]).toEqual(['rgba16float', 8, 2]);
  });

  it('makes a layer per image for arrays and cubes', () => {
    const { root } = recordingRoot();
    const faces = Array.from({ length: 6 }, () => blank(4, 4));

    expect(raw(images.array(root, faces.slice(0, 3)), root).depthOrArrayLayers).toBe(3);
    expect(raw(images.cube(root, faces), root).depthOrArrayLayers).toBe(6);
  });

  it('rejects images it cannot store together', () => {
    const { root } = recordingRoot();
    const hdr = images.decodeHdr(hdrFile());

    expect(() => images.cube(root, [blank(4, 4)])).toThrow('6 faces');
    expect(() =>
      images.cube(
        root,
        Array.from({ length: 6 }, () => blank(4, 2)),
      ),
    ).toThrow('square');
    expect(() => images.array(root, [blank(4, 4), blank(2, 2)])).toThrow('one size');
    expect(() => images.array(root, [hdr, blank(8, 2)])).toThrow("can't mix");
    expect(() => images.create(root, { width: 2, height: 2, pixels: new Uint8Array(4) })).toThrow(
      RangeError,
    );
    expect(() => images.create(root, blank(0, 0))).toThrow('texels a side');
  });
});
