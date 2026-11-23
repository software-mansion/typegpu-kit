/**
 * Puts images on the GPU from files, pixels or Radiance HDR, as colour or data, with mips, layers
 * of an array or a cube, and samplers.
 *
 * @module
 * @submits setup `create`, `array` and `cube` generate mips with TypeGPU's `generateMipmaps`,
 * which submits its own commands, unless `mips` is false. With `packed`, or a `format` other than
 * the image's, they first draw each layer through a pipeline of their own and submit it
 * @owns textures, samplers, a baker with `packed` or a narrower `format`: the textures and
 * samplers each call returns, of which you destroy the textures; a staging texture it destroys and
 * a baker that draws it into the texture, used once while creating
 * @requires rg11b10ufloat-renderable option `format: 'rg11b10ufloat'` needs it
 */

import * as baker from './baker.ts';
import {
  d,
  std,
  type RenderFlag,
  type SampledFlag,
  type TgpuRoot,
  type TgpuSampler,
  type TgpuTexture,
} from 'typegpu';

/** This type holds an encoded image file, such as a PNG, JPEG, WebP or Radiance `.hdr`. */
export interface EncodedImage {
  readonly bytes: Uint8Array;
  readonly mimeType: string;
}

/**
 * This type holds unpremultiplied RGBA pixels, with 4 bytes per pixel in rows from top to
 * bottom.
 */
export interface DecodedImage {
  readonly width: number;
  readonly height: number;
  readonly pixels: Uint8Array;
}

/**
 * This type holds linear RGBA pixels as half floats, 4 per pixel in rows from top to bottom, as an
 * `rgba16float` texture stores them. `decodeHdr` makes it.
 */
export interface HdrImage {
  /** Reports the width in pixels. */
  readonly width: number;
  /** Reports the height in pixels. */
  readonly height: number;
  /** Holds the half-float values, 4 per pixel. */
  readonly pixels: Uint16Array;
}

/** This type accepts a decoded image: 8-bit pixels, half-float pixels or a bitmap. */
export type Picture = DecodedImage | HdrImage | ImageBitmap;

/** This type accepts an image to decode, or one decoded already. */
export type ImageSource = EncodedImage | Picture;

/**
 * Decodes an image file into pixels, or into a bitmap on a runtime with `createImageBitmap`, such
 * as React Native's JS thread before a scene's worker gets the result.
 */
export type Decode = (image: EncodedImage) => Promise<DecodedImage | ImageBitmap>;

/** The MIME type of Radiance `.hdr` files, which `decode` reads with `decodeHdr`. */
export const HDR = 'image/vnd.radiance';

/**
 * Says whether this runtime can decode PNG, JPEG and WebP files without a `Decode` of yours.
 *
 * @returns whether `createImageBitmap` exists
 */
export const decodesItself = () => typeof createImageBitmap === 'function';

const blobTakesBytes = () => {
  try {
    return new Blob([new Uint8Array(1)]).size === 1;
  } catch {
    return false;
  }
};

const sourceOf = ({ bytes, mimeType }: EncodedImage): ImageBitmapSource => {
  // React Native's Blob rejects bytes, and its createImageBitmap takes them raw
  if (!blobTakesBytes()) return bytes as unknown as ImageBitmapSource;
  // A Blob part needs an ArrayBuffer behind it, so copy a shared one's range
  const part =
    bytes.buffer instanceof ArrayBuffer ? (bytes as Uint8Array<ArrayBuffer>) : bytes.slice();
  return new Blob([part], { type: mimeType });
};

/**
 * Decodes an image file with `createImageBitmap`. On the web it decodes unpremultiplied and
 * without colour conversion, so alpha can hold data such as height. On React Native, call it on
 * the JS thread and give the bitmaps to a scene on a worker, which has no `createImageBitmap`.
 * React Native's decoder premultiplies alpha and then divides it back, so colour where alpha is
 * near 0 is lost, and it ignores `colorSpaceConversion`, so a colour profile in the file can change
 * the bytes. Keep data out of alpha there, or decode to pixels with your own `Decode`.
 *
 * @param image an encoded PNG, JPEG or WebP, such as one from `gltf.read`
 * @returns a bitmap for `create`, or for a model's `images`
 */
export function decodeBitmap(image: EncodedImage): Promise<ImageBitmap> {
  return createImageBitmap(sourceOf(image), {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
}

/**
 * Decodes an image file. A Radiance `.hdr` file, by its MIME type `images.HDR`, decodes with
 * `decodeHdr` on any thread. Other files decode with `custom`, or with `decodeBitmap` where
 * `createImageBitmap` exists. Rejects on a runtime without either.
 *
 * @param image the encoded file and its MIME type
 * @param custom decodes files other than HDR, such as on a worker without `createImageBitmap`
 * @returns the pixels or bitmap for `create`
 */
export async function decode(image: EncodedImage, custom?: Decode): Promise<Picture> {
  if (image.mimeType === HDR) return decodeHdr(image.bytes);
  if (custom) return custom(image);
  if (!decodesItself()) {
    throw new Error(
      'This runtime lacks `createImageBitmap`. Pass a `Decode` function, or decode the image ' +
        'on a thread that has `createImageBitmap`, for example with `images.decodeBitmap`.',
    );
  }
  return decodeBitmap(image);
}

const HALF_MAX = 0x7bff;
const HALF_ONE = 0x3c00;
/** The largest side `decodeHdr` reads. */
const LARGEST = 16384;
const ENDED = 'The HDR file ends before all its rows are complete.';
const BAD_RUN =
  'An HDR run length is invalid. Check for a run that crosses the end of its row or a zero-length channel run.';

// Half-float bits for m × 2^(e − 136), the value of mantissa byte m under RGBE exponent byte e, at
// e << 8 | m. Built on first use; 128 KB
let halves: Uint16Array | undefined;

const halvesOf = () => {
  if (halves) return halves;
  halves = new Uint16Array(0x10000);
  for (let e = 1; e < 256; e++) {
    for (let m = 1; m < 256; m++) {
      // m = 1.f × 2^k gives the exponent field k + e − 136 + 15 and the top bits of f as the
      // fraction, which a byte's 7 bits below its leading one always fit exactly
      const k = 31 - Math.clz32(m);
      const exponent = k + e - 121;
      let bits: number;
      if (exponent >= 31) bits = HALF_MAX;
      else if (exponent > 0) bits = (exponent << 10) | ((m << (10 - k)) & 0x3ff);
      else {
        // Below the smallest normal half, m × 2^(e − 112) counts subnormal steps, rounded half to
        // even
        const steps = m * 2 ** (e - 112);
        bits = Math.round(steps);
        if (bits - steps === 0.5 && bits & 1) bits--;
      }
      halves[(e << 8) | m] = bits;
    }
  }
  return halves;
};

/**
 * Decodes a Radiance `.hdr` file into linear half floats for an `rgba16float` texture. It reads
 * flat and run-length encoded RGBE scanlines in the standard `-Y +X` order, and clamps values
 * above 65 504, a half float's largest. It runs on any thread, a React Native worker included.
 * Throws for another orientation, an XYZE file, a side above 16 384 or a truncated or corrupt
 * file.
 *
 * @param bytes the whole file
 * @returns pixels for `create`
 */
export function decodeHdr(bytes: Uint8Array): HdrImage {
  let at = 0;
  const line = () => {
    const end = bytes.indexOf(0x0a, at);
    if (end < 0) throw new Error('The HDR file ends inside its header.');
    // Hermes has no TextDecoder, and the header is ASCII
    const text = String.fromCharCode(...bytes.subarray(at, Math.min(end, at + 256)));
    at = end + 1;
    return text;
  };

  const magic = line();
  if (!magic.startsWith('#?')) throw new Error('This is not a Radiance HDR file.');
  for (let header = line(); header.trim() !== ''; header = line()) {
    if (header.startsWith('FORMAT=') && header.trim() !== 'FORMAT=32-bit_rle_rgbe') {
      throw new Error(`HDR files in ${header.slice(7).trim()} are not supported; use RGBE.`);
    }
  }
  const size = /^-Y (\d+) \+X (\d+)$/.exec(line().trim());
  if (!size) throw new Error('HDR files must store rows from the top, `-Y height +X width`.');
  const height = Number(size[1]);
  const width = Number(size[2]);
  if (width < 1 || height < 1 || width > LARGEST || height > LARGEST) {
    throw new Error(
      `The HDR file is ${width} × ${height}; textures hold 1 to 16 384 texels a side.`,
    );
  }

  const half = halvesOf();
  const pixels = new Uint16Array(width * height * 4);
  const row = new Uint8Array(width * 4);
  const n = bytes.length;
  // Rows of at least 8 pixels may start with 2, 2 and the width, marking run-length encoding
  const marks = width >= 8;

  for (let y = 0; y < height; y++) {
    if (at + 4 > n) throw new Error(ENDED);

    if (marks && bytes[at] === 2 && bytes[at + 1] === 2 && bytes[at + 2] < 0x80) {
      if (((bytes[at + 2] << 8) | bytes[at + 3]) !== width) {
        throw new Error('An HDR row declares a different width from the file header.');
      }
      at += 4;
      // Each channel runs separately: a count above 128 repeats one byte, others copy bytes
      for (let channel = 0; channel < 4; channel++) {
        for (let x = 0; x < width;) {
          if (at >= n) throw new Error(ENDED);
          const count = bytes[at++];
          if (count > 128) {
            if (x + count - 128 > width) throw new Error(BAD_RUN);
            if (at >= n) throw new Error(ENDED);
            const value = bytes[at++];
            for (const end = x + count - 128; x < end; x++) row[x * 4 + channel] = value;
          } else {
            if (count === 0 || x + count > width) throw new Error(BAD_RUN);
            if (at + count > n) throw new Error(ENDED);
            for (const end = x + count; x < end; x++) row[x * 4 + channel] = bytes[at++];
          }
        }
      }
    } else {
      // Flat pixels; the old encoding marks a repeat of the previous pixel with 1, 1, 1, count
      let shift = 0;
      for (let x = 0; x < width;) {
        if (at + 4 > n) throw new Error(ENDED);
        const r = bytes[at];
        const g = bytes[at + 1];
        const b = bytes[at + 2];
        const e = bytes[at + 3];
        at += 4;
        if (r === 1 && g === 1 && b === 1 && x > 0) {
          const count = e * 2 ** shift;
          if (!(count <= width - x)) throw new Error(BAD_RUN);
          for (const end = x + count; x < end; x++) row.copyWithin(x * 4, (x - 1) * 4, x * 4);
          shift += 8;
        } else {
          row[x * 4] = r;
          row[x * 4 + 1] = g;
          row[x * 4 + 2] = b;
          row[x * 4 + 3] = e;
          x++;
          shift = 0;
        }
      }
    }

    for (let i = 0, o = y * width * 4; i < width * 4; i += 4, o += 4) {
      const e = row[i + 3] << 8;
      pixels[o] = half[e | row[i]];
      pixels[o + 1] = half[e | row[i + 1]];
      pixels[o + 2] = half[e | row[i + 2]];
      pixels[o + 3] = HALF_ONE;
    }
  }

  return { width, height, pixels };
}

/** These are the formats `create`, `array` and `cube` pick. */
export type Format = 'rgba8unorm-srgb' | 'rgba8unorm' | 'rgba16float' | 'rg11b10ufloat';

/** Turns a texel of an image as stored into linear RGBA, in a shader. */
export type Unpack = (texel: d.v4f) => d.v4f;

/** This texture holds one image, or one per layer of an array or a cube. */
export type Texture<S extends readonly number[] = [number, number]> = TgpuTexture<{
  size: S;
  format: Format;
}> &
  SampledFlag &
  RenderFlag;

export interface Options {
  /**
   * Says what 8-bit images hold: `'srgb'` for colour, which sampling turns into linear light, or
   * `'linear'` for data such as normals, roughness or height. Half-float images are linear
   * either way. Default `'srgb'`.
   */
  readonly color?: 'srgb' | 'linear';
  /** Generates every mip level down to 1 × 1. Default `true`. */
  readonly mips?: boolean;
  /**
   * Converts light stored in an 8-bit image, such as RGBM, to linear light once, at upload, so the
   * texture filters and mips like any HDR texture. A `'use gpu'` function that takes each texel as
   * stored, unfiltered, and returns linear RGBA. The texture takes `format`. It reads raw values,
   * so it takes no `color`. Default none.
   * @gpu
   */
  readonly packed?: Unpack;
  /**
   * Sets the format of a texture of light, from `packed` or half-float images.
   * `'rg11b10ufloat'` takes half the memory and drops alpha. It needs the device feature
   * `'rg11b10ufloat-renderable'`. 8-bit images without `packed` take the format `color` picks.
   * Default `'rgba16float'`.
   */
  readonly format?: 'rgba16float' | 'rg11b10ufloat';
}

/**
 * Unpacks RGBM, light stored as `rgb × a × range` in linear values, for `packed`.
 *
 * @param range the brightest light the encoding holds
 * @returns a function for `packed`
 */
export const rgbm = (range: number): Unpack => {
  const fromRgbm: Unpack = (texel) => {
    'use gpu';
    return d.vec4f(texel.xyz * (texel.w * range), 1);
  };
  return fromRgbm;
};

const isHdr = (picture: Picture): picture is HdrImage =>
  'pixels' in picture && picture.pixels instanceof Uint16Array;

const formatOf = (picture: Picture, color: 'srgb' | 'linear'): Format =>
  isHdr(picture) ? 'rgba16float' : color === 'linear' ? 'rgba8unorm' : 'rgba8unorm-srgb';

function layered<S extends [number, number] | [number, number, number]>(
  root: TgpuRoot,
  pictures: readonly Picture[],
  options: Options,
  what: string,
  size: (width: number, height: number) => S,
): Texture<S> {
  const { mips = true, packed } = options;
  if (packed && options.color) {
    throw new Error(`${what} with \`packed\` reads raw texels, so it takes no \`color\`.`);
  }
  const color = packed ? 'linear' : (options.color ?? 'srgb');
  const first = pictures[0];
  if (!first) throw new Error(`${what} needs at least one image.`);
  const { width, height } = first;
  const largest = root.device.limits.maxTextureDimension2D;
  if (width < 1 || height < 1 || width > largest || height > largest) {
    throw new Error(
      `${what} is 1 to ${largest} texels a side on this device, got ${width} × ${height}.`,
    );
  }
  const stored = formatOf(first, color);
  const light = packed !== undefined || stored === 'rgba16float';
  if (options.format && !light) {
    throw new Error(
      `${what} of 8-bit colour or data takes its format from \`color\`, so leave out \`format\`, or give \`packed\` to store light.`,
    );
  }
  const format = light ? (options.format ?? 'rgba16float') : stored;
  if (format === 'rg11b10ufloat' && !root.device.features.has('rg11b10ufloat-renderable')) {
    throw new Error(
      "An 'rg11b10ufloat' texture needs the device feature 'rg11b10ufloat-renderable'.",
    );
  }

  for (const picture of pictures) {
    if (picture.width !== width || picture.height !== height) {
      throw new Error(
        `${what} needs images of one size: ${width} × ${height}, got ${picture.width} × ${picture.height}.`,
      );
    }
    if (formatOf(picture, color) !== stored) {
      throw new Error(`${what} can't mix half-float and 8-bit images.`);
    }
    if ('pixels' in picture && picture.pixels.length !== width * height * 4) {
      throw new RangeError(
        `A ${width} × ${height} image holds ${width * height * 4} channel values, got ${picture.pixels.length}.`,
      );
    }
  }

  const levels = mips ? Math.floor(Math.log2(Math.max(width, height))) + 1 : 1;
  const texture = root
    .createTexture({ size: size(width, height), format, mipLevelCount: levels })
    .$usage('sampled', 'render');

  if (packed === undefined && format === stored) {
    upload(texture, pictures);
  } else {
    const raw = root
      .createTexture({ size: size(width, height), format: stored })
      .$usage('sampled', 'render');
    upload(raw, pictures);
    unpack(root, raw, texture, packed);
    raw.destroy();
  }
  if (levels > 1) texture.generateMipmaps();
  return texture;
}

/** A texture `layered` writes or draws into, of any size. */
type Layers = TgpuTexture & SampledFlag & RenderFlag;

function upload(texture: Layers, pictures: readonly Picture[]) {
  pictures.forEach((picture, layer) => {
    if ('pixels' in picture) {
      texture.write(picture.pixels, {
        origin: [0, 0, layer],
        size: [picture.width, picture.height, 1],
      });
    } else {
      texture.write(picture, { origin: [0, 0, layer] });
    }
  });
}

/**
 * Draws each layer of `raw` through `packed` into the first level of `texture`. Without `packed`,
 * as for a half-float image given a narrower `format`, each texel stays as it is.
 */
function unpack(root: TgpuRoot, raw: Layers, texture: Layers, packed: Unpack | undefined) {
  const stored = raw.createView(d.texture2dArray(d.f32));
  const unpacked = baker.create(root, texture, {
    texel: (uv, layer) => {
      'use gpu';
      const at = d.vec2i(uv * d.vec2f(std.textureDimensions(stored.$)));
      const value = std.textureLoad(stored.$, at, layer, 0);
      if (packed !== undefined) return packed(value);
      return d.vec4f(value);
    },
  });
  const encoder = root.createCommandEncoder();
  unpacked.record(encoder);
  encoder.submit();
}

/**
 * Creates a sampled texture from one image. 8-bit images become `rgba8unorm-srgb`, or
 * `rgba8unorm` with `color: 'linear'`. Half-float images, and 8-bit ones with `packed`, become
 * `format`. Throws if the pixel count doesn't match the size, the image is empty or larger than
 * the device allows, `packed` comes with `color`, or `format` with 8-bit colour or data.
 *
 * @param picture pixels from `decode`, `decodeHdr` or your own code, or a bitmap
 * @returns a texture you destroy, with every mip level unless `mips` is false
 */
export function create(root: TgpuRoot, picture: Picture, options: Options = {}): Texture {
  return layered(root, [picture], options, 'A texture', (w, h): [number, number] => [w, h]);
}

/**
 * Creates a sampled 2D array texture with one image per layer, such as frames of an animation or
 * terrain materials. Sample a view of `d.texture2dArray(d.f32)`. Throws for no images, or images
 * of different sizes or kinds.
 *
 * @param pictures the layers, all one size
 * @returns a texture you destroy
 */
export function array(
  root: TgpuRoot,
  pictures: readonly Picture[],
  options: Options = {},
): Texture<[number, number, number]> {
  return layered(root, pictures, options, 'A texture array', (w, h): [number, number, number] => [
    w,
    h,
    pictures.length,
  ]);
}

/**
 * Creates a cube map from six square faces in WebGPU's order: +x, −x, +y, −y, +z, −z. Sample a
 * view of `d.textureCube(d.f32)`. Throws unless there are six square faces of one size.
 *
 * @param faces the six faces
 * @returns a texture you destroy, of six layers
 */
export function cube(
  root: TgpuRoot,
  faces: readonly Picture[],
  options: Options = {},
): Texture<[number, number, 6]> {
  if (faces.length !== 6) throw new Error(`A cube map has 6 faces, got ${faces.length}.`);
  if (faces[0].width !== faces[0].height) {
    throw new Error(`A cube map's faces are square, got ${faces[0].width} × ${faces[0].height}.`);
  }
  return layered(root, faces, options, 'A cube map', (w, h): [number, number, 6] => [w, h, 6]);
}

export interface SamplerOptions {
  /** Says what happens past the edge: WebGPU's address mode on every axis. Default `'repeat'`. */
  readonly wrap?: GPUAddressMode;
  /** Filters within and between mip levels. Default `'linear'`. */
  readonly filter?: GPUFilterMode;
  /**
   * Sets the samples at grazing angles, 1 to 16. WebGPU allows more than 1 only with linear
   * filtering, so `'nearest'` uses 1. Default 1.
   */
  readonly anisotropy?: number;
}

/**
 * Creates a filtering sampler for textures from this module.
 *
 * @returns a sampler, which needs no destroy
 */
export function sampler(root: TgpuRoot, options: SamplerOptions = {}): TgpuSampler {
  const { wrap = 'repeat', filter = 'linear', anisotropy = 1 } = options;
  const samples = filter === 'linear' ? Math.min(Math.max(Math.round(anisotropy), 1), 16) : 1;
  return root
    .createSampler({
      addressModeU: wrap,
      addressModeV: wrap,
      addressModeW: wrap,
      magFilter: filter,
      minFilter: filter,
      mipmapFilter: filter,
      maxAnisotropy: samples,
    })
    .$name('imageSampler');
}
