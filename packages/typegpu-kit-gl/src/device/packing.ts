/** Conversions between readback values and a texture format's own texel layout. */

import type { FormatInfo } from './formats.ts';

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function toHalf(value: number): number {
  f32[0] = value;
  const bits = u32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  const e = exponent - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - e;
    const half = mantissa >>> shift;
    const round = (mantissa >>> (shift - 1)) & 1;
    return sign | (half + round);
  }
  const half = sign | (e << 10) | (mantissa >>> 13);
  return half + ((mantissa >>> 12) & 1);
}

export function fromHalf(half: number): number {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >>> 10) & 0x1f;
  const mantissa = half & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** An unsigned float with `mantissaBits` and a 5-bit exponent, as in rg11b10ufloat. */
function toSmallFloat(value: number, mantissaBits: number): number {
  if (!(value > 0)) return 0;
  const half = toHalf(value);
  const exponent = (half >>> 10) & 0x1f;
  const mantissa = half & 0x3ff;
  if (exponent === 0x1f) return (0x1f << mantissaBits) | (mantissa ? 1 : 0);
  return (exponent << mantissaBits) | (mantissa >>> (10 - mantissaBits));
}

function fromSmallFloat(bits: number, mantissaBits: number): number {
  const exponent = bits >>> mantissaBits;
  const mantissa = bits & ((1 << mantissaBits) - 1);
  return fromHalf((exponent << 10) | (mantissa << (10 - mantissaBits)));
}

export function packRg11b10(r: number, g: number, b: number): number {
  return (toSmallFloat(r, 6) | (toSmallFloat(g, 6) << 11) | (toSmallFloat(b, 5) << 22)) >>> 0;
}

export function unpackRg11b10(packed: number): [number, number, number] {
  return [
    fromSmallFloat(packed & 0x7ff, 6),
    fromSmallFloat((packed >>> 11) & 0x7ff, 6),
    fromSmallFloat((packed >>> 22) & 0x3ff, 5),
  ];
}

const clampRgb9e5 = (v: number) => Math.min(Math.max(v || 0, 0), 65408);
const tenBits = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 1023);

export function packRgb9e5(r: number, g: number, b: number): number {
  const clamp = clampRgb9e5;
  const [cr, cg, cb] = [clamp(r), clamp(g), clamp(b)];
  const max = Math.max(cr, cg, cb);
  let exponent = Math.max(-16, Math.floor(Math.log2(max || 2 ** -16))) + 16;
  let scale = 2 ** (exponent - 15 - 9);
  if (Math.round(max / scale) === 512) {
    exponent += 1;
    scale *= 2;
  }
  const pack = (v: number) => Math.round(v / scale) & 0x1ff;
  return (pack(cr) | (pack(cg) << 9) | (pack(cb) << 18) | (exponent << 27)) >>> 0;
}

/** Packs RGBA unorm values from 0 to 1 into rgb10a2unorm's word. */
export function packRgb10a2(r: number, g: number, b: number, a: number): number {
  const ten = tenBits;
  const two = Math.round(Math.min(Math.max(a, 0), 1) * 3);
  return (ten(r) | (ten(g) << 10) | (ten(b) << 20) | (two << 30)) >>> 0;
}

export type TexelArray =
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array
  | Uint32Array
  | Int32Array
  | Float32Array;

export function texelArray(info: FormatInfo, count: number): TexelArray {
  const length = count * info.components;
  switch (info.array) {
    case 'u8':
      return new Uint8Array(length);
    case 'i8':
      return new Int8Array(length);
    case 'u16':
    case 'f16':
      return new Uint16Array(length);
    case 'i16':
      return new Int16Array(length);
    case 'u32':
      return new Uint32Array(length);
    case 'i32':
      return new Int32Array(length);
    case 'f32':
      return new Float32Array(length);
  }
}

/** Views bytes as the typed array the format's GL upload type takes. */
export function viewAs(info: FormatInfo, bytes: Uint8Array): TexelArray {
  const { buffer, byteOffset, byteLength } = bytes;
  const make = <T>(Ctor: new (b: ArrayBufferLike, o: number, l: number) => T, size: number): T => {
    if (byteOffset % size === 0) return new Ctor(buffer, byteOffset, Math.floor(byteLength / size));
    const copy = bytes.slice();
    return new Ctor(copy.buffer, 0, Math.floor(byteLength / size));
  };
  switch (info.array) {
    case 'u8':
      return bytes;
    case 'i8':
      return make(Int8Array, 1);
    case 'u16':
    case 'f16':
      return make(Uint16Array, 2);
    case 'i16':
      return make(Int16Array, 2);
    case 'u32':
      return make(Uint32Array, 4);
    case 'i32':
      return make(Int32Array, 4);
    case 'f32':
      return make(Float32Array, 4);
  }
}

/**
 * Converts RGBA texels read back from GL into the format's own layout: `rgba` holds four values
 * a texel, as `UNSIGNED_BYTE`, `FLOAT`, `UNSIGNED_INT` or `INT` reads them.
 */
export function packTexels(
  info: FormatInfo,
  rgba: Uint8Array | Float32Array | Uint32Array | Int32Array,
  out: TexelArray,
  stride = 4,
): void {
  const count = rgba.length / stride;
  const n = info.components;
  const { format } = info;
  if (format === 'rg11b10ufloat') {
    for (let i = 0; i < count; i++)
      out[i] = packRg11b10(rgba[i * 4]!, rgba[i * 4 + 1]!, rgba[i * 4 + 2]!);
    return;
  }
  if (format === 'rgb9e5ufloat') {
    for (let i = 0; i < count; i++)
      out[i] = packRgb9e5(rgba[i * 4]!, rgba[i * 4 + 1]!, rgba[i * 4 + 2]!);
    return;
  }
  if (format === 'rgb10a2uint') {
    for (let i = 0; i < count; i++) {
      const [r, g, b, a] = [rgba[i * 4]!, rgba[i * 4 + 1]!, rgba[i * 4 + 2]!, rgba[i * 4 + 3]!];
      out[i] = (r | (g << 10) | (b << 20) | (a << 30)) >>> 0;
    }
    return;
  }
  if (format === 'rgb10a2unorm' && rgba instanceof Float32Array) {
    for (let i = 0; i < count; i++)
      out[i] = packRgb10a2(rgba[i * 4]!, rgba[i * 4 + 1]!, rgba[i * 4 + 2]!, rgba[i * 4 + 3]!);
    return;
  }
  const convert =
    info.array === 'f16'
      ? toHalf
      : info.kind === 'snorm'
        ? (v: number) => Math.round(Math.min(Math.max(v, -1), 1) * 127)
        : format === 'depth16unorm'
          ? (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 65535)
          : undefined;
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < n; c++) {
      const source = info.bgra && c < 3 ? 2 - c : c;
      const value = rgba[i * stride + source]!;
      out[i * n + c] = convert ? convert(value) : value;
    }
  }
}

/** Swaps red and blue of 4-byte texels, for bgra8 formats stored as RGBA. */
export function swapRedBlue(bytes: Uint8Array): Uint8Array {
  const out = bytes.slice();
  for (let i = 0; i + 3 < out.length; i += 4) {
    out[i] = bytes[i + 2]!;
    out[i + 2] = bytes[i]!;
  }
  return out;
}
