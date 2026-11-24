import type { CpuTexture, CpuTextureLevel } from './types.ts';
import { vector } from './values.ts';

type Texels = Uint8Array | Int8Array | Uint16Array | Float32Array | Uint32Array | Int32Array;

export function decodeHalf(word: number): number {
  const sign = word & 0x8000 ? -1 : 1;
  const exponent = (word >>> 10) & 31;
  const fraction = word & 1023;
  return exponent === 31
    ? fraction
      ? NaN
      : sign * Infinity
    : sign * (exponent === 0 ? fraction * 2 ** -24 : (1 + fraction / 1024) * 2 ** (exponent - 15));
}

const round = (n: number) => {
  const floor = Math.floor(n);
  return n - floor === 0.5 ? floor + (floor % 2) : Math.round(n);
};

export function encodeHalf(value: number): number {
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const magnitude = Math.abs(value);
  if (Number.isNaN(magnitude)) return sign | 0x7e00;
  if (magnitude >= 65520) return sign | 0x7c00;
  if (magnitude === 0) return sign;

  if (magnitude < 2 ** -14) return sign | round(magnitude * 2 ** 24);
  const exponent = Math.floor(Math.log2(magnitude));
  const significand = round(magnitude * 2 ** (10 - exponent));
  return sign | ((exponent + 14) * 1024 + significand);
}

function formatOf(format: string) {
  const match = /^(r|rg|rgba|bgra)(8|16|32)(uint|sint|float|unorm|snorm)(-srgb)?$/.exec(format);
  if (!match) throw new Error(`@typegpu/kit-gl CPU: unsupported texture format ${format}`);
  return {
    channels: match[1]!.length === 4 ? 4 : match[1]!.length,
    bits: Number(match[2]),
    kind: match[3]!,
    bgra: match[1] === 'bgra',
    srgb: !!match[4],
  };
}

function texel(
  texture: CpuTexture,
  level: CpuTextureLevel,
  x: number,
  y: number,
  z: number,
): number[] {
  const format = formatOf(texture.format);
  if (
    x < 0 ||
    y < 0 ||
    z < 0 ||
    x >= level.width ||
    y >= level.height ||
    z >= level.depthOrArrayLayers
  )
    return vector([0, 0, 0, 0]);
  const offset = ((z * level.height + y) * level.width + x) * format.channels;
  const data = level.data as Texels;
  const result = [0, 0, 0, 1];
  for (let i = 0; i < format.channels; i++) {
    let value = data[offset + i]!;
    if (format.kind === 'float' && format.bits === 16) value = decodeHalf(value);
    if (format.kind === 'unorm') value /= 2 ** format.bits - 1;
    if (format.kind === 'snorm') value = Math.max(-1, value / (2 ** (format.bits - 1) - 1));
    if (format.srgb && i < 3)
      value = value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    result[format.bgra && i < 3 ? 2 - i : i] = value;
  }
  return vector(result);
}

function wrap(value: number, size: number, mode: GPUAddressMode = 'clamp-to-edge'): number {
  if (mode === 'repeat') return ((value % size) + size) % size;
  if (mode === 'mirror-repeat') {
    const n = ((value % (size * 2)) + size * 2) % (size * 2);
    return n < size ? n : size * 2 - 1 - n;
  }
  return Math.min(Math.max(value, 0), size - 1);
}

export const textureOps = {
  textureDimensions(texture: CpuTexture, lod = 0) {
    const level = texture.levels[lod]!;
    return vector(
      texture.dimension === '3d'
        ? [level.width, level.height, level.depthOrArrayLayers]
        : [level.width, level.height],
    );
  },
  textureNumLevels(texture: CpuTexture) {
    return texture.levels.length;
  },
  textureNumLayers(texture: CpuTexture) {
    return texture.levels[0]!.depthOrArrayLayers;
  },
  textureLoad(texture: CpuTexture, coords: number[], index = 0, lod = 0) {
    const array = texture.dimension === '2d-array';
    const level = texture.levels[array ? lod : index];
    if (!level) return vector([0, 0, 0, 0]);
    return texel(texture, level, coords[0]!, coords[1]!, array ? index : (coords[2] ?? 0));
  },
  textureStore(
    texture: CpuTexture,
    coords: number[],
    valueOrLayer: number[] | number,
    arrayValue?: number[],
  ) {
    const layer = typeof valueOrLayer === 'number' ? valueOrLayer : (coords[2] ?? 0);
    const value = typeof valueOrLayer === 'number' ? arrayValue! : valueOrLayer;
    const level = texture.levels[0]!;
    const x = coords[0]!,
      y = coords[1]!;
    if (
      x < 0 ||
      y < 0 ||
      layer < 0 ||
      x >= level.width ||
      y >= level.height ||
      layer >= level.depthOrArrayLayers
    )
      return;
    const format = formatOf(texture.format);
    const offset = ((layer * level.height + y) * level.width + x) * format.channels;
    const data = level.data as Texels;
    for (let i = 0; i < format.channels; i++) {
      let component = value[format.bgra && i < 3 ? 2 - i : i]!;
      if (format.srgb && i < 3)
        component =
          component <= 0.0031308 ? component * 12.92 : 1.055 * component ** (1 / 2.4) - 0.055;
      if (format.kind === 'float' && format.bits === 16) component = encodeHalf(component);
      if (format.kind === 'unorm')
        component = Math.round(Math.min(Math.max(component, 0), 1) * (2 ** format.bits - 1));
      if (format.kind === 'snorm')
        component = Math.round(Math.min(Math.max(component, -1), 1) * (2 ** (format.bits - 1) - 1));
      data[offset + i] = component;
    }
  },
  textureSampleLevel(
    texture: CpuTexture,
    sampler: GPUSamplerDescriptor,
    coords: number[],
    layerOrLod: number,
    arrayLod?: number,
  ) {
    if (!['2d', '2d-array', '3d'].includes(texture.dimension))
      throw new Error(`@typegpu/kit-gl CPU: unsupported sampled dimension ${texture.dimension}`);
    const array = texture.dimension === '2d-array';
    const layer = array ? Math.round(layerOrLod) : 0;
    const requested = array ? arrayLod! : layerOrLod;
    const lod = Math.min(
      Math.max(requested, sampler.lodMinClamp ?? 0),
      sampler.lodMaxClamp ?? 32,
      texture.levels.length - 1,
    );
    const sample = (index: number) => {
      const level = texture.levels[index]!;
      const linear = (lod <= 0 ? sampler.magFilter : sampler.minFilter) === 'linear';
      const p = [
        coords[0]! * level.width,
        coords[1]! * level.height,
        (coords[2] ?? 0) * level.depthOrArrayLayers,
      ];
      const at = (x: number, y: number, z: number) =>
        texel(
          texture,
          level,
          wrap(x, level.width, sampler.addressModeU),
          wrap(y, level.height, sampler.addressModeV),
          array
            ? layer
            : texture.dimension === '3d'
              ? wrap(z, level.depthOrArrayLayers, sampler.addressModeW)
              : 0,
        );
      if (!linear) return at(Math.floor(p[0]!), Math.floor(p[1]!), Math.floor(p[2]!));
      const base = p.map((v) => Math.floor(v - 0.5));
      const fraction = p.map((v, i) => v - 0.5 - base[i]!);
      const result = [0, 0, 0, 0];
      const depth = texture.dimension === '3d' ? 2 : 1;
      for (let z = 0; z < depth; z++)
        for (let y = 0; y < 2; y++)
          for (let x = 0; x < 2; x++) {
            const weight =
              (x ? fraction[0]! : 1 - fraction[0]!) *
              (y ? fraction[1]! : 1 - fraction[1]!) *
              (depth === 1 ? 1 : z ? fraction[2]! : 1 - fraction[2]!);
            const value = at(base[0]! + x, base[1]! + y, base[2]! + z);
            for (let c = 0; c < 4; c++) result[c]! += value[c]! * weight;
          }
      return vector(result);
    };
    if (sampler.mipmapFilter !== 'linear') return sample(Math.round(lod));
    const low = Math.floor(lod),
      high = Math.min(low + 1, texture.levels.length - 1),
      fraction = lod - low;
    const a = sample(low),
      b = sample(high);
    return vector(a.map((value, i) => value * (1 - fraction) + b[i]! * fraction));
  },
};
