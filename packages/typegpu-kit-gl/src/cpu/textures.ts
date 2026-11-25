import type { CpuTexture, CpuTextureLevel } from './types.ts';
import { vector } from './values.ts';

type Texels = Uint8Array | Int8Array | Uint16Array | Float32Array | Uint32Array | Int32Array;

function decodeHalfBits(word: number): number {
  const sign = word & 0x8000 ? -1 : 1;
  const exponent = (word >>> 10) & 31;
  const fraction = word & 1023;
  return exponent === 31
    ? fraction
      ? NaN
      : sign * Infinity
    : sign * (exponent === 0 ? fraction * 2 ** -24 : (1 + fraction / 1024) * 2 ** (exponent - 15));
}

const halfValues = Float32Array.from({ length: 65536 }, (_, word) => decodeHalfBits(word));
export function decodeHalf(word: number): number {
  return halfValues[word & 65535]!;
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

interface Format {
  channels: number;
  bits: number;
  kind: string;
  bgra: boolean;
  srgb: boolean;
}
const formats = new Map<string, Format>();
function formatOf(format: string): Format {
  const known = formats.get(format);
  if (known) return known;
  if (format.startsWith('depth'))
    return { channels: 1, bits: 32, kind: 'float', bgra: false, srgb: false };
  const match = /^(r|rg|rgba|bgra)(8|16|32)(uint|sint|float|unorm|snorm)(-srgb)?$/.exec(format);
  if (!match) throw new Error(`@typegpu/kit-gl CPU: unsupported texture format ${format}`);
  const value = {
    channels: match[1]!.length === 4 ? 4 : match[1]!.length,
    bits: Number(match[2]),
    kind: match[3]!,
    bgra: match[1] === 'bgra',
    srgb: !!match[4],
  };
  formats.set(format, value);
  return value;
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
    result[format.bgra && i < 3 ? 2 - i : i] =
      format.kind === 'uint' || format.kind === 'sint' ? value : Math.fround(value);
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

function cubeFace(direction: number[]): [number, number, number] {
  const [x, y, z] = direction as [number, number, number],
    ax = Math.abs(x),
    ay = Math.abs(y),
    az = Math.abs(z);
  if (ax >= ay && ax >= az)
    return x >= 0
      ? [0, (-z / ax + 1) / 2, (-y / ax + 1) / 2]
      : [1, (z / ax + 1) / 2, (-y / ax + 1) / 2];
  if (ay >= az)
    return y >= 0
      ? [2, (x / ay + 1) / 2, (z / ay + 1) / 2]
      : [3, (x / ay + 1) / 2, (-z / ay + 1) / 2];
  return z >= 0
    ? [4, (x / az + 1) / 2, (-y / az + 1) / 2]
    : [5, (-x / az + 1) / 2, (-y / az + 1) / 2];
}

function cubeDirection(face: number, u: number, v: number): number[] {
  const a = u * 2 - 1,
    b = v * 2 - 1;
  return [
    [1, -b, -a],
    [-1, -b, a],
    [a, 1, b],
    [a, -1, -b],
    [a, -b, 1],
    [-a, -b, -1],
  ][face]!;
}

function sampleDepth(
  texture: CpuTexture,
  sampler: GPUSamplerDescriptor,
  coords: number[],
  layer: number,
  requested: number,
  compare?: number,
): number {
  const lod = Math.min(
    Math.max(requested, sampler.lodMinClamp ?? 0),
    sampler.lodMaxClamp ?? 32,
    texture.levels.length - 1,
  );
  const sample = (index: number) => {
    const level = texture.levels[index]!;
    const at = (x: number, y: number) => {
      const value =
        (level.data as Float32Array)[
          (layer * level.height + wrap(y, level.height, sampler.addressModeV)) * level.width +
            wrap(x, level.width, sampler.addressModeU)
        ] ?? 0;
      if (compare === undefined) return value;
      switch (sampler.compare) {
        case 'always':
          return 1;
        case 'never':
          return 0;
        case 'less':
          return Number(compare < value);
        case 'less-equal':
          return Number(compare <= value);
        case 'greater':
          return Number(compare > value);
        case 'greater-equal':
          return Number(compare >= value);
        case 'equal':
          return Number(compare === value);
        default:
          return Number(compare !== value);
      }
    };
    const x = coords[0]! * level.width,
      y = coords[1]! * level.height;
    if ((lod <= 0 ? sampler.magFilter : sampler.minFilter) !== 'linear')
      return at(Math.floor(x), Math.floor(y));
    const bx = Math.floor(x - 0.5),
      by = Math.floor(y - 0.5),
      fx = x - 0.5 - bx,
      fy = y - 0.5 - by;
    return (
      (at(bx, by) * (1 - fx) + at(bx + 1, by) * fx) * (1 - fy) +
      (at(bx, by + 1) * (1 - fx) + at(bx + 1, by + 1) * fx) * fy
    );
  };
  if (sampler.mipmapFilter !== 'linear') return sample(Math.round(lod));
  const low = Math.floor(lod),
    fraction = lod - low;
  return (
    sample(low) * (1 - fraction) + sample(Math.min(low + 1, texture.levels.length - 1)) * fraction
  );
}

export const textureOps = {
  textureSampleCompareLevel(
    texture: CpuTexture,
    sampler: GPUSamplerDescriptor,
    coords: number[],
    layerOrReference: number,
    reference?: number,
  ): number {
    const array = texture.dimension === '2d-array' || texture.dimension === 'cube-array';
    return textureOps.textureSampleLevel(
      texture,
      sampler,
      coords,
      array ? layerOrReference : 0,
      array ? 0 : undefined,
      array ? reference : layerOrReference,
    ) as number;
  },
  textureDimensions(texture: CpuTexture, lod = 0) {
    const level = texture.levels[lod]!;
    if (texture.dimension === '1d') return level.width;
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
  textureLoad(texture: CpuTexture, coords: number[] | number, index = 0, lod = 0) {
    if (typeof coords === 'number') coords = [coords, 0];
    const array = texture.dimension === '2d-array';
    const level = texture.levels[array ? lod : index];
    if (!level) return texture.format.startsWith('depth') ? 0 : vector([0, 0, 0, 0]);
    const result = texel(texture, level, coords[0]!, coords[1]!, array ? index : (coords[2] ?? 0));
    return texture.format.startsWith('depth') ? result[0]! : result;
  },
  textureStore(
    texture: CpuTexture,
    coords: number[] | number,
    valueOrLayer: number[] | number,
    arrayValue?: number[],
  ) {
    if (typeof coords === 'number') coords = [coords, 0];
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
    coords: number[] | number,
    layerOrLod: number,
    arrayLod?: number,
    compare?: number,
  ): number[] | number {
    if (typeof coords === 'number') coords = [coords, 0.5];
    if (!['1d', '2d', '2d-array', '3d', 'cube', 'cube-array'].includes(texture.dimension))
      throw new Error(`@typegpu/kit-gl CPU: unsupported sampled dimension ${texture.dimension}`);
    if (
      texture.format.startsWith('depth') &&
      (texture.dimension === '2d' || texture.dimension === '2d-array')
    )
      return Math.fround(
        sampleDepth(
          texture,
          sampler,
          coords,
          texture.dimension === '2d-array' ? Math.round(layerOrLod) : 0,
          texture.dimension === '2d-array' ? arrayLod! : layerOrLod,
          compare,
        ),
      );
    const cube = texture.dimension.startsWith('cube');
    const array = texture.dimension === '2d-array' || texture.dimension === 'cube-array';
    const face = cube ? cubeFace(coords) : undefined;
    const layer = (array ? Math.round(layerOrLod) : 0) * (cube ? 6 : 1) + (face?.[0] ?? 0);
    if (face) coords = [face[1], face[2]];
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
      const at = (x: number, y: number, z: number) => {
        let texelLayer =
          array || cube
            ? layer
            : texture.dimension === '3d'
              ? wrap(z, level.depthOrArrayLayers, sampler.addressModeW)
              : 0;
        if (cube && (x < 0 || y < 0 || x >= level.width || y >= level.height)) {
          const next = cubeFace(
            cubeDirection(face![0], (x + 0.5) / level.width, (y + 0.5) / level.height),
          );
          texelLayer = layer - face![0] + next[0];
          x = Math.floor(next[1] * level.width);
          y = Math.floor(next[2] * level.height);
        }
        const result = texel(
          texture,
          level,
          wrap(x, level.width, cube ? 'clamp-to-edge' : sampler.addressModeU),
          wrap(y, level.height, cube ? 'clamp-to-edge' : sampler.addressModeV),
          texelLayer,
        );
        if (compare !== undefined) {
          const value = result[0]!;
          const success =
            sampler.compare === 'always'
              ? true
              : sampler.compare === 'never'
                ? false
                : sampler.compare === 'less'
                  ? compare < value
                  : sampler.compare === 'less-equal'
                    ? compare <= value
                    : sampler.compare === 'greater'
                      ? compare > value
                      : sampler.compare === 'greater-equal'
                        ? compare >= value
                        : sampler.compare === 'equal'
                          ? compare === value
                          : compare !== value;
          return [Number(success), 0, 0, 1];
        }
        return result;
      };
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
    const finish = (value: number[]) => {
      for (let i = 0; i < value.length; i++) value[i] = Math.fround(value[i]!);
      return texture.format.startsWith('depth') ? value[0]! : value;
    };
    if (sampler.mipmapFilter !== 'linear') return finish(sample(Math.round(lod)));
    const low = Math.floor(lod),
      high = Math.min(low + 1, texture.levels.length - 1),
      fraction = lod - low;
    const a = sample(low),
      b = sample(high);
    return finish(vector(a.map((value, i) => value * (1 - fraction) + b[i]! * fraction)));
  },
};
