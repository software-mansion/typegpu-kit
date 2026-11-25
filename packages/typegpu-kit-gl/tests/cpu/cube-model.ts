import { textureOps, encodeHalf, decodeHalf } from '../../src/cpu/textures.ts';
import type { CpuTexture } from '../../src/cpu/types.ts';

function faceOf([x, y, z]: number[]): [number, number, number] {
  const ax = Math.abs(x!),
    ay = Math.abs(y!),
    az = Math.abs(z!);
  if (ax >= ay && ax >= az)
    return x! >= 0
      ? [0, (-z! / ax + 1) / 2, (-y! / ax + 1) / 2]
      : [1, (z! / ax + 1) / 2, (-y! / ax + 1) / 2];
  if (ay >= az)
    return y! >= 0
      ? [2, (x! / ay + 1) / 2, (z! / ay + 1) / 2]
      : [3, (x! / ay + 1) / 2, (-z! / ay + 1) / 2];
  return z! >= 0
    ? [4, (x! / az + 1) / 2, (-y! / az + 1) / 2]
    : [5, (-x! / az + 1) / 2, (-y! / az + 1) / 2];
}
function directionOf(face: number, u: number, v: number): number[] {
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
/** A diagnostic model of a permitted hardware cube-corner policy, never used by dispatch. */
export function cubeSampleModel(
  texture: CpuTexture,
  sampler: GPUSamplerDescriptor,
  direction: number[],
  requested: number,
  quantized = true,
  truncateLod = false,
  weightOffsets: readonly [number, number] = [0, 0],
) {
  const [face, u, v] = faceOf(direction);
  const lod = Math.min(
    Math.max(requested, sampler.lodMinClamp ?? 0),
    sampler.lodMaxClamp ?? 32,
    texture.levels.length - 1,
  );
  const load = (mip: number, x: number, y: number, layer: number) => {
    const level = texture.levels[mip]!;
    return textureOps.textureLoad(
      { ...texture, dimension: '2d-array' },
      [Math.max(0, Math.min(level.width - 1, x)), Math.max(0, Math.min(level.height - 1, y))],
      layer,
      mip,
    ) as number[];
  };
  const sample = (mip: number) => {
    const level = texture.levels[mip]!,
      px = u * level.width - 0.5,
      py = v * level.height - 0.5,
      bx = Math.floor(px),
      by = Math.floor(py);
    const wx = quantized
        ? Math.round((px - bx) * 256) / 256
        : Math.max(0, Math.min(1, px - bx + weightOffsets[0])),
      wy = quantized
        ? Math.round((py - by) * 256) / 256
        : Math.max(0, Math.min(1, py - by + weightOffsets[1]));
    const result = [0, 0, 0, 0];
    for (let y = 0; y < 2; y++)
      for (let x = 0; x < 2; x++) {
        let tx = bx + x,
          ty = by + y,
          layer = face;
        const outsideX = tx < 0 || tx >= level.width,
          outsideY = ty < 0 || ty >= level.height;
        if (outsideX && outsideY) {
          const corner = faceOf(directionOf(face, tx < 0 ? 0 : 1, ty < 0 ? 0 : 1));
          layer = corner[0];
          tx = Math.floor(corner[1] * level.width);
          ty = Math.floor(corner[2] * level.height);
        } else if (outsideX !== outsideY) {
          const adjacent = faceOf(
            directionOf(face, (tx + 0.5) / level.width, (ty + 0.5) / level.height),
          );
          layer = adjacent[0];
          tx = Math.floor(adjacent[1] * level.width);
          ty = Math.floor(adjacent[2] * level.height);
        }
        const value = load(mip, tx, ty, layer),
          weight = (x ? wx : 1 - wx) * (y ? wy : 1 - wy);
        for (let c = 0; c < 4; c++) result[c]! += value[c]! * weight;
      }
    return result.map((value) => (quantized ? decodeHalf(encodeHalf(value)) : value));
  };
  const low = Math.floor(lod),
    high = Math.min(low + 1, texture.levels.length - 1),
    weight = quantized
      ? (truncateLod ? Math.floor : Math.round)((lod - low) * 256) / 256
      : lod - low;
  const a = sample(low),
    b = sample(high);
  return a.map((value, c) => {
    const result = value * (1 - weight) + b[c]! * weight;
    return quantized ? decodeHalf(encodeHalf(result)) : Math.fround(result);
  });
}

function expand(value: number, sign: number) {
  const bits = encodeHalf(value);
  if ((bits & 0x7fff) === 0) return sign * decodeHalf(1);
  return decodeHalf(bits + (value < 0 ? -sign : sign));
}

/** Bounds one 8-bit weight step and one adjacent half at each filtering stage. */
export function cubeSampleEnvelope(
  texture: CpuTexture,
  sampler: GPUSamplerDescriptor,
  direction: number[],
  requested: number,
) {
  const lod = Math.min(
    Math.max(requested, sampler.lodMinClamp ?? 0),
    sampler.lodMaxClamp ?? 32,
    texture.levels.length - 1,
  );
  const low = Math.floor(lod),
    high = Math.min(low + 1, texture.levels.length - 1);

  const levels = [low, high].map((mip) => {
    const corners = [-1, 1].flatMap((x) =>
      [-1, 1].map((y) =>
        cubeSampleModel(texture, sampler, direction, mip, false, false, [x / 256, y / 256]),
      ),
    );
    return Array.from({ length: 4 }, (_, c) => [
      expand(Math.min(...corners.map((v) => v[c]!)), -1),
      expand(Math.max(...corners.map((v) => v[c]!)), 1),
    ]);
  });
  const weights = [-1, 1].map((sign) => Math.max(0, Math.min(1, lod - low + sign / 256)));
  return Array.from({ length: 4 }, (_, c) => {
    const bottom = Math.min(
      ...weights.map((w) => levels[0]![c]![0]! * (1 - w) + levels[1]![c]![0]! * w),
    );
    const top = Math.max(
      ...weights.map((w) => levels[0]![c]![1]! * (1 - w) + levels[1]![c]![1]! * w),
    );
    return [expand(bottom, -1), expand(top, 1)];
  });
}
