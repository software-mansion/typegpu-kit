import { expect, it } from 'vitest';
import { textureOps } from '../../src/cpu/textures.ts';
import type { CpuTexture } from '../../src/cpu/types.ts';

it('returns a scalar zero for an out-of-range depth mip load', () => {
  const texture: CpuTexture = {
    dimension: '2d',
    format: 'depth32float',
    levels: [{ width: 1, height: 1, depthOrArrayLayers: 1, data: new Float32Array([0.5]) }],
  };
  expect(textureOps.textureLoad(texture, [0, 0], 0)).toBe(0.5);
  for (const mip of [-1, 1, 8]) {
    const result = textureOps.textureLoad(texture, [0, 0], mip);
    expect(result).toBe(0);
    expect(Number(result) + 1).toBe(1);
  }
  const color = { ...texture, format: 'r32float' as const };
  expect(textureOps.textureLoad(color, [0, 0], 8)).toEqual([0, 0, 0, 0]);
});
