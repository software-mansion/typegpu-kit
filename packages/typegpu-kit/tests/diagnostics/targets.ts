import type { SampledFlag, TgpuRoot } from 'typegpu';
import { targets } from '../../src/index.ts';

declare const root: TgpuRoot;
const size = { width: 1, height: 1 };

export const misspelt = targets.create(root, size, {
  hdr: { format: 'rgba16float', sampled: true },
  depth: { format: 'depth24plus', smapled: true },
});

export const mistyped = targets.create(root, size, {
  hdr: { format: 'rgba16float', sampled: true },
  bloom: { format: 'rgba16float', mips: 'three' },
});

// The other targets keep their formats and flags
export const kept: SampledFlag[] = [misspelt.textures.hdr, mistyped.textures.hdr];
