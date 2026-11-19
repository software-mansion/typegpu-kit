import { mipmaps } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d } from 'typegpu';

/** The kit's combines the figure switches between, by their names in `mipmaps`. */
const COMBINES = { mean: mipmaps.mean, largest: mipmaps.largest } as const;

export type CombineName = keyof typeof COMBINES;

/** A level's texels, as rows. */
export type Level = readonly (readonly number[])[];

export interface Texel {
  readonly level: number;
  readonly x: number;
  readonly y: number;
}

/**
 * Level 0, depths that grow away from the eye: a wall at 9 behind a floor that nears from 7 to 4,
 * a box at 2 and a pole at 3, one texel wide.
 */
const FIRST: Level = [
  [9, 9, 9, 9, 9, 9, 9, 9],
  [9, 9, 9, 9, 9, 9, 3, 9],
  [9, 2, 2, 2, 2, 9, 3, 9],
  [9, 2, 2, 2, 2, 9, 3, 9],
  [7, 2, 2, 2, 2, 7, 3, 7],
  [6, 2, 2, 2, 2, 6, 3, 6],
  [5, 2, 2, 2, 2, 5, 3, 5],
  [4, 4, 4, 4, 4, 4, 4, 4],
];

/** The texels of the level before that `block` in mipmaps.ts loads for one texel, in its order. */
export function blockOf({ level, x, y }: Texel, above: Level): Texel[] {
  const lastX = above[0].length - 1;
  const lastY = above.length - 1;
  const near = { x: Math.min(x * 2, lastX), y: Math.min(y * 2, lastY) };
  const far = { x: Math.min(x * 2 + 1, lastX), y: Math.min(y * 2 + 1, lastY) };
  return [
    { level: level - 1, x: near.x, y: near.y },
    { level: level - 1, x: far.x, y: near.y },
    { level: level - 1, x: near.x, y: far.y },
    { level: level - 1, x: far.x, y: far.y },
  ];
}

/** The side of the level after one of `size`, as WebGPU halves a mip level. */
const side = (size: number) => Math.max(1, size >> 1);

/** The level after `above`, as `record` fills it: the kit's `combine` of each texel's block. */
function halve(above: Level, level: number, combine: mipmaps.Combine): Level {
  return Array.from({ length: side(above.length) }, (_row, y) =>
    Array.from({ length: side(above[0].length) }, (_texel, x) => {
      const [a, b, c, e] = blockOf({ level, x, y }, above).map((texel) =>
        d.vec4f(above[texel.y][texel.x]),
      );
      return combine(a, b, c, e).x;
    }),
  );
}

/**
 * A depth texture with a `mipmaps` for each combine. The recording device runs no kernel, so each
 * dispatch `record` records fills its level on the CPU, through the same combine.
 */
export function createProgram() {
  const { root, log } = recordingRoot();
  const texture = root
    .createTexture({ size: [FIRST[0].length, FIRST.length], format: 'r32float', mipLevelCount: 4 })
    .$usage('storage', 'sampled');
  const pyramids = {
    mean: mipmaps.create(root, texture, { combine: COMBINES.mean }),
    largest: mipmaps.create(root, texture, { combine: COMBINES.largest }),
  };

  return {
    /** Records the mipmaps of `combine`, and returns level 0 and each level a dispatch filled. */
    levels(combine: CombineName): Level[] {
      log.events.length = 0;
      const encoder = root.createCommandEncoder();
      pyramids[combine].record(encoder);
      encoder.submit();

      const levels = [FIRST];
      for (const event of log.events) {
        if (event === 'dispatch')
          levels.push(halve(levels.at(-1)!, levels.length, COMBINES[combine]));
      }
      return levels;
    },
  };
}
