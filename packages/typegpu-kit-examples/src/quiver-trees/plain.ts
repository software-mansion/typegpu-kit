const TAU = Math.PI * 2;

/**
 * Poly Haven CC0 scans `quiver_tree_01`, `quiver_tree_02`, `namaqualand_boulder_02` and
 * `namaqualand_boulder_05`. These files have about one third of the source triangles and
 * 512-texel textures.
 */
export const MODELS = ['quiver-tree-01', 'quiver-tree-02', 'boulder-02', 'boulder-05'] as const;
export type ModelName = (typeof MODELS)[number];

/** Plain and initial clearing radii in metres. */
export const RADIUS = 260;
export const CLEARING = 16;

/** Model frequencies and minimum and maximum scales. */
const MIX: readonly { share: number; scale: readonly [number, number] }[] = [
  { share: 0.35, scale: [1.6, 3.2] },
  { share: 0.25, scale: [1.8, 3.6] },
  { share: 0.2, scale: [0.8, 2.4] },
  { share: 0.2, scale: [1.2, 3.5] },
];
/** Minimum and maximum brightness multipliers relative to the model. */
const SHADES = [0.8, 1.2] as const;

export interface Planted {
  readonly model: number;
  readonly position: [number, number, number];
  readonly rotation: [number, number, number, number];
  readonly scale: number;
  /** Brightness multiplier relative to the model. */
  readonly shade: number;
}

const between = ([low, high]: readonly [number, number]) => low + Math.random() * (high - low);

/** Scatters trees and boulders between the clearing and plain edge with random rotations. */
export function plant(count: number): Planted[] {
  return Array.from({ length: count }, () => {
    const out = Math.sqrt(between([CLEARING ** 2, RADIUS ** 2]));
    const angle = Math.random() * TAU;
    const yaw = Math.random() * TAU;

    let pick = Math.random();
    let model = 0;
    while (model < MIX.length - 1 && pick >= MIX[model].share) pick -= MIX[model++].share;

    return {
      model,
      position: [Math.cos(angle) * out, 0, Math.sin(angle) * out],
      rotation: [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)],
      scale: between(MIX[model].scale),
      shade: between(SHADES),
    };
  });
}
