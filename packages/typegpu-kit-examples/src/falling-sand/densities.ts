import { d, std, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import {
  type Cells,
  GRID,
  gridKernel,
  gridTexture,
  GROUPS,
  HEIGHT,
  Kind,
  lifeOf,
  Matter,
  matterOf,
  WIDTH,
} from './cells.ts';

/** Vapour sampling radius and blur width in cells. */
const VAPOUR = { reach: 4, spread: 2.8 };
/** Density multiplier from the last lifetime step to the first. */
const YOUTH = { old: 0.35, young: 1 };
/** Vertical and horizontal sampling reach for filling gaps in falling streams, in cells. */
const STREAM = { rows: 6, columns: 1 };

/**
 * Blurs steam and smoke density and fills gaps in falling water, lava and sand. Makes discrete
 * falling cells appear as continuous streams.
 */
export function createDensities(root: TgpuRoot, cells: Cells) {
  const vapour = gridTexture(root);
  const streams = gridTexture(root);
  const smooth = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });

  const measure = gridKernel(function measure({ gid }) {
    'use gpu';
    if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
    const at = d.vec2i(gid.xy);

    let mist = d.vec2f();
    let weights = d.f32(0);
    for (let y = -VAPOUR.reach; y <= VAPOUR.reach; y++) {
      for (let x = -VAPOUR.reach; x <= VAPOUR.reach; x++) {
        const offset = d.vec2f(d.f32(x), d.f32(y));
        const weight = std.exp(-std.dot(offset, offset) / (VAPOUR.spread * VAPOUR.spread));
        const cell = cells.cellAt(at + d.vec2i(x, y));
        const matter = matterOf(cell);
        const youth = std.mix(YOUTH.old, YOUTH.young, d.f32(lifeOf(cell)) / 255);
        if (matter === Matter.steam) mist.x += weight * youth;
        if (matter === Matter.smoke) mist.y += weight * youth;
        weights += weight;
      }
    }

    let above = d.vec3f();
    let below = d.vec3f();
    for (let y = -STREAM.rows; y <= STREAM.rows; y++) {
      for (let x = -STREAM.columns; x <= STREAM.columns; x++) {
        const p = at + d.vec2i(x, y);
        const cell = cells.cellAt(p);
        const matter = matterOf(cell);
        if (matter !== Matter.water && matter !== Matter.lava && matter !== Matter.sand) continue;
        // Only cells with gas below count as falling
        if (cells.kindAt(p + d.vec2i(0, 1)) !== Kind.gas) continue;
        const falling = d.vec3f(
          d.f32(matter === Matter.water),
          d.f32(matter === Matter.lava),
          d.f32(matter === Matter.sand),
        );
        if (y <= 0) above += falling;
        if (y >= 0) below += falling;
      }
    }

    // Fill between falling cells without extending beyond a stream's ends
    const falls = std.min(above, below);
    const counted = (STREAM.rows + 1) * (STREAM.columns * 2 + 1);
    std.textureStore(vapour.writing.$, gid.xy, d.vec4f(mist / weights, 0, 1));
    std.textureStore(streams.writing.$, gid.xy, d.vec4f(falls / counted, 1));
  });
  const measuring = root.createComputePipeline({ compute: measure });

  return {
    /** Interpolated steam density in `x` and smoke density in `y` at cell coordinates. */
    vapourAt: (p: d.v2f) => {
      'use gpu';
      return std.textureSampleLevel(vapour.reading.$, smooth.$, p / GRID, 0).xy;
    },
    /** Water, lava and sand stream densities from 0 to 1, nonzero only between falling cells. */
    streamsAt: (cell: d.v2i) => {
      'use gpu';
      return std.textureLoad(streams.reading.$, cell, 0).xyz;
    },
    initAsync: () => measuring.initAsync(),
    measure(pass: TgpuComputePass) {
      pass.setPipeline(measuring);
      pass.dispatchWorkgroups(GROUPS.x, GROUPS.y);
    },
    destroy() {
      vapour.destroy();
      streams.destroy();
    },
  };
}

export type Densities = ReturnType<typeof createDensities>;
