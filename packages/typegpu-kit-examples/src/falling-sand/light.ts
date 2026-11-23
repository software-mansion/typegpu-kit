import { pipelines } from '@typegpu/kit';
import { perlin2d } from '@typegpu/noise';
import * as hrc from '@typegpu/radiance-cascades/holographic';
import {
  d,
  std,
  tgpu,
  type SampledFlag,
  type TgpuComputePass,
  type TgpuRoot,
  type TgpuTexture,
  type TgpuUniform,
} from 'typegpu';
import type { Densities } from './densities.ts';
import {
  type Cells,
  FIRE_LIFE,
  GRID,
  gridKernel,
  gridTexture,
  GROUPS,
  HEIGHT,
  Kind,
  lifeOf,
  Matter,
  matterOf,
  shadeOf,
  WIDTH,
} from './cells.ts';

const TRACED_LEVELS = 3;
const SECTOR_SHARE = 1 / 4;
/** How much of an ember's drawn glow lights the cave. */
const EMBER_LIGHT = 1;

/** Extinction multiplier for grains falling through air. */
const FALLING_GRAINS = 0.15;
const FLICKER = { scale: 0.3, perSecond: 7 };
/** Hot, crust-free emission from falling lava. */
const FALLING_LAVA = d.vec3f(2.6, 1, 0.22);
/** Lava emission multiplier for the light field. Keeps lava the main light source. */
const LAVA_LIGHT = 5;
/** Falling-lava density threshold and extinction. */
const RIBBON = { least: 0.04, full: 0.3, extinction: 1.2 };

/** Crystal brightness multiplier when buried. */
const BURIED = 0.12;
/** Sample offsets three and six cells away to detect surrounding grains and air. */
const AROUND = [3, 6].flatMap((reach) =>
  [-1, 0, 1].flatMap((y) =>
    [-1, 0, 1].filter((x) => x !== 0 || y !== 0).map((x) => ({ x: x * reach, y: y * reach })),
  ),
);
/** Crystal pulse rates and colours. High crystals are violet, and lower crystals are cyan. */
const CRYSTAL = {
  violet: d.vec3f(1.2, 0.36, 2.3),
  violetAbove: 45,
  pulse: 0.6,
  depth: 0.15,
};
/** Crystal emission multiplier for the field. Keeps lava the main light source. */
const CRYSTAL_LIGHT = 0.7;

/**
 * Cell emission. Fire fades with age, burning wood flickers, and crystal colour depends on
 * height.
 */
export const glowOf = (base: d.v3f, cell: number, p: d.v2f, seconds: number) => {
  'use gpu';
  const matter = matterOf(cell);
  if (matter === Matter.fire) {
    const young = d.f32(lifeOf(cell)) / FIRE_LIFE;
    return base * (0.4 + young) + d.vec3f(1, 1, 0.5) * young * young;
  }
  if (matter === Matter.burning) {
    const flicker = perlin2d.sample(p * FLICKER.scale + d.vec2f(seconds * FLICKER.perSecond, 0));
    return base * (0.5 + flicker * 0.6 + shadeOf(cell) * 0.3);
  }
  if (matter === Matter.crystal) {
    let hue = d.vec3f(base);
    if (p.y < CRYSTAL.violetAbove) hue = d.vec3f(CRYSTAL.violet);
    const pulse = perlin2d.sample(d.vec2f(p.x * 0.02, seconds * CRYSTAL.pulse)) * CRYSTAL.depth;
    return hue * (0.85 + shadeOf(cell) * 0.3 + pulse);
  }
  return d.vec3f(base);
};

/**
 * Builds a holographic radiance field from cells, embers and reflected light from walls and
 * steam in the last field update.
 */
export function createLight(
  root: TgpuRoot,
  options: {
    readonly cells: Cells;
    readonly densities: Densities;
    /** Ember drawing at the light-field resolution. */
    readonly embers: TgpuTexture & SampledFlag;
    readonly seconds: TgpuUniform<d.F32>;
    readonly bounce: boolean;
  },
) {
  const { cells, densities, seconds } = options;
  const glow = gridTexture(root);
  const thickness = gridTexture(root);
  const emberRead = options.embers.createView(d.texture2d(d.f32));
  const smooth = root.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const bounce = root.createUniform(d.u32, options.bounce ? 1 : 0);
  const field = hrc.create({
    root,
    size: { width: WIDTH, height: HEIGHT },
    tracedLevels: TRACED_LEVELS,
    transmission: 'rgb',
    medium: (pixel) => {
      'use gpu';
      return {
        emission: std.textureLoad(glow.reading.$, pixel, 0).rgb,
        extinction: std.textureLoad(thickness.reading.$, pixel, 0).rgb,
      };
    },
  });
  const radiance = field.output.createView(d.texture2d(d.f32));
  const sectors = field.sectors.createView(d.texture2dArray(d.f32));

  /** Mean light at a cell. */
  const lightAt = (cell: d.v2i) => {
    'use gpu';
    return std.textureLoad(radiance.$, cell, 0).rgb;
  };
  /** Interpolated mean light at cell coordinates. */
  const radianceAt = (p: d.v2f) => {
    'use gpu';
    return std.textureSampleLevel(radiance.$, smooth.$, p / GRID, 0).rgb;
  };
  /** Samples incoming light `reach` cells outward along a solid face's `normal`. */
  const faceAt = (cell: d.v2i, normal: d.v2f, reach: number) => {
    'use gpu';
    const outside = (d.vec2f(cell) + 0.5 + normal * reach) / GRID;
    const weights = d.vec4f(
      std.max(normal.x, 0),
      std.max(normal.y, 0),
      std.max(-normal.x, 0),
      std.max(-normal.y, 0),
    );
    let light = d.vec3f();
    for (const sector of tgpu.unroll([0, 1, 2, 3])) {
      light +=
        std.textureSampleLevel(sectors.$, smooth.$, outside, sector, 0).rgb * weights[sector];
    }
    // Each sector contributes a quarter of the mean. Divide by that fraction to recover side
    // lighting
    return light / (SECTOR_SHARE * std.max(std.abs(normal.x) + std.abs(normal.y), 1));
  };
  /** Crystal brightness multiplier, from 1 in open air to `BURIED` among grains. */
  const shineAt = (cell: d.v2i) => {
    'use gpu';
    let open = d.f32(0);
    let buried = d.f32(0);
    for (const offset of tgpu.unroll(AROUND)) {
      const kind = cells.kindAt(cell + d.vec2i(offset.x, offset.y));
      if (kind === Kind.powder) buried += 1;
      if (kind === Kind.gas || kind === Kind.liquid) open += 1;
    }
    const covered = std.smoothstep(0.15, 0.75, buried / std.max(open + buried, 1));
    return std.mix(1, BURIED, covered);
  };
  /** Medium properties at a cell for the diagnostic view. */
  const mediumAt = (cell: d.v2i) => {
    'use gpu';
    return d.vec4f(
      std.textureLoad(glow.reading.$, cell, 0).rgb,
      std.textureLoad(thickness.reading.$, cell, 0).x,
    );
  };

  /** Whether air lies left or right of a cell. */
  const airBeside = (at: d.v2i) => {
    'use gpu';
    return (
      cells.matterAt(at + d.vec2i(-1, 0)) === Matter.empty ||
      cells.matterAt(at + d.vec2i(1, 0)) === Matter.empty
    );
  };

  /** What a cell gives off into the field. Falling lava burns hotter, and buried crystal dims. */
  const givenAt = (cell: number, at: d.v2i, p: d.v2f) => {
    'use gpu';
    const base = glowOf(cells.traitsOf(cell).glow, cell, p, seconds.$);
    const matter = matterOf(cell);
    if (matter === Matter.crystal) return base * CRYSTAL_LIGHT * shineAt(at);
    if (matter !== Matter.lava) return base;
    if (airBeside(at)) return FALLING_LAVA * LAVA_LIGHT;
    return base * LAVA_LIGHT;
  };

  const fill = gridKernel(function fill({ gid }) {
    'use gpu';
    if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
    const at = d.vec2i(gid.xy);
    const p = d.vec2f(gid.xy) + 0.5;
    const cell = cells.cellAt(at);
    const traits = cells.traitsOf(cell);
    let thick = d.vec3f(traits.extinction);
    let given = givenAt(cell, at, p);

    // Grains beside or above air pass more light, so falling streams stay lit
    const below = cells.matterAt(at + d.vec2i(0, 1));
    if (traits.kind === Kind.powder && (below === Matter.empty || airBeside(at))) {
      thick *= FALLING_GRAINS;
    }

    // Emit along the full falling-lava ribbon to avoid disconnected points of light
    const falls = densities.streamsAt(at).y;
    if (traits.kind === Kind.gas && falls > RIBBON.least) {
      const full = std.smoothstep(RIBBON.least, RIBBON.full, falls);
      thick = std.max(thick, d.vec3f(RIBBON.extinction * full));
      given = FALLING_LAVA * LAVA_LIGHT * full;
    }

    // Scale emission and extinction together to preserve their ratio
    let emission = given * thick;
    if (bounce.$ !== 0) {
      const matter = matterOf(cell);
      const relief = cells.reliefAt(at);
      if (relief.z === 1 && std.dot(traits.glow, traits.glow) === 0) {
        emission += traits.albedo * faceAt(at, relief.xy, d.f32(1)) * thick;
      } else if (matter === Matter.steam || matter === Matter.smoke) {
        emission += traits.albedo * radianceAt(p) * thick;
      }
    }
    emission += std.textureLoad(emberRead.$, gid.xy, 0).rgb * EMBER_LIGHT;
    std.textureStore(glow.writing.$, gid.xy, d.vec4f(emission, 1));
    std.textureStore(thickness.writing.$, gid.xy, d.vec4f(thick, 1));
  });
  const filling = root.createComputePipeline({ compute: fill });

  return {
    bounce,
    lightAt,
    radianceAt,
    shineAt,
    faceAt,
    mediumAt,
    initAsync: () => pipelines.compile([field, filling]),
    solve(pass: TgpuComputePass) {
      pass.setPipeline(filling);
      pass.dispatchWorkgroups(GROUPS.x, GROUPS.y);
      field.run({ pass });
    },
    destroy() {
      glow.destroy();
      thickness.destroy();
      field.destroy();
    },
  };
}

export type Light = ReturnType<typeof createLight>;
