import { backends } from '@typegpu/kit';
import { byStage, type Inject } from '@typegpu/kit/inject';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';

export const WIDTH = 320;
export const HEIGHT = 180;
const CELLS = WIDTH * HEIGHT;
/** The grid's size in cells, which divides cell coordinates into texture coordinates. */
export const GRID = d.vec2f(WIDTH, HEIGHT);
export const WORKGROUP = 8;
/** Workgroups that cover the grid, one thread a cell. */
export const GROUPS = { x: Math.ceil(WIDTH / WORKGROUP), y: Math.ceil(HEIGHT / WORKGROUP) };

/**
 * A compute entry point over a two-dimensional grid of threads, `WORKGROUP` square. Give it a
 * named function, as in `gridKernel(function step({ gid }) {…})`, to name the entry point in WGSL.
 */
export const gridKernel = tgpu.computeFn({
  workgroupSize: [WORKGROUP, WORKGROUP],
  in: { gid: d.builtin.globalInvocationId },
});

/** What a cell holds, in its low byte. */
export const Matter = {
  empty: 0,
  rock: 1,
  sand: 2,
  water: 3,
  lava: 4,
  obsidian: 5,
  wood: 6,
  burning: 7,
  fire: 8,
  smoke: 9,
  steam: 10,
  crystal: 11,
  ash: 12,
  spring: 13,
  drain: 14,
};

/** Movement categories for fixed solids, powders, liquids and gases. */
export const Kind = { fixed: 0, powder: 1, liquid: 2, gas: 3 };

/** Per-cell extinction threshold for treating a cell as solid. */
export const OPAQUE = 4;
/** Search depth for nearby open cells, in cells. */
export const RELIEF_CELLS = 5;

export const Material = d.struct({
  /** Linear colour where light falls on it. */
  albedo: d.vec3f,
  /** Heavier cells sink through lighter ones. Fixed cells never move. */
  density: d.f32,
  /** Linear radiance it gives off. */
  glow: d.vec3f,
  /** Per-step probability of falling, or rising for gases, when space permits. */
  fall: d.f32,
  /** Extinction per cell and colour channel for the radiance field. */
  extinction: d.vec3f,
  /** Per-step probability of sideways movement. */
  flow: d.f32,
  /** Per-step probability of lifetime decay. */
  decay: d.f32,
  kind: d.u32,
});

type Traits = d.InferInput<typeof Material>;

const solid = [OPAQUE, OPAQUE, OPAQUE] as const;
const none = [0, 0, 0] as const;
const FIXED = 100;
const BASALT = [0.17, 0.16, 0.22] as const;

const material = (traits: Partial<Traits> & Pick<Traits, 'kind'>): Traits => ({
  albedo: none,
  density: 1,
  glow: none,
  fall: 0,
  extinction: solid,
  flow: 0,
  decay: 0,
  ...traits,
});

const TRAITS: Record<keyof typeof Matter, Traits> = {
  empty: material({ kind: Kind.gas, extinction: [0.004, 0.0045, 0.005] }),
  rock: material({ kind: Kind.fixed, density: FIXED, albedo: BASALT }),
  sand: material({ kind: Kind.powder, density: 5, fall: 0.8, albedo: [0.86, 0.68, 0.38] }),
  water: material({
    kind: Kind.liquid,
    density: 3,
    fall: 1,
    flow: 1,
    albedo: [0.05, 0.34, 0.36],
    extinction: [0.24, 0.06, 0.05],
  }),
  lava: material({
    kind: Kind.liquid,
    density: 4,
    fall: 1,
    flow: 1,
    albedo: [0.3, 0.06, 0.02],
    glow: [3.6, 1.05, 0.14],
  }),
  obsidian: material({ kind: Kind.powder, density: 3.5, fall: 1, albedo: [0.13, 0.1, 0.2] }),
  wood: material({ kind: Kind.fixed, density: FIXED, albedo: [0.36, 0.2, 0.09] }),
  burning: material({
    kind: Kind.fixed,
    density: FIXED,
    decay: 0.1,
    albedo: [0.06, 0.025, 0.012],
    glow: [2.6, 0.62, 0.08],
  }),
  fire: material({
    kind: Kind.gas,
    density: 0.6,
    fall: 0.5,
    flow: 0.5,
    decay: 1,
    glow: [4, 1.5, 0.22],
    extinction: [0.5, 0.5, 0.5],
  }),
  smoke: material({
    kind: Kind.gas,
    density: 0.5,
    fall: 0.4,
    flow: 0.5,
    decay: 0.25,
    albedo: [0.2, 0.18, 0.18],
    extinction: [0.25, 0.25, 0.25],
  }),
  steam: material({
    kind: Kind.gas,
    density: 0.3,
    fall: 0.6,
    flow: 0.7,
    decay: 0.18,
    albedo: [0.75, 0.78, 0.82],
    extinction: [0.04, 0.04, 0.04],
  }),
  crystal: material({
    kind: Kind.fixed,
    density: FIXED,
    albedo: [0.2, 0.5, 0.6],
    glow: [0.25, 1.9, 2.6],
    extinction: [2, 2, 2],
  }),
  ash: material({
    kind: Kind.powder,
    density: 1.5,
    fall: 0.35,
    flow: 0.2,
    albedo: [0.22, 0.21, 0.2],
  }),
  spring: material({ kind: Kind.fixed, density: FIXED, albedo: BASALT }),
  drain: material({ kind: Kind.fixed, density: FIXED, albedo: BASALT }),
};

/** Material traits indexed by `Matter` id, as `TRAITS` lists them in its order. */
export const MATERIALS = Object.values(TRAITS);

const Grid = d.arrayOf(d.u32, CELLS);
const Table = d.arrayOf(Material, MATERIALS.length);

/** Lifetime a painted fire cell starts with, in decay steps. Fire fades as it ages. */
export const FIRE_LIFE = 40;
/** Lifetime of wood that paint or an ember sets alight, in decay steps. */
export const BURN_LIFE = 255;

const SHADE_SHIFT = 8;
const LIFE_SHIFT = 16;
const FLOW_SHIFT = 24;
export const LIFE_ONE = 1 << LIFE_SHIFT;
const FLOW_BIT = 1 << FLOW_SHIFT;

/** Packs a cell from its four bytes, low to high: matter, shade, life and flow. */
const pack = (matter: number, shade: number, life: number, flow: number) => {
  'use gpu';
  return (
    d.u32(matter) |
    ((d.u32(shade) & 0xff) << SHADE_SHIFT) |
    ((d.u32(life) & 0xff) << LIFE_SHIFT) |
    (d.u32(flow) << FLOW_SHIFT)
  );
};

/** Creates a cell with material lifetime, random shade and random flow direction. */
export const cellOf = (matter: number, life: number) => {
  'use gpu';
  const shade = d.u32(randf.sample() * 256);
  const right = d.u32(randf.sample() * 2);
  return pack(matter, shade, life, right);
};

/** Creates a rock cell with `shade` from 0 to 255. */
export const rockOf = (shade: number) => {
  'use gpu';
  return pack(Matter.rock, shade, 0, 0);
};

export const matterOf = (cell: number) => {
  'use gpu';
  return cell & 0xff;
};

/** Random shade from 0 to 1, fixed at birth. */
export const shadeOf = (cell: number) => {
  'use gpu';
  return d.f32((cell >>> SHADE_SHIFT) & 0xff) / 255;
};

export const lifeOf = (cell: number) => {
  'use gpu';
  return (cell >>> LIFE_SHIFT) & 0xff;
};

/**
 * Creates a spring emitting `pours` with a per-step probability in units of 1/255. Keeps the
 * shade of `rock`, the rock cell it replaces.
 */
export const springOf = (pours: number, chance: number, rock: number) => {
  'use gpu';
  return pack(Matter.spring, rock >>> SHADE_SHIFT, pours, chance);
};

/**
 * Creates a drain for touching grains and liquids. `water` enables draining water. Probability
 * is per step in units of 1/255. Keeps the shade of `rock`, the rock cell it replaces.
 */
export const drainOf = (chance: number, water: boolean, rock: number) => {
  'use gpu';
  return pack(Matter.drain, rock >>> SHADE_SHIFT, chance, d.u32(water));
};

export const drainsWater = (cell: number) => {
  'use gpu';
  return cell >>> FLOW_SHIFT !== 0;
};

/** Creates a crystal cell with facet shade and base-to-tip progress from 0 to 1. */
export const crystalOf = (facet: number, along: number) => {
  'use gpu';
  return pack(Matter.crystal, std.saturate(facet) * 255, std.saturate(along) * 255, 0);
};

/** Spring emission probability per step, from 0 to 1. */
export const pourOf = (cell: number) => {
  'use gpu';
  return d.f32(cell >>> FLOW_SHIFT) / 255;
};

export const rightward = (cell: number) => {
  'use gpu';
  return (cell & FLOW_BIT) !== 0;
};

export const turned = (cell: number) => {
  'use gpu';
  return cell ^ FLOW_BIT;
};

/** Whether `p` lies in the grid. */
const inside = (p: d.v2i) => {
  'use gpu';
  return p.x >= 0 && p.y >= 0 && p.x < WIDTH && p.y < HEIGHT;
};

/** A grid-sized texture that a kernel writes and later passes read. */
export const gridTexture = (root: TgpuRoot) => {
  const texture = root
    .createTexture({ size: [WIDTH, HEIGHT], format: 'rgba16float' })
    .$usage('storage', 'sampled');
  return {
    writing: texture.createView(d.textureStorage2d('rgba16float', 'write-only')),
    reading: texture.createView(d.texture2d(d.f32)),
    destroy() {
      texture.destroy();
    },
  };
};

/** Stores cells in rows from the top left and material traits indexed by id. */
export function createCells(root: TgpuRoot) {
  const buffer = root.createBuffer(Grid).$usage('storage');
  const writable = buffer.as('mutable');
  const materials = root.createReadonly(Table, MATERIALS);
  const grid = tgpu.accessor(Grid, byStage(buffer));

  /** Reads a cell by column and row. Returns rock outside the grid. */
  const cellAt = (p: d.v2i) => {
    'use gpu';
    if (!inside(p)) return d.u32(Matter.rock);
    return grid.$[d.u32(p.y) * WIDTH + d.u32(p.x)];
  };

  /** Writes a cell from a compute shader. Ignores coordinates outside the grid. */
  const put = (p: d.v2i, cell: number) => {
    'use gpu';
    if (!inside(p)) return;
    writable.$[d.u32(p.y) * WIDTH + d.u32(p.x)] = cell;
  };

  const traitsOf = (cell: number) => {
    'use gpu';
    return Material(materials.$[matterOf(cell)]);
  };

  const matterAt = (p: d.v2i) => {
    'use gpu';
    return matterOf(cellAt(p));
  };

  const kindAt = (p: d.v2i) => {
    'use gpu';
    return traitsOf(cellAt(p)).kind;
  };

  /** Tests solid-cell extinction, including rock and sand but excluding water and air. */
  const opaqueAt = (p: d.v2i) => {
    'use gpu';
    return materials.$[matterAt(p)].extinction.x >= OPAQUE;
  };

  /** Direction and distance from each solid cell to open cells nearby, for lighting solid rims. */
  const relief = gridTexture(root);

  const shape = gridKernel(function shape({ gid }) {
    'use gpu';
    if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
    const at = d.vec2i(gid.xy);
    if (!opaqueAt(at)) {
      std.textureStore(relief.writing.$, gid.xy, d.vec4f());
      return;
    }

    let depth = d.f32(RELIEF_CELLS + 1);
    let out = d.vec2f();
    for (let y = -RELIEF_CELLS; y <= RELIEF_CELLS; y++) {
      for (let x = -RELIEF_CELLS; x <= RELIEF_CELLS; x++) {
        if (opaqueAt(at + d.vec2i(x, y))) continue;
        const offset = d.vec2f(d.f32(x), d.f32(y));
        depth = std.min(depth, d.f32(std.max(std.abs(x), std.abs(y))));
        out += offset / std.dot(offset, offset);
      }
    }

    let normal = d.vec2f();
    if (std.dot(out, out) > 0) normal = std.normalize(out);
    std.textureStore(relief.writing.$, gid.xy, d.vec4f(normal, depth, 0));
  });
  const shaping = root.createComputePipeline({ compute: shape });

  /**
   * Returns direction to nearby open cells in `xy` and distance in `z`. Distance is 1 at the
   * surface, up to `RELIEF_CELLS + 1` inside, and 0 in open cells.
   */
  const reliefAt = (p: d.v2i) => {
    'use gpu';
    return std.textureLoad(relief.reading.$, p, 0);
  };

  // Kernels where each cell gathers its own from its neighbours read them from a copy taken
  // before they run, as they would race with the cells they write otherwise. Only backends whose
  // compute runs on the CPU use them, and only those keep a copy
  const kept =
    backends.of(root).compute === 'cpu' ? root.createBuffer(Grid).$usage('storage') : undefined;
  const keeping = kept?.as('mutable');
  const copying = keeping
    ? root.createComputePipeline({
        compute: gridKernel(function keep({ gid }) {
          'use gpu';
          if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
          const i = gid.y * WIDTH + gid.x;
          keeping.$[i] = grid.$[i] as number;
        }),
      })
    : undefined;
  const fromKept: Inject = (cfg) => (kept ? cfg.with(grid, kept.as('readonly')) : cfg);

  return {
    cellAt,
    put,
    /**
     * Copies the cells for kernels that read the copy through `fromKept`, where compute runs on
     * the CPU. Elsewhere it records nothing.
     */
    keep(pass: TgpuComputePass) {
      if (!copying) return;
      pass.setPipeline(copying);
      pass.dispatchWorkgroups(GROUPS.x, GROUPS.y);
    },
    fromKept,
    traitsOf,
    matterAt,
    kindAt,
    opaqueAt,
    reliefAt,
    initAsync: () => shaping.initAsync(),
    /** Recomputes relief after cells move. */
    shape(pass: TgpuComputePass) {
      pass.setPipeline(shaping);
      pass.dispatchWorkgroups(GROUPS.x, GROUPS.y);
    },
    destroy() {
      buffer.destroy();
      kept?.destroy();
      materials.buffer.destroy();
      relief.destroy();
    },
  };
}

export type Cells = ReturnType<typeof createCells>;
