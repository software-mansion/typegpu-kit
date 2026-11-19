import { pipelines } from '@typegpu/kit';
import { perlin2d, randf } from '@typegpu/noise';
import { sdLine, sdRoundedBox2d } from '@typegpu/sdf';
import { d, std, tgpu, type TgpuCommandEncoder, type TgpuRoot } from 'typegpu';
import {
  BURN_LIFE,
  cellOf,
  type Cells,
  crystalOf,
  drainOf,
  FIRE_LIFE,
  gridKernel,
  GROUPS,
  HEIGHT,
  Kind,
  Matter,
  matterOf,
  rockOf,
  springOf,
  WIDTH,
  WORKGROUP,
} from './cells.ts';

/** Strokes painted per frame. Later strokes stretch the last one. */
const STROKES = 64;
const Stroke = d.struct({ start: d.vec2f, finish: d.vec2f, radius: d.f32, matter: d.u32 });
/** Queued strokes, and a seed that changes each time they are painted. */
const Batch = d.struct({ count: d.u32, seed: d.u32 });

/** What the pointer paints, and the matter it puts down. `erase` empties cells, rock included. */
const PAINTED = {
  sand: Matter.sand,
  water: Matter.water,
  lava: Matter.lava,
  wood: Matter.wood,
  fire: Matter.fire,
  crystal: Matter.crystal,
  rock: Matter.rock,
  erase: Matter.empty,
};
export type Paint = keyof typeof PAINTED;
export const PAINTS = Object.keys(PAINTED) as Paint[];
/** Fraction of painted area filled by loose material. Leaves gaps between grains. */
const POURED = 0.45;
/** Base-to-tip progress of painted crystal, from 0 to 1. */
const PAINTED_CRYSTAL = 0.6;

/** Rounded-box bounds before noise roughening. */
const HALL = { center: d.vec2f(160, 88), half: d.vec2f(152, 84), round: 36 };
/** Outline-noise frequencies per cell and displacement amplitudes in cells. */
const ROUGH = {
  broad: 0.021,
  broadCells: 30,
  mid: 0.07,
  midCells: 12,
  fine: 0.22,
  fineCells: 5,
};
/** Roughness multipliers for the floor and rock overhangs. */
const ROUGHER = { floor: 0.25, massif: 0.45 };
/** Coordinate-warp frequency per cell and displacement in cells. Breaks straight outlines. */
const WARP = { scale: 0.03, cells: 20 };

/** Floor rows under the water overhang, in the basin and under the lava overhang. */
const FLOOR = { grotto: 131, basin: 168, right: 124 };
/** Basin-wall columns and their feet at the basin floor. */
const BASIN = { left: 94, leftFoot: 114, rightFoot: 228, right: 246, dip: 3 };
/** Dam position and lava-bed row. Lava spills over the dam into water. */
const DAM = { x: 178, top: 128, width: 4, bed: 150 };
/** Initial water and lava surface rows. Lava fills the area to the right of the dam. */
const FILLED = { water: 142, lava: 129 };

/** A rock overhang tapering from `wallRadius` at `wall` to `tipRadius` at `tip`, in cells. */
const Massif = d.struct({ wall: d.vec2f, tip: d.vec2f, wallRadius: d.f32, tipRadius: d.f32 });
/** The overhangs above the water and the lava. */
const massifs = tgpu.const(d.arrayOf(Massif, 2), [
  { wall: d.vec2f(-12, 70), tip: d.vec2f(118, 64), wallRadius: 17, tipRadius: 9 },
  { wall: d.vec2f(332, 64), tip: d.vec2f(200, 58), wallRadius: 17, tipRadius: 10 },
]);

/**
 * A vertical opening `half` cells either side of column `x`, with a spring at row `top` pouring
 * `pours` with a per-step probability of `chance` in units of 1/255.
 */
const Cleft = d.struct({ x: d.f32, half: d.f32, top: d.f32, pours: d.u32, chance: d.u32 });
/** The openings through the water and lava overhangs, and a crack above the grotto. */
const clefts = tgpu.const(d.arrayOf(Cleft, 3), [
  { x: 112, half: 3, top: 63, pours: Matter.water, chance: 102 },
  { x: 208, half: 2.6, top: 57, pours: Matter.lava, chance: 64 },
  { x: 52, half: 2.1, top: 72, pours: Matter.sand, chance: 20 },
]);
/** Open-cleft depth below each spring, in cells. */
const CLEFT_DEPTH = 30;
/** Per-step probability in units of 1/255 that a stalactite tip drips. */
const DRIP = 1;

/**
 * Drain regions in rock, between corners `low` and `high`. Left-wall outlets remove excess water
 * and sand. Floor seeps remove lava and grit less often. Probabilities use units of 1/255.
 */
const SEEPS = {
  water: { low: d.vec2f(80, 140), high: d.vec2f(136, 151) },
  sand: { low: d.vec2f(88, 131), high: d.vec2f(108, 138) },
  floor: { top: 148, chance: 24 },
  bed: 1,
  outlet: 255,
};

/** Stalactite column, base half-width and length in cells. */
const Stalactite = d.struct({ x: d.f32, half: d.f32, length: d.f32 });
const STALACTITES = [
  { x: 40, half: 4, length: 14 },
  { x: 60, half: 4.5, length: 15 },
  { x: 78, half: 6.5, length: 25 },
  { x: 101, half: 2.5, length: 10 },
  { x: 125, half: 5, length: 31 },
  { x: 176, half: 3, length: 13 },
  { x: 214, half: 7.5, length: 27 },
  { x: 237, half: 3, length: 16 },
  { x: 266, half: 5.5, length: 21 },
  { x: 291, half: 3.5, length: 11 },
] as const;
const stalactites = tgpu.const(d.arrayOf(Stalactite, STALACTITES.length), [...STALACTITES]);
const CEILING_CELLS = 10;

/** Crystal prisms ordered front to back, with base, tip and half-width in cells. */
const Prism = d.struct({ base: d.vec2f, tip: d.vec2f, width: d.f32 });
const PRISMS: d.InferInput<typeof Prism>[] = [
  // Sand can bury this cluster
  { base: [52, 134], tip: [52, 109], width: 2.2 },
  { base: [49, 134], tip: [42, 116], width: 1.8 },
  { base: [55, 134], tip: [63, 115], width: 1.8 },
  { base: [46, 134], tip: [37, 125], width: 1.4 },
  { base: [58, 134], tip: [68, 124], width: 1.4 },
  { base: [150, -2], tip: [152, 30], width: 2.4 },
  { base: [144, -2], tip: [136, 22], width: 1.7 },
  { base: [157, -2], tip: [167, 24], width: 1.8 },
  { base: [139, 0], tip: [129, 13], width: 1.3 },
  { base: [324, 32], tip: [301, 24], width: 2.1 },
  { base: [324, 38], tip: [306, 43], width: 1.6 },
  { base: [324, 26], tip: [310, 15], width: 1.3 },
  { base: [100, 170], tip: [117, 151], width: 2 },
  { base: [98, 166], tip: [104, 149], width: 1.4 },
  { base: [104, 172], tip: [125, 162], width: 1.5 },
];

/** Wooden beams from `base` to `tip`, `radius` cells thick. */
const Timber = d.struct({ base: d.vec2f, tip: d.vec2f, radius: d.f32 });
const TIMBERS: d.InferInput<typeof Timber>[] = [
  { base: [258, 66], tip: [258, 128], radius: 1.7 },
  { base: [296, 66], tip: [296, 128], radius: 1.7 },
  { base: [252, 90], tip: [302, 90], radius: 1.3 },
  { base: [252, 108], tip: [302, 108], radius: 1.3 },
  { base: [259, 91], tip: [295, 107], radius: 1 },
  { base: [295, 91], tip: [259, 107], radius: 1 },
];

/** Signed distance to a roughened outline, in cells. */
const roughAt = (p: d.v2f) => {
  'use gpu';
  return (
    perlin2d.sample(p * ROUGH.broad) * ROUGH.broadCells +
    perlin2d.sample(p * ROUGH.mid + 11) * ROUGH.midCells +
    perlin2d.sample(p * ROUGH.fine + 23) * ROUGH.fineCells
  );
};

/** Floor surface row at a column. */
const floorAt = (x: number) => {
  'use gpu';
  const down = std.smoothstep(BASIN.left, BASIN.leftFoot, x) * (FLOOR.basin - FLOOR.grotto);
  const up = std.smoothstep(BASIN.rightFoot, BASIN.right, x) * (FLOOR.basin - FLOOR.right);
  const middle = (BASIN.leftFoot + BASIN.rightFoot) / 2;
  const half = (BASIN.rightFoot - BASIN.leftFoot) / 2;
  // `**` compiles to `pow`, which is NaN for a negative base on most GPUs
  const centred = (x - middle) / half;
  const dip = std.saturate(1 - centred * centred) * BASIN.dip;
  const bed = std.smoothstep(DAM.x - 2, DAM.x + 5, x) * (FLOOR.basin + dip - DAM.bed);
  const across = (x - DAM.x) / DAM.width;
  const dam = std.max(FLOOR.basin + dip - bed - DAM.top, 0) * std.exp(-across * across);
  return FLOOR.grotto + down - up + dip - bed - dam;
};

/** Signed distance from `p` to an overhang, in cells. */
const massifAt = (p: d.v2f, massif: d.Infer<typeof Massif>) => {
  'use gpu';
  const span = massif.tip - massif.wall;
  const h = std.saturate(std.dot(p - massif.wall, span) / std.dot(span, span));
  return std.distance(p, massif.wall + span * h) - std.mix(massif.wallRadius, massif.tipRadius, h);
};

/** Horizontal distance from `p` to a stalactite's wavering axis. */
const spikeOffset = (p: d.v2f, spike: d.Infer<typeof Stalactite>) => {
  'use gpu';
  const wobble = perlin2d.sample(d.vec2f(p.y * 0.08, spike.x)) * 3;
  return std.abs(p.x - spike.x + wobble);
};

/** Stalactite half-width at row `y`, including roughness. */
const spikeWidth = (spike: d.Infer<typeof Stalactite>, y: number) => {
  'use gpu';
  const down = (y - CEILING_CELLS) / spike.length;
  const lumpy = 0.8 + perlin2d.sample(d.vec2f(y * 0.3, spike.x)) * 0.5;
  return spike.half * std.max(1 - down, 0) ** 1.3 * lumpy;
};

/** Whether `p` lies in a cleft's opening below its spring. */
const cleftAt = (p: d.v2f, cleft: d.Infer<typeof Cleft>) => {
  'use gpu';
  const wobble = std.abs(perlin2d.sample(d.vec2f(p.y * 0.35, cleft.x))) * 1.2 + 0.4;
  const beside = std.abs(p.x - cleft.x) < cleft.half + wobble;
  return beside && p.y > cleft.top + 1 && p.y < cleft.top + CLEFT_DEPTH;
};

/** Whether `p` lies in the row of a cleft's spring. */
const springAt = (p: d.v2f, cleft: d.Infer<typeof Cleft>) => {
  'use gpu';
  return std.abs(p.x - cleft.x) < cleft.half && p.y > cleft.top && p.y < cleft.top + 1;
};

const solidAt = (p: d.v2f) => {
  'use gpu';
  const rough = roughAt(p);
  const bend = d.vec2f(perlin2d.sample(p * WARP.scale + 31), perlin2d.sample(p * WARP.scale + 57));
  const q = p + bend * WARP.cells;
  let rock = sdRoundedBox2d(q - HALL.center, HALL.half, HALL.round) + rough > 0;
  if (q.y > floorAt(q.x) + rough * ROUGHER.floor) rock = true;

  for (const massif of tgpu.unroll(massifs.$)) {
    if (massifAt(q, massif) + rough * ROUGHER.massif < 0) rock = true;
  }

  for (let i = d.u32(0); i < STALACTITES.length; i++) {
    const spike = stalactites.$[i];
    if (spikeOffset(p, spike) < spikeWidth(spike, p.y)) rock = true;
  }

  for (const cleft of tgpu.unroll(clefts.$)) {
    if (cleftAt(p, cleft)) rock = false;
  }
  return rock;
};

/** Tests the bottom centre cell of a stalactite, where drips originate. */
const dripsAt = (p: d.v2f) => {
  'use gpu';
  let drips = false;
  for (let i = d.u32(0); i < STALACTITES.length; i++) {
    const spike = stalactites.$[i];
    const off = spikeOffset(p, spike);
    const width = spikeWidth(spike, p.y);
    const below = spikeWidth(spike, p.y + 1);
    if (off < 0.5 && off < width && off >= below) drips = true;
  }
  return drips;
};

/** Whether `p` lies strictly between corners `low` and `high`. */
const within = (p: d.v2f, low: d.v2f, high: d.v2f) => {
  'use gpu';
  return std.all(std.gt(p, low)) && std.all(std.lt(p, high));
};

/** Rock shade from horizontal strata and grain. */
const strataAt = (p: d.v2f) => {
  'use gpu';
  const beds = perlin2d.sample(p * d.vec2f(0.025, 0.11)) * 0.5 + 0.5;
  const grain = perlin2d.sample(p * 0.4) * 0.5 + 0.5;
  return std.saturate(beds * 0.7 + grain * 0.3);
};

const rockAt = (p: d.v2f) => {
  'use gpu';
  return rockOf((strataAt(p) * 0.65 + randf.sample() * 0.35) * 255);
};

/** Maps prism coordinates from -1 to 1 to a dark side, lit side or front facet. */
const facetOf = (across: number) => {
  'use gpu';
  if (across < -0.34) return d.f32(0.5);
  if (across > 0.34) return d.f32(0.75);
  return d.f32(1);
};

/** Initialises the cave and paints pointer strokes. */
export function createCave(root: TgpuRoot, cells: Cells) {
  const strokes = root.createReadonly(d.arrayOf(Stroke, STROKES));
  const batch = root.createUniform(Batch);
  const prisms = root.createReadonly(d.arrayOf(Prism, PRISMS.length), PRISMS);
  const timbers = root.createReadonly(d.arrayOf(Timber, TIMBERS.length), TIMBERS);
  // `solidAt` over the grid and a cell around it, so `start` reads neighbours instead of
  // repeating the noise: Windows shader compilers inline each call
  const solid = root.createMutable(d.arrayOf(d.u32, (WIDTH + 2) * (HEIGHT + 2)));

  const crystalAt = (p: d.v2f) => {
    'use gpu';
    for (let i = d.u32(0); i < PRISMS.length; i++) {
      const prism = prisms.$[i];
      const axis = prism.tip - prism.base;
      const length = std.length(axis);
      const along = std.dot(p - prism.base, axis) / length;
      const across = ((p.x - prism.base.x) * axis.y - (p.y - prism.base.y) * axis.x) / length;
      const reach = prism.width * std.saturate((length - along) / (prism.width * 1.8));
      if (along > -1 && along < length && std.abs(across) < reach) {
        return crystalOf(facetOf(across / prism.width), along / length);
      }
    }
    return d.u32(Matter.empty);
  };

  /** Reads `solidAt` at the centre `p` of a cell in the grid or a cell outside it. */
  const solidNear = (p: d.v2f) => {
    'use gpu';
    const at = d.vec2u(p + 0.5);
    return solid.$[at.y * (WIDTH + 2) + at.x] === 1;
  };

  /** Rock, a drain where the seeps run, or air where the noise leaves an isolated speck. */
  const wallAt = (p: d.v2f, stone: number) => {
    'use gpu';
    const open =
      d.u32(!solidNear(p + d.vec2f(1, 0))) +
      d.u32(!solidNear(p - d.vec2f(1, 0))) +
      d.u32(!solidNear(p + d.vec2f(0, 1))) +
      d.u32(!solidNear(p - d.vec2f(0, 1)));
    if (open >= 3) return d.u32(Matter.empty);
    if (open === 0) return stone;

    const water = within(p, SEEPS.water.low, SEEPS.water.high);
    const sand = within(p, SEEPS.sand.low, SEEPS.sand.high);
    if (water || sand) return drainOf(SEEPS.outlet, water, stone);
    const floor = p.x > BASIN.left && p.x < DAM.x - DAM.width && p.y > SEEPS.floor.top;
    if (floor) return drainOf(SEEPS.floor.chance, false, stone);
    const bed = p.x > DAM.x + DAM.width && p.x < BASIN.right && p.y > DAM.top + 4;
    if (bed) return drainOf(SEEPS.bed, false, stone);
    return stone;
  };

  /** Water and lava either side of the dam, the timber beams, or air. */
  const hollowAt = (p: d.v2f) => {
    'use gpu';
    const inBasin = p.x > BASIN.left && p.x < BASIN.right;
    if (inBasin && p.x > DAM.x && p.y > FILLED.lava) return cellOf(Matter.lava, 0);
    if (inBasin && p.x < DAM.x && p.y > FILLED.water) return cellOf(Matter.water, 0);
    for (let i = d.u32(0); i < TIMBERS.length; i++) {
      const timber = timbers.$[i];
      if (sdLine(p, timber.base, timber.tip) < timber.radius) return cellOf(Matter.wood, 0);
    }
    return d.u32(Matter.empty);
  };

  /** The cell the cave starts with at `p`. */
  const startAt = (p: d.v2f) => {
    'use gpu';
    const stone = rockAt(p);
    for (const cleft of tgpu.unroll(clefts.$)) {
      if (springAt(p, cleft)) return springOf(cleft.pours, cleft.chance, stone);
    }
    if (dripsAt(p) && solidNear(p - d.vec2f(0, 1))) return springOf(Matter.water, DRIP, stone);

    const crystal = crystalAt(p);
    if (crystal !== Matter.empty) return crystal;
    if (solidNear(p)) return wallAt(p, stone);
    return hollowAt(p);
  };

  /** What a stroke of `matter` leaves where `cell` was. */
  const paintedOver = (cell: number, matter: number, p: d.v2f) => {
    'use gpu';
    const now = cells.traitsOf(cell).kind;
    if (matter === Matter.empty) return d.u32(Matter.empty);
    if (matter === Matter.rock) return rockAt(p);
    if (matter === Matter.crystal) {
      return crystalOf(facetOf(randf.sample() * 2 - 1), PAINTED_CRYSTAL);
    }
    if (cells.traitsOf(matter).kind === Kind.fixed) return cellOf(matter, 0);
    if (matter === Matter.fire && now === Kind.fixed) {
      if (matterOf(cell) === Matter.wood) return cellOf(Matter.burning, BURN_LIFE);
      return d.u32(cell);
    }
    if (now === Kind.gas && randf.sample() < POURED) return cellOf(matter, FIRE_LIFE);
    return d.u32(cell);
  };

  const outline = gridKernel(function outline({ gid }) {
    'use gpu';
    if (gid.x >= WIDTH + 2 || gid.y >= HEIGHT + 2) return;
    solid.$[gid.y * (WIDTH + 2) + gid.x] = d.u32(solidAt(d.vec2f(gid.xy) - 0.5));
  });

  const start = gridKernel(function start({ gid }) {
    'use gpu';
    if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
    randf.seed2(d.vec2f(gid.xy));
    cells.put(d.vec2i(gid.xy), startAt(d.vec2f(gid.xy) + 0.5));
  });

  const paint = gridKernel(function paint({ gid }) {
    'use gpu';
    if (gid.x >= WIDTH || gid.y >= HEIGHT) return;
    randf.seed3(d.vec3f(d.vec2f(gid.xy), d.f32(batch.$.seed)));
    const at = d.vec2i(gid.xy);
    const p = d.vec2f(gid.xy) + 0.5;
    let cell = cells.cellAt(at);
    for (let i = d.u32(0); i < batch.$.count; i++) {
      const stroke = strokes.$[i];
      let distance = std.distance(p, stroke.start);
      // A zero-length segment makes `sdLine` divide by zero, so measure point distance instead
      if (std.any(std.ne(stroke.start, stroke.finish))) {
        distance = sdLine(p, stroke.start, stroke.finish);
      }
      if (distance > stroke.radius) continue;
      cell = paintedOver(cell, stroke.matter, p);
    }
    cells.put(at, cell);
  });

  const outlining = root.createComputePipeline({ compute: outline });
  const starting = root.createComputePipeline({ compute: start });
  const painting = root.createComputePipeline({ compute: paint });
  const outlineGroups = [
    Math.ceil((WIDTH + 2) / WORKGROUP),
    Math.ceil((HEIGHT + 2) / WORKGROUP),
  ] as const;

  const pending: d.InferInput<typeof Stroke>[] = [];
  let batches = 0;
  let resetting = true;

  return {
    initAsync: () => pipelines.compile([outlining, starting, painting]),
    /** Queues a stroke in cell coordinates, with y pointing down. */
    stroke(
      from: readonly [number, number],
      to: readonly [number, number],
      radius: number,
      what: Paint,
    ) {
      const stroke = { start: from, finish: to, radius, matter: PAINTED[what] };
      if (pending.length < STROKES) pending.push(stroke);
      else pending[STROKES - 1] = { ...stroke, start: pending[STROKES - 1].start };
    },
    /** Resets the cave or paints queued strokes. Returns false if no work was queued. */
    record(encoder: TgpuCommandEncoder) {
      if (!resetting && pending.length === 0) return false;
      const pass = encoder.beginComputePass();
      if (resetting) {
        pass.setPipeline(outlining);
        pass.dispatchWorkgroups(...outlineGroups);
        pass.setPipeline(starting);
        pass.dispatchWorkgroups(GROUPS.x, GROUPS.y);
      }
      resetting = false;
      if (pending.length > 0) {
        strokes.buffer.write(pending);
        batch.write({ count: pending.length, seed: batches++ % 65536 });
        pending.length = 0;
        pass.setPipeline(painting);
        pass.dispatchWorkgroups(GROUPS.x, GROUPS.y);
      }
      pass.end();
      return true;
    },
    reset() {
      resetting = true;
      pending.length = 0;
    },
    destroy() {
      for (const rows of [strokes, prisms, timbers, solid]) rows.buffer.destroy();
    },
  };
}
