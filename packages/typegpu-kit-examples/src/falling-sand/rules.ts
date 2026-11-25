import { pipelines, type Counters } from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import {
  BURN_LIFE,
  cellOf,
  type Cells,
  drainsWater,
  FIRE_LIFE,
  gridKernel,
  HEIGHT,
  Kind,
  LIFE_ONE,
  lifeOf,
  Matter,
  matterOf,
  pourOf,
  rightward,
  turned,
  WIDTH,
  WORKGROUP,
} from './cells.ts';

/** 2 × 2 block grid with one extra block per axis for shifted steps. */
const BLOCKS_X = WIDTH / 2 + 1;
const BLOCKS_Y = HEIGHT / 2 + 1;
/** Rows per parity. */
const ROWS = HEIGHT / 2;
/** Workgroups over the blocks, and over one parity's rows. */
const BLOCK_GROUPS = { x: Math.ceil(BLOCKS_X / WORKGROUP), y: Math.ceil(BLOCKS_Y / WORKGROUP) };
const ROW_GROUPS = Math.ceil(ROWS / WORKGROUP);

/** Liquid-flow passes per simulation step. Each covers both row parities. */
const FLOW_PASSES = 3;
/** Probability of reversing behind another liquid. A low value preserves flow direction. */
const TURNS = 0.1;
/** Per-step probability of sinking through liquid or non-air gas. */
const SINK = 0.35;
/** Per-step probability of a falling liquid drifting sideways by one cell. */
const SPRAY = 0.05;
/** Sideways-drift multiplier for gas with nothing above it. */
const LOOSE_DRIFT = 0.3;

const LAVA_IGNITES = 0.05;
const FIRE_SPREADS = 0.006;
const EMBER_SPREADS = 0.002;
/** Per-step probabilities of lava melting sand and obsidian on contact. */
const MELTS_SAND = 0.03;
const MELTS_OBSIDIAN = 0.01;
const DOUSED = 0.08;
const FLAMES = 0.04;
const SMOKES = 0.3;
const CONDENSES = 0.35;
const CHARS = 0.4;
/** Chance a step that a lava surface or a burning cell spits an ember. */
const SPITS = 0.0025;
/** Per-step probability of young steam spreading sideways into air and halving its lifetime. */
const BILLOWS = 0.4;
/** Steam lifetime threshold for spreading, in decay steps. */
const YOUNG_STEAM = 90;

/** Minimum and maximum lifetimes in decay steps. */
const STEAM_LIFE = { least: 60, most: 170 };
const SMOKE_LIFE = { least: 80, most: 200 };
const FLAME_LIFE = { least: 12, most: FIRE_LIFE };
const KINDLED_LIFE = { least: 180, most: BURN_LIFE };

export const SPRINGS = true;

const Controls = d.struct({ springs: d.u32 });

const between = tgpu.fn(
  [d.f32, d.f32],
  d.f32,
)((least, most) => {
  'use gpu';
  return least + randf.sample() * (most - least);
});

/**
 * Moves cells within disjoint 2 × 2 blocks. Shifts the blocks between steps so threads never
 * write overlapping cells and all neighbour pairs share a block within four steps.
 */
export function createRules(
  root: TgpuRoot,
  options: {
    readonly cells: Cells;
    readonly counts: Pick<Counters<'cooled' | 'burnt'>, 'add'>;
    /** Emits from a cell in y-down coordinates, with `heat` from 0 to 1. */
    readonly spark: (at: d.v2f, heat: number) => void;
  },
) {
  const { cells, counts, spark } = options;
  const tick = root.createMutable(d.u32);
  const controls = root.createUniform(Controls, { springs: SPRINGS ? 1 : 0 });

  const ignite = () => {
    'use gpu';
    counts.add.burnt(1);
    return cellOf(Matter.burning, between(KINDLED_LIFE.least, KINDLED_LIFE.most));
  };

  /** Contact reaction for `a` and `b`. */
  const act = (a: number, b: number) => {
    'use gpu';
    const actor = matterOf(a);
    const other = matterOf(b);
    let first = d.u32(a);
    let second = d.u32(b);
    switch (actor) {
      case Matter.lava:
        if (other === Matter.water) {
          first = cellOf(Matter.obsidian, 0);
          second = cellOf(Matter.steam, between(STEAM_LIFE.least, STEAM_LIFE.most));
          counts.add.cooled(1);
        } else if (other === Matter.wood && randf.sample() < LAVA_IGNITES) {
          second = ignite();
        } else if (other === Matter.sand && randf.sample() < MELTS_SAND) {
          second = d.u32(Matter.empty);
        } else if (other === Matter.obsidian && randf.sample() < MELTS_OBSIDIAN) {
          second = cellOf(Matter.lava, 0);
        }
        break;
      case Matter.burning:
      case Matter.fire: {
        const burning = actor === Matter.burning;
        if (other === Matter.wood) {
          const spreading = burning ? EMBER_SPREADS : FIRE_SPREADS;
          if (randf.sample() < spreading) second = ignite();
        } else if (other === Matter.water) {
          if (randf.sample() < DOUSED) {
            if (burning) first = cellOf(Matter.wood, 0);
            else first = cellOf(Matter.steam, between(STEAM_LIFE.least, STEAM_LIFE.most));
            second = cellOf(Matter.steam, between(STEAM_LIFE.least, STEAM_LIFE.most));
          }
        } else if (burning && other === Matter.empty && randf.sample() < FLAMES) {
          second = cellOf(Matter.fire, between(FLAME_LIFE.least, FLAME_LIFE.most));
        }
        break;
      }
      case Matter.spring: {
        const pours = lifeOf(a);
        const open = cells.traitsOf(b).kind === Kind.gas;
        if (open && controls.$.springs !== 0 && randf.sample() < pourOf(a)) {
          second = cellOf(pours, 0);
        } else if (pours === Matter.lava && other === Matter.obsidian) {
          // Melt cooled crust around the lava spout to keep it open
          second = cellOf(Matter.lava, 0);
        }
        break;
      }
      case Matter.steam: {
        const life = lifeOf(a);
        if (other === Matter.empty && life > YOUNG_STEAM && randf.sample() < BILLOWS) {
          first = cellOf(Matter.steam, life / 2);
          second = cellOf(Matter.steam, life / 2);
        }
        break;
      }
      case Matter.drain: {
        const kind = cells.traitsOf(b).kind;
        const takes = kind === Kind.powder || (kind === Kind.liquid && other !== Matter.water);
        const seeps = d.f32(lifeOf(a)) / 255;
        if ((takes || (other === Matter.water && drainsWater(a))) && randf.sample() < seeps) {
          second = d.u32(Matter.empty);
        }
        break;
      }
    }
    return d.vec2u(first, second);
  };

  /** Tests contact reactions in both directions. */
  const react = (a: number, b: number) => {
    'use gpu';
    const once = act(a, b);
    const twice = act(once.y, once.x);
    return d.vec2u(twice.y, twice.x);
  };

  /** Reacts the cells in a block's `x` and `y`, then those in its `z` and `w`. */
  const reactPairs = (block: d.v4u) => {
    'use gpu';
    return d.vec4u(react(block.x, block.y), react(block.z, block.w));
  };

  /** Decays a cell's lifetime and transforms expired cells. */
  const age = (cell: number) => {
    'use gpu';
    const decay = cells.traitsOf(cell).decay;
    if (decay === 0 || randf.sample() >= decay) return d.u32(cell);
    if (lifeOf(cell) > 1) return cell - LIFE_ONE;
    switch (matterOf(cell)) {
      case Matter.fire:
        if (randf.sample() < SMOKES) {
          return cellOf(Matter.smoke, between(SMOKE_LIFE.least, SMOKE_LIFE.most));
        }
        break;
      case Matter.steam:
        if (randf.sample() < CONDENSES) return cellOf(Matter.water, 0);
        break;
      case Matter.burning:
        if (randf.sample() < CHARS) return cellOf(Matter.ash, 0);
        break;
    }
    return d.u32(Matter.empty);
  };

  const isLiquid = (cell: number) => {
    'use gpu';
    return cells.traitsOf(cell).kind === Kind.liquid;
  };

  const isVapour = (cell: number) => {
    'use gpu';
    return cells.traitsOf(cell).kind === Kind.gas && matterOf(cell) !== Matter.empty;
  };

  const rests = (cell: number, below: number) => {
    'use gpu';
    const under = cells.traitsOf(below);
    return under.kind === Kind.fixed || under.density >= cells.traitsOf(cell).density;
  };

  const capped = (cell: number, above: number) => {
    'use gpu';
    const over = cells.traitsOf(above);
    return over.kind === Kind.fixed || over.density <= cells.traitsOf(cell).density;
  };

  /** Tests sideways liquid movement into lighter movable material. */
  const yields = (mover: number, target: number) => {
    'use gpu';
    const into = cells.traitsOf(target);
    return into.kind !== Kind.fixed && into.density < cells.traitsOf(mover).density;
  };

  /** Reverses a blocked liquid at walls, or with low probability behind other liquid. */
  const turns = (ahead: number) => {
    'use gpu';
    return cells.traitsOf(ahead).kind === Kind.fixed || randf.sample() < TURNS;
  };

  /** Tests vertical exchange. Heavier matter sinks, and gas rises through air. */
  const sinks = (top: number, bottom: number) => {
    'use gpu';
    const upper = cells.traitsOf(top);
    const lower = cells.traitsOf(bottom);
    if (upper.kind === Kind.fixed || lower.kind === Kind.fixed) return false;
    if (upper.density <= lower.density) return false;
    let chance = d.f32(SINK);
    if (matterOf(bottom) === Matter.empty) chance = upper.fall;
    else if (matterOf(top) === Matter.empty) chance = lower.fall;
    return randf.sample() < chance;
  };

  /** Tests a falling liquid drifting one cell sideways as it falls. */
  const sprays = (mover: number, below: number, target: number) => {
    'use gpu';
    if (!isLiquid(mover)) return false;
    if (matterOf(below) !== Matter.empty || matterOf(target) !== Matter.empty) return false;
    return randf.sample() < SPRAY;
  };

  /** Tests a grain or liquid sliding down past `beside` into `target`. */
  const slides = (mover: number, below: number, target: number, beside: number) => {
    'use gpu';
    const moving = cells.traitsOf(mover);
    if (moving.kind !== Kind.powder && moving.kind !== Kind.liquid) return false;
    const into = cells.traitsOf(target);
    if (into.kind === Kind.fixed || into.density >= moving.density) return false;
    const under = cells.traitsOf(below);
    if (under.kind !== Kind.fixed && under.density < moving.density) return false;
    if (under.kind === Kind.fixed && cells.traitsOf(beside).kind === Kind.fixed) return false;
    return randf.sample() < moving.fall;
  };

  /** Tests a gas rising past `beside` into `target`. */
  const rises = (mover: number, above: number, target: number, beside: number) => {
    'use gpu';
    const moving = cells.traitsOf(mover);
    if (moving.kind !== Kind.gas || matterOf(mover) === Matter.empty) return false;
    const into = cells.traitsOf(target);
    if (into.kind === Kind.fixed || into.density <= moving.density) return false;
    const over = cells.traitsOf(above);
    if (over.kind !== Kind.fixed && over.density > moving.density) return false;
    if (over.kind === Kind.fixed && cells.traitsOf(beside).kind === Kind.fixed) return false;
    return randf.sample() < moving.fall;
  };

  /**
   * Moves supported liquids in the top row along their flow directions, reversing where
   * blocked.
   */
  const pour = (left: number, right: number, belowLeft: number, belowRight: number) => {
    'use gpu';
    let l = d.u32(left);
    let r = d.u32(right);
    if (isLiquid(l) && rightward(l) && rests(l, belowLeft)) {
      if (!yields(l, r)) {
        if (turns(r)) l = turned(l);
      } else if (randf.sample() < cells.traitsOf(l).flow) return d.vec2u(r, l);
    }
    if (isLiquid(r) && !rightward(r) && rests(r, belowRight)) {
      if (!yields(r, l)) {
        if (turns(l)) r = turned(r);
      } else if (randf.sample() < cells.traitsOf(r).flow) return d.vec2u(r, l);
    }
    return d.vec2u(l, r);
  };

  /** Drifts bottom-row gas into air, mainly when the row above traps it. */
  const drift = (left: number, right: number, aboveLeft: number, aboveRight: number) => {
    'use gpu';
    const towardsRight = randf.sample() < 0.5;
    const mover = std.select(right, left, towardsRight);
    const target = std.select(left, right, towardsRight);
    const above = std.select(aboveRight, aboveLeft, towardsRight);
    if (!isVapour(mover) || matterOf(target) !== Matter.empty) return d.vec2u(left, right);
    let chance = cells.traitsOf(mover).flow;
    if (!capped(mover, above)) chance *= LOOSE_DRIFT;
    if (randf.sample() < chance) return d.vec2u(right, left);
    return d.vec2u(left, right);
  };

  /** Spits an ember from lava or a burning cell with air above it. */
  const spits = (over: number, hot: number, at: d.v2i) => {
    'use gpu';
    if (matterOf(over) !== Matter.empty) return;
    const matter = matterOf(hot);
    if (matter !== Matter.lava && matter !== Matter.burning) return;
    if (randf.sample() < SPITS) spark(d.vec2f(at) + 0.5, randf.sample());
  };

  const step = gridKernel(function step({ gid }) {
    'use gpu';
    if (gid.x >= BLOCKS_X || gid.y >= BLOCKS_Y) return;
    // Shift rows every step to let falling cells cross blocks. Shift columns every other step
    // to alternate left and right positions within each top row
    const phase = tick.$ & 3;
    const shift = d.vec2i(d.i32(((phase + 1) >>> 1) & 1), d.i32(phase & 1));
    const origin = d.vec2i(gid.xy) * 2 - shift;
    if (origin.x >= WIDTH || origin.y >= HEIGHT) return;
    randf.seed4(d.vec4f(d.vec2f(gid.xy), d.f32(tick.$ & 0xffff), d.f32(tick.$ >>> 16)));

    // The block's top left in x, top right in y, bottom left in z and bottom right in w
    const atTopRight = origin + d.vec2i(1, 0);
    const atBottomLeft = origin + d.vec2i(0, 1);
    const atBottomRight = origin + d.vec2i(1, 1);
    let block = d.vec4u(
      cells.cellAt(origin),
      cells.cellAt(atTopRight),
      cells.cellAt(atBottomLeft),
      cells.cellAt(atBottomRight),
    );

    // React the rows, the columns, then the diagonals. Each swizzle gathers two pairs into the
    // lanes `reactPairs` pairs, and the second one puts them back
    block = reactPairs(block);
    block = reactPairs(block.xzyw).xzyw;
    block = reactPairs(block.xwyz).xzwy;
    block = d.vec4u(age(block.x), age(block.y), age(block.z), age(block.w));
    spits(block.x, block.z, atBottomLeft);
    spits(block.y, block.w, atBottomRight);

    // Each move swaps two cells, as its swizzle shows. Mirror half the blocks to avoid
    // directional bias
    const mirrored = randf.sample() < 0.5;
    if (mirrored) block = block.yxwz;
    if (sprays(block.x, block.z, block.w)) block = block.wyzx;
    if (sinks(block.x, block.z)) block = block.zyxw;
    if (sinks(block.y, block.w)) block = block.xwzy;
    if (slides(block.x, block.z, block.w, block.y)) block = block.wyzx;
    if (slides(block.y, block.w, block.z, block.x)) block = block.xzyw;
    if (rises(block.z, block.x, block.y, block.w)) block = block.xzyw;
    if (rises(block.w, block.y, block.x, block.z)) block = block.wyzx;
    if (mirrored) block = block.yxwz;

    const top = pour(block.x, block.y, block.z, block.w);
    const bottom = drift(block.z, block.w, top.x, top.y);
    cells.put(origin, top.x);
    cells.put(atTopRight, top.y);
    cells.put(atBottomLeft, bottom.x);
    cells.put(atBottomRight, bottom.y);
  });

  /**
   * Moves liquids in one row parity, two cells per thread. Supporting rows have the opposite
   * parity, so this dispatch never writes them.
   */
  const flowing = (parity: number, offset: number) =>
    gridKernel(function flow({ gid }) {
      'use gpu';
      if (gid.x >= BLOCKS_X || gid.y >= ROWS) return;
      const left = d.vec2i(d.i32(gid.x) * 2 - offset, d.i32(gid.y) * 2 + parity);
      const right = left + d.vec2i(1, 0);
      const l = cells.cellAt(left);
      const r = cells.cellAt(right);
      if (!isLiquid(l) && !isLiquid(r)) return;
      randf.seed4(d.vec4f(d.vec2f(left), d.f32(tick.$ & 0xffff), d.f32(tick.$ >>> 16) + offset));
      const below = d.vec2i(0, 1);
      const pair = pour(l, r, cells.cellAt(left + below), cells.cellAt(right + below));
      cells.put(left, pair.x);
      cells.put(right, pair.y);
    });

  const ticking = root.createComputePipeline({
    compute: tgpu.computeFn({ workgroupSize: [1] })(() => {
      'use gpu';
      tick.$ += 1;
    }),
  });
  const stepping = root.createComputePipeline({ compute: step });
  const flowingAt = (parity: number, offset: number) =>
    root.createComputePipeline({ compute: flowing(parity, offset) });
  /** Both row parities at column offsets 0 and 1. */
  const pairings = [0, 1].map((offset) => [flowingAt(0, offset), flowingAt(1, offset)]);
  // Passes alternate column offsets so liquid can move one cell per pass
  const flows = Array.from({ length: FLOW_PASSES }, (_, pass) => pairings[pass % 2]).flat();

  return {
    controls,
    initAsync: () => pipelines.compile([ticking, stepping, ...pairings.flat()]),
    /** Steps all cells, then moves liquids along their rows. */
    step(pass: TgpuComputePass) {
      pass.setPipeline(ticking);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(stepping);
      pass.dispatchWorkgroups(BLOCK_GROUPS.x, BLOCK_GROUPS.y);
      for (const flow of flows) {
        pass.setPipeline(flow);
        pass.dispatchWorkgroups(BLOCK_GROUPS.x, ROW_GROUPS);
      }
    },
    destroy() {
      tick.buffer.destroy();
    },
  };
}
