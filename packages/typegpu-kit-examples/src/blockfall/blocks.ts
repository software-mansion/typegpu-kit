import { instances, materials, meshPool, type Frame } from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';

/** Seconds a move takes on screen. */
const MOVE_TIME = 0.08;

/** Seconds a cleared block flashes before it goes. */
export const CLEAR_TIME = 0.3;

/** Each kind's pigment, linear: vermilion, amber, saffron, jade, turquoise, lapis and amethyst. */
export const PALETTE = [
  d.vec3f(0.85, 0.22, 0.14),
  d.vec3f(0.95, 0.5, 0.12),
  d.vec3f(0.95, 0.78, 0.2),
  d.vec3f(0.18, 0.62, 0.36),
  d.vec3f(0.08, 0.6, 0.7),
  d.vec3f(0.2, 0.32, 0.85),
  d.vec3f(0.55, 0.25, 0.8),
].map((color) => srgbToLinear(color));

/** Edge length slightly below cell size, leaving gaps between blocks. */
const SIZE = 0.94;
/** The radius of a block's bevel, in cells. */
const BEVEL = 0.035;

/** Seconds after blocks start to change until they hold still: a flash, plus a frame or two. */
const SETTLE = Math.max(MOVE_TIME, CLEAR_TIME) + 0.05;

/** Final flash brightness exceeds white so cleared blocks bloom. */
const FLASH = 6;

/** A ghost's opacity, which also tells its material from a block's. */
const GHOST_OPACITY = 0.3;
/** Initial ghost positions below the well, hidden until a piece moves them. */
const HIDDEN: Cell = [0, -2, 0];

/** Frame-clock time in seconds. */
const clock = tgpu.accessor(d.f32);

/**
 * Movement progress from 0 to 1 after `since` seconds. The CPU also uses it to resume motion
 * from the visible position.
 */
const eased = (since: number) => {
  'use gpu';
  return std.smoothstep(0, 1, since / MOVE_TIME);
};

/** Flash progress from 0 to 1. */
const flashed = (clearedAt: number) => {
  'use gpu';
  if (clearedAt < 0) return d.f32(0);
  return std.saturate((clock.$ - clearedAt) / CLEAR_TIME);
};

/** How much of its colour a block gives off, settled and falling, so the piece in play stands out. */
const LIFT = { settled: 0, falling: 0.08 };
/**
 * The paint: its roughness and how far it varies from block to block, how much pale stone mutes
 * the pigment, how much lighter or darker a block's coat may be, and how far the chamfers wear
 * through to the stone.
 */
const PAINT = { roughness: 0.8, spread: 0.12, chalk: 0.08, shade: 0.14, worn: 0.5 };
/** The pale limestone the blocks are cut from. */
export const LIMESTONE = srgbToLinear(d.vec3f(0.76, 0.71, 0.62));

/** Chalky pigment of `color`, a shade lighter or darker by `seed`, from 0 to 1. */
export const pigmentOf = (color: d.v3f, seed: number) => {
  'use gpu';
  return std.mix(color, LIMESTONE, PAINT.chalk) * (1 + (seed - 0.5) * PAINT.shade);
};

/** The paint's roughness, varied by `seed`, from 0 to 1. */
export const roughnessOf = (seed: number) => {
  'use gpu';
  return PAINT.roughness + (seed - 0.5) * PAINT.spread;
};
/**
 * The brass inlaid in the plinth: its colour, how strongly it shows, its glow, and its line
 * half-width in cells.
 */
const BRASS = {
  color: srgbToLinear(d.vec3f(0.85, 0.66, 0.4)),
  strength: 0.8,
  glow: 0.08,
  width: 0.012,
};
/** The fewest pixels across a line is drawn; a line any thinner fades instead. */
const LINE_PIXELS = 1.5;
/** How many pixels either side of a line's edge its coverage softens over. */
const LINE_SOFTNESS = 0.75;
/** A ghost's fill, as a share of its paint, and the glow of its outline. */
const GHOST = { fill: 0.5, outline: 2.4 };

/**
 * How much of a line of half-width `width` covers a point `distance` from its middle, where a
 * `pixel` spans that many of the same units, as Ben Golus's pristine grid draws it. Its edges
 * soften over `LINE_SOFTNESS` pixels either side, and a line thinner than `LINE_PIXELS` is drawn
 * that wide and fades by how much thinner it is.
 *
 * A bright line on a dark floor blends in linear light, where a pixel half covered looks much
 * more than half as bright. Squaring the coverage evens that out, so a shallow line keeps one
 * weight as it crosses rows of pixels instead of swelling where it straddles two.
 */
const line = (distance: number, width: number, pixel: number) => {
  'use gpu';
  const half = width / pixel;
  const drawn = std.max(half, LINE_PIXELS / 2);
  const away = std.abs(distance) / pixel;
  const covered = 1 - std.smoothstep(drawn - LINE_SOFTNESS, drawn + LINE_SOFTNESS, away);
  return covered * covered * (half / drawn);
};

/** How much of a line covers a point `over` past its end, softened as `line` softens its edges. */
const capped = (over: number, width: number, pixel: number) => {
  'use gpu';
  const drawn = std.max(width / pixel, LINE_PIXELS / 2);
  return 1 - std.smoothstep(drawn - LINE_SOFTNESS, drawn + LINE_SOFTNESS, over / pixel);
};

/**
 * Painted limestone. Eases from `origin` to the target over `MOVE_TIME`, starting at `movedAt`,
 * and shrinks over `CLEAR_TIME` from `clearedAt`, negative before clearing, by the frame clock.
 * The paint wears through on the chamfers, and the piece in play gives off a little of its colour.
 * A ghost is a faint fill with a glowing outline. A floor, whose `grid` holds the well's half
 * width and depth, is dark polished stone with the well's cells inlaid in brass.
 */
const Block = instances.define('Block', {
  host: {
    origin: d.vec3f,
    movedAt: d.f32,
    clearedAt: d.f32,
    falling: d.f32,
    grid: d.vec2f,
    seed: d.f32,
  },

  deform: (vertex, block) => {
    'use gpu';
    const at = std.mix(block.origin, block.position, eased(clock.$ - block.movedAt));
    const cleared = flashed(block.clearedAt);

    // Divide world displacement by scale because deformation uses local block units
    return vertex.position * (1 - cleared * cleared) + (at - block.position) / block.scale;
  },

  carry: (vertexIndex) => {
    'use gpu';
    return d.vec4f(meshPool.positionAt(vertexIndex), 0);
  },

  material: (point, block, base, carried) => {
    'use gpu';
    const p = point.position;
    const n = std.abs(point.normal);

    // Derivatives run before any branch, in uniform control flow. They come from positions,
    // since the distances below jump between lines and where faces meet
    const across = d.vec2f(
      std.length(d.vec2f(std.dpdx(p.x), std.dpdy(p.x))),
      std.length(d.vec2f(std.dpdx(p.z), std.dpdy(p.z))),
    );
    const pixel = std.max(across, d.vec2f(1e-5));
    const span = std.max(std.length(std.fwidth(carried.xyz)) * 0.7, 1e-5);
    // Each line of the grid is a segment across the well, ending in a soft cap at its border
    const rel = p.xz - block.position.xz;
    const nearest = std.round(std.clamp(rel, block.grid * -1, block.grid));
    const off = rel - nearest;
    const over = std.max(std.abs(rel) - block.grid, d.vec2f(0));
    const grid = std.max(
      line(off.x, BRASS.width, pixel.x) * capped(over.y, BRASS.width, pixel.y),
      line(off.y, BRASS.width, pixel.y) * capped(over.x, BRASS.width, pixel.x),
    );
    const local = std.abs(carried.xyz);
    const face = std.select(
      std.select(local.xy, local.xz, n.y > n.z),
      local.zy,
      n.x > n.y && n.x > n.z,
    );
    const outline = line(std.max(face.x, face.y) - 0.5 + 0.035, 0.03, span);

    if (block.grid.x > 0) {
      const gold = std.select(d.f32(0), grid * BRASS.strength, point.normal.y > 0.5);
      const brass = materials.glossy(
        std.mix(base.color, BRASS.color, gold),
        std.mix(base.roughness, 0.3, gold),
      );
      brass.emission = BRASS.color * (BRASS.glow * gold);
      brass.metallic = gold;
      return brass;
    }

    // Chalky pigment, a shade lighter or darker on each block. The chamfer is where the normal
    // leaves its face's axis, since blocks never turn, and there the paint wears to the stone
    const pigment = pigmentOf(base.color, block.seed);
    const chamfer = std.smoothstep(0.04, 0.25, 1 - std.max(std.max(n.x, n.y), n.z));
    const color = std.mix(pigment, LIMESTONE, chamfer * PAINT.worn);

    if (block.opacity < 0.99) {
      const lit = materials.matte(color * GHOST.fill);
      lit.emission = base.color * (GHOST.outline * outline);
      return lit;
    }

    const lit = materials.glossy(color, roughnessOf(block.seed));
    const lift = std.mix(LIFT.settled, LIFT.falling, block.falling);
    const flash = FLASH * flashed(block.clearedAt);
    lit.emission = base.color * (lift + flash * 0.4) + d.vec3f(flash * 0.2);
    return lit;
  },
});

export type Cell = readonly [x: number, y: number, z: number];
type Triple = [number, number, number];
type BlockSet = instances.Instances<typeof Block>;

/** Block handle. Invalid after removal. */
export type BlockId = instances.Id<typeof Block>;

/** CPU row state for resuming motion from the visible position. */
interface Row {
  readonly set: BlockSet;
  readonly id: BlockId;
  kind: number;
  /** Origin, target and last requested cell centres. Motion starts at `movedAt`. */
  readonly origin: Triple;
  readonly to: Triple;
  readonly goal: Triple;
  movedAt: number;
  /** Flash start time. -1 means inactive, and NaN requests a start on the next update. */
  clearedAt: number;
}

export interface BlocksOptions {
  /** Maximum solid-block count. */
  readonly capacity: number;
  /** One ghost block per piece cube. */
  readonly ghosts: number;
  /** Frame-clock time in seconds. */
  readonly time: TgpuUniform<d.F32>;
  /** A slab drawn with the solid blocks, by its centre and size. */
  readonly plinth: { readonly at: Cell; readonly size: Cell; readonly color: d.v3f };
  /** The well's width and depth, which the plinth's grid covers, centred on the plinth. */
  readonly grid: readonly [width: number, depth: number];
  /** The scenery's meshes, added to the same pool. */
  readonly scenery?: readonly meshes.Geometry[];
}

/**
 * Draws carved solid blocks, their plinth and landing ghosts as rounded-box instances. The GPU
 * eases blocks between cells. Row writes occur only when motion or a flash starts.
 */
export function createBlocks(root: TgpuRoot, options: BlocksOptions) {
  const { capacity } = options;

  const shape = meshes.roundedBox({ radius: BEVEL });
  const [width, height, depth] = options.plinth.size;
  const slab = meshes.roundedBox({ width, height, depth, radius: 0.12 });
  const scenery = options.scenery ?? [];
  const pool = meshPool.create(root, meshPool.roomFor([shape, slab, ...scenery]));
  const cube = pool.add(shape);
  const plinth = pool.add(slab);
  const sceneryMeshes = scenery.map((mesh) => pool.add(mesh));

  const solid = instances.create(root, pool, { kind: Block, capacity: capacity + 1 });
  const ghosts = instances.create(root, pool, { kind: Block, capacity: options.ghosts });
  const timed: Inject = (cfg) => cfg.with(clock, options.time);
  const solidRenderer = solid.renderer().pipe(timed);
  const ghostRenderer = ghosts.renderer({ blend: 'translucent' }).pipe(timed);

  const rows = new Map<BlockId, Row>();
  const starting = new Set<Row>();
  /** When solid blocks last started to change, and whether one came or went since `update`. */
  let stirredAt = -Infinity;
  let stirred = false;

  const written = {
    position: [0, 0, 0] as Triple,
    origin: [0, 0, 0] as Triple,
    movedAt: 0,
    clearedAt: -1,
  };

  const write = (row: Row) => {
    written.position = row.to;
    written.origin = row.origin;
    written.movedAt = row.movedAt;
    written.clearedAt = row.clearedAt;
    row.set.write(row.id, written);
  };

  let placed = 0;
  const placeBlock = (set: BlockSet, kind: number, cell: Cell, opacity = 1): Row => {
    const centre: Triple = [cell[0] + 0.5, cell[1] + 0.5, cell[2] + 0.5];
    const id = set.add(cube, {
      position: centre,
      scale: SIZE,
      ...materials.matte(PALETTE[kind]),
      // Golden-ratio steps spread the coats' shades evenly
      seed: (placed++ * 0.618034) % 1,
      opacity,
      origin: centre,
      movedAt: 0,
      clearedAt: -1,
    });

    return {
      set,
      id,
      kind,
      origin: [...centre],
      to: [...centre],
      goal: [...centre],
      movedAt: 0,
      clearedAt: -1,
    };
  };

  /** Sets the row's goal to `cell`, so the next `update` starts it easing there. */
  const ease = (row: Row, cell: Cell) => {
    for (let axis = 0; axis < 3; axis++) row.goal[axis] = cell[axis] + 0.5;
    starting.add(row);
  };

  /** Starts easing the row from its visible position to its goal at `time`. */
  const restart = (row: Row, time: number) => {
    const moved = eased(time - row.movedAt);
    for (let axis = 0; axis < 3; axis++) {
      row.origin[axis] = row.origin[axis] + (row.to[axis] - row.origin[axis]) * moved;
      row.to[axis] = row.goal[axis];
    }
    row.movedAt = time;
  };

  /** Whether the requested cell differs from the current motion target. */
  const redirected = (row: Row) =>
    row.goal[0] !== row.to[0] || row.goal[1] !== row.to[1] || row.goal[2] !== row.to[2];

  const at = [...options.plinth.at] as Triple;
  solid.add(plinth, {
    position: at,
    ...materials.glossy(options.plinth.color, 0.4),
    origin: at,
    movedAt: 0,
    clearedAt: -1,
    grid: [options.grid[0] / 2, options.grid[1] / 2],
  });

  const ghostRows = Array.from({ length: options.ghosts }, () =>
    placeBlock(ghosts, 0, HIDDEN, GHOST_OPACITY),
  );

  return {
    /** Adds a block of `kind` at `cell` without easing. */
    add(kind: number, cell: Cell) {
      const row = placeBlock(solid, kind, cell);
      rows.set(row.id, row);
      stirred = true;
      return row.id;
    },

    /** Eases a block from its visible position to `cell`. */
    move(id: BlockId, cell: Cell) {
      ease(rows.get(id)!, cell);
    },

    /** Starts a clear flash lasting `CLEAR_TIME` seconds. */
    flash(id: BlockId) {
      const row = rows.get(id)!;
      row.clearedAt = Number.NaN;
      starting.add(row);
    },

    kindOf: (id: BlockId) => rows.get(id)!.kind,

    /** Marks a block as part of the falling piece, which shines brightest. */
    fall(id: BlockId, falling: boolean) {
      solid.write(id, { falling: falling ? 1 : 0 });
    },

    remove(id: BlockId) {
      stirred = true;
      starting.delete(rows.get(id)!);
      rows.delete(id);
      solid.remove(id);
    },

    /** Eases ghost `i` to `cell`, in the colour of `kind`. */
    ghost(i: number, kind: number, cell: Cell) {
      const row = ghostRows[i];
      if (row.kind !== kind) ghosts.write(row.id, { color: PALETTE[kind] });
      row.kind = kind;
      ease(row, cell);
    },

    /**
     * Whether no solid block has moved, flashed, come or gone since a little over `CLEAR_TIME`
     * before `time`, so shadows drawn since then still hold.
     */
    still: (time: number) => time - stirredAt > SETTLE,

    /** Writes rows starting motion or flashes, then flushes solid and ghost instances. */
    update(frame: Frame) {
      if (stirred || starting.size > 0) stirredAt = frame.time;
      stirred = false;
      for (const row of starting) {
        if (redirected(row)) restart(row, frame.time);
        if (Number.isNaN(row.clearedAt)) row.clearedAt = frame.time;
        write(row);
      }
      starting.clear();

      solid.flush();
      ghosts.flush();
    },

    pool,
    sceneryMeshes,
    solid: solidRenderer,
    /** Add ghosts after solid blocks so they blend over them. */
    ghosts: ghostRenderer,

    destroy() {
      solid.destroy();
      ghosts.destroy();
      pool.destroy();
    },
  };
}

export type Blocks = ReturnType<typeof createBlocks>;
