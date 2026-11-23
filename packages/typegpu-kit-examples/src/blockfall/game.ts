import { clock, coroutines, type Frame, type Coroutines } from '@typegpu/kit';
import { CLEAR_TIME, type BlockId, type Blocks } from './blocks.ts';

export const WIDTH = 6;
export const DEPTH = 6;
export const HEIGHT = 12;
/** Cells per well layer and total cells. */
export const LAYER = WIDTH * DEPTH;
export const CELLS = LAYER * HEIGHT;
export const PIECE = 4;

type Cell = [x: number, y: number, z: number];

/** Spawn origin at the well centre, two cells below the top. */
const SPAWN: Cell = [WIDTH / 2 - 1, HEIGHT - 2, DEPTH / 2 - 1];

/** Each piece has `PIECE` cubes. The last two require three dimensions. */
// oxfmt-ignore
const SHAPES: readonly (readonly Cell[])[] = [
  [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
  [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]],
  [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]],
  [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]],
  [[-1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0]],
  [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]],
  [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
];

/** Offsets tried, in order, when a rotation collides. */
// oxfmt-ignore
const KICKS: readonly Cell[] = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0]];

/** Score by cleared-layer count, from zero to four, multiplied by level. */
const POINTS = [0, 100, 300, 600, 1000];
/** Layers to clear for each level. */
const LINES_PER_LEVEL = 6;

/** Initial drop interval in seconds, multiplier per level and minimum interval. */
const DROP = { first: 0.8, keep: 0.85, least: 0.1 };
/** Delay in seconds between game-over layer flashes, ordered top down. */
const CASCADE = 0.04;

/** Seconds between drops at `level`. */
const dropStep = (level: number) => Math.max(DROP.least, DROP.first * DROP.keep ** (level - 1));

/** Linear grid index for `x, y, z`, ordered by layer from the bottom. */
const at = (x: number, y: number, z: number) => y * LAYER + z * WIDTH + x;

/** Whether column `x, z` lies inside the well's walls. */
const inside = (x: number, z: number) => x >= 0 && x < WIDTH && z >= 0 && z < DEPTH;

/** Rotates a cell a quarter turn around `axis` (0 x, 1 y, 2 z), counter-clockwise for turn 1. */
function rotate([x, y, z]: Cell, axis: number, turn: number): Cell {
  if (axis === 0) return [x, -turn * z, turn * y];
  if (axis === 1) return [turn * z, y, -turn * x];
  return [-turn * y, turn * x, z];
}

export interface Score {
  readonly score: number;
  readonly lines: number;
  readonly level: number;
}

export interface BlockfallOptions {
  readonly blocks: Blocks;
  /** Runs layer clears and the game-over cascade. */
  readonly scheduler: Coroutines;
  /** Receives the score at startup and after each clear or game over. */
  readonly onScore: (score: Score) => void;
  /** Receives each cleared block's cell and kind as it breaks. Default none. */
  readonly onBreak?: (x: number, y: number, z: number, kind: number) => void;
}

/** Runs game rules on the CPU. */
export function createBlockfall(options: BlockfallOptions) {
  const { blocks, scheduler, onScore, onBreak } = options;
  const grid: (BlockId | undefined)[] = Array.from({ length: CELLS });
  /** The block ids of layer `y`, from the bottom. */
  const layer = (y: number) => grid.slice(y * LAYER, (y + 1) * LAYER);
  /** The layers with no gaps, from the bottom. */
  const fullLayers = () =>
    Array.from({ length: HEIGHT }, (_, y) => y).filter((y) =>
      layer(y).every((id) => id !== undefined),
    );

  /** Whether `cell` lies inside the walls, on or above the floor, and holds no block. */
  const free = ([x, y, z]: Cell) =>
    inside(x, z) && y >= 0 && (y >= HEIGHT || grid[at(x, y, z)] === undefined);

  /** The falling piece: its kind, its cells around its origin, and their blocks. */
  const piece = { kind: 0, cells: [] as Cell[], origin: [0, 0, 0] as Cell, ids: [] as BlockId[] };

  const state = { score: 0, lines: 0, level: 1 };
  /** Drops the piece a cell a step, at the level's rate. */
  const gravity = clock.fixedStep({ step: dropStep(1), maxSteps: 1 });
  // Until the game spawns the first piece
  gravity.paused = true;
  const startLevel = (level: number) => {
    state.level = level;
    gravity.step = dropStep(level);
    gravity.reset();
  };

  /** `cell` of a piece at `origin`, in the well. */
  const world = ([x, y, z]: Cell, [ox, oy, oz] = piece.origin): Cell => [x + ox, y + oy, z + oz];

  const fits = (cells: readonly Cell[], origin: Cell) =>
    cells.every((cell) => free(world(cell, origin)));

  const landing = (): Cell => {
    const [x, y, z] = piece.origin;
    let floor = y;
    while (fits(piece.cells, [x, floor - 1, z])) floor -= 1;
    return [x, floor, z];
  };

  /** Eases piece blocks toward their target cells and ghosts toward landing cells. */
  const show = () => {
    const below = landing();

    piece.cells.forEach((cell, i) => {
      blocks.move(piece.ids[i], world(cell));
      blocks.ghost(i, piece.kind, world(cell, below));
    });
  };

  /** Tries cells and origin, easing blocks to the target if they fit. */
  const attempt = (cells: Cell[], origin: Cell) => {
    if (!fits(cells, origin)) return false;

    piece.cells = cells;
    piece.origin = origin;
    show();

    return true;
  };

  /** Starts a new piece above the well. Returns false when it has no room, ending the game. */
  const spawn = () => {
    piece.kind = Math.floor(Math.random() * SHAPES.length);
    piece.cells = SHAPES[piece.kind].map((cell): Cell => [...cell]);
    piece.origin = [...SPAWN];
    if (!fits(piece.cells, piece.origin)) return false;

    const above: Cell = [piece.origin[0], HEIGHT + 1, piece.origin[2]];
    piece.ids = piece.cells.map((cell) => blocks.add(piece.kind, world(cell, above)));
    for (const id of piece.ids) blocks.fall(id, true);
    show();
    gravity.paused = false;
    return true;
  };

  /** Sets the piece into the grid where it is and stops its fall. */
  const lock = () => {
    piece.cells.forEach((cell, i) => {
      const [x, y, z] = world(cell);
      grid[at(x, y, z)] = piece.ids[i];
      blocks.fall(piece.ids[i], false);
    });
    gravity.paused = true;
  };

  const playing = () => !gravity.paused;
  /** Resumes once `lock` stops the falling piece. */
  const landed = coroutines.wait({ ready: () => !playing() });
  const flashed = coroutines.seconds(CLEAR_TIME);
  const cascaded = coroutines.seconds(CASCADE);

  /** Flashes and removes full layers, then drops blocks above them. */
  function* clear(layers: readonly number[]) {
    for (const y of layers) {
      for (const id of layer(y)) blocks.flash(id!);
    }

    yield* flashed;

    let removed = 0;
    for (let y = 0; y < HEIGHT; y++) {
      const cleared = layers.includes(y);
      for (let z = 0; z < DEPTH; z++) {
        for (let x = 0; x < WIDTH; x++) {
          const id = grid[at(x, y, z)];
          if (id === undefined) continue;
          grid[at(x, y, z)] = undefined;

          if (cleared) {
            onBreak?.(x, y, z, blocks.kindOf(id));
            blocks.remove(id);
          } else {
            grid[at(x, y - removed, z)] = id;
            blocks.move(id, [x, y - removed, z]);
          }
        }
      }

      if (cleared) removed += 1;
    }

    state.lines += layers.length;
    state.score += POINTS[layers.length] * state.level;
    const level = 1 + Math.floor(state.lines / LINES_PER_LEVEL);
    if (level !== state.level) startLevel(level);
    onScore(state);
  }

  /** Flashes the well layer by layer from the top, then empties it and resets the score. */
  function* gameOver() {
    for (let y = HEIGHT - 1; y >= 0; y--) {
      for (const id of layer(y)) if (id !== undefined) blocks.flash(id);
      yield* cascaded;
    }

    yield* flashed;
    for (const id of grid) if (id !== undefined) blocks.remove(id);
    grid.fill(undefined);

    Object.assign(state, { score: 0, lines: 0 });
    startLevel(1);
    onScore(state);
  }

  /** Plays game after game: drops pieces and clears the layers they fill until one has no room. */
  function* play() {
    for (;;) {
      while (spawn()) {
        yield* landed;
        const full = fullLayers();
        if (full.length > 0) yield* clear(full);
      }
      yield* gameOver();
    }
  }

  scheduler.start(play());
  onScore(state);

  return {
    /** Advances the drop clock once per frame. Each drop moves the piece one cell. */
    update(frame: Frame) {
      for (let steps = gravity.update(frame.dt); steps > 0; steps--) {
        const [x, y, z] = piece.origin;
        if (!attempt(piece.cells, [x, y - 1, z])) lock();
      }
    },

    move(dx: number, dz: number) {
      if (!playing()) return;

      const [x, y, z] = piece.origin;
      attempt(piece.cells, [x + dx, y, z + dz]);
    },

    /** A quarter turn around `axis` (0 x, 1 y, 2 z); `turn` is 1 or -1. */
    rotate(axis: number, turn: number) {
      if (!playing()) return;

      const cells = piece.cells.map((cell) => rotate(cell, axis, turn));
      KICKS.some((kick) => attempt(cells, world(kick)));
    },

    drop() {
      if (!playing()) return;

      attempt(piece.cells, landing());
      lock();
    },
  };
}
