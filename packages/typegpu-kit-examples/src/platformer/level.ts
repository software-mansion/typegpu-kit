/**
 * One character per tile, top row first: `#` ground, `=` bricks, `o` a coin, `w` a walker,
 * `f` a fork, `b` a bat, `@` the start and `F` the flag.
 */
export const MAP = [
  '',
  '',
  '',
  '                                                                                  ====',
  '                                                                                  ====',
  '                  ooo                                ooooo                        ====                    o o',
  '                                                       w      b   o o o           ====      b',
  '          o o o   ===                  b             =====                    b   ====                    ooo',
  '                          o               o                       =====           ====      o             ###',
  '          =====          o o             o o   ====                               ====     o o            ###',
  '                                    ##                                    ##      ====                    ###',
  '   @                 w         w  ####        f           w            w  ##    f  oo   w        ##       ###      F',
  '#########################   #############   #################  ############################   ########################',
  '#########################   #############   #################  ############################   ########################',
];

export const EMPTY = 0;
export const GROUND = 1;
export const BRICKS = 2;

export const WALKER = 0;
export const BAT = 1;
/** A walker with spikes on top, which only a slide defeats. */
export const FORK = 2;

type Point = readonly [x: number, y: number];

export interface Spawn {
  readonly kind: typeof WALKER | typeof BAT | typeof FORK;
  /** Centre of its tile. */
  readonly at: Point;
  /** Walker travel limits between walls and floor edges. */
  readonly patrol: Point;
}

export interface Tiles {
  readonly width: number;
  readonly height: number;
  /** Tile kinds, row by row from the bottom. */
  readonly tiles: Uint32Array;
}

export interface Level extends Tiles {
  /** Object positions at tile centres. */
  readonly start: Point;
  readonly flag: Point;
  readonly coins: readonly Point[];
  readonly critters: readonly Spawn[];
}

const TILE_KINDS: Record<string, number> = { '#': GROUND, '=': BRICKS };

/** Whether tile `x, y` blocks the hero. Level sides are walls, and the bottom is open. */
export function solidAt(level: Tiles, x: number, y: number) {
  if (x < 0 || x >= level.width) return true;
  if (y < 0 || y >= level.height) return false;
  return level.tiles[y * level.width + x] !== EMPTY;
}

/** Reads tiles, marks and walker patrols from a map such as `MAP`. */
export function parseLevel(map: readonly string[]): Level {
  const width = Math.max(...map.map((row) => row.length));
  const height = map.length;
  const tiles = new Uint32Array(width * height);

  const marks: Record<string, Point[]> = {};
  map.forEach((row, line) => {
    const y = height - 1 - line;
    for (let x = 0; x < row.length; x++) {
      const char = row[x];
      const kind = TILE_KINDS[char];
      if (kind !== undefined) tiles[y * width + x] = kind;
      else if (char !== ' ') (marks[char] ??= []).push([x + 0.5, y + 0.5]);
    }
  });

  const level: Tiles = { width, height, tiles };

  /** Whether a walker on tile `x, y` stands on a floor, with room above for its body. */
  const walkable = (x: number, y: number) =>
    !solidAt(level, x, y) && !solidAt(level, x, y + 1) && solidAt(level, x, y - 1);

  /** Centre of the last walkable tile from `x` in direction `step`. */
  const patrolEnd = (x: number, y: number, step: number) => {
    let end = Math.floor(x);
    while (walkable(end + step, y)) end += step;
    return end + 0.5;
  };

  const spawn =
    (kind: Spawn['kind']) =>
    ([x, y]: Point): Spawn => ({
      kind,
      at: [x, y],
      patrol: [patrolEnd(x, Math.floor(y), -1), patrolEnd(x, Math.floor(y), 1)],
    });

  return {
    ...level,
    start: marks['@']?.[0] ?? [0.5, 0.5],
    flag: marks.F?.[0] ?? [width - 0.5, 0.5],
    coins: marks.o ?? [],
    critters: [
      ...(marks.w ?? []).map(spawn(WALKER)),
      ...(marks.f ?? []).map(spawn(FORK)),
      ...(marks.b ?? []).map(spawn(BAT)),
    ],
  };
}
