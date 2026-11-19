/**
 * One character per tile, top row first: `#` ground, `=` bricks, `o` a coin, `w` a walker,
 * `b` a bat, `@` the start and `F` the flag.
 */
export const MAP = [
  '',
  '',
  '',
  '',
  '',
  '                  ooo                                ooooo',
  '                                                       w      b   o o o                     b',
  '          o o o   ===                  b             =====                    b   o o o',
  '                          o               o                       =====                     o',
  '          =====          o o             o o   ====                               =====    o o',
  '                                    ##                                    ##',
  '   @                 w         w  ####        w           w            w  ##    w       w        ##  F',
  '#########################   #############   #################  ############################   ##########',
  '#########################   #############   #################  ############################   ##########',
];

export const EMPTY = 0;
export const GROUND = 1;
export const BRICKS = 2;

export const WALKER = 0;
export const BAT = 1;

type Point = readonly [x: number, y: number];

export interface Spawn {
  readonly kind: typeof WALKER | typeof BAT;
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

  /** Centre of the last tile from `x` in direction `step` that is open with floor below. */
  const patrolEnd = (x: number, y: number, step: number) => {
    let end = Math.floor(x);
    while (!solidAt(level, end + step, y) && solidAt(level, end + step, y - 1)) end += step;
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
    critters: [...(marks.w ?? []).map(spawn(WALKER)), ...(marks.b ?? []).map(spawn(BAT))],
  };
}
