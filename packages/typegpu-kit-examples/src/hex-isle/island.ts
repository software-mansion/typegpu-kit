/** Binary glTF models from Kay Lousberg's KayKit Medieval Hexagon Pack (CC0). */
export const MODELS = [
  'grass',
  'water',
  'coast-a',
  'coast-b',
  'coast-c',
  'coast-d',
  'home-a',
  'home-b',
  'church',
  'tavern',
  'windmill',
  'windmill-fan',
  'well',
  'market',
  'blacksmith',
  'trees-large',
  'trees-medium',
  'tree',
  'mountain',
  'hills',
  'rock',
  'cloud-big',
  'cloud-small',
] as const;

export type ModelName = (typeof MODELS)[number];

type Vec3 = [number, number, number];

/**
 * Maps hex tiles to characters spaced two columns apart. Odd rows shift one character right,
 * giving horizontal and diagonal neighbours. Land tiles start as grass, sea-facing edges become
 * beaches, and remaining tiles become sea.
 */
export const MAP = [
  '      . . . .        ',
  '     . T ^ Y . .     ',
  '    . Y ^ ^ T T .    ',
  '   . . n T . m . .   ',
  '  . . h . c . y . .  ',
  '   . H w t k . H .   ',
  '    . b . . h . .    ',
  '     . y . r . .     ',
  '      . . . .        ',
];

const STANDING: Record<string, ModelName> = {
  h: 'home-a',
  H: 'home-b',
  c: 'church',
  t: 'tavern',
  m: 'windmill',
  w: 'well',
  k: 'market',
  b: 'blacksmith',
  T: 'trees-large',
  Y: 'trees-medium',
  y: 'tree',
  '^': 'mountain',
  n: 'hills',
  r: 'rock',
};

/**
 * Buildings with illuminated windows. Excludes the market because its awning shares the window
 * atlas cell.
 */
const HOUSES = new Set<ModelName>([
  'home-a',
  'home-b',
  'church',
  'tavern',
  'windmill',
  'blacksmith',
]);

/** Neighbour offsets in map-character coordinates, starting at +x and turning toward +z. */
const AROUND = [
  [2, 0],
  [1, 1],
  [-1, 1],
  [-2, 0],
  [-1, -1],
  [1, -1],
] as const;

/** Beach models indexed by exposed edge count and first exposed edge. */
const BEACHES = [
  { model: 'coast-a', first: 1 },
  { model: 'coast-b', first: 1 },
  { model: 'coast-c', first: 0 },
  { model: 'coast-d', first: 5 },
] as const;

/** Windmill hub position from the glTF nodes. */
const HUB: Vec3 = [0, 0.957, 0.332];
/** Radians a second the windmill's fan turns. */
const FAN_SPIN = 1.2;

/** Minimum and maximum window brightness across five night-time levels. */
const GLOW = [0.6, 1] as const;

/** World units above a house's tile where its lamp hangs. */
const LAMP_HEIGHT = 0.4;

/** World units from the island's centre that the sea reaches. */
const SEA = 12;

/** World units the clouds drift before they come round again. */
export const LOOP = 44;

const CLOUDS = [
  { model: 'cloud-big', position: [-9, 5.5, -3], drift: 0.5 },
  { model: 'cloud-small', position: [4, 6.2, 4], drift: 0.65 },
  { model: 'cloud-big', position: [16, 5.2, 7], drift: 0.55 },
  { model: 'cloud-small', position: [-18, 6.5, 1], drift: 0.6 },
] as const;
/** Cloud scale multiplier relative to the KayKit model. */
const CLOUD_SCALE = 1.3;

export interface Piece {
  readonly model: ModelName;
  readonly position: Vec3;
  /** Radians around +y. */
  readonly yaw: number;
  readonly scale?: number;
  /** Radians a second about the mesh's z axis. */
  readonly spin?: number;
  /** World units a second along +x, round the loop. */
  readonly drift?: number;
  /** Window brightness at night. */
  readonly glow?: number;
}

export interface Island {
  readonly pieces: readonly Piece[];
  /** Lamp position above each house. */
  readonly lamps: readonly Vec3[];
}

const turned = ([x, y, z]: Vec3, yaw: number): Vec3 => [
  x * Math.cos(yaw) + z * Math.sin(yaw),
  y,
  z * Math.cos(yaw) - x * Math.sin(yaw),
];

/** Builds `map` around the origin with sea extending to `sea` world units. */
export function layout(map: readonly string[] = MAP, sea = SEA): Island {
  const land = new Map<string, string>();
  let left = Infinity;
  let right = -Infinity;

  map.forEach((line, row) => {
    for (let column = 0; column < line.length; column++) {
      const tile = line[column];
      if (tile === ' ') continue;
      if ((column + row) % 2 !== 0) {
        throw new Error(`The tile at column ${column}, row ${row} is off the grid.`);
      }
      if (tile !== '.' && !STANDING[tile]) throw new Error(`No model stands for '${tile}'.`);

      land.set(`${column},${row}`, tile);
      left = Math.min(left, column);
      right = Math.max(right, column);
    }
  });

  const middle = { column: (left + right) / 2, row: (map.length - 1) / 2 };
  const at = (column: number, row: number): Vec3 => [
    column - middle.column,
    0,
    (row - middle.row) * Math.sqrt(3),
  ];

  const pieces: Piece[] = [];
  const lamps: Vec3[] = [];

  for (const [key, tile] of land) {
    const [column, row] = key.split(',').map(Number) as [number, number];
    const position = at(column, row);
    const shore = AROUND.map(([dc, dr]) => !land.has(`${column + dc},${row + dr}`));
    const edges = shore.filter(Boolean).length;

    if (edges === 0) {
      pieces.push({ model: 'grass', position, yaw: 0 });
    } else {
      // Rotate the beach to align its exposed edges with the map
      const first = shore.findIndex((wet, edge) => wet && !shore[(edge + 5) % 6]);
      const beach = BEACHES[edges - 1];
      const run = shore.every((wet, edge) => wet === (edge - first + 6) % 6 < edges);
      if (!beach || !run || tile !== '.') {
        throw new Error(
          tile === '.'
            ? `The tile at column ${column}, row ${row} has no beach that fits its coast.`
            : `The ${STANDING[tile]} at column ${column}, row ${row} stands on a beach.`,
        );
      }

      pieces.push({ model: beach.model, position, yaw: ((beach.first - first) * Math.PI) / 3 });
    }

    const model = STANDING[tile];
    if (!model) continue;

    // Vary rotation and window brightness by position
    const yaw = (((column * 3 + row * 5) % 6) * Math.PI) / 3;
    const step = ((column * 7 + row * 3) % 5) / 4;
    const glow = HOUSES.has(model) ? GLOW[0] + (GLOW[1] - GLOW[0]) * step : 0;
    pieces.push({ model, position, yaw, glow });

    if (HOUSES.has(model)) lamps.push([position[0], LAMP_HEIGHT, position[2]]);
    if (model === 'windmill') {
      const [x, y, z] = turned(HUB, yaw);
      pieces.push({
        model: 'windmill-fan',
        position: [position[0] + x, y, position[2] + z],
        yaw,
        spin: FAN_SPIN,
      });
    }
  }

  const rows = Math.ceil(sea / Math.sqrt(3));
  for (let row = Math.round(middle.row) - rows; row <= middle.row + rows; row++) {
    for (let column = Math.floor(middle.column - sea); column <= middle.column + sea; column++) {
      if ((column + row) % 2 !== 0 || land.has(`${column},${row}`)) continue;

      const position = at(column, row);
      if (Math.hypot(position[0], position[2]) <= sea) {
        pieces.push({ model: 'water', position, yaw: 0 });
      }
    }
  }

  for (const cloud of CLOUDS) {
    pieces.push({ ...cloud, position: [...cloud.position], yaw: 0, scale: CLOUD_SCALE });
  }

  return { pieces, lamps };
}
