import { instances, meshPool } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import type { TgpuRoot } from 'typegpu';
import { quat, vec3, type Quat } from 'wgpu-matrix';
import type { Model } from '../models.ts';

type Vec3 = [number, number, number];

/**
 * The parts of `models/market.glb`. Buildings and props are from Kay Lousberg's KayKit Medieval
 * Hexagon Pack, potions, the mug and the book from KayKit Adventurers, and the table, banners,
 * string lights, lamps, candles and logs from Kenney's Fantasy Town, Holiday, Graveyard and
 * Survival kits. Lamp glass, bulbs, flames and window panes are split into materials named
 * `glow`.
 */
export const MODELS = [
  'market-red',
  'market-blue',
  'market-yellow',
  'tavern',
  'well',
  'home-a',
  'home-b',
  'barrel',
  'bucket',
  'crate-a',
  'crate-open',
  'crate-fruit',
  'crate-goods',
  'sack',
  'flag-red',
  'flag-blue',
  'tree',
  'potion-huge-blue',
  'potion-large-red',
  'potion-medium-green',
  'potion-small-orange',
  'potion-large-green',
  'potion-medium-red',
  'potion-small-blue',
  'stall-red',
  'post',
  'table',
  'banner-green',
  'lights',
  'street-lamp',
  'wall-lamp',
  'lantern',
  'candles',
  'logs',
] as const;

export type ModelName = (typeof MODELS)[number];
type Part = Pick<Model, 'meshes' | 'images'>;
export type Models = Readonly<Record<ModelName, Part>>;

/** KayKit's hexagon buildings and props stand this many times their size beside the characters. */
const HEX = 4.5;
/** Kenney's town pieces stand this many times their size. */
const TOWN = 2.6;

/** The potion table, by the eye: where its middle stands, and how it is scaled. */
const TABLE = { at: [0.9, 0, 3.4] as Vec3, scale: 2.6 };
/** Half the table's length along the street's width, and half its depth, once scaled. */
const HALF_LENGTH = 0.5 * TABLE.scale;
const HALF_DEPTH = 0.325 * TABLE.scale;
/** Where the fire burns, on top of a barrel in the street. */
export const FIRE: Vec3 = [4.6, 1.1, -4.6];
/**
 * The rows of string lights across the street: where each crosses and how high its wire is tied
 * to a pole at either side. The lantern hangs from the second row's wire.
 */
const ROWS = [
  { z: 5.2, high: 4.4 },
  { z: 0.6, high: 4.7 },
  { z: -5.4, high: 4.4 },
  { z: -9.0, high: 4.7 },
  { z: -14.8, high: 4.4 },
];
const LANTERN_ROW = 1;
/** The poles stand this far from the street's middle, this thick, and rise this far over the tie. */
const POLE_X = 4.3;
const POLE = 1.6;
const POLE_OVER = 0.25;
/** Half the post model's width, before it is scaled. */
const POST_HALF_WIDTH = 0.08;
/** Swags of lights on each wire, and how far the wire sags between its poles. */
const SWAGS = 5;
const SAG = 0.35;
/** The wires the lights hang from, and the lantern's cord. */
const WIRE = { color: [0.02, 0.02, 0.02] as Vec3, roughness: 0.8 };
const CORD = 1.2;
/** How much the lantern on the cord is scaled. */
const HUNG_LANTERN = 2.2;
/** Where a wall lamp's bulb sits in the lamp's own space. */
const WALL_LAMP_BULB: Vec3 = [0, -0.45, 0.42];

/** How bright each glowing part is, as a factor on its texture's colour or, for panes, alone. */
const WINDOW: Vec3 = [5, 2.4, 0.7];
const LAMP_GLASS: Vec3 = [5, 4, 3];
const BULBS: Vec3 = [5, 3.2, 1.2];
const RED_BULBS: Vec3 = [7, 1.2, 0.35];
const CANDLE: Vec3 = [5, 4, 3];
/** The embers' glow in the burning logs. */
const EMBERS: Vec3 = [0.5, 0.12, 0.02];
/** The faint light of each brew in its bottle. */
const BREW = {
  red: [0.035, 0.003, 0.003] as Vec3,
  blue: [0.003, 0.015, 0.05] as Vec3,
  green: [0.006, 0.035, 0.009] as Vec3,
  orange: [0.04, 0.015, 0.0015] as Vec3,
};

/** A light that casts no shadow. */
interface Bulb {
  readonly position: Vec3;
  readonly aim?: Vec3;
  /** Inner and outer half-angles in radians. */
  readonly cone?: readonly [number, number];
  readonly color: Vec3;
  readonly intensity: number;
  readonly range: number;
}

/** The spots under the markets' awnings, the candles on the table and the standing lamps. */
const AWNING_SPOT = {
  cone: [0.55, 1.1],
  color: [1, 0.7, 0.4],
  intensity: 3,
  range: 4.5,
} satisfies Omit<Bulb, 'position' | 'aim'>;
const CANDLE_LIGHT = { color: [1, 0.62, 0.3], intensity: 0.5, range: 2.4 } satisfies Partial<Bulb>;
const LANTERN_LIGHT = {
  color: [1, 0.66, 0.32],
  intensity: 1.2,
  range: 3.5,
} satisfies Partial<Bulb>;
const STREET_LIGHT = { color: [1, 0.72, 0.4], intensity: 2.5, range: 4.5 } satisfies Partial<Bulb>;

const UP = [0, 1, 0] as const;
const X_AXIS = [1, 0, 0] as const;
const Z_AXIS = [0, 0, 1] as const;
const DOWN = [0, -1, 0] as const;

/** Whether a mesh is a glowing part: lamp glass, a bulb, a flame or a window pane. */
export const isGlow = (mesh: Part['meshes'][number]) =>
  (mesh.material.name ?? '').endsWith('/glow');

/** The lowest and highest point of a model, before it is scaled. */
const extent = (model: Pick<Model, 'meshes'>) => {
  const low: Vec3 = [Infinity, Infinity, Infinity];
  const high: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const { positions } of model.meshes) {
    for (let i = 0; i < positions.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        low[k] = Math.min(low[k], positions[i + k]);
        high[k] = Math.max(high[k], positions[i + k]);
      }
    }
  }
  return { low, high };
};

interface Placing {
  readonly at: Vec3;
  readonly yaw?: number;
  /** Turns the model this way instead of by `yaw`. */
  readonly rotation?: Quat;
  readonly scale?: number;
  /** Stands the model's lowest point at `at[1]` rather than its origin. */
  readonly stand?: boolean;
  readonly roughness?: number;
  /** The glowing part's emission factor. */
  readonly glow?: Vec3;
  /** The body's own emission. */
  readonly emission?: Vec3;
  /** Draws the body without casting shadows, as lamps do. */
  readonly lamp?: boolean;
}

/**
 * The market's pool and instances, and what places each model in them. Solid things cast shadows;
 * lamps, their glass and the glowing panes don't.
 */
function createBuilder(root: TgpuRoot, models: Models) {
  const cord = meshes.cylinder({ radialSegments: 8 });
  const pool = meshPool.create(
    root,
    meshPool.roomFor([...MODELS.map((name) => models[name]), cord]),
  );

  // Each model is its body and, where it has one, its glowing part, which sets its own emission
  const bodies = new Map<ModelName, meshPool.Model>();
  const glows = new Map<ModelName, meshPool.Model>();
  for (const name of MODELS) {
    const { meshes: all, images } = models[name];
    const glow = all.filter(isGlow);
    bodies.set(name, pool.add({ meshes: all.filter((mesh) => !isGlow(mesh)), images }));
    if (glow.length > 0) glows.set(name, pool.add({ meshes: glow, images }));
  }
  const rope = pool.add(cord);
  const ready = Promise.all([...bodies.values(), ...glows.values()].map((model) => model.ready));

  const solid = instances.create(root, pool, { capacity: 512 });
  // The lantern and its cord swing, so they keep last frame's rows for TAA's motion
  const glowing = instances.create(root, pool, { capacity: 256, history: true });
  const extents = new Map(MODELS.map((name) => [name, extent(models[name])]));

  /** How far `placing` raises a model so its lowest point stands at `at[1]`. */
  const liftOf = (name: ModelName, { scale = HEX, stand = false }: Placing) =>
    stand ? -extents.get(name)!.low[1] * scale : 0;

  return {
    pool,
    solid,
    glowing,
    rope,
    ready,
    /** The lights that cast no shadow, in the order they are placed. */
    bulbs: [] as Bulb[],
    bodyOf: (name: ModelName) => bodies.get(name)!,
    glowOf: (name: ModelName) => glows.get(name)!,
    extentOf: (name: ModelName) => extents.get(name)!,
    topOf: (name: ModelName, scale: number) => extents.get(name)!.high[1] * scale,

    /** Places a model's body and glowing part. */
    place(name: ModelName, placing: Placing) {
      const { at, yaw = 0, scale = HEX, roughness, glow, emission, lamp = false } = placing;
      const position: Vec3 = [at[0], at[1] + liftOf(name, placing), at[2]];
      const rotation = placing.rotation ?? quat.fromAxisAngle(UP, yaw);
      const fields = {
        position,
        rotation,
        scale,
        ...(roughness === undefined ? {} : { roughness }),
      };
      (lamp ? glowing : solid).add(bodies.get(name)!, {
        ...fields,
        ...(emission ? { emission } : {}),
      });
      const lit = glows.get(name);
      if (lit) glowing.add(lit, { ...fields, ...(glow ? { emission: glow } : {}) });
    },

    /** Where a point given in a model's own space lands once placed. */
    onto(name: ModelName, placing: Placing, local: Vec3): Vec3 {
      const { at, yaw = 0, scale = HEX } = placing;
      const turned = vec3.transformQuat(vec3.scale(local, scale), quat.fromAxisAngle(UP, yaw));
      return [at[0] + turned[0], at[1] + liftOf(name, placing) + turned[1], at[2] + turned[2]];
    },
  };
}

type Builder = ReturnType<typeof createBuilder>;

/** The street: stalls down both sides, the tavern closing its end, houses, awnings and goods. */
function buildStreet({ place, bulbs }: Builder) {
  place('market-red', { at: [-7.2, 0, -0.5], yaw: Math.PI / 2 });
  place('market-blue', { at: [7.6, 0, -5.5], yaw: -Math.PI / 2 });
  place('market-yellow', { at: [-7.2, 0, -10.5], yaw: Math.PI / 2 });
  place('tavern', { at: [1.5, 0, -21], glow: WINDOW });
  place('home-b', { at: [8.5, 0, -14.5], yaw: -Math.PI / 2, glow: WINDOW });
  place('home-a', { at: [-7.5, 0, -18.5], yaw: Math.PI / 2, glow: WINDOW });
  place('well', { at: [-1.6, 0, -8.5], yaw: 0.4 });
  place('tree', { at: [-12, 0, -6], yaw: 0.7 });
  place('tree', { at: [12.5, 0, -10], yaw: 2.1, scale: 5 });
  place('stall-red', { at: [5.6, 0, -11.5], yaw: -Math.PI / 2, scale: TOWN });
  place('flag-red', { at: [-5.2, 0, 3.6], yaw: 0.6 });
  place('flag-blue', { at: [5.6, 0, -9.5], yaw: -0.4 });

  // Awning bulbs over each market's goods, spots looking down at them
  for (const [x, z, facing] of [
    [-4.6, -2.5, 1],
    [-4.6, 1.5, 1],
    [5.0, -7.5, -1],
    [5.0, -3.5, -1],
    [-4.6, -12.5, 1],
    [-4.6, -8.5, 1],
  ] as const) {
    bulbs.push({ position: [x, 2.6, z], aim: [facing * 0.5, -1, 0], ...AWNING_SPOT });
  }

  // The crates the fire throws towards the eye, and goods by the stalls
  const crates: [ModelName, number, number, number][] = [
    ['crate-a', 4.3, -2.8, 0.3],
    ['crate-a', 5.15, -3.0, -0.45],
    ['crate-open', 5.5, -2.3, 0.9],
    ['barrel', 3.7, -2.2, 0.4],
    ['sack', 5.1, -1.2, 0.8],
    ['crate-goods', -4.5, -10, 1.5],
    ['crate-a', -0.4, -13.5, 0.4],
    ['barrel', 0.4, -14.2, 0],
    ['sack', -0.9, -14.6, 1.1],
    ['barrel', -4.3, -3.6, 0],
    ['barrel', -4.4, -4.6, 0.5],
    ['crate-fruit', -3.6, 2.6, 1.4],
    ['crate-goods', -3.8, -0.4, 1.7],
    ['sack', -3.2, 1.2, 0.3],
    ['bucket', -0.6, -8.0, 0],
    ['barrel', 3.4, -15.5, 0.2],
    ['crate-a', 4.3, -16.2, 0.5],
  ];
  for (const [name, x, z, yaw] of crates) place(name, { at: [x, 0, z], yaw });
}

/**
 * The potion seller's table by the eye, its goods in a row along the front and candles behind.
 * Returns the height of its top and where along the street its front row stands.
 */
function buildTable({ place, onto, topOf, bulbs }: Builder) {
  place('table', { at: TABLE.at, yaw: Math.PI / 2, scale: TABLE.scale });
  const top = topOf('table', TABLE.scale);
  const front = TABLE.at[2] + HALF_DEPTH - 0.35;
  const potions: [ModelName, number, number, Vec3][] = [
    ['potion-medium-red', -1.0, 0.1, BREW.red],
    ['potion-small-orange', -0.62, 0.12, BREW.orange],
    ['potion-small-blue', -0.32, 0.1, BREW.blue],
    ['potion-large-green', 0.05, -0.05, BREW.green],
    ['potion-huge-blue', 0.46, -0.1, BREW.blue],
    ['potion-large-red', 0.84, -0.08, BREW.red],
    ['potion-medium-green', 1.16, 0.06, BREW.green],
  ];
  for (const [name, along, yaw, emission] of potions) {
    place(name, {
      at: [TABLE.at[0] + along, top, front + Math.sin(along * 3) * 0.08],
      yaw,
      scale: 1,
      stand: true,
      roughness: 0.15,
      emission,
    });
  }
  const candles: Placing = {
    at: [TABLE.at[0] - 0.75, top, TABLE.at[2] - 0.3],
    scale: 2,
    stand: true,
    glow: CANDLE,
  };
  place('candles', candles);
  bulbs.push({ position: onto('candles', candles, [0, 0.25, 0]), ...CANDLE_LIGHT });
  return { top, front };
}

/**
 * Lanterns standing on crates and barrels, the seller's lamp on a post at the table's end, the
 * street lamps and the banner with a lamp over it. Returns the bulbs of the lamps over the table
 * and the banner.
 */
function buildLamps({ place, onto, solid, bodyOf, bulbs }: Builder) {
  for (const [x, y, z] of [
    [-3.6, 0.95, 2.6],
    [-4.3, 0.95, -3.6],
    [3.7, 1.15, -2.2],
    [3.4, 0.95, -15.5],
  ] as const) {
    const standing: Placing = { at: [x, y, z], scale: 1.4, lamp: true, glow: LAMP_GLASS };
    place('lantern', standing);
    bulbs.push({ position: onto('lantern', standing, [0, 0.2, 0]), ...LANTERN_LIGHT });
  }

  // The seller's lamp, whose spot falls along the potions
  const corner: Vec3 = [TABLE.at[0] + HALF_LENGTH + 0.12, 0, TABLE.at[2] + 0.2];
  solid.add(bodyOf('post'), { position: corner, scale: [1.8, 2.9, 1.8] });
  const hook: Placing = {
    at: [corner[0] - POST_HALF_WIDTH * 1.8, 2.4, corner[2]],
    yaw: -Math.PI / 2,
    scale: 1.5,
    lamp: true,
    glow: LAMP_GLASS,
  };
  place('wall-lamp', hook);
  const counter = onto('wall-lamp', hook, WALL_LAMP_BULB);

  for (const at of [
    [-4.2, 0, -6.2],
    [3.6, 0, -12.6],
  ] as const) {
    const lamp: Placing = { at: [...at], scale: 1.7, glow: LAMP_GLASS };
    place('street-lamp', lamp);
    bulbs.push({ position: onto('street-lamp', lamp, [0, 1.5, 0]), ...STREET_LIGHT });
  }

  // The red stall's banner on a post, with a lamp on an arm over it
  const bannerPost: Vec3 = [-4.2, 0, 2.6];
  const bannerPostScale = 3.8;
  const postFace = bannerPost[2] + POST_HALF_WIDTH * bannerPostScale;
  place('post', { at: bannerPost, scale: bannerPostScale });
  place('banner-green', {
    at: [bannerPost[0], 0.9, postFace - 0.38 * TOWN],
    yaw: -Math.PI / 2,
    scale: TOWN,
  });
  const arm: Placing = {
    at: [bannerPost[0], 3.75, postFace],
    scale: 1.6,
    lamp: true,
    glow: LAMP_GLASS,
  };
  place('wall-lamp', arm);
  return { counter, sign: onto('wall-lamp', arm, WALL_LAMP_BULB) };
}

/**
 * String lights across the street: a wire tied to a pole at each side sags between them in a
 * parabola, and swags of lights hang from it, each tilted along the wire. Returns the points where
 * each row's swags meet.
 */
function buildStringLights({ place, solid, glowing, rope, bodyOf, extentOf }: Builder) {
  const tie = POLE_X - POST_HALF_WIDTH * POLE;
  const sagAt = (x: number) => SAG * (1 - (x / tie) ** 2);
  const swag = extentOf('lights');
  const width = swag.high[0] - swag.low[0];
  return ROWS.map(({ z, high }, r) => {
    for (const x of [-POLE_X, POLE_X]) {
      solid.add(bodyOf('post'), {
        position: [x, 0, z],
        scale: [POLE, high + POLE_OVER, POLE],
      });
    }
    const points: Vec3[] = [];
    for (let k = 0; k <= SWAGS; k++) {
      const x = -tie + ((2 * tie) / SWAGS) * k;
      points.push([x, high - sagAt(x), z]);
    }
    for (let k = 0; k < SWAGS; k++) {
      const [ax, ay] = points[k];
      const [bx, by] = points[k + 1];
      const length = Math.hypot(bx - ax, by - ay);
      const slope = Math.atan2(by - ay, bx - ax);
      const roll = quat.fromAxisAngle(Z_AXIS, slope);
      const scale = length / width;
      // The swag's wire ends at its top corners, so its middle hangs below the chord's
      const below = vec3.transformQuat([0, swag.high[1] * scale, 0], roll);
      const at: Vec3 = [(ax + bx) / 2 - below[0], (ay + by) / 2 - below[1], z];
      place('lights', {
        at,
        rotation: roll,
        scale,
        lamp: true,
        glow: r % 2 === 0 ? BULBS : RED_BULBS,
      });
      // The wire, straight between two swags' ends
      glowing.add(rope, {
        position: [(ax + bx) / 2, (ay + by) / 2, z],
        rotation: quat.fromAxisAngle(Z_AXIS, Math.atan2(-(bx - ax), by - ay)),
        scale: [0.025, length, 0.025],
        ...WIRE,
      });
    }
    return points;
  });
}

/**
 * Hangs a lantern on a cord from `anchor`. Returns what swings it by `x` and `z` radians and
 * returns where its light is, in a vector it reuses.
 */
function hangLantern({ glowing, rope, bodyOf, glowOf, topOf }: Builder, anchor: Vec3) {
  const cord = glowing.add(rope, WIRE);
  const body = glowing.add(bodyOf('lantern'), { scale: HUNG_LANTERN });
  const glass = glowing.add(glowOf('lantern'), { scale: HUNG_LANTERN, emission: LAMP_GLASS });
  const height = topOf('lantern', HUNG_LANTERN);

  const tilt = quat.create();
  const across = quat.create();
  const down = vec3.create();
  const light = vec3.create();
  const hung = { position: vec3.create(), rotation: tilt, scale: [0.03, CORD, 0.03] as Vec3 };
  const held = { position: vec3.create(), rotation: tilt };

  return (x: number, z: number) => {
    quat.fromAxisAngle(X_AXIS, x, across);
    quat.multiply(across, quat.fromAxisAngle(Z_AXIS, z, tilt), tilt);
    vec3.transformQuat(DOWN, tilt, down);
    vec3.addScaled(anchor, down, CORD / 2, hung.position);
    glowing.write(cord, hung);
    vec3.addScaled(anchor, down, CORD + height, held.position);
    glowing.write(body, held);
    glowing.write(glass, held);
    return vec3.addScaled(anchor, down, CORD + height * 0.45, light) as Readonly<Float32Array>;
  };
}

/**
 * Builds the market: KayKit stalls, a tavern and houses along a street, a well, crates and barrels,
 * a potion seller's table by the eye, string lights across the street, a lantern on a cord, lamps
 * and a fire pit.
 */
export function createStalls(root: TgpuRoot, models: Models) {
  const builder = createBuilder(root, models);
  const { place, solid, glowing, pool } = builder;
  buildStreet(builder);
  const { top, front } = buildTable(builder);
  const lamps = buildLamps(builder);
  const wires = buildStringLights(builder);
  const swing = hangLantern(builder, wires[LANTERN_ROW][3]);
  swing(0, 0);

  place('barrel', { at: [FIRE[0], 0, FIRE[2]], scale: 5.5, roughness: 0.9 });
  place('logs', {
    at: [FIRE[0], FIRE[1] - 0.12, FIRE[2]],
    scale: 3.2,
    lamp: true,
    emission: EMBERS,
  });

  const { counter, sign } = lamps;
  return {
    solid,
    glowing,
    ready: builder.ready,
    /** The awning spots, the candles, the standing lanterns and the street lamps. */
    bulbs: builder.bulbs as readonly Bulb[],
    /** The lamp over the table and the lamp over the banner, and what each looks at. */
    spots: {
      counter: {
        lamp: counter,
        aim: [TABLE.at[0] - counter[0], top - counter[1], front - counter[2]] as Vec3,
      },
      sign: { lamp: sign, aim: [0, -1, -0.35] as Vec3 },
    },
    /** Where the seller stands behind her table. */
    seller: [TABLE.at[0] + 0.2, 0, TABLE.at[2] - HALF_DEPTH - 0.45] as Vec3,
    /** Where a customer stops, at the table's left end. */
    customer: [TABLE.at[0] - HALF_LENGTH - 0.45, 0, TABLE.at[2] - 0.8] as Vec3,
    /** Where the potions stand, the focus at the start. */
    potions: [TABLE.at[0], top + 0.3, front] as Vec3,
    swing,

    destroy() {
      solid.destroy();
      glowing.destroy();
      pool.destroy();
    },
  };
}
