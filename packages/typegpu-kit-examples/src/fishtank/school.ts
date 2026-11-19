import * as grid from '@typegpu/kit/grid';
import { srgbToLinear } from '@typegpu/color';
import { instances, materials, meshPool, pipelines } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { randf } from '@typegpu/noise';
import { random } from 'math/random';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot, type TgpuUniform } from 'typegpu';
import type { Food } from './food.ts';
import { TANK, type Water } from './water.ts';

const SchoolSettings = d.struct({
  separation: d.f32,
  alignment: d.f32,
  cohesion: d.f32,
  wander: d.f32,
  appetite: d.f32,
});

type Rgb = readonly [number, number, number];

interface Body {
  /** The greatest height and width as shares of length. */
  readonly height: number;
  readonly width: number;
  /** The mesh's rings along the body and sides round each ring. */
  readonly rings: number;
  readonly sides: number;
}

interface Species {
  readonly count: number;
  /** The shortest and longest fish, nose to tail, in world units. */
  readonly length: readonly [number, number];
  /** Top speed in world units a second. */
  readonly speed: number;
  /** Back and belly colours in sRGB. */
  readonly back: Rgb;
  readonly belly: Rgb;
  readonly body: Body;
}

/** Small fish in large numbers, a middling species, and a few tall, slow ones. */
const SPECIES: readonly Species[] = [
  {
    count: 2400,
    length: [0.22, 0.3],
    speed: 2.2,
    back: [0.1, 0.45, 0.95],
    belly: [0.95, 0.25, 0.22],
    body: { height: 0.3, width: 0.16, rings: 8, sides: 6 },
  },
  {
    count: 900,
    length: [0.3, 0.4],
    speed: 2.8,
    back: [0.85, 0.82, 0.6],
    belly: [0.2, 0.28, 0.55],
    body: { height: 0.26, width: 0.14, rings: 10, sides: 6 },
  },
  {
    count: 40,
    length: [0.9, 1.2],
    speed: 1.1,
    back: [0.95, 0.92, 0.82],
    belly: [0.22, 0.2, 0.18],
    body: { height: 0.95, width: 0.12, rings: 16, sides: 8 },
  },
];

/** Tail-fin start from nose 0 to tail 1, and its thickness and height as shares of the body's. */
const FIN = { from: 0.75, thickness: 0.1, height: 0.6 };
/** Minimum body thickness as a fraction of length. Keeps normals finite at the tail-fin edge. */
const THINNEST = 0.004;
/** Tail beat in radians a second, its phase change nose to tail, and sway as a share of length. */
const BEAT = { rate: 10, wave: 5, sway: 0.14 };
const ROUGHNESS = 0.35;

/** World units a fish sees its neighbours within. */
const SIGHT = 1;
/** The tank's lowest and highest corners, which the walls keep fish within. */
const LOW = d.vec3f(-TANK.x / 2, 0, -TANK.z / 2);
const HIGH = d.vec3f(TANK.x / 2, TANK.y, TANK.z / 2);
/** World units from a wall at which it starts pushing a fish away. */
const MARGIN = 0.6;
/** Wall push one world unit inside `MARGIN`. */
const PUSH = 8;
/** World units a fish smells food from. */
const SMELL = 6;
/** World units a fish bites food from. */
const REACH = 0.35;
/** The slowest a fish swims, as a share of its top speed. */
const SLOWEST = 1 / 3;

/** Where fish start in world units: x and z either side of the centre, y a range of heights. */
const START = { x: 6, y: [1, TANK.y - 1] as const, z: 3 };
/** The speed a fish starts at, as a share of its top speed. */
const CRUISE = 0.5;
/** Maximum fractional variation from a species' top speed. */
const VARIETY = 0.15;

const WORKGROUP = 64;
const TAU = Math.PI * 2;

/** Sorted snapshot of fish positions and velocities. Steering reads it while fish move. */
const Neighbour = d.struct({ position: d.vec3f, species: d.u32, velocity: d.vec3f });

/**
 * Builds a one-unit fish facing +z, tapering from a spindle body into a flat tail fin. `u` runs
 * from nose to tail, and `v` runs around the body from its back for material shading.
 */
const bodyOf = ({ height, width, rings, sides }: Body) => {
  const across = (t: number) => {
    'use gpu';
    const body = std.sin(Math.PI * std.pow(t, 0.7));
    const fin = std.smoothstep(FIN.from, 1, t);

    return d.vec2f(
      std.max(width * body * std.mix(1, FIN.thickness, fin), THINNEST),
      height * std.mix(body, 1, fin * FIN.height),
    );
  };

  // Clockwise winding seen from the nose makes the triangles face outward
  return meshes.parametric(
    {
      at: (t, around) => {
        'use gpu';
        const size = across(t);
        const angle = around * TAU;
        return d.vec3f(-std.sin(angle) * size.x, std.cos(angle) * size.y, 0.5 - t);
      },

      normalAt: (t, around) => {
        'use gpu';
        const size = std.max(across(t), d.vec2f(THINNEST));
        const angle = around * TAU;
        return std.normalize(d.vec3f(-std.sin(angle) / size.x, std.cos(angle) / size.y, 0));
      },
    },
    { cols: rings, rows: sides },
  );
};

export interface SchoolOptions {
  readonly water: Water;
  readonly capacity: number;
  /** Seconds per `step`. */
  readonly step: number;
  /** Seconds since the last fixed step, from `sim.ahead`. Fish extrapolate by it. */
  readonly ahead: TgpuUniform<d.F32>;
  readonly time: TgpuUniform<d.F32>;
  readonly food: Food;
  readonly settings: d.InferInput<typeof SchoolSettings>;
}

/** Steers fish of three species as boids in a neighbour grid, and draws them as instances. */
export function createSchool(root: TgpuRoot, options: SchoolOptions) {
  const { capacity, step, ahead, time, water } = options;

  /**
   * Each species uses one mesh. Kernels write fish positions. `deform` animates tails and
   * extrapolates along local +z, which faces the velocity.
   */
  const Fish = instances.define('Fish', {
    host: {
      belly: d.vec3f,
      /** Top speed in world units a second. */
      speed: d.f32,
      /** Tail phase offset prevents neighbours swimming in step. */
      phase: d.f32,
    },
    device: { velocity: d.vec3f },
    transform: 'device',

    deform: (vertex, fish) => {
      'use gpu';
      const t = vertex.uv.x;
      const beat = std.sin(time.$ * BEAT.rate + fish.phase - t * BEAT.wave) * BEAT.sway * t * t;
      const swum = std.length(fish.velocity) * ahead.$;
      return vertex.position + d.vec3f(beat, 0, swum / fish.scale.z);
    },

    material: (point, fish) => {
      'use gpu';
      const up = std.cos(point.uv.y * TAU);
      const color = std.mix(fish.belly, fish.color, std.smoothstep(-0.4, 0.5, up));
      return materials.glossy(color, ROUGHNESS);
    },
  });

  const shapes = SPECIES.map(({ body }) => bodyOf(body));
  const pool = meshPool.create(root, meshPool.roomFor(shapes));
  const bodies = shapes.map((shape) => pool.add(shape));
  const school = instances.create(root, pool, { kind: Fish, capacity });
  const settings = root.createUniform(SchoolSettings, options.settings);

  const fish = school.gpu;
  const pellet = options.food.table.gpu;
  const cells = grid.create(root, {
    ...school.live(),
    low: LOW,
    size: TANK,
    sight: SIGHT,
    item: Neighbour,
    position: fish.position,
    itemOf: (i: number) => {
      'use gpu';
      return Neighbour({
        position: fish.position(i),
        species: fish.mesh(i),
        velocity: fish.velocity(i),
      });
    },
  });

  const steerKernel = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!fish.alive(i)) return;

    const position = fish.position(i);
    const velocity = fish.velocity(i);
    const species = fish.mesh(i);
    randf.seed4(d.vec4f(position, d.f32(i)));

    // Separate from all nearby fish, but align and gather only with the same species
    let away = d.vec3f();
    let heading = d.vec3f();
    let centre = d.vec3f();
    let mates = d.f32(0);

    const walk = cells.walk(position);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(walk, r);

      for (let j = d.u32(0); j < range.count; j++) {
        const other = cells.neighbour(range, j);
        const offset = other.position - position;
        const distance2 = std.dot(offset, offset);
        if (distance2 > SIGHT * SIGHT || distance2 < 1e-6) continue;

        away -= offset / distance2;
        if (other.species === species) {
          heading += other.velocity;
          centre += offset;
          mates += 1;
        }
      }
    }

    let force = away * settings.$.separation;
    if (mates > 0) {
      force += (heading / mates - velocity) * settings.$.alignment;
      force += (centre / mates) * settings.$.cohesion;
    }

    force += (std.clamp(position, LOW + MARGIN, HIGH - MARGIN) - position) * PUSH;

    // Head for the nearest pellet within smell, and bite it within reach
    let nearest = d.f32(SMELL * SMELL);
    let toward = d.vec3f();
    let meal = d.i32(-1);
    for (let k = d.u32(0); k < pellet.span(); k++) {
      if (!pellet.alive(k)) continue;

      const offset = options.food.pelletPosition(k, time.$) - position;
      const distance2 = std.dot(offset, offset);
      if (distance2 < nearest) {
        nearest = distance2;
        toward = d.vec3f(offset);
        meal = d.i32(k);
      }
    }

    if (meal >= 0) {
      force += std.normalize(toward) * settings.$.appetite;
      if (nearest < REACH * REACH) options.food.bite(d.u32(meal));
    }

    force += randf.inUnitSphere() * settings.$.wander;

    const speed = fish.speed(i);
    const pushed = velocity + force * step;
    const length = std.max(std.length(pushed), 1e-4);

    const moved = fish.device.$[i];
    moved.velocity = (pushed / length) * std.clamp(length, speed * SLOWEST, speed);
    moved.position = position + moved.velocity * step;
    moved.rotation = instances.toward(moved.velocity);
  });

  const steerPipeline = root.createComputePipeline({ compute: steerKernel });

  return {
    settings,

    renderer: school.renderer().pipe(water.inject()),
    flush: school.flush,
    initAsync: () => pipelines.compile([cells, steerPipeline]),

    /** Adds every species' fish at random through the tank, each heading its own way. */
    populate() {
      SPECIES.forEach(({ count, length, speed, back, belly }, species) => {
        const color = srgbToLinear(d.vec3f(...back));
        const underside = srgbToLinear(d.vec3f(...belly));
        const cruise = speed * CRUISE;

        for (let i = 0; i < count; i++) {
          const angle = random.float(Math.random, 0, TAU);
          const velocity = d.vec3f(Math.cos(angle) * cruise, 0, Math.sin(angle) * cruise);

          school.add(bodies[species], {
            position: [
              random.float(Math.random, -START.x, START.x),
              random.float(Math.random, ...START.y),
              random.float(Math.random, -START.z, START.z),
            ],
            rotation: instances.toward(velocity),
            scale: random.float(Math.random, ...length),
            color,
            belly: underside,
            speed: speed * random.float(Math.random, 1 - VARIETY, 1 + VARIETY),
            phase: random.float(Math.random, 0, TAU),
            velocity,
          });
        }
      });
    },

    step(pass: TgpuComputePass) {
      const workgroups = Math.ceil(school.span / WORKGROUP);
      cells.sort(pass);
      pass.setPipeline(steerPipeline);
      pass.dispatchWorkgroups(workgroups);
    },

    destroy() {
      school.destroy();
      pool.destroy();
      cells.destroy();
    },
  };
}
