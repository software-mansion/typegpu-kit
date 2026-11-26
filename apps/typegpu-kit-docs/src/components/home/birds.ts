import { cameras, lighting, materials, passes, shading, table } from '@typegpu/kit';
import * as grid from '@typegpu/kit/grid';
import { perlin3d, randf } from '@typegpu/noise';
import {
  d,
  std,
  tgpu,
  type TgpuComputePass,
  type TgpuFn,
  type TgpuRoot,
  type TgpuUniform,
} from 'typegpu';
import { KNOBS } from './knobs.ts';

/** The most birds the table holds: 32 × 32, so the table's plate lays out square. */
export const CAPACITY = 1024;

/** The air the birds keep to: its least corner and size, metres. */
export const LOW = d.vec3f(-7, 0.4, -7);
export const AIR = d.vec3f(14, 1.8, 14);

/** A cell of the grid, the farthest a bird sees and a lamp reaches. */
export const CELL = 1.4;

/** Cells of the grid across x and z. */
export const COLUMNS = Math.ceil(AIR.x / CELL);

/** Seconds a step. */
export const STEP = 1 / 60;

/** One bird in this many is a lamp. */
export const EVERY = 8;

const HEEDS = 12;
/** Metres within which a bird keeps away from another. */
const CLOSE = 0.65;
const MIDDLE = d.vec2f(LOW.x + AIR.x / 2, LOW.z + AIR.z / 2);
const HALF = d.vec2f(AIR.x / 2, AIR.z / 2);
const FLOOR = LOW.y + 0.4;
const CEILING = LOW.y + AIR.y - 0.4;
const TURNS = Math.PI * 2.4;
const WORKGROUP = 64;

export const Bird = table.define('Bird', {
  device: { position: d.vec3f, velocity: d.vec3f, force: d.vec3f },
});

/** Flocks the birds keep to, each on its own air. */
const FLOCKS = 3;

/** Snapshot copied by the grid for neighbour reads and steering rules. */
export const Seen = d.struct({ position: d.vec3f, velocity: d.vec3f, flock: d.u32 });
export type Seen = d.Infer<typeof Seen>;

/** The bird in row `i` as the grid copies it. */
export const seenOf = tgpu.fn((i: number) => {
  'use gpu';
  return Seen({ position: Bird.position(i), velocity: Bird.velocity(i), flock: i % FLOCKS });
});

const random = (low: number, high: number) => low + Math.random() * (high - low);

/** A bird somewhere in the air, flying level in any direction. */
export const bornAt = () => {
  const angle = random(0, Math.PI * 2);
  return {
    position: [
      random(LOW.x + 1, LOW.x + AIR.x - 1),
      random(LOW.y + 0.3, LOW.y + AIR.y - 0.3),
      random(LOW.z + 1, LOW.z + AIR.z - 1),
    ] as [number, number, number],
    velocity: [Math.cos(angle) * 3, 0, Math.sin(angle) * 3] as [number, number, number],
    force: [0, 0, 0] as [number, number, number],
  };
};

/** The sun's direction, `elevation` degrees above the horizon, from the frame's far left. */
export const degreesUp = (elevation: number) => {
  const up = (elevation * Math.PI) / 180;
  const round = 2.35;
  return [Math.cos(up) * Math.cos(round), Math.sin(up), Math.cos(up) * Math.sin(round)] as [
    number,
    number,
    number,
  ];
};

/** Neighbour separation, weighted by inverse distance, plus the same flock's mean heading, mean offset and count. Uses the bird's heading when no flockmates are nearby. */
export const Near = d.struct({
  away: d.vec3f,
  heading: d.vec3f,
  centre: d.vec3f,
  mates: d.f32,
});

export type Near = d.Infer<typeof Near>;

/** A bird's rule, a `'use gpu'` function: the force on it, from itself and what it found near. */
export type Rule = (me: Seen, near: Near) => d.v3f;

/** The rule the steering calls, which each of its pipelines fills. */
const rule = tgpu.slot<TgpuFn<(me: typeof Seen, near: typeof Near) => d.Vec3f>>();

/** The rules' numbers. */
export const Tuned = d.struct({
  /** How hard a flocking bird closes in on its neighbours. */
  toward: d.f32,
  /** The ring an orbiting bird keeps to, metres from the middle. */
  radius: d.f32,
  /** Neighbours that make a crowd, for a huddling bird. */
  crowd: d.f32,
});

/** The rules' numbers, which the steering fills. */
export const tuned = tgpu.accessor(Tuned);

/** What the steering reads each step, written when it changes. */
export const Steer = d.struct({
  /** The pointer on the birds' height, 1 in `w` while it is over the frame. */
  pointer: d.vec4f,
  time: d.f32,
  /** How far a bird heeds its neighbours, metres, up to a cell. */
  sight: d.f32,
  /** A press on the lockup, from 1 down to 0: the flock scatters. */
  startle: d.f32,
});

/** How the birds look, written when it changes. */
const Look = d.struct({
  /** Metres from beak to tail. */
  length: d.f32,
  /** 0 for a dot, 1 for a chevron, between as one becomes the other. */
  shape: d.f32,
});

/** A lamp's range, metres, and its intensity; 0 while the lamps are out. */
const Lamp = d.struct({ range: d.f32, intensity: d.f32 });

/** Whether bird `slot` is a lamp. */
export const isLamp = (slot: number) => {
  'use gpu';
  return slot % EVERY === 0;
};

/** Seconds since the flock started, for the air. */
const airTime = tgpu.accessor(d.f32);

/**
 * The way the air flows for `me`, level and of length 1, turned home near the air's round edge. Each
 * flock has air of its own, so the flocks part.
 */
export const airAt = (me: d.Infer<typeof Seen>) => {
  'use gpu';
  const p = me.position;
  const own = d.vec2f(d.f32(me.flock) * 13.7, d.f32(me.flock) * -7.1);
  const angle = perlin3d.sample(d.vec3f(p.xz * 0.16 + own, airTime.$ * 0.08)) * TURNS;
  const across = (p.xz - MIDDLE) / HALF;
  const out = std.length(across);
  const home = (across / std.max(out, 1e-4)) * (std.smoothstep(0.65, 0.95, out) * -3);
  return std.normalize(d.vec3f(std.cos(angle) + home.x, 0, std.sin(angle) + home.y));
};

/** Back towards the middle across the round's edge, and from the floor and the ceiling. */
const walls = (p: d.v3f) => {
  'use gpu';
  const across = (p.xz - MIDDLE) / HALF;
  const out = std.length(across);
  const back = (across / std.max(out, 1e-4)) * (std.smoothstep(0.82, 1, out) * -12);
  const up = std.max(FLOOR - p.y, 0) - std.max(p.y - CEILING, 0);
  return d.vec3f(back.x, up * 8, back.y);
};

/** The pointer's push and the startle's, on a bird at `p`. */
const scare = (p: d.v3f, steer: d.InferGPU<typeof Steer>) => {
  'use gpu';
  const from = d.vec3f(p.x - steer.pointer.x, 0, p.z - steer.pointer.z);
  const near = std.smoothstep(2.2, 0.3, std.length(from)) * steer.pointer.w;
  const out = d.vec3f(p.x - MIDDLE.x, 0.4, p.z - MIDDLE.y);
  return std.normalize(from + 1e-4) * (near * 40) + std.normalize(out) * (steer.startle * 30);
};

const moveBy = (i: number, force: d.v3f) => {
  'use gpu';
  const velocity = Bird.velocity(i);
  const pushed = velocity + force * STEP;
  const length = std.max(std.length(pushed), 1e-5);
  const moved = Bird.device.$[i];
  moved.velocity = (pushed / length) * std.clamp(length, 0.4, 5);
  moved.position = Bird.position(i) + moved.velocity * STEP;
  moved.force = d.vec3f(force);
};

export interface SteeringOptions<Name extends string> {
  /** The rules a step can take, by name. */
  readonly rules: Record<Name, Rule>;
  /** How far a bird heeds its neighbours, metres, up to a cell. Default a cell. */
  readonly sight?: number;
}

/**
 * Moves the birds a step under a rule. `flock` reads the grid; `drift`, while the grid is out,
 * passes no neighbours. Compiles each rule both ways.
 */
export function createSteering<Name extends string>(
  root: TgpuRoot,
  birds: table.Table<typeof Bird>,
  cells: grid.Grid<typeof Seen>,
  options: SteeringOptions<Name>,
) {
  const steer = root.createUniform(Steer, {
    pointer: d.vec4f(),
    time: 0,
    sight: options.sight ?? CELL,
    startle: 0,
  });
  const tuning = root.createUniform(Tuned, {
    toward: KNOBS.toward.value,
    radius: KNOBS.radius.value,
    crowd: KNOBS.crowd.value,
  });
  const rules = options.rules;
  const flock = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!Bird.alive(i)) return;
    const s = steer.$;
    const position = Bird.position(i);
    const velocity = Bird.velocity(i);
    const sight = s.sight;
    const own = i % FLOCKS;
    randf.seed4(d.vec4f(position, d.f32(i)));

    let away = d.vec3f();
    let heading = d.vec3f();
    let centre = d.vec3f();
    let mates = d.f32(0);
    const walk = cells.walk(position);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      if (mates >= HEEDS) break;
      const range = cells.range(walk, r);
      for (let j = d.u32(0); j < range.count; j++) {
        const other = cells.neighbour(range, j);
        const offset = other.position - position;
        const distance2 = std.dot(offset, offset);
        if (distance2 > sight * sight || distance2 < 1e-8) continue;
        if (distance2 < CLOSE * CLOSE) away -= offset / distance2;
        if (other.flock !== own) continue;
        heading += other.velocity;
        centre += offset;
        mates += 1;
        if (mates >= HEEDS) break;
      }
    }
    const among = std.max(mates, 1);
    const near = Near({
      away,
      heading: std.select(velocity, heading / among, mates > 0),
      centre: centre / among,
      mates,
    });
    const me = Seen({ position, velocity, flock: own });
    moveBy(i, rule.$(me, near) + walls(position) + scare(position, s));
  });

  const drift = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (!Bird.alive(i)) return;
    const position = Bird.position(i);
    const velocity = Bird.velocity(i);
    const near = Near({ away: d.vec3f(), heading: velocity, centre: d.vec3f(), mates: 0 });
    const me = Seen({ position, velocity, flock: i % FLOCKS });
    moveBy(i, rule.$(me, near) + walls(position) + scare(position, steer.$));
  });

  const air = () => {
    'use gpu';
    return steer.$.time;
  };
  const names = Object.keys(rules) as Name[];
  const pipelinesOf = (kernel: typeof flock, onGrid: boolean) =>
    Object.fromEntries(
      names.map((name) => {
        const base = root.pipe(birds.inject());
        const cfg = onGrid ? base.pipe(cells.inject()) : base;
        const called = tgpu.fn([Seen, Near], d.vec3f)(rules[name]);
        return [
          name,
          cfg
            .with(airTime, air)
            .with(tuned, tuning)
            .with(rule, called)
            .createComputePipeline({ compute: kernel }),
        ];
      }),
    ) as Record<Name, ReturnType<typeof root.createComputePipeline>>;
  const flocking = pipelinesOf(flock, true);
  const drifting = pipelinesOf(drift, false);

  return {
    /** What the steering reads each step: the pointer, the time, the sight and the startle. */
    settings: steer,
    /** The rules' numbers. */
    tuning,
    initAsync: () =>
      Promise.all(
        [...Object.values(flocking), ...Object.values(drifting)].map((each) =>
          (each as (typeof flocking)[Name]).initAsync(),
        ),
      ),
    /** One step under `name`'s rule: on the grid, sorted first in the same pass, or without it. */
    step(pass: TgpuComputePass, onGrid: boolean, name: Name) {
      if (birds.span === 0) return;
      const groups = Math.ceil(birds.span / WORKGROUP);
      pass.setPipeline((onGrid ? flocking : drifting)[name]);
      pass.dispatchWorkgroups(groups);
    },
  };
}

const Corner = d.struct({
  position: d.vec4f,
  world: d.vec3f,
  normal: d.vec3f,
  lamp: d.f32,
});

const Lit = d.struct({ color: d.vec4f });

/** The ground's and the birds' materials. */
const GREY = d.vec3f(0.62, 0.63, 0.66);
const GROUND = d.vec3f(0.05, 0.052, 0.055);
/** A lamp bird's glow, linear: a blue that burns to ice at its brightest. */
const LAMP = d.vec3f(0.2, 0.55, 2.2);

/**
 * One bird in `EVERY` as a blue lamp, lighting the ground and the birds around it: `light` is the
 * lighting's point light at a bird's row.
 */
export function createLamps(
  root: TgpuRoot,
  birds: table.Table<typeof Bird>,
  initial: { range: number; intensity: number },
) {
  const settings = root.createUniform(Lamp, initial);
  const light = tgpu
    .fn((slot: number) => {
      'use gpu';
      return lighting.Light({
        position: Bird.position(slot),
        range: settings.$.range,
        color: LAMP,
        intensity: std.select(0, settings.$.intensity, isLamp(slot)),
      });
    })
    .pipe(birds.inject());
  return { settings, light };
}

export type Lamps = ReturnType<typeof createLamps>;

/**
 * Faces chevrons towards the eye around their flight direction. Lamp birds glow blue; others use
 * the injected lighting. `settings.shape` turns them from dots into chevrons.
 */
export function birdsDrawing(
  root: TgpuRoot,
  birds: table.Table<typeof Bird>,
  lamps: Lamps,
  options: { length: number },
): passes.Drawing<typeof Corner, d.Infer<typeof Lit>> & { settings: TgpuUniform<typeof Look> } {
  const look = root.createUniform(Look, { length: options.length, shape: 1 });
  const at = (vertexIndex: number, slot: number) => {
    'use gpu';
    if (!Bird.alive(slot)) return Corner();
    const view = cameras.current.$;
    const position = Bird.position(slot);
    const velocity = Bird.velocity(slot);
    const forward = velocity / std.max(std.length(velocity), 1e-4);
    const toEye = std.normalize(view.position - position);
    const side = std.normalize(std.cross(forward, toEye) + d.vec3f(1e-4, 0, 0));

    const chevron = [
      d.vec2f(0.5, 0),
      d.vec2f(-0.5, 0.45),
      d.vec2f(-0.15, 0),
      d.vec2f(0.5, 0),
      d.vec2f(-0.15, 0),
      d.vec2f(-0.5, -0.45),
    ];
    const dot = [
      d.vec2f(0.12, 0),
      d.vec2f(0, 0.12),
      d.vec2f(-0.12, 0),
      d.vec2f(0.12, 0),
      d.vec2f(-0.12, 0),
      d.vec2f(0, -0.12),
    ];
    const shape = std.mix(dot[vertexIndex], chevron[vertexIndex] * look.$.length, look.$.shape);
    const world = position + forward * shape.x + side * shape.y;
    return Corner({
      position: cameras.toClip(world),
      world,
      normal: std.normalize(std.cross(side, forward)),
      lamp: std.select(d.f32(0), d.f32(1), isLamp(slot)),
    });
  };

  return {
    settings: look,
    label: 'birds',
    Vertex: Corner,
    at,
    pixelOf: (corner) => {
      'use gpu';
      let normal = std.normalize(corner.normal);
      if (std.dot(normal, shading.towardsSun()) < 0) normal = normal * -1;
      const glow = std.saturate(lamps.settings.$.intensity) * corner.lamp;
      const material = materials.glossy(std.mix(GREY, d.vec3f(0.05), glow), 0.7);
      material.emission = LAMP * (glow * 1.3);
      return Lit({
        color: d.vec4f(shading.shade(material, shading.pointAt(corner.world, normal)), 1),
      });
    },
    color: passes.shaded,
    flat: ['lamp'],
    primitive: { cullMode: 'none' },
    inject: () => (cfg) => cfg.pipe(birds.inject()),
    draw: (pass) => {
      if (birds.span > 0) pass.draw(6, birds.span);
    },
  };
}

const GroundCorner = d.struct({ position: d.vec4f, world: d.vec3f, normal: d.vec3f });

/** A flat ground at y = 0, `reach` metres every way, lit by the sun and the lamps. */
export function groundDrawing(
  reach: number,
): passes.Drawing<typeof GroundCorner, d.Infer<typeof Lit>> {
  const at = (vertexIndex: number, _instance: number) => {
    'use gpu';
    const corners = [
      d.vec2f(-1, -1),
      d.vec2f(1, -1),
      d.vec2f(1, 1),
      d.vec2f(-1, -1),
      d.vec2f(1, 1),
      d.vec2f(-1, 1),
    ];
    const xz = corners[vertexIndex] * reach;
    const world = d.vec3f(xz.x, 0, xz.y);
    return GroundCorner({
      position: cameras.toClip(world),
      world,
      normal: d.vec3f(0, 1, 0),
    });
  };

  return {
    label: 'ground',
    Vertex: GroundCorner,
    at,
    pixelOf: (corner) => {
      'use gpu';
      const ground = materials.glossy(GROUND, 0.95);
      const lit = shading.shade(ground, shading.pointAt(corner.world, d.vec3f(0, 1, 0)));
      return Lit({ color: d.vec4f(lit, 1) });
    },
    color: passes.shaded,
    draw: (pass) => pass.draw(6),
  };
}
