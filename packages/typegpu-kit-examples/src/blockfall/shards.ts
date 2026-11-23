import { instances, materials, meshPool, particles, type CameraBase } from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { d, std, tgpu, type TgpuRoot, type TgpuUniform } from 'typegpu';
import { LIMESTONE, PALETTE, pigmentOf, roughnessOf } from './blocks.ts';
import { DEPTH, LAYER, PIECE, WIDTH } from './game.ts';

/** Chips a block breaks into. */
const CHIPS = 4;
/** Downward acceleration of a chip, in cells per second squared. */
const GRAVITY = 14;
/** How far a chip keeps from the well's walls, in cells. */
const MARGIN = 0.12;
/** A chip's size range along its three axes, in cells. */
const CHIP = { small: [0.2, 0.08, 0.16], large: [0.5, 0.22, 0.42] };
/** A chip's life range in seconds; it shrinks away over the last `FADE` of it. */
const LIFE = [0.9, 1.4];
const FADE = 0.35;
/** Seconds a landed chip takes to tip onto its nearest face. */
const TIP = 0.1;
/** The sandstone dust a block leaves where it stood. */
const SAND = [0.3, 0.26, 0.22, 0.2] as const;

/** Frame-clock time in seconds. */
const clock = tgpu.accessor(d.f32);

/** The unit quaternion turning by `angle` about the unit `axis`. */
const turn = (axis: d.v3f, angle: number) => {
  'use gpu';
  return d.vec4f(axis * std.sin(angle / 2), std.cos(angle / 2));
};

/** The rotation `a` after `b`, of unit quaternions. */
const after = (a: d.v4f, b: d.v4f) => {
  'use gpu';
  return d.vec4f(
    b.xyz * a.w + a.xyz * b.w + std.cross(a.xyz, b.xyz),
    a.w * b.w - std.dot(a.xyz, b.xyz),
  );
};

/** The heights of the unit axes `rotation` turns: how far each points up. */
const heightsOf = (rotation: d.v4f) => {
  'use gpu';
  const q = rotation;
  return d.vec3f(
    2 * (q.x * q.y + q.w * q.z),
    1 - 2 * (q.x * q.x + q.z * q.z),
    2 * (q.y * q.z - q.w * q.x),
  );
};

/** Seconds until something thrown up at `rise` lands `drop` below where it started. */
const landing = (drop: number, rise: number) => {
  'use gpu';
  const below = std.max(drop, 0);
  return (rise + std.sqrt(rise * rise + 2 * GRAVITY * below)) / GRAVITY;
};

/** How far a box of size `scale`, turned by `rotation`, reaches below its centre. */
const reachOf = (rotation: d.v4f, scale: d.v3f) => {
  'use gpu';
  return std.dot(std.abs(heightsOf(rotation)), scale) * 0.5;
};

/**
 * The least turn after `rotation` that stands the axis nearest the vertical upright, so the box
 * lies on a face, turned about the vertical as it was.
 */
const flattened = (rotation: d.v4f) => {
  'use gpu';
  const height = std.abs(heightsOf(rotation));
  const axis = std.select(
    std.select(d.vec3f(0, 0, 1), d.vec3f(0, 1, 0), height.y >= height.z),
    d.vec3f(1, 0, 0),
    height.x >= height.y && height.x >= height.z,
  );
  const tilted = instances.rotate(rotation, axis);
  const upright = d.vec3f(0, std.sign(tilted.y), 0);
  const level = std.normalize(d.vec4f(std.cross(tilted, upright), 1 + std.dot(tilted, upright)));
  return after(level, rotation);
};

/**
 * A chip of a broken block: painted on the block's outer faces, raw limestone where it broke
 * away. It flies from `position` at `velocity`, spinning about `spin.xyz` at `spin.w` radians a
 * second, lands on the floor at `rest`, tips onto a face over `TIP` seconds, and shrinks away by
 * `bornAt + life`.
 */
const Chip = instances.define('Chip', {
  host: { velocity: d.vec3f, spin: d.vec4f, bornAt: d.f32, life: d.f32, rest: d.f32, seed: d.f32 },

  place: (chip) => {
    'use gpu';
    const age = clock.$ - chip.bornAt;
    const rise = chip.velocity.y;
    const phase = chip.seed * 6.283;
    // It lands about when its lowest corner meets the floor: first guessed from its smallest
    // half extent, then from how it is turned then
    const smallest = std.min(std.min(chip.scale.x, chip.scale.y), chip.scale.z) * 0.5;
    const guess = landing(chip.position.y - chip.rest - smallest, rise);
    const guessed = turn(chip.spin.xyz, phase + chip.spin.w * guess);
    const landed = landing(chip.position.y - chip.rest - reachOf(guessed, chip.scale), rise);
    const flying = std.min(age, landed);

    const touched = turn(chip.spin.xyz, phase + chip.spin.w * landed);
    const tipped = std.smoothstep(0, TIP, age - landed);
    const rotation = std.normalize(std.mix(touched, flattened(touched), tipped));
    const turning = std.select(
      rotation,
      turn(chip.spin.xyz, phase + chip.spin.w * flying),
      age < landed,
    );
    const size = std.select(std.saturate((chip.life - age) / (chip.life * FADE)), 0, age < 0);

    // It rests on its lowest corner as it tips and shrinks away
    const resting = chip.rest + reachOf(turning, chip.scale) * size;
    const fallen = chip.position.y + rise * flying - 0.5 * GRAVITY * flying * flying;

    return instances.Placement({
      rotation: turning,
      position: d.vec3f(
        std.clamp(chip.position.x + chip.velocity.x * flying, MARGIN, WIDTH - MARGIN),
        std.select(std.max(fallen, resting), resting, age >= landed),
        std.clamp(chip.position.z + chip.velocity.z * flying, MARGIN, DEPTH - MARGIN),
      ),
      scale: chip.scale * size,
    });
  },

  visible: (chip) => {
    'use gpu';
    const age = clock.$ - chip.bornAt;
    return age >= 0 && age < chip.life;
  },

  carry: (vertexIndex) => {
    'use gpu';
    return d.vec4f(meshPool.vertexAt(vertexIndex).normal, 0);
  },

  material: (pixel, base) => {
    'use gpu';
    const chip = pixel.instance;
    const carried = pixel.carried;
    const broken = carried.y < -0.5 || carried.x < -0.5;
    const color = std.select(pigmentOf(base.color, chip.seed), LIMESTONE * 0.8, broken);
    return materials.glossy(color, roughnessOf(chip.seed));
  },
});

const random = (low: number, high: number) => low + Math.random() * (high - low);

/** Lays a dust patch flat on the floor, turned and stretched by its seed. */
const lying = (dust: particles.Particle, corner: d.v2f) => {
  'use gpu';
  const angle = dust.seed * 6.283;
  const along = d.vec3f(std.cos(angle), 0, std.sin(angle));
  const across = d.vec3f(-along.z, 0, along.x);
  const stretch = 1.2 + std.fract(dust.seed * 7.31) * 0.6;
  return dust.position + (along * (corner.x * stretch) + across * corner.y) * (dust.size * 0.5);
};

/** Shades a dust patch with a ragged edge that spreads and thins with its age. */
const ragged = (dust: particles.Particle, local: d.v2f) => {
  'use gpu';
  const aged = dust.age / dust.life;
  const angle = std.atan2(local.y, local.x);
  const edge =
    1 + 0.25 * std.sin(angle * 3 + dust.seed * 40) + 0.12 * std.sin(angle * 5 + dust.seed * 17);
  const reach = std.length(local) / (edge * (0.6 + 0.4 * std.sqrt(aged)));
  const body = std.saturate(1 - reach);
  const alpha = dust.color.w * body * body * (1 - aged) * std.smoothstep(0, 0.1, aged);
  return d.vec4f(dust.color.xyz * alpha, alpha);
};

export interface ShardsOptions {
  /** The length of a step in seconds; see `particles.Options.step`. */
  readonly step: number;
  readonly camera: CameraBase;
  /** Frame-clock time in seconds. */
  readonly time: TgpuUniform<d.F32>;
  /** The pool holding `chip`. */
  readonly pool: meshPool.MeshPool;
  /** A unit box, which a chip stretches. */
  readonly chip: meshPool.Mesh;
}

/** Breaks cleared blocks into painted stone chips and a little sandstone dust. */
export function createShards(root: TgpuRoot, options: ShardsOptions) {
  const { camera, step } = options;

  // One piece can clear at most one layer per cube, and all cleared blocks break together
  const capacity = CHIPS * PIECE * LAYER;
  const chips = instances.create(root, options.pool, { kind: Chip, capacity });
  const timed: Inject = (cfg) => cfg.with(clock, options.time);
  const ids = Array.from({ length: capacity }, () =>
    chips.add(options.chip, { ...materials.matte(PALETTE[0]), bornAt: -1e6, life: 1 }),
  );
  let next = 0;
  let brokeAt = -Infinity;

  const dust = particles.create(root, camera, {
    step,
    capacity: PIECE * LAYER,
    blend: 'over',
    bursts: PIECE * LAYER,
    place: lying,
    shade: ragged,
    settings: { gravity: [0, 0, 0], drag: 3 },
  });

  const chip = {
    position: [0, 0, 0] as [number, number, number],
    velocity: [0, 0, 0] as [number, number, number],
    spin: [0, 0, 0, 0] as [number, number, number, number],
    scale: [0, 0, 0] as [number, number, number],
    color: PALETTE[0],
    bornAt: 0,
    life: 0,
    rest: 0,
    seed: 0,
  };

  return {
    chips,
    /** The chips' drawing, timed by the frame clock. */
    drawing: chips.renderer().pipe(timed),
    dust,

    /** Whether any chip is still moving or shrinking at `time`, so shadows must follow it. */
    flying: (time: number) => time - brokeAt < LIFE[1],

    /** Breaks the block at cell `x, y, z`, of kind `kind`, at `time`. */
    break(x: number, y: number, z: number, kind: number, time: number) {
      chip.color = PALETTE[kind];
      chip.bornAt = time;
      brokeAt = time;
      chip.rest = y;
      for (let i = 0; i < CHIPS; i++) {
        const dx = random(-0.35, 0.35);
        const dy = random(-0.35, 0.35);
        const dz = random(-0.35, 0.35);
        chip.position[0] = x + 0.5 + dx;
        chip.position[1] = y + 0.5 + dy;
        chip.position[2] = z + 0.5 + dz;
        for (let axis = 0; axis < 3; axis++) {
          chip.scale[axis] = random(CHIP.small[axis], CHIP.large[axis]);
        }
        chip.velocity[0] = dx * 4 + random(-0.6, 0.6);
        chip.velocity[1] = random(2, 4.5);
        chip.velocity[2] = dz * 4 + random(-0.6, 0.6);

        const sx = random(-1, 1);
        const sy = random(-1, 1);
        const sz = random(-1, 1);
        const length = Math.hypot(sx, sy, sz) || 1;
        chip.spin[0] = sx / length;
        chip.spin[1] = sy / length;
        chip.spin[2] = sz / length;
        chip.spin[3] = random(4, 11);
        chip.life = random(LIFE[0], LIFE[1]);
        chip.seed = Math.random();
        chips.write(ids[next], chip);
        next = (next + 1) % capacity;
      }

      dust.burst({
        count: 1,
        position: [x + 0.5, y + 0.1, z + 0.5],
        area: [0.4, 0, 0.4],
        scatter: 0.4,
        size: 1.1,
        life: 1.5,
        variance: 0.4,
        color: SAND,
      });
    },

    destroy() {
      chips.destroy();
      dust.destroy();
    },
  };
}
