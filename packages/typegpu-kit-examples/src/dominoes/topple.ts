import {
  counters,
  Particle,
  picker,
  pipelines,
  shading,
  type CameraBase,
  type Particles,
  type Surface,
} from '@typegpu/kit';
import { randf } from '@typegpu/noise';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import type { Dominoes } from './dominoes.ts';
import { HEIGHT, SPACING, THICKNESS, WIDTH } from './layout.ts';

/** Seconds per step of the fall. */
export const STEP = 1 / 120;

/**
 * Angular acceleration from gravity, in radians per second squared. Slower than a physical
 * domino.
 */
export const GRAVITY = 700;
/** Tilt at which the centre of mass passes the pivot and gravity tips the domino forward. */
export const TIP = Math.atan(THICKNESS / HEIGHT);
/** The share of a domino's spin it hands the one it strikes. */
export const CARRY = 0.85;
/** The share of a leaning domino's weight that pushes the one it leans on. */
export const PUSH = 0.3;
/** Angular speed imparted by a tap, enough to tip the domino. */
export const KICK = 6;
/** Seconds the reset wave takes from the last domino to the first. It slows toward the start. */
export const RISE = 5;
/** Seconds each domino takes to rise. */
export const LIFT = 0.5;
/** Seconds a reset runs: the wave, the last domino's lift and a few steps for it to settle. */
const RESET = RISE + LIFT + 0.2;

/**
 * Reset start time as a fraction of `RISE`, at progress `back` from the last domino. The wave
 * travels nearly five times faster at the end than at the start.
 */
export const rewound = (back: number) => {
  'use gpu';
  return back * (0.35 + 0.65 * back);
};

const WORKGROUP = 64;

/** Next-step values, copied to the table after all dominoes have read the current step. */
const Motion = d.struct({ angle: d.f32, spin: d.f32, hit: d.f32 });

/**
 * Resting angle against a neighbour tilted by `next` radians. The top corner meets the
 * neighbour back face at distance `SPACING`.
 */
export const leanOn = (next: number) => {
  'use gpu';
  const reach = (SPACING * std.cos(next) - THICKNESS) / HEIGHT;
  return next + std.asin(std.clamp(reach, -1, 1));
};

/**
 * One step of a domino's turn from `angle` and `spin`: gravity tips it past `TIP`, `pushed` is
 * the weight of a neighbour leaning on it, and the ground stops it. Returns the next angle and
 * spin.
 */
export const swing = (angle: number, spin: number, pushed: number) => {
  'use gpu';
  let turning = spin;
  if (angle > 0 || spin > 0) {
    turning += GRAVITY * (std.sin(angle - TIP) + pushed * PUSH) * STEP;
  }
  const next = angle + turning * STEP;
  if (next <= 0) return d.vec2f();
  return d.vec2f(next, turning);
};

/** Rotates toward the fall direction, then tilts forward by `angle`. */
const rotationOf = (yaw: number, angle: number) => {
  'use gpu';
  const turn = d.vec2f(std.sin(yaw / 2), std.cos(yaw / 2));
  const tip = d.vec2f(std.sin(angle / 2), std.cos(angle / 2));
  return d.vec4f(turn.y * tip.x, turn.x * tip.y, -turn.x * tip.x, turn.y * tip.y);
};

/**
 * Shades fading dust with sun and sky light. Fill its lighting accessors with
 * `tgpu.fn(puff).pipe(light.inject())`.
 */
export const puff = (particle: Particle, local: d.v2f) => {
  'use gpu';
  const env = shading.settings();
  const sun = shading.sunShare(shading.pointAt(particle.position, d.vec3f(0, 1, 0)));
  const light =
    env.sunColor * (env.sunIntensity * sun * 0.3) + env.skyTop * (env.skyIntensity * 0.6);
  const dot = std.saturate(1 - std.length(local));
  const left = 1 - particle.age / particle.life;
  const alpha = dot * dot * left * particle.color.w;
  return d.vec4f(particle.color.xyz * light * alpha, alpha);
};

/** Mostly transparent, pale warm grey. */
const DUST = d.vec4f(0.85, 0.78, 0.68, 0.14);

/**
 * Computes next angles from current neighbour states, then copies them into the table. GPU
 * picking turns a tap into an initial spin. Counters count fallen dominoes, and impacts emit
 * dust.
 */
export function createTopple(
  root: TgpuRoot,
  options: {
    view: CameraBase;
    surface: Surface;
    dominoes: Dominoes;
    dust: Particles<typeof Particle>;
  },
) {
  const { view, surface, dominoes, dust } = options;
  const { set } = dominoes;
  const capacity = set.capacity;

  const motion = root.createMutable(d.arrayOf(Motion, capacity));
  /** Seconds since reset began, or -1 when inactive. */
  const rising = root.createUniform(d.f32, -1);
  const counts = counters.create(root, ['fallen'], { hz: 10 });

  const domino = set.gpu;

  /** The way domino `i` falls. */
  const forwardOf = (i: number) => {
    'use gpu';
    const yaw = domino.yaw(i);
    return d.vec3f(std.sin(yaw), 0, std.cos(yaw));
  };

  /** Centre position after pivoting by `angle` around the bottom front edge. */
  const middleOf = (i: number, angle: number) => {
    'use gpu';
    const forward = forwardOf(i);
    const edge = domino.foot(i) + forward * (THICKNESS / 2);
    const up = (HEIGHT / 2) * std.cos(angle) + (THICKNESS / 2) * std.sin(angle);
    const ahead = (HEIGHT / 2) * std.sin(angle) - (THICKNESS / 2) * std.cos(angle);
    return edge + d.vec3f(0, up, 0) + forward * ahead;
  };

  /**
   * Chooses the next domino that limits the lean most. Returns id 0 at the end of a line, where
   * the domino falls flat.
   */
  const restsOn = (i: number) => {
    'use gpu';
    const next = domino.next(i);
    if (
      next.y !== 0 &&
      leanOn(domino.angle(domino.slotOf(next.y))) < leanOn(domino.angle(domino.slotOf(next.x)))
    ) {
      return next.y;
    }
    return next.x;
  };

  /** Maximum lean against domino id `next`. Id 0 allows falling flat. */
  const leanLimit = (next: number) => {
    'use gpu';
    if (next === 0) return d.f32(Math.PI / 2);
    return leanOn(domino.angle(domino.slotOf(next)));
  };

  /** Emits dust at the back foot of the struck domino `i`. */
  const strike = (i: number) => {
    'use gpu';
    const forward = forwardOf(i);
    const side = d.vec3f(forward.z, 0, -forward.x);
    const at = domino.foot(i) - forward * (THICKNESS / 2);
    randf.seed2(d.vec2f(at.xz));
    for (let k = 0; k < 3; k++) {
      const across = randf.sample() * 2 - 1;
      dust.emit(
        Particle({
          position: at + side * (across * WIDTH * 0.5) + d.vec3f(0, 0.3, 0),
          age: 0,
          velocity: side * (across * 6) + d.vec3f(0, 2 + randf.sample() * 3, 0) - forward * 3,
          life: 0.7 + randf.sample() * 0.5,
          color: DUST,
          size: 2 + randf.sample() * 1.5,
          seed: randf.sample(),
        }),
      );
    }
  };

  /** Steps each domino: impacts start its spin, gravity tips it, and what it leans on stops it. */
  const fall = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= domino.span() || !domino.alive(i)) return;
    let angle = domino.angle(i);
    let spin = domino.spin(i);

    if (rising.$ >= 0) {
      // During reset, `hit` stores the starting angle. Limit the lean against the rising
      // neighbour so dominoes cannot pass through each other
      const begin = rewound(1 - domino.order(i)) * RISE;
      const t = std.saturate((rising.$ - begin) / LIFT);
      let from = domino.hit(i);
      if (t === 0) from = angle;
      angle = std.min(from * (1 - std.smoothstep(0, 1, t)), leanLimit(restsOn(i)));
      if (t === 1) from = 0;
      motion.$[i] = Motion({ angle, spin: 0, hit: from });
      return;
    }

    const was = angle;
    let pushed = d.f32(0);
    const prev = domino.prev(i);
    if (prev !== 0) {
      const p = domino.slotOf(prev);
      const before = domino.angle(p);
      if (angle === 0 && spin === 0 && domino.hit(p) > 0) {
        spin = domino.hit(p) * CARRY;
        strike(i);
      }
      if (before >= leanOn(angle) - 0.001) pushed = std.max(std.sin(before - TIP), 0);
    }

    const swung = swing(angle, spin, pushed);
    angle = swung.x;
    spin = swung.y;

    const on = restsOn(i);
    const limit = leanLimit(on);
    let hit = d.f32(0);
    if (angle >= limit) {
      hit = spin;
      angle = limit;
      if (on === 0) spin = 0;
      else spin = std.min(spin, domino.spin(domino.slotOf(on)));
    }

    if (was < Math.PI / 4 && angle >= Math.PI / 4) counts.add.fallen(1);
    motion.$[i] = Motion({ angle, spin, hit });
  });

  const place = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= domino.span() || !domino.alive(i)) return;
    const moved = Motion(motion.$[i]);
    const row = domino.device.$[i];
    row.angle = moved.angle;
    row.spin = moved.spin;
    row.hit = moved.hit;
    row.rotation = rotationOf(domino.yaw(i), moved.angle);
    row.position = middleOf(i, moved.angle);
  });

  const hits = picker.create(root, view, {
    capacity,
    size: surface,
    position: domino.position,
    radius: () => {
      'use gpu';
      return HEIGHT / 2;
    },
    // Only standing dominoes accept a push
    alive: (i: number) => {
      'use gpu';
      return domino.alive(i) && domino.angle(i) === 0;
    },
  });

  const push = tgpu.computeFn({ workgroupSize: [1] })(() => {
    'use gpu';
    const i = hits.found();
    if (i === picker.none || domino.spin(i) > 0) return;
    domino.device.$[i].spin = KICK;
  });

  const falling = root.createComputePipeline({ compute: fall });
  const placing = root.createComputePipeline({ compute: place });
  const pushing = root.createComputePipeline({ compute: push });

  let tapped: [x: number, y: number] | undefined;
  /** The CPU's copy of `rising`. */
  let risen = -1;

  return {
    counts,
    initAsync: () => pipelines.compile([hits, falling, placing, pushing]),

    /** Picks and pushes a domino at `x, y`, from 0 to 1 across the canvas from its top left. */
    tap(x: number, y: number) {
      tapped = [x, y];
    },

    /** Stands dominoes up from the last to the first over `RISE` seconds. */
    rise() {
      risen = 0;
    },

    get rising() {
      return risen >= 0;
    },

    /**
     * Applies the picked push and runs `steps` simulation steps. `dt` is seconds since the last
     * frame.
     */
    record(pass: TgpuComputePass, steps: number, dt: number) {
      if (risen >= 0) {
        rising.write(risen);
        risen += dt;
        if (risen > RESET) {
          risen = -1;
          rising.write(-1);
        }
      }

      if (tapped) {
        hits.pick(pass, ...tapped);
        pass.setPipeline(pushing);
        pass.dispatchWorkgroups(1);
        tapped = undefined;
      }

      const groups = Math.ceil(set.span / WORKGROUP);
      for (let step = 0; step < steps; step++) {
        pass.setPipeline(falling);
        pass.dispatchWorkgroups(groups);
        pass.setPipeline(placing);
        pass.dispatchWorkgroups(groups);
      }
    },

    destroy() {
      hits.destroy();
      counts.destroy();
      motion.buffer.destroy();
    },
  };
}
