import { machines, picker, pipelines } from '@typegpu/kit';
import { byStage } from '@typegpu/kit/inject';
import * as grid from '@typegpu/kit/grid';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { heightAt, LOOP_METRES, loopAt } from './land.ts';

export const CAPACITY = 8192;
/** Model-to-metre scale. A horse is about 2 m tall at the head. */
export const HORSE_SCALE = 0.42;
export const STEP_SECONDS = 1 / 60;
const WORKGROUP = 64;

/** Metres a gallop's and a walk's cycle covers at scale 1. */
const GALLOP_STRIDE = 6.6;
const WALK_STRIDE = 2.1;
/** Held moving poses: phases of a cycle, each at evenly spaced mixes from walk to gallop. */
export const PHASES = 24;
export const MIXES = 9;
/** Idle and grazing poses with independent clocks, assigned by horse index. */
export const RESTING_POSES = 6;
export const POSES = PHASES * MIXES + RESTING_POSES * 2;
/** Walk-to-gallop mix change per second, so a gait change passes through every held mix. */
const GAIT_PER_SECOND = 1.25;

/** Walking, full-gallop and panic speeds in metres per second. */
const WALK_SPEED = 1.7;
const GALLOP_SPEED = 11.5;
const PANIC_SPEED = 15.5;
/** Gait-switch speed thresholds. Walks below the lower one and gallops above the upper one. */
const GALLOP_FROM = 5.2;
const WALK_FROM = 4.2;
const RESTING_SPEED = 0.35;
/** Maximum urgency for occasional grazing. */
const GRAZING_BELOW = 0.3;
/** Average grazing starts per second for calm walking horses. */
const GRAZES_PER_SECOND = 0.03;
/** Minimum grazing duration and mean additional duration in seconds. */
const GRAZE_SECONDS = 4;
const GRAZE_MORE_SECONDS = 3.5;

const SIGHT_METRES = 7;
const PERSONAL_METRES = 3.2;
const RESPONSE_PER_SECOND = 1.6;
const SEPARATION = 16;
const ALIGNMENT = 0.9;
const COHESION = 0.2;
/** Lateral track spread in loop-radius units. */
export const LANE_SPREAD = 0.15;
const LANE_PULL = 9;
/** Position-restoring rate per second along the herd. */
const CATCH_UP_PER_SECOND = 0.12;
const CATCH_UP_SHARE = 0.3;
const OBSTACLE_REACH_METRES = 9;
const OBSTACLE_PUSH = 26;
const LOOK_AHEAD_SECONDS = 0.8;
const EYE_REACH_METRES = 6;
const EYE_PUSH = 90;
const FEAR_METRES = 26;
const FLEE = 30;
const CLAP_METRES_PER_SECOND = 70;
const CLAP_BAND_METRES = 9;
const CLAP_REACH_METRES = 260;
const CLAP_PUSH = 40;
const PANIC_SECONDS = 2.6;
/** Panic retention per neighbour hop and propagation rate. */
const PANIC_KEPT = 0.86;
const PANIC_SPREAD_PER_SECOND = 4;
/** Track area each horse takes at running density, in square metres. */
const SQUARE_METRES_EACH = 10;
/** Grid bounds include the track and a margin. */
const LOW = d.vec3f(-LOOP_METRES[0] * 1.3, -20, -LOOP_METRES[1] * 1.45);
const SIZE = d.vec3f(LOOP_METRES[0] * 2.6, 60, LOOP_METRES[1] * 2.9);

export const Horse = d.struct({
  position: d.vec3f,
  panic: d.f32,
  velocity: d.vec2f,
  /** Ground-plane facing, retained while stationary. */
  facing: d.vec2f,
  /** Gait-cycle progress from 0 to 1. */
  phase: d.f32,
  /** Pose id. */
  pose: d.u32,
  /** Mix from walk at 0 to gallop at 1, easing towards the gait's. */
  gait: d.f32,
});

/** Per-horse traits, fixed until the herd is populated again. */
const Trait = d.struct({
  /** Lateral track position from -1 to 1. */
  lane: d.f32,
  /** Longitudinal offset from the herd centre in metres. */
  place: d.f32,
  /** Speed multiplier relative to herd speed. */
  pace: d.f32,
  /** Size multiplier. */
  size: d.f32,
});

const Controls = d.struct({
  count: d.u32,
  /** Skin-to-card transition distance in metres. */
  handOff: d.f32,
  /** Herd-centre angle around the oval in radians. */
  middle: d.f32,
  urgency: d.f32,
  time: d.f32,
  /** Seconds the frame runs past the last step, which drawing moves the horses on by. */
  ahead: d.f32,
  /** Followed horse id, or `picker.none`. `watched` holds its state. */
  followed: d.u32,
});

/** Followed horse position and heading for camera readback, at the herd clock's `time`. */
const Watched = d.struct({ position: d.vec3f, velocity: d.vec2f, time: d.f32 });

const Threat = d.struct({
  pointer: d.vec3f,
  /** Pointer fear from 0 to 1. Full strength while moving over ground, fading when still. */
  fear: d.f32,
  clap: d.vec3f,
  /** Seconds since the last thunderclap. */
  clapAge: d.f32,
  /** Camera position and avoidance radius while the camera is low among the horses. */
  eye: d.vec2f,
  eyeRadius: d.f32,
});

/** Ground-plane position and avoidance radius of a rock or tree. */
const Obstacle = d.struct({ place: d.vec2f, radius: d.f32 });

const Neighbour = d.struct({ position: d.vec3f, panic: d.f32, velocity: d.vec2f });

/** Track angle at a position, measured on the herd clock. */
export const aroundAt = (x: number, z: number) => {
  'use gpu';
  return std.atan2(z / LOOP_METRES[1], x / LOOP_METRES[0]);
};

/** Push away along `offset`, from 0 at `reach` metres outside `radius` to `push` at `radius`. */
const avoid = tgpu.fn(
  [d.vec2f, d.f32, d.f32, d.f32],
  d.vec2f,
)((offset, radius, reach, push) => {
  'use gpu';
  const distance = std.length(offset);
  const gap = distance - radius;
  if (gap < reach) {
    const close = 1 - std.max(gap, 0) / reach;
    return (offset / std.max(distance, 0.1)) * push * close * close;
  }
  return d.vec2f();
});

/** Track length per radian at `around`, in metres. */
const metresPerRadian = (around: number) => {
  'use gpu';
  const along = LOOP_METRES[0] * std.sin(around);
  const across = LOOP_METRES[1] * std.cos(around);
  return std.sqrt(along * along + across * across);
};

/** The way clockwise along the track at `place`, turned back towards `lane`, from -1 to 1. */
const routeAt = (place: d.v2f, lane: number) => {
  'use gpu';
  const around = aroundAt(place.x, place.y);
  const along = std.normalize(
    d.vec2f(LOOP_METRES[0] * std.sin(around), -LOOP_METRES[1] * std.cos(around)),
  );
  const outward = std.normalize(
    place / d.vec2f(LOOP_METRES[0] * LOOP_METRES[0], LOOP_METRES[1] * LOOP_METRES[1]),
  );
  const off = 1 + lane * LANE_SPREAD - loopAt(place.x, place.y);
  return std.normalize(along + outward * off * LANE_PULL);
};

/**
 * Speed in metres per second added to `cruise` that brings a horse at `place` back to its place,
 * `metres` along the track from the herd's middle at `middle` radians: positive behind it and
 * negative ahead of it. The herd moves towards decreasing angles.
 */
const catchUpAt = (place: d.v2f, metres: number, middle: number, cruise: number) => {
  'use gpu';
  const around = aroundAt(place.x, place.y);
  let behind = around - middle + metres / metresPerRadian(around);
  behind -= std.round(behind / (Math.PI * 2)) * Math.PI * 2;
  return std.clamp(
    behind * metresPerRadian(around) * CATCH_UP_PER_SECOND,
    -cruise * CATCH_UP_SHARE,
    cruise * CATCH_UP_SHARE,
  );
};

/** CPU equivalent of `metresPerRadian`. */
export function metresAround(around: number) {
  return Math.hypot(LOOP_METRES[0] * Math.sin(around), LOOP_METRES[1] * Math.cos(around));
}

/** `angle` brought within half a turn of zero. */
export const wrap = (angle: number) => angle - Math.round(angle / (Math.PI * 2)) * Math.PI * 2;

/** Track position at `around`. The herd travels clockwise viewed from above. */
export function trackAt(around: number, out: [number, number] = [0, 0]): [number, number] {
  out[0] = LOOP_METRES[0] * Math.cos(around);
  out[1] = LOOP_METRES[1] * Math.sin(around);
  return out;
}

/** Track tangent at `around`. */
export function trackHeading(around: number, out: [number, number] = [0, 0]): [number, number] {
  const length = metresAround(around);
  out[0] = (LOOP_METRES[0] * Math.sin(around)) / length;
  out[1] = (-LOOP_METRES[1] * Math.cos(around)) / length;
  return out;
}

/** Track length in metres occupied by `count` horses at running density. */
function herdLength(count: number) {
  const width = LOOP_METRES[1] * LANE_SPREAD * 2;
  return (count * SQUARE_METRES_EACH) / width;
}

/** Cruising speed for the given urgency before adding panic. */
export function cruiseAt(urgency: number) {
  return WALK_SPEED + (GALLOP_SPEED - WALK_SPEED) * urgency;
}

interface HerdOptions {
  readonly count: number;
  /** Herd-centre angle around the oval to start the herd at, in radians. */
  readonly middle: number;
  /** Metres from the eye beyond which horses are drawn as cards. */
  readonly handOff: number;
  readonly obstacles: readonly {
    readonly place: readonly [number, number];
    readonly radius: number;
  }[];
  /**
   * Pose ids ordered by moving poses, a cycle of phases per mix from walk to gallop, then idle
   * poses and grazing poses.
   */
  readonly poses: readonly number[];
  /** Random-number source for the horses' traits. Default `Math.random`. */
  readonly random?: (() => number) | undefined;
}

/** Moves horses on the GPU using neighbour steering, gait selection and panic propagation. */
export function createHerd(root: TgpuRoot, options: HerdOptions) {
  const random = options.random ?? Math.random;
  const state = root.createBuffer(d.arrayOf(Horse, CAPACITY)).$usage('storage');
  const traitData = root.createBuffer(d.arrayOf(Trait, CAPACITY)).$usage('storage');
  let middle = options.middle;
  let drawnMiddle = middle;
  const clock = { middle, urgency: 1, time: 0, ahead: 0 };
  const controls = root.createUniform(Controls, {
    ...clock,
    count: options.count,
    handOff: options.handOff,
    followed: picker.none,
  });
  const watched = root.createBuffer(Watched).$usage('storage');
  const watching = watched.as('mutable');
  const threat = root.createUniform(Threat);
  const obstacles = root.createReadonly(
    d.arrayOf(Obstacle, options.obstacles.length),
    options.obstacles.map(({ place, radius }) => ({ place: d.vec2f(...place), radius })),
  );
  const poseIds = root.createReadonly(d.arrayOf(d.u32, POSES), [...options.poses]);
  const alive = (i: number) => {
    'use gpu';
    return i < controls.$.count;
  };
  /** The horses, written by `step` and read by renderers. */
  const horses = tgpu.accessor(d.arrayOf(Horse), byStage(state));
  const traits = tgpu.accessor(d.arrayOf(Trait), traitData.as('readonly'));

  /** Size-adjusted speed of horse `i` in metres per second. */
  const paceOf = (i: number) => {
    'use gpu';
    return std.length(horses.$[i].velocity) / traits.$[i].size;
  };

  const fast = (i: number) => {
    'use gpu';
    return paceOf(i) > GALLOP_FROM;
  };

  const slow = (i: number) => {
    'use gpu';
    return paceOf(i) < WALK_FROM;
  };

  const calm = (i: number) => {
    'use gpu';
    return controls.$.urgency < GRAZING_BELOW && horses.$[i].panic < 0.1;
  };

  const restless = (i: number) => {
    'use gpu';
    return !calm(i);
  };

  /**
   * Selects walking or galloping by speed. Separate thresholds prevent gait flicker. Calm horses
   * occasionally graze.
   */
  const Gait = machines.define('Gait', {
    states: {
      galloping: { next: [{ to: 'walking', when: slow }] },
      walking: {
        next: [
          { to: 'galloping', when: fast },
          { to: 'grazing', when: calm, rate: GRAZES_PER_SECOND },
        ],
      },
      grazing: {
        next: [
          { to: 'walking', when: restless },
          { to: 'walking', after: GRAZE_SECONDS, rate: 1 / GRAZE_MORE_SECONDS },
        ],
      },
    },
  });
  const gaits = machines.create(root, Gait, { capacity: CAPACITY, step: STEP_SECONDS, alive });

  const cells = grid.create(root, {
    capacity: CAPACITY,
    low: LOW,
    size: SIZE,
    sight: SIGHT_METRES,
    item: Neighbour,
    alive,
    position: (i) => {
      'use gpu';
      return d.vec3f(horses.$[i].position);
    },
    itemOf: (i) => {
      'use gpu';
      const horse = horses.$[i];
      return Neighbour({
        position: horse.position,
        panic: horse.panic,
        velocity: horse.velocity,
      });
    },
  });

  const poseOf = (i: number, speed: number, gait: number, phase: number, mixed: number) => {
    'use gpu';
    if (speed < RESTING_SPEED) {
      const resting = i % RESTING_POSES;
      if (gait === Gait.grazing) return poseIds.$[PHASES * MIXES + RESTING_POSES + resting];
      return poseIds.$[PHASES * MIXES + resting];
    }
    const level = d.u32(std.round(mixed * (MIXES - 1)));
    return poseIds.$[level * PHASES + (d.u32(phase * PHASES) % PHASES)];
  };

  const step = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const i = gid.x;
    if (i >= controls.$.count) return;
    const horse = Horse(horses.$[i]);
    const trait = traits.$[i];
    const gait = gaits.gpu.state(i);
    const place = horse.position.xz;

    let away = d.vec2f();
    let heading = d.vec2f();
    let centre = d.vec2f();
    let mates = d.f32(0);
    let panicked = d.f32(0);
    const near = cells.around(horse.position);
    for (let r = d.u32(0); r < grid.RANGES; r++) {
      const range = cells.range(near, r);
      for (let j = d.u32(0); j < range.count; j++) {
        const other = cells.neighbour(range, j);
        const offset = other.position.xz - place;
        const distance2 = std.dot(offset, offset);
        if (distance2 > SIGHT_METRES * SIGHT_METRES || distance2 < 1e-6) continue;
        if (distance2 < PERSONAL_METRES * PERSONAL_METRES) away -= offset / distance2;
        heading += other.velocity;
        centre += offset;
        mates += 1;
        panicked = std.max(panicked, other.panic);
      }
    }

    const route = routeAt(place, trait.lane);
    const cruise = std.mix(WALK_SPEED, GALLOP_SPEED, controls.$.urgency);
    const catchUp = catchUpAt(place, trait.place, controls.$.middle, cruise);

    let panic = horse.panic * std.exp(-STEP_SECONDS / PANIC_SECONDS);
    const spread = panicked * PANIC_KEPT;
    if (spread > panic)
      panic += (spread - panic) * std.saturate(PANIC_SPREAD_PER_SECOND * STEP_SECONDS);
    let flee = d.vec2f();
    const threats = threat.$;
    if (threats.fear > 0) {
      const offset = place - threats.pointer.xz;
      const distance = std.length(offset);
      if (distance < FEAR_METRES) {
        const fear = (1 - distance / FEAR_METRES) * threats.fear;
        panic = std.max(panic, fear);
        flee += (offset / std.max(distance, 0.1)) * FLEE * fear;
      }
    }
    const reach = threats.clapAge * CLAP_METRES_PER_SECOND;
    if (reach < CLAP_REACH_METRES) {
      const offset = place - threats.clap.xz;
      const distance = std.length(offset);
      if (std.abs(distance - reach) < CLAP_BAND_METRES) {
        panic = std.max(panic, 1 - reach / CLAP_REACH_METRES);
        flee += (offset / std.max(distance, 0.1)) * CLAP_PUSH;
      }
    }

    const grazing = gait === Gait.grazing;
    let wanted = (std.mix(cruise, PANIC_SPEED, panic) + catchUp) * trait.pace;
    if (grazing) wanted = 0;

    // Avoid obstacles using a point ahead of the horse
    const ahead = place + horse.velocity * LOOK_AHEAD_SECONDS;
    let swerve = d.vec2f();
    for (let k = d.u32(0); k < obstacles.$.length; k++) {
      const obstacle = obstacles.$[k];
      const offset = ahead - obstacle.place;
      swerve += avoid(offset, obstacle.radius, OBSTACLE_REACH_METRES, OBSTACLE_PUSH);
    }
    if (threats.eyeRadius > 0) {
      swerve += avoid(place - threats.eye, threats.eyeRadius, EYE_REACH_METRES, EYE_PUSH);
    }

    let force = (route * wanted - horse.velocity) * RESPONSE_PER_SECOND;
    force += away * SEPARATION + swerve + flee;
    if (mates > 0) {
      force += (heading / mates - horse.velocity) * ALIGNMENT * (1 - panic * 0.5);
      force += (centre / mates) * COHESION;
    }

    let velocity = horse.velocity + force * STEP_SECONDS;
    const speed = std.length(velocity);
    const fastest = std.max(wanted * 1.15, PANIC_SPEED * panic);
    if (speed > fastest) velocity *= fastest / std.max(speed, 1e-4);
    if (grazing) velocity *= 0.9;
    const moved = std.length(velocity);
    const next = place + velocity * STEP_SECONDS;

    const galloping = std.select(d.f32(0), d.f32(1), gait === Gait.galloping);
    const turn = GAIT_PER_SECOND * STEP_SECONDS;
    const mixed = horse.gait + std.clamp(galloping - horse.gait, -turn, turn);
    const stride = std.mix(WALK_STRIDE, GALLOP_STRIDE, mixed) * trait.size;
    const phase = std.fract(horse.phase + (moved * STEP_SECONDS) / stride);

    let facing = d.vec2f(horse.facing);
    if (moved > RESTING_SPEED) facing = std.normalize(std.mix(facing, velocity / moved, 0.25));

    const position = d.vec3f(next.x, heightAt(next.x, next.y), next.y);
    if (i === controls.$.followed) {
      watching.$ = Watched({ position, velocity, time: controls.$.time });
    }
    horses.$[i] = Horse({
      position,
      panic,
      velocity,
      facing,
      phase,
      pose: poseOf(i, moved, gait, phase, mixed),
      gait: mixed,
    });
  });

  const stepping = root.createComputePipeline({ compute: step });
  const traitValues: d.Infer<typeof Trait>[] = [];
  let count = options.count;

  const populate = (wanted: number) => {
    if (!Number.isInteger(wanted) || wanted < 1 || wanted > CAPACITY) {
      throw new RangeError(`The herd holds 1 to ${CAPACITY} horses, got ${wanted}.`);
    }
    count = wanted;
    const length = herdLength(count);
    const states: d.InferInput<typeof Horse>[] = [];
    traitValues.length = 0;
    for (let i = 0; i < count; i++) {
      // Lower edge density gives the herd uneven sides
      const lane = random() + random() - 1;
      // Concentrate horses near the herd centre
      const place = ((random() + random() + random()) / 3 - 0.5) * length;
      const size = 0.9 + random() * 0.2;
      const pace = 0.95 + random() * 0.1;
      const around = middle - place / metresAround(middle);
      const radius = 1 + lane * LANE_SPREAD;
      const [x, z] = trackAt(around).map((value) => value * radius) as [number, number];
      const [hx, hz] = trackHeading(around);
      const phase = random();
      traitValues.push({ lane, place, pace, size });
      states.push({
        position: d.vec3f(x, heightAt(x, z), z),
        panic: 0,
        velocity: d.vec2f(hx * GALLOP_SPEED, hz * GALLOP_SPEED),
        facing: d.vec2f(hx, hz),
        phase,
        pose: options.poses[(MIXES - 1) * PHASES + Math.floor(phase * PHASES)],
        gait: 1,
      });
      gaits.start(i, Gait.galloping);
    }
    state.write(states);
    traitData.write(traitValues);
    controls.patch({ count });
  };
  populate(options.count);

  return {
    controls,
    threat,
    /** Followed horse state written by `step` for readback. */
    watched,
    traits: traitValues,
    get count() {
      return count;
    },
    /** Herd-centre angle around the oval, which leads the horses clockwise. */
    get middle() {
      return middle;
    },
    populate,
    gaits,
    /** Herd-centre angle as drawn, moved on by the `ahead` that `advance` was given. */
    get drawnMiddle() {
      return drawnMiddle;
    },
    /** Herd-clock seconds that the steps so far have simulated. */
    get time() {
      return clock.time;
    },
    /**
     * Moves the herd's centre on by `seconds` at the cruising speed of `urgency`, for the next
     * steps, and draws the horses `ahead` seconds past the last step.
     */
    advance(seconds: number, urgency: number, ahead = 0) {
      const cruise = cruiseAt(urgency);
      middle = wrap(middle - (cruise * seconds) / metresAround(middle));
      drawnMiddle = wrap(middle - (cruise * ahead) / metresAround(middle));
      clock.middle = middle;
      clock.urgency = urgency;
      clock.time += seconds;
      clock.ahead = ahead;
      controls.patch(clock);
    },
    step(pass: TgpuComputePass) {
      cells.sort(pass);
      pass.setPipeline(stepping);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP));
      gaits.step(pass);
    },
    /** The horses, which kernels write and renderers read. */
    horses,
    /** Where horse `i` is drawn: its last step's position, moved on by its velocity for `ahead`. */
    drawnAt: (i: number) => {
      'use gpu';
      const horse = horses.$[i];
      return horse.position + d.vec3f(horse.velocity.x, 0, horse.velocity.y) * controls.$.ahead;
    },
    /** Returns the size multiplier of horse `i`. */
    sizeOf: (i: number) => {
      'use gpu';
      return traits.$[i].size;
    },
    initAsync: () => pipelines.compile([cells, stepping, gaits]),
    destroy() {
      for (const resource of [
        state,
        traitData,
        watched,
        obstacles.buffer,
        poseIds.buffer,
        gaits,
        cells,
      ]) {
        resource.destroy();
      }
    },
  };
}

export type Herd = ReturnType<typeof createHerd>;
