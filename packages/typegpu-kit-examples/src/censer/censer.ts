import { perlin2d } from '@typegpu/noise';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import { createChain, LENGTH, SIZE } from './chain.ts';

type Vec3 = [number, number, number];
type Mat3 = [number, number, number, number, number, number, number, number, number];

/** The body as a capsule along the chain: half its length and its radius. */
const HALF = 0.32 * SIZE;
const RADIUS = 0.5 * SIZE;
/** How near the body's middle a pointer's ray must pass to take hold of it, in metres. */
const REACH = 0.32;
/**
 * The vents breathe out in puffs from seven places around the top of the lid, each a soft blob
 * this wide, this far out from the chain and this high above the body's middle, just clear of its
 * solid.
 */
const VENTS = 7;
const PUFF = 0.085;
const VENT_OUT = 0.13;
const VENT_UP = 0.37;
/**
 * Smoke and heat at the middle of a puff at the default emission, and its jet's acceleration,
 * per second.
 */
const SMOKE = 1.3;
const HEAT = 17;
const JET = 30;
/**
 * Air within this many metres of the body is dragged along with it, at up to this rate per second,
 * and air in a puff takes on its vent's motion at this rate, so the censer stirs what it passes.
 */
const SHELL = 0.1;
const SHELL_RATE = 40;
const INHERIT = 30;
/** How fast each vent's breath comes and goes, per second. */
const BREATHING = 2.3;
/** The smoke's step, in seconds, over which each puff is swept along the body's path. */
export const STEP = 1 / 60;

const Pose = d.struct({
  /** Turns the model of the hanging censer to the chain's direction. */
  rotation: d.mat3x3f,
  /** The middle of the body, and its velocity. */
  middle: d.vec3f,
  velocity: d.vec3f,
  /** The body's angular velocity, in radians per second. */
  spin: d.vec3f,
  /** The smoke's time in seconds, which the vents breathe to. */
  time: d.f32,
  /** The smoke and heat the vents add per second at the middle of a puff. */
  vents: d.vec2f,
});

const poseLayout = tgpu.bindGroupLayout({ pose: { uniform: Pose } });

/** How far a point of the body's frame lies outside the body. */
const distanceTo = (body: d.v3f) => {
  'use gpu';
  const along = d.vec3f(0, std.clamp(body.y, -HALF, HALF), 0);
  return std.length(body - along) - RADIUS;
};

/** A point of the body, measured from its middle, along and across the chain. */
const bodyOf = (position: d.v3f) => {
  'use gpu';
  return std.transpose(poseLayout.$.pose.rotation) * (position - poseLayout.$.pose.middle);
};

/**
 * The puff of the vent nearest a point: its jet's direction in the body's frame, and how much of
 * it reaches the point. Each vent breathes on its own rhythm and sweeps its puff along the way
 * it moved over the last step, so a fast swing leaves a trail rather than a string of beads.
 */
const puffAt = (position: d.v3f) => {
  'use gpu';
  const body = bodyOf(position);
  const swept = std.transpose(poseLayout.$.pose.rotation) * poseLayout.$.pose.velocity * STEP;
  const margin = PUFF * 3 + std.length(swept);
  const rim = std.length(body.xz) - VENT_OUT;
  if (std.abs(body.y - VENT_UP) > margin || std.abs(rim) > margin) return d.vec4f();
  const around = std.atan2(body.z, body.x) * (VENTS / (2 * Math.PI));
  const vent = std.round(around);
  const angle = vent * ((2 * Math.PI) / VENTS);
  const out = d.vec3f(std.cos(angle), 0, std.sin(angle));
  const middle = out * VENT_OUT + d.vec3f(0, VENT_UP, 0);
  const length2 = std.max(std.dot(swept, swept), 1e-8);
  const back = std.saturate(std.dot(middle - body, swept) / length2);
  const offset = body - (middle - swept * back);
  const spread = PUFF / (PUFF + std.sqrt(length2) * 0.56);
  const reach = std.exp(-std.dot(offset, offset) / (PUFF * PUFF)) * spread;
  const breath = perlin2d.sample(d.vec2f(poseLayout.$.pose.time * BREATHING, vent * 7.3 + 11));
  const pulse = std.saturate(breath * 2 + 0.5);
  const jet = std.normalize(out * 0.3 + d.vec3f(0, 1.5, 0));
  return d.vec4f(jet, reach * pulse * pulse);
};

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * A thurible hanging by its chain: its pose, the solid and smoke it gives the air, and a hand that
 * may take hold of it and throw it.
 */
export function createCenser(root: TgpuRoot) {
  const chain = createChain();
  // Plain arrays, which TypeGPU writes as they are, where setting a vector's components would make
  // garbage that Hermes collects in frames
  const state = {
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1] as Mat3,
    middle: chain.middle,
    velocity: chain.velocity,
    spin: chain.spin,
    time: 0,
    vents: [SMOKE, HEAT] as [number, number],
  };
  // Two poses, so the two steps a slow frame records each read their own: each swing writes the
  // other one, and the pass binds it for the step and the draws that follow
  const posed = [0, 1].map(() => {
    const buffer = root.createBuffer(Pose).$usage('uniform').$name('censer pose');
    return { buffer, group: root.createBindGroup(poseLayout, { pose: buffer }) };
  });
  let last = 0;
  // The plane the hand moves in, through where it took hold and facing along the ray, and where it
  // pulls to
  const anchor: Vec3 = [0, 0, 0];
  const facing: Vec3 = [0, 0, 1];
  const pulled: Vec3 = [0, 0, 0];

  /** Writes the chain's state as the body's pose. */
  const write = (time: number) => {
    const x = chain.turn[0];
    const y = chain.turn[1];
    const z = chain.turn[2];
    const w = chain.turn[3];
    // Column by column
    const r = state.rotation;
    r[0] = 1 - 2 * (y * y + z * z);
    r[1] = 2 * (x * y + z * w);
    r[2] = 2 * (x * z - y * w);
    r[3] = 2 * (x * y - z * w);
    r[4] = 1 - 2 * (x * x + z * z);
    r[5] = 2 * (y * z + x * w);
    r[6] = 2 * (x * z + y * w);
    r[7] = 2 * (y * z - x * w);
    r[8] = 1 - 2 * (x * x + y * y);
    state.time = time;
    last = 1 - last;
    posed[last].buffer.write(state);
  };

  return {
    chain,
    /** Places a point of the model, given from the pivot as the censer hangs at rest. */
    place: (position: d.v3f) => {
      'use gpu';
      return (
        poseLayout.$.pose.rotation * (position + d.vec3f(0, LENGTH, 0)) + poseLayout.$.pose.middle
      );
    },
    /** Turns a direction of the model as the censer hangs. */
    turn: (direction: d.v3f) => {
      'use gpu';
      return poseLayout.$.pose.rotation * direction;
    },
    /** The middle of the body, where the embers burn. */
    middle: () => {
      'use gpu';
      return d.vec3f(poseLayout.$.pose.middle);
    },
    /** How far a point lies outside the censer's body, in metres. */
    distance: (position: d.v3f) => {
      'use gpu';
      return distanceTo(bodyOf(position));
    },
    /** The censer's body as a solid, moving as it does. */
    solid: (position: d.v3f) => {
      'use gpu';
      if (distanceTo(bodyOf(position)) > 0) return d.vec4f();
      const moving =
        poseLayout.$.pose.velocity +
        std.cross(poseLayout.$.pose.spin, position - poseLayout.$.pose.middle);
      return d.vec4f(moving, 1);
    },
    /**
     * The motion air near the body is dragged toward: its own, in a thin shell around it and in
     * each puff.
     */
    carry: (position: d.v3f) => {
      'use gpu';
      const outside = distanceTo(bodyOf(position));
      const shell = std.saturate(1 - outside / SHELL) * std.step(0, outside);
      const rate = SHELL_RATE * shell * shell + INHERIT * puffAt(position).w;
      const moving =
        poseLayout.$.pose.velocity +
        std.cross(poseLayout.$.pose.spin, position - poseLayout.$.pose.middle);
      return d.vec4f(moving, rate);
    },
    /** Smoke and heat from the vents in the lid, in puffs. */
    source: (position: d.v3f) => {
      'use gpu';
      const puff = puffAt(position);
      return poseLayout.$.pose.vents * puff.w;
    },
    /** The jet each puff leaves its vent with, up and out of the lid. */
    force: (position: d.v3f) => {
      'use gpu';
      const puff = puffAt(position);
      return poseLayout.$.pose.rotation * puff.xyz * (JET * puff.w);
    },
    /**
     * Scales the smoke and heat the vents breathe out, 1 by default. Hotter puffs rise faster, so
     * more smoke billows higher rather than piling up.
     */
    emit(rate: number) {
      state.vents[0] = SMOKE * rate;
      state.vents[1] = HEAT * rate;
    },
    /**
     * Binds the pose the last swing wrote, for the step that follows it and the draws: set it on
     * each pass that reads the censer.
     */
    get pose() {
      return posed[last].group;
    },
    /** Swings the censer `dt` seconds on, to `time`, writing its pose into the other group. */
    swing(dt: number, time: number) {
      chain.advance(dt);
      write(time);
    },
    /**
     * Takes hold of the body if a pointer's ray passes near its middle, and returns whether it did.
     * The hand then moves in the plane through the grip that faces along the ray.
     */
    grab(origin: ArrayLike<number>, direction: ArrayLike<number>) {
      const middle = chain.middle;
      const to = [middle[0] - origin[0], middle[1] - origin[1], middle[2] - origin[2]];
      const along = dot(to, direction);
      if (along <= 0 || dot(to, to) - along * along > REACH * REACH) return false;
      for (let i = 0; i < 3; i++) {
        facing[i] = direction[i];
        anchor[i] = origin[i] + direction[i] * along;
      }
      chain.hold(chain.localOf(anchor));
      return true;
    },
    /** Moves the hand to where a pointer's ray meets its plane. */
    pull(origin: ArrayLike<number>, direction: ArrayLike<number>) {
      const square = dot(direction, facing);
      if (!chain.held || square < 1e-3) return;
      for (let i = 0; i < 3; i++) pulled[i] = anchor[i] - origin[i];
      const t = dot(pulled, facing) / square;
      for (let i = 0; i < 3; i++) pulled[i] = origin[i] + direction[i] * t;
      chain.pull(pulled);
    },
    /** Lets go, leaving the censer swinging as the hand threw it. */
    release() {
      chain.release();
    },
  };
}

export type Censer = ReturnType<typeof createCenser>;
