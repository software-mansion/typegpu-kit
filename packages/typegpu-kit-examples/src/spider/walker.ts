import { clamp, deltaAngle, quat, type Quat, vec3, type Vec3 } from 'math';
import { fabrik3 } from 'math/ik';
import { spring, spring3 } from 'math/time';

/** Leg segments ordered from the hip through coxa, femur and tibia. */
const BONES = [0.5, 1, 1.35] as const;

/** Initial gait settings, editable while walking. */
export const GAIT = {
  /** World units a second the body walks at most. */
  speed: 1.8,
  /** Maximum planted-foot displacement before stepping. */
  stride: 0.75,
  /** Seconds a step takes. */
  step: 0.2,
  /** Foot lift height in world units. */
  lift: 0.4,
};
export type Gait = typeof GAIT;

/** Planted-foot tolerance as a fraction of stride during its group's step. */
const PLANTED = 0.15;
/** Hip offsets from the body centre, laterally and longitudinally, ordered front to back. */
const HIPS = { across: 0.32, along: [0.42, 0.14, -0.14, -0.42] };
/** Forward or backward leg splay per unit longitudinal hip offset. */
const SPLAY = 1.4;
/** Horizontal foot distance from its hip. */
const SPREAD = 1.65;
/** Body height above feet and height-following rate per second. */
const RIDE = 0.75;
const RISING = 8;
/** Foot spans front to back and across in world units, turning height differences into tilt. */
const SPAN = { along: 1.2, across: 1.4 };
/** Seconds the body takes to come to a stop. */
const SETTLE = 1;
/** Maximum turn rate in radians per second and turning response time in seconds. */
const TURN = 1.6;
const TURNING = 0.35;
/** Foot-target lookahead in seconds after liftoff. */
const LEAD = 0.25;
/** World units a second below which the body keeps its heading instead of turning to its motion. */
const STILL = 0.15;

const UP: Vec3 = [0, 1, 0];

/** Bone angle limits below and above horizontal, in radians. */
const RISE = {
  coxa: [0.2, 1.05],
  femur: [0.5, 1.4],
  tibia: [2.6, 0.2],
} satisfies Record<string, readonly [number, number]>;
const BASE_HINGE = fabrik3.BaseboneConstraintType.GLOBAL_HINGE;
const { GLOBAL_HINGE } = fabrik3.JointType;

interface Leg {
  readonly chain: fabrik3.Chain3;
  /** Hip position in body coordinates. */
  readonly hip: Vec3;
  /** Horizontal leg reach direction in body coordinates. */
  readonly out: Vec3;
  /** Two stepping groups, with neighbouring legs in opposite groups. */
  readonly group: number;
  readonly foot: Vec3;
  readonly from: Vec3;
  readonly to: Vec3;
  /** Step progress from 0 to 1. Negative while planted. */
  step: number;
}

/**
 * Uses pmndrs/math CPU arrays for a spring-driven body and FABRIK legs. Feet stay planted until
 * they exceed their stride. Reads `gait` each step.
 */
export function createWalker(heightAt: (x: number, z: number) => number, gait: Readonly<Gait>) {
  const ground = (point: Vec3) => (point[1] = heightAt(point[0], point[2]));

  const body = spring3.create([0, heightAt(0, 0) + RIDE, 0]);
  const heading = spring.create(0);
  const rotation: Quat = quat.create();
  const goal: Vec3 = [0, 0, 0];

  const world = (out: Vec3, local: Vec3) =>
    vec3.add(out, vec3.transformQuat(out, local, rotation), body.value);

  const legs: Leg[] = [-1, 1].flatMap((side) =>
    HIPS.along.map((along, k): Leg => {
      const hip: Vec3 = [side * HIPS.across, 0, along];
      const out = vec3.normalize(vec3.create(), [side, 0, along * SPLAY]);
      const chain = fabrik3.createChain3();
      chain.maxIterations = 8;
      fabrik3.addBone(chain, [0, 0, 0], [0, BONES[0], 0]);
      fabrik3.addConsecutiveBone(chain, UP, BONES[1]);
      fabrik3.addConsecutiveBone(chain, UP, BONES[2]);
      return {
        chain,
        hip,
        out,
        group: (k + (side > 0 ? 1 : 0)) % 2,
        foot: vec3.create(),
        from: vec3.create(),
        to: vec3.create(),
        step: -1,
      };
    }),
  );

  const at = vec3.create();
  const across = vec3.create();
  const home = vec3.create();
  const lead = vec3.create();
  const ahead = vec3.create();

  /** Predicts the resting foot position after `seconds` of body movement and rotation. */
  const homeOf = (leg: Leg, out: Vec3, seconds = LEAD) => {
    vec3.scaleAndAdd(at, leg.hip, leg.out, SPREAD);
    vec3.transformQuat(at, at, rotation);
    const turn = heading.velocity * seconds;
    const [x, , z] = at;
    out[0] = x * Math.cos(turn) + z * Math.sin(turn) + body.value[0] + body.velocity[0] * seconds;
    out[2] = z * Math.cos(turn) - x * Math.sin(turn) + body.value[2] + body.velocity[2] * seconds;
    ground(out);
    return out;
  };

  /** Raises the knees into arches so the first FABRIK solve bends the right way. */
  const arch = (leg: Leg) => {
    const [coxa, femur, tibia] = leg.chain.bones;
    world(coxa!.start, leg.hip);
    vec3.transformQuat(across, leg.out, rotation);
    vec3.scaleAndAdd(coxa!.end, coxa!.start, across, BONES[0] * 0.7);
    coxa!.end[1] += BONES[0] * 0.7;
    vec3.copy(femur!.start, coxa!.end);
    vec3.scaleAndAdd(femur!.end, femur!.start, across, BONES[1] * 0.8);
    femur!.end[1] += BONES[1] * 0.6;
    vec3.copy(tibia!.start, femur!.end);
    vec3.copy(tibia!.end, leg.foot);
  };

  for (const leg of legs) {
    homeOf(leg, leg.foot);
    arch(leg);
  }

  /** Starts a step for each displaced foot in `group` and returns how many lifted. */
  const lift = (group: number) => {
    let lifted = 0;
    for (const leg of legs) {
      if (leg.group !== group) continue;
      if (vec3.distance(leg.foot, homeOf(leg, home)) < gait.stride * PLANTED) continue;
      vec3.copy(leg.from, leg.foot);
      vec3.copy(leg.to, home);
      leg.step = 0;
      lifted += 1;
    }
    return lifted;
  };

  /** First overextended planted leg, or none while any leg is stepping. */
  const strayedLeg = () => {
    for (const leg of legs) if (leg.step >= 0) return undefined;
    for (const leg of legs) {
      if (vec3.distance(leg.foot, homeOf(leg, home)) > gait.stride) return leg;
    }
    return undefined;
  };

  /**
   * Constrains the leg to the vertical plane through hip and foot. Positive angles rise above the
   * floor: coxa and femur rise while the tibia falls, keeping the knee up.
   */
  const hinge = (leg: Leg) => {
    vec3.subtract(across, leg.foot, leg.chain.base);
    across[1] = 0;
    vec3.normalize(across, across);
    const axis = vec3.normalize(at, vec3.cross(at, across, UP));
    const [, femur, tibia] = leg.chain.bones;
    fabrik3.setBaseboneHingeConstraint(leg.chain, BASE_HINGE, axis, ...RISE.coxa, across);
    fabrik3.setHingeJoint(femur!.joint, GLOBAL_HINGE, axis, ...RISE.femur, across);
    fabrik3.setHingeJoint(tibia!.joint, GLOBAL_HINGE, axis, ...RISE.tibia, across);
  };

  /** Springs the body toward the goal and turns it to its motion, capping speed and turn rate. */
  const steer = (dt: number) => {
    ground(goal);
    vec3.subtract(ahead, goal, body.value);
    ahead[1] = 0;
    const far = vec3.length(ahead);
    vec3.scaleAndAdd(ahead, body.value, ahead, Math.min(1, (gait.speed * SETTLE) / (far || 1)));
    spring3.damp(body, ahead, SETTLE, dt);

    const speed = Math.hypot(body.velocity[0], body.velocity[2]);
    const facing = speed > STILL ? Math.atan2(body.velocity[0], body.velocity[2]) : heading.value;
    const turn = clamp(deltaAngle(heading.value, facing), -TURN * TURNING, TURN * TURNING);
    spring.dampAngle(heading, heading.value + turn, TURNING, dt);
  };

  /** Rides at the feet's mean height, tilted by their height differences along and across. */
  const ride = (dt: number) => {
    const half = legs.length / 2;
    let height = 0;
    let pitch = 0;
    let roll = 0;
    for (let i = 0; i < legs.length; i++) {
      const leg = legs[i];
      height += leg.foot[1] / legs.length;
      pitch += ((leg.hip[2] > 0 ? 1 : -1) * leg.foot[1]) / half;
      roll += ((i < half ? -1 : 1) * leg.foot[1]) / half;
    }
    body.value[1] += (height + RIDE - body.value[1]) * clamp(dt * RISING, 0, 1);
    quat.setAxisAngle(rotation, UP, heading.value);
    quat.rotateX(rotation, rotation, -Math.atan2(pitch, SPAN.along));
    quat.rotateZ(rotation, rotation, Math.atan2(roll, SPAN.across));
  };

  /** The group that stepped last. */
  let stepped = 1;

  /**
   * Lifts the other group once a planted foot strays, or the stray's own group if the other has
   * nothing to step, so one group can't take every step in fast turns.
   */
  const pace = () => {
    const strayed = strayedLeg();
    if (!strayed) return;
    if (lift(1 - stepped) > 0) stepped = 1 - stepped;
    else if (lift(strayed.group) > 0) stepped = strayed.group;
  };

  return {
    legs: legs as readonly Leg[],
    /** Body position and rotation. Read-only to callers. */
    body: body.value,
    rotation,

    /** Ground-plane walking target. */
    goal,

    step(dt: number) {
      steer(dt);
      ride(dt);
      pace();

      for (const leg of legs) {
        if (leg.step >= 0) {
          // Update foot targets during steps as the body moves and turns
          homeOf(leg, leg.to, LEAD - leg.step * gait.step);
          leg.step = Math.min(1, leg.step + dt / gait.step);
          const t = leg.step * leg.step * (3 - 2 * leg.step);
          vec3.lerp(leg.foot, leg.from, leg.to, t);
          leg.foot[1] += Math.sin(leg.step * Math.PI) * gait.lift;
          if (leg.step === 1) leg.step = -1;
        }

        world(lead, leg.hip);
        fabrik3.setBaseLocation(leg.chain, lead);
        hinge(leg);
        fabrik3.solve(leg.chain, leg.foot);
      }
    },
  };
}
