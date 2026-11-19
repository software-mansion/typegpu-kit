import { clamp } from 'math';
import { mat4, vec3, type Mat4, type Vec3, type Vec3Arg } from 'wgpu-matrix';

type Triple = readonly [number, number, number];

/** Hinge at the end of the preceding link. */
export interface Joint {
  /**
   * Position in the preceding joint's rotated frame. The first joint uses world coordinates on
   * the bench.
   */
  readonly offset: Triple;
  /** Rotation axis in joint-local coordinates. */
  readonly axis: Triple;
  /** Radians it may turn either way from straight. */
  readonly range: readonly [number, number];
}

const UP: Triple = [0, 1, 0];
const ACROSS: Triple = [1, 0, 0];

/** Link lengths in world units for the pedestal, shoulder housing, upper arm and forearm. */
export const LINKS = { pedestal: 0.3, housing: 0.4, upper: 1.5, fore: 1.25 };

/** Turntable, shoulder, elbow and wrist. Shoulder and elbow bend mainly forward. */
export const JOINTS: readonly Joint[] = [
  { offset: [0, LINKS.pedestal, 0], axis: UP, range: [-Infinity, Infinity] },
  { offset: [0, LINKS.housing, 0], axis: ACROSS, range: [-0.5, 1.7] },
  { offset: [0, LINKS.upper, 0], axis: ACROSS, range: [0, 2.6] },
  { offset: [0, LINKS.fore, 0], axis: ACROSS, range: [-2.2, 2.2] },
];

/** Index of the wrist, the last joint. */
export const WRIST = JOINTS.length - 1;

/** Initial joint angles in radians. */
const START: readonly number[] = [0.6, 0.5, 1.2, 0.8];

/** Offset from wrist to fingertips. */
export const TIP: Triple = [0, 0.84, 0];

/** The gripper's boxes, each centred `at` along the wrist's +y, which points to the tip. */
export const HAND = {
  palm: { at: 0.22, width: 0.9, height: 0.14, depth: 0.26 },
  finger: { at: 0.565, width: 0.07, height: 0.55, depth: 0.2 },
  /** Distance of each finger from the palm centre while open. */
  open: 0.41,
};

/** Removes the part of `v` along unit `axis`, in place. */
const flatten = (v: Vec3, axis: Vec3) => vec3.addScaled(v, axis, -vec3.dot(v, axis), v);

/**
 * Solves the hinge chain from `START` on the CPU. `frames` contains rotated world transforms,
 * and `tip` is the chain endpoint.
 */
export function createArm() {
  const angles = [...START];
  const frames: Mat4[] = JOINTS.map(() => mat4.identity());
  const tip = vec3.create();

  const at = vec3.create();
  const axis = vec3.create();
  const end = vec3.create();
  const from = vec3.create();
  const to = vec3.create();
  const across = vec3.create();

  const pose = () => {
    for (let i = 0; i < JOINTS.length; i++) {
      const joint = JOINTS[i];
      const frame = frames[i];
      if (i === 0) mat4.translation(joint.offset, frame);
      else mat4.translate(frames[i - 1], joint.offset, frame);
      mat4.axisRotate(frame, joint.axis, angles[i], frame);
    }
    vec3.transformMat4(TIP, frames[WRIST], tip);
  };
  pose();

  /** Rotates joint `i` to bring `from` as close to `to` as its hinge axis and range allow. */
  const turn = (i: number) => {
    const joint = JOINTS[i];
    vec3.normalize(vec3.transformMat4Upper3x3(joint.axis, frames[i], axis), axis);
    flatten(from, axis);
    flatten(to, axis);

    const sine = vec3.dot(axis, vec3.cross(from, to, across));
    const turned = angles[i] + Math.atan2(sine, vec3.dot(from, to));
    angles[i] = clamp(turned, joint.range[0], joint.range[1]);
    pose();
  };

  return {
    angles,
    frames: frames as readonly Mat4[],
    tip: tip as Readonly<Vec3>,

    /** Recomputes frames and tip after direct changes to `angles`. */
    pose,

    /**
     * Runs `sweeps` cyclic-coordinate-descent iterations to bring joint `last` toward `goal`.
     * Adjusts preceding joints from nearest to farthest.
     */
    solve(goal: Vec3Arg, last: number, sweeps: number) {
      for (let sweep = 0; sweep < sweeps; sweep++) {
        for (let i = last - 1; i >= 0; i--) {
          mat4.getTranslation(frames[last], end);
          mat4.getTranslation(frames[i], at);
          vec3.sub(end, at, from);
          vec3.sub(goal, at, to);
          turn(i);
        }
      }
    },

    /** Turns joint `i` to bring the tip as close to `direction` as possible. */
    aim(i: number, direction: Vec3Arg) {
      mat4.getTranslation(frames[i], at);
      vec3.sub(tip, at, from);
      vec3.copy(direction, to);
      turn(i);
    },
  };
}
