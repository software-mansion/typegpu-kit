import { Camera, cameras, type OrbitCamera } from '@typegpu/kit';
import { recordingRoot } from '@typegpu/kit/testing';
import { d } from 'typegpu';
import { floatsOf } from './program.ts';

export type Vec3 = [number, number, number];

/** The eyes the trail keeps. */
export const TRAIL = 24;

/** Where the eye is in the uniform, in floats. */
const EYE = d.memoryLayoutOf(Camera, (camera) => camera.position).offset / 4;

interface From {
  readonly yaw: number;
  readonly pitch: number;
  readonly distance: number;
}

export function createProgram({ damping = 14 } = {}) {
  const recording = recordingRoot();
  const size = { width: 16, height: 9 };

  const create = (from: From, value: number) =>
    cameras.orbit(recording.root, {
      size,
      target: [0, 0, 0],
      ...from,
      pitchRange: [-0.3, 1.4],
      distanceRange: [3, 9.5],
      damping: value,
    });

  const uniformOf = (camera: OrbitCamera) => floatsOf(recording.bytes(camera.uniform.buffer));

  let orbit = create({ yaw: 0.7, pitch: 0.5, distance: 7 }, damping);
  let floats = uniformOf(orbit);
  const eye = (): Vec3 => [floats[EYE], floats[EYE + 1], floats[EYE + 2]];

  const trail: Vec3[] = [];
  let frames = 0;
  let moving = false;

  /** A frame: `update(dt)`, and whether it wrote the uniform. */
  const frame = (dt: number) => {
    const writes = recording.log.writes.length;
    orbit.update(dt);
    moving = recording.log.writes.length > writes;
    frames += 1;

    if (moving) trail.push(eye());
    if (trail.length > TRAIL) trail.shift();
  };

  frame(0);

  return {
    /** Where the camera is heading; write it to move the camera. */
    get goal() {
      return orbit.goal;
    },

    /** The eye, from the uniform. */
    eye,
    /** The eye at the frames it moved in, oldest first. */
    trail: trail as readonly Vec3[],

    get frames() {
      return frames;
    },

    get moving() {
      return moving;
    },

    /** Every uniform write so far. */
    get writes() {
      return recording.log.writes.length;
    },

    frame,

    /** Damping is an option, so a new value takes a new camera, starting where the old one is. */
    damp(value: number) {
      const [x, y, z] = eye();
      const distance = Math.hypot(x, y, z);
      const next = create(
        { yaw: Math.atan2(x, z), pitch: Math.asin(y / distance), distance },
        value,
      );

      Object.assign(next.goal, orbit.goal);
      orbit = next;
      floats = uniformOf(orbit);
      frame(0);
    },

    destroy() {},
  };
}
