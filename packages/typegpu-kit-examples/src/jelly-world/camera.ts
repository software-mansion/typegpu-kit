import { cameras, type Surface } from '@typegpu/kit';
import { std, type TgpuRoot } from 'typegpu';
import { MIDDLE } from './tray.ts';

/**
 * The camera's poses over the bare tray and in the revealed world: its pitch and its target's
 * offset in x and z from the tray's middle, as pairs for views `aspects[1]` wide or wider and
 * `aspects[0]` or narrower, blended between. A pose backs off to fit `across` to the width, no
 * closer than `nearest`. The host's orbit turns the view within `pitches` and zooms it within
 * `distances`, or out to the bare tray's distance where that is farther.
 */
const VIEW = {
  bare: { pitch: [0.958, 0.958], x: [0, 0.08], z: [0, 0], across: 3.2, nearest: 2.9 },
  world: { pitch: [0.55, 0.9], x: [0, 0], z: [0, -0.5], across: 2.7, nearest: 1.5 },
  aspects: [0.5, 1.3],
  fov: Math.PI / 4,
  pitches: [0.08, 1.45] as const,
  distances: [0.8, 6],
};
/** Seconds the view takes to lower into the revealed world. */
const DESCENT = 2.4;

/** A camera pose: its pitch, distance and its target's offset from the tray's middle. */
interface Pose {
  pitch: number;
  distance: number;
  x: number;
  z: number;
}

const createPose = (): Pose => ({ pitch: 0, distance: 0, x: 0, z: 0 });
const mix = (from: number, to: number, t: number) => from + (to - from) * t;

/** Writes `pose` as fitted to a view `aspect` wide to 1 high, `portrait` of the way upright. */
function fit(pose: (typeof VIEW)['bare' | 'world'], aspect: number, portrait: number, out: Pose) {
  out.pitch = mix(pose.pitch[0], pose.pitch[1], portrait);
  out.distance = Math.max(pose.nearest, pose.across / aspect);
  out.x = mix(pose.x[0], pose.x[1], portrait);
  out.z = mix(pose.z[0], pose.z[1], portrait);
}

/** Writes the poses over the bare tray and in the revealed world for a view `aspect` wide. */
function framing(aspect: number, out: { bare: Pose; world: Pose }) {
  const portrait = 1 - std.smoothstep(VIEW.aspects[0], VIEW.aspects[1], aspect);
  fit(VIEW.bare, aspect, portrait, out.bare);
  fit(VIEW.world, aspect, portrait, out.world);
  return out;
}

/**
 * The orbit camera over the tray. Until the host's orbit turns or zooms it, it follows the framing,
 * so it refits as the surface turns or resizes, and lowers into the jelly world once that is
 * revealed.
 */
export function createCamera(root: TgpuRoot, surface: Surface) {
  const aspect = () => surface.width / surface.height;
  const poses = { bare: createPose(), world: createPose() };
  const opening = framing(aspect(), poses).bare;
  const view = cameras.orbit(root, {
    size: surface,
    target: [MIDDLE[0] + opening.x, MIDDLE[1], MIDDLE[2] + opening.z],
    distance: opening.distance,
    pitch: opening.pitch,
    fov: VIEW.fov,
    near: 0.1,
    far: 20,
    pitchRange: VIEW.pitches,
    distanceRange: [VIEW.distances[0], Infinity],
  });
  /** Whether the view still follows the framing, and how far it has lowered into the world. */
  let framed = true;
  let descent = 0;
  /** The goal as the framing last left it, to tell when the host's orbit moved it. */
  const left = { yaw: view.goal.yaw, pitch: view.goal.pitch, distance: view.goal.distance };

  return {
    view,

    /**
     * Follows the framing, lowering into the world while it is `revealed`, until the host's orbit
     * moves the goal. Then keeps its distance within `distances` or the bare framing.
     */
    follow(dt: number, revealed: boolean) {
      const pose = view.goal;
      framed &&=
        pose.yaw === left.yaw && pose.pitch === left.pitch && pose.distance === left.distance;
      if (!framed) {
        const farthest = Math.max(VIEW.distances[1], framing(aspect(), poses).bare.distance);
        pose.distance = Math.min(pose.distance, farthest);
        return;
      }
      if (revealed) descent = Math.min(1, descent + dt / DESCENT);
      const k = std.smoothstep(0, 1, descent);
      const { bare, world } = framing(aspect(), poses);
      pose.pitch = left.pitch = mix(bare.pitch, world.pitch, k);
      pose.distance = left.distance = mix(bare.distance, world.distance, k);
      pose.target[0] = MIDDLE[0] + mix(bare.x, world.x, k);
      pose.target[2] = MIDDLE[2] + mix(bare.z, world.z, k);
    },
  };
}
