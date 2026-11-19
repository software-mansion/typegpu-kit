import * as table from '../table.ts';
import { unfilled } from '../inject.ts';
import { meshes } from '../geometry/index.ts';
import { d, std, tgpu } from 'typegpu';
import { influencesAt } from './meshPool.ts';

// Keep pose bindings here so instances can read them without importing the module that defines Pose

/**
 * This schema holds a unit quaternion, position and scale per axis. Instance transforms use
 * world space; joint transforms use parent space.
 */
export const Placement = d.struct({ rotation: d.vec4f, position: d.vec3f, scale: d.vec3f });
export type Placement = d.InferGPU<typeof Placement>;

/**
 * This schema selects a clip or line and whether it loops. Its cycle is at `offset + time * speed /
 * duration` cycles, where a line's duration mixes its clips' by `mix`. Clip `0xffffffff` plays
 * nothing.
 */
export const Playing = d.struct({
  clip: d.u32,
  loops: d.u32,
  offset: d.f32,
  speed: d.f32,
  mix: d.f32,
});

/**
 * This schema holds a layer's current and preceding plays. The fade starts at `fadeStart` and
 * lasts `fade` seconds.
 */
export const Track = d.struct({
  playing: Playing,
  previous: Playing,
  fadeStart: d.f32,
  fade: d.f32,
  weight: d.f32,
});

/**
 * This table holds a track for the base and for each of three layers. Each track is a field of its
 * own, so a CPU write to one leaves the others as kernels wrote them. `born` marks a pose added
 * since the last `animate`.
 */
export const Pose = table.define('Pose', {
  device: { track0: Track, track1: Track, track2: Track, track3: Track, born: d.u32 },
});

/**
 * This schema holds the time `animate` poses at and where the halves of the joints buffer start:
 * the one it writes this frame and the one it wrote the frame before.
 */
export const PoseTime = d.struct({ time: d.f32, now: d.u32, before: d.u32 });

const missing = unfilled('the poses', 'figures', 'a `poses.create`');
/** Reads joints in pose order. Each joint occupies the top three rows of its skinning matrix. */
export const poseJoints = tgpu.accessor(d.arrayOf(d.vec4f), missing);
export const jointCount = tgpu.accessor(d.u32, missing);
/** Gives the first row of the joints' half that vertices read. */
export const jointHalf = tgpu.accessor(d.u32, missing);

const Rows = d.struct({ x: d.vec4f, y: d.vec4f, z: d.vec4f });

const blended = (i: number, id: number) => {
  'use gpu';
  // Read jointCount first so missing injection reports the poses rather than their table
  const joints = jointCount.$;
  const influences = influencesAt(i);
  const first = Pose.slotOf(id) * joints;
  const half = jointHalf.$;
  let x = d.vec4f();
  let y = d.vec4f();
  let z = d.vec4f();
  for (const k of tgpu.unroll([0, 1, 2, 3])) {
    const at = half + (first + influences.joints[k]) * 3;
    const weight = influences.weights[k];
    x += poseJoints.$[at] * weight;
    y += poseJoints.$[at + 1] * weight;
    z += poseJoints.$[at + 2] * weight;
  }
  return Rows({ x, y, z });
};

/** Transforms `position` for vertex `i` by the weighted joints in pose `id`. */
export const skinnedPosition = (i: number, position: d.v3f, id: number) => {
  'use gpu';
  const rows = blended(i, id);
  const p = d.vec4f(position, 1);
  return d.vec3f(std.dot(rows.x, p), std.dot(rows.y, p), std.dot(rows.z, p));
};

/** Transforms the position and normal of vertex `i` by the weighted joints in pose `id`. */
export const skinned = (i: number, vertex: d.InferGPU<typeof meshes.Vertex>, id: number) => {
  'use gpu';
  const rows = blended(i, id);
  const p = d.vec4f(vertex.position, 1);
  const n = vertex.normal;
  return meshes.Vertex({
    position: d.vec3f(std.dot(rows.x, p), std.dot(rows.y, p), std.dot(rows.z, p)),
    normal: d.vec3f(std.dot(rows.x.xyz, n), std.dot(rows.y.xyz, n), std.dot(rows.z.xyz, n)),
    uv: vertex.uv,
  });
};
