/**
 * This module animates skinned models by computing pose joints on the GPU each frame. It bakes
 * skin clips into a buffer. Each pose plays a clip or a line of clips on its base, blends from the
 * preceding one, and plays up to three layers over it, each masked to some joints and replacing or
 * adding. Vertices in an instance kind with `pose` follow those joints, and read last frame's in
 * code compiled with `history.previously`.
 *
 * @module
 * @unstable
 * @injects definition every pose's joints, which the vertices of a kind with a `pose` read, and the
 * plays the verbs write, for a pipeline piped `figures.inject()`
 * @binds 1 storage, 2 uniform: a skinned vertex, every pose's joints, the poses' slots and the
 * clock that says which half of the joints to read
 * @binds 2 storage, 2 uniform: `jointPosition`, every pose's joints, the joints' bind origins,
 * the poses' slots and the clock
 * @binds 2 storage, 2 uniform: a kernel calling `play`, `once`, `stop`, `mix`, `weigh` or
 * `seconds`, the plays, the clips, the clock and the poses' slots
 * @owns a table, 4 buffers, a uniform, a pipeline: the plays, the clips' keys and spans, the joints'
 * parents, inverse binds, bind origins and masks, every pose's joints in two halves, the clock,
 * and the kernel that finds them. Where compute runs on the CPU, three pipelines instead: one
 * finds each joint, and two copy fresh poses into the last frame's half and clear their `born`
 * @records `animate(pass, time)` once a frame in a compute pass, after the kernels that call the
 * verbs and before the passes that draw the poses: it uploads what changed and finds every pose's
 * joints.
 */

import * as backends from '../backends.ts';
import * as table from '../table.ts';
import { temporal } from '../history.ts';
import { unfilled, type Inject } from '../inject.ts';
import { type RowId } from '../table.ts';
import { d, std, tgpu, type TgpuComputePass, type TgpuRoot } from 'typegpu';
import { mat4d, quatd } from 'wgpu-matrix';
import type { ModelData } from './meshPool.ts';
import {
  PoseTime,
  jointCount,
  jointHalf,
  Placement,
  Playing,
  Pose,
  poseJoints,
  Track,
} from './skinning.ts';

export { Placement, skinned, skinnedPosition } from './skinning.ts';

type Triple = readonly [number, number, number];

/** This type describes a joint returned by `gltf.read`. */
export interface JointData {
  /** Names the joint targeted by clip channels. */
  readonly name: string;
  /** Gives the parent joint index, which must precede this joint. A root joint uses -1. */
  readonly parent: number;
  /** Sets the rest position in parent space. */
  readonly position: Triple;
  /** Sets rotation as a unit quaternion `[x, y, z, w]`. */
  readonly rotation: readonly [number, number, number, number];
  readonly scale: Triple;
  /**
   * Transforms model space to joint space at binding. The matrix contains 16 floats in
   * column-major order.
   */
  readonly inverseBind: ArrayLike<number>;
}

/** This type describes the joints that move a model, as returned by `gltf.read`. */
export interface SkinData {
  readonly joints: readonly JointData[];
}

/**
 * This type holds animation keys for a joint position, rotation or scale, as returned by
 * `gltf.read`.
 */
export interface ChannelData {
  /** Names the joint that this channel moves. */
  readonly joint: string;
  readonly path: 'position' | 'rotation' | 'scale';
  /**
   * Selects key interpolation. Each `'cubic'` key contains an incoming tangent, value and
   * outgoing tangent.
   */
  readonly interpolation: 'linear' | 'step' | 'cubic';
  /** Lists key times in seconds, in increasing order. */
  readonly times: ArrayLike<number>;
  /**
   * Holds three floats per key for position or scale, or four for rotation. Cubic interpolation
   * uses three times as many.
   */
  readonly values: ArrayLike<number>;
}

/**
 * This type describes an animation returned by `gltf.read`. Joints without channels stay in
 * their rest transforms.
 */
export interface ClipData {
  /** Names the clip, from the glTF animation. */
  readonly name: string;
  /** Gives the time of the last key in seconds. */
  readonly duration: number;
  /** Lists the keys of each joint position, rotation and scale the clip moves. */
  readonly channels: readonly ChannelData[];
}

/**
 * This clip has no keys and keeps every joint at rest. Use it for poses driven only by `local`.
 */
export const rest: ClipData = { name: 'rest', duration: 0, channels: [] };

/** These options describe a layer played over the base. */
export interface LayerOptions {
  /**
   * Names the joints whose subtrees the layer moves, such as `['spine_02']` for the upper body.
   * Throws for a name the skin lacks. Default every joint.
   */
  readonly joints?: readonly string[];
  /**
   * Adds how the layer's clip differs from its first key to the pose beneath, rather than
   * replacing it. Default `false`.
   */
  readonly add?: boolean;
}

export interface Options<L extends string = string> {
  /** Limits the pose count to an integer from 1 to 65 535. */
  readonly capacity: number;
  /**
   * Sets a finite, positive number of baked keys per second. Poses blend the two keys around
   * their playback time. Default 30.
   */
  readonly hz?: number;
  /**
   * Names up to three layers over the base, applied in this order. `base` names the base and
   * can't name a layer. Default none.
   */
  readonly layers?: Readonly<Record<L, LayerOptions>>;
  /**
   * Returns a joint transform in parent space after clip playback and layers. The callback
   * receives the played transform, joint index and pose slot. Use it to aim a head, reach towards
   * a target or read a simulation buffer indexed by `slotOf`. Default the clip transform.
   * @gpu
   */
  readonly local?: (played: Placement, joint: number, slot: number) => Placement;
}

/** These options control clip playback. `L` names the layers besides the base, none by default. */
export interface Play<L extends string = never> {
  /** Names the layer that plays the clip. Default `'base'`. */
  readonly layer?: L | 'base';
  /**
   * Scales playback speed. A value of 1 uses the original speed, 0 pauses and negative values
   * play backwards. Default 1.
   */
  readonly speed?: number;
  /**
   * Controls whether playback repeats. A value of `false` holds the final key at the end.
   * Default `true`.
   */
  readonly loop?: boolean;
  /** Sets the starting offset within the clip or cycle in seconds. Default 0. */
  readonly from?: number;
  /**
   * Sets the blend duration from the preceding clip in seconds. A layer that shows only in part,
   * fading in or out, fades the clip in from the share it shows. `add` ignores it. Default 0.
   */
  readonly fade?: number;
  /**
   * Sets how much of the layer shows, from 0 to 1, as `weigh` does. The base ignores it. Default
   * the layer's weight, which starts at 1.
   */
  readonly weight?: number;
  /**
   * Places a line between its clips: 0 plays the first, 1 the second and 1.5 halfway between the
   * second and third. A clip ignores it. Default 0.
   */
  readonly mix?: number;
}

/** These options control how a layer stops. `L` names the layers besides the base. */
export interface Stop<L extends string = never> {
  /** Names the layer that stops. The base doesn't stop, so stopping it throws. Default `'base'`. */
  readonly layer?: L | 'base';
  /**
   * Sets the fade back to the pose beneath in seconds. A layer still fading in fades out from the
   * share it shows. Default 0.
   */
  readonly fade?: number;
}

/**
 * This id identifies a pose for an instance `pose` field. Removing the pose makes the id stale.
 */
export type Id = RowId<typeof Pose>;

/**
 * This set holds poses of one skin. Each pose plays an entry named by `C` on its base and on the
 * layers named by `L`.
 */
export interface Poses<C extends string = string, L extends string = string> extends Disposable {
  /** Gives the maximum pose count. */
  readonly capacity: number;
  /** Gives the live pose count. */
  readonly count: number;
  /** Gives the number of joints placed by each pose. */
  readonly joints: number;
  /** Gives each clip's or line's index, for kernels calling `poses.play` and `poses.once`. */
  readonly clips: Readonly<Record<C, number>>;
  /** Gives each layer's index, with the base at 0, for kernels calling the verbs. */
  readonly layers: Readonly<Record<L | 'base', number>>;
  /**
   * Returns how many seconds one cycle of `clip` lasts at speed 1. A line's cycle lasts between its
   * clips' at `mix`, as `Play.mix` places it. Throws for a clip the poses lack.
   */
  duration(clip: C, mix?: number): number;
  /**
   * Adds a pose playing `clip` on its base from `time` on the clock passed to `animate`. Its
   * layers play nothing. Ignores `layer`, `weight` and `fade`. Throws if the set is full or the
   * clip was not supplied at creation, listing the clips it has.
   */
  add(clip: C, time: number, play?: Play<L>): Id;
  /**
   * Starts `clip` at `time` on the base or `play.layer`, and blends from the preceding clip over
   * `fade` seconds. An interrupted blend starts from the last selected clip. Throws for a clip or
   * layer the poses lack.
   */
  play(id: Id, clip: C, time: number, play?: Play<L>): void;
  /**
   * Places the line playing on the base or `on.layer` at `value` from `time`, keeping its phase. A
   * call with the value it has writes nothing.
   */
  mix(id: Id, value: number, time: number, on?: Pick<Play<L>, 'layer'>): void;
  /**
   * Sets how much of `on.layer` shows, from 0 to 1, at once. The base always shows whole, so
   * weighing it throws.
   */
  weigh(id: Id, weight: number, on?: Pick<Play<L>, 'layer'>): void;
  /** Fades `stop.layer` out over `stop.fade` seconds from `time`, back to the pose beneath. */
  stop(id: Id, time: number, stop?: Stop<L>): void;
  /** Returns how many seconds into its clip or cycle the base or `on.layer` is at `time`. */
  seconds(id: Id, time: number, on?: Pick<Play<L>, 'layer'>): number;
  /**
   * Returns where `joint` of pose `id` is at `time`, as `animate` places it: a column-major matrix
   * from the joint's space to the model's, with the joint's position in its last column. It plays
   * the clips, lines, layers, fades and weights set through these methods on the CPU, and leaves
   * out `local` and plays from kernels. Throws for a joint the skin lacks or a name two joints
   * share.
   */
  jointAt(id: Id, joint: string, time: number): d.m4x4f;
  /** Frees pose `id`, making the id stale. */
  remove(id: Id): void;
  /** Tells whether `id` names a live pose. */
  has(id: Id): boolean;
  /** Returns the pose slot passed to `local`. Throws for a stale id. */
  slotOf(id: Id): number;
  /**
   * Uploads changes and computes every pose joint at `time`, with one workgroup per pose and one
   * thread per joint. Record
   * this once per frame in a compute pass before drawing, also while paused, so last frame's
   * joints stay one frame old.
   */
  animate(pass: TgpuComputePass, time: number): void;
  /**
   * Fills joint and play resources for pipelines created from the configuration you pipe it
   * into. Apply this to the renderer of a kind with `pose`, and to kernels calling the verbs.
   */
  inject(): Inject;
  /** Compiles the kernel that finds the joints. Calling it is optional. */
  initAsync(): Promise<void>;
  /** Destroys the table and the key, span, rig and joint buffers. */
  destroy(): void;
}

const Key = Placement;
const KEY = 12;
/**
 * The first entries name each clip or line, at the index of its first clip's span and with the
 * count of its clips. Each clip's span follows, at its first key and with its key intervals.
 */
const Span = d.struct({ first: d.u32, count: d.u32, duration: d.f32 });
const Joint = d.struct({ inverseBind: d.mat4x4f, mask: d.vec4f, origin: d.vec3f, parent: d.i32 });
const NONE = 0xffffffff;
const LAYERS = 3;
const TRACKS = ['track0', 'track1', 'track2', 'track3'] as const;

const missing = unfilled('the poses', 'figures', 'a `poses.create`');
const spans = tgpu.accessor(d.arrayOf(Span), missing);
const rigged = tgpu.accessor(d.arrayOf(Joint), missing);
const clock = tgpu.accessor(PoseTime, missing);

const nlerp = (a: d.v4f, b: d.v4f, t: number) => {
  'use gpu';
  const near = std.select(b * -1, b, std.dot(a, b) >= 0);
  return std.normalize(std.mix(a, near, t));
};

const blend = (a: d.Infer<typeof Key>, b: d.Infer<typeof Key>, t: number) => {
  'use gpu';
  return Key({
    rotation: nlerp(a.rotation, b.rotation, t),
    position: std.mix(a.position, b.position, t),
    scale: std.mix(a.scale, b.scale, t),
  });
};

/** The product of two quaternions, `a` after `b`. */
const turned = (a: d.v4f, b: d.v4f) => {
  'use gpu';
  return d.vec4f(
    a.w * b.xyz + b.w * a.xyz + std.cross(a.xyz, b.xyz),
    a.w * b.w - std.dot(a.xyz, b.xyz),
  );
};

/** Adds `delta`, a difference from a first key, onto `key` by `weight`. */
const onto = (key: d.Infer<typeof Key>, delta: d.Infer<typeof Key>, weight: number) => {
  'use gpu';
  return Key({
    rotation: std.normalize(
      turned(nlerp(d.vec4f(0, 0, 0, 1), delta.rotation, weight), key.rotation),
    ),
    position: key.position + delta.position * weight,
    scale: key.scale * std.mix(d.vec3f(1), delta.scale, weight),
  });
};

/** How `key` differs from `first`, the difference `onto` adds. */
const differenceOf = (key: d.Infer<typeof Key>, first: d.Infer<typeof Key>) => {
  'use gpu';
  const back = first.rotation * d.vec4f(-1, -1, -1, 1);
  return Key({
    rotation: turned(key.rotation, back),
    position: key.position - first.position,
    scale: key.scale / first.scale,
  });
};

const Around = d.struct({ a: Span, b: Span, t: d.f32, duration: d.f32 });

/** The two clips of an entry around `mix`, the share of the second, and the cycle's length. */
const aroundOf = (playing: d.Infer<typeof Playing>) => {
  'use gpu';
  const entry = spans.$[playing.clip];
  const last = entry.count - 1;
  const mix = std.clamp(playing.mix, 0, d.f32(last));
  const k = std.min(d.u32(mix), std.max(last, 1) - 1);
  const a = Span(spans.$[entry.first + k]);
  const b = Span(spans.$[entry.first + std.min(k + 1, last)]);
  const t = mix - d.f32(k);
  return Around({ a, b, t, duration: std.max(std.mix(a.duration, b.duration, t), 1e-6) });
};

const phaseOf = (playing: d.Infer<typeof Playing>, now: number, duration: number) => {
  'use gpu';
  return playing.offset + (now * playing.speed) / duration;
};

const cycleOf = (playing: d.Infer<typeof Playing>, phase: number) => {
  'use gpu';
  return std.select(std.saturate(phase), std.fract(phase), playing.loops === 1);
};

const shareOf = (track: d.Infer<typeof Track>, now: number) => {
  'use gpu';
  if (track.fade <= 0) return d.f32(1);
  return std.saturate((now - track.fadeStart) / track.fade);
};

const slotTrack = (slot: number, layer: number) => {
  'use gpu';
  if (layer === 1) return Pose.track1(slot);
  if (layer === 2) return Pose.track2(slot);
  if (layer === 3) return Pose.track3(slot);
  return Pose.track0(slot);
};

const store = (slot: number, layer: number, track: d.Infer<typeof Track>) => {
  'use gpu';
  const row = Pose.device.$[slot];
  if (layer === 1) row.track1 = Track(track);
  else if (layer === 2) row.track2 = Track(track);
  else if (layer === 3) row.track3 = Track(track);
  else row.track0 = Track(track);
};

/** The first cycle of `clip` at `mix` 0, from `from` seconds, starting now at speed 1. */
const started = (clip: number, loops: number, from: number) => {
  'use gpu';
  const now = clock.$.time;
  const fresh = Playing({ clip, loops, offset: 0, speed: 1, mix: 0 });
  return Playing({
    clip,
    loops,
    offset: (from - now) / aroundOf(fresh).duration,
    speed: 1,
    mix: 0,
  });
};

/** How much of a track shows at `now`, before its weight and mask: 0 stopped, 1 whole. */
const shownOf = (track: d.Infer<typeof Track>, now: number) => {
  'use gpu';
  const share = shareOf(track, now);
  const fading = track.previous.clip !== NONE && share < 1;
  if (track.playing.clip !== NONE) return std.select(share, d.f32(1), fading);
  return std.select(d.f32(0), 1 - share, fading);
};

const begin = (id: number, layer: number, playing: d.Infer<typeof Playing>, fade: number) => {
  'use gpu';
  const slot = Pose.slotOf(id);
  const track = slotTrack(slot, layer);
  const time = clock.$.time;
  // A track shown whole blends from its clip, one shown in part fades in from that share
  const shown = shownOf(track, time);
  let previous = Playing(track.playing);
  let fadeStart = time;
  if (shown < 1) {
    previous = Playing({ clip: NONE, loops: 0, offset: 0, speed: 0, mix: 0 });
    fadeStart = time - fade * shown;
  }
  store(slot, layer, Track({ playing, previous, fadeStart, fade, weight: track.weight }));
};

/**
 * Plays entry `clip` looping on `layer` of pose `id` from `from` seconds, at speed 1, blending
 * from the preceding clip over `fade` seconds. Call it from a kernel created from
 * `root.pipe(figures.inject())` recorded before `animate`.
 */
export const play = /* @__PURE__ */ tgpu
  .fn([d.u32, d.u32, d.u32, d.f32, d.f32])((id, clip, layer, fade, from) => {
    'use gpu';
    if (layer > LAYERS) return;
    begin(id, layer, started(clip, d.u32(1), from), fade);
  })
  .$name('poses_play');

/** Plays entry `clip` once from its start on `layer` of pose `id`, holding its last key. */
export const once = /* @__PURE__ */ tgpu
  .fn([d.u32, d.u32, d.u32, d.f32])((id, clip, layer, fade) => {
    'use gpu';
    if (layer > LAYERS) return;
    begin(id, layer, started(clip, d.u32(0), d.f32(0)), fade);
  })
  .$name('poses_once');

/** Fades `layer` of pose `id` out over `fade` seconds. The base doesn't stop. */
export const stop = /* @__PURE__ */ tgpu
  .fn([d.u32, d.u32, d.f32])((id, layer, fade) => {
    'use gpu';
    if (layer === 0 || layer > LAYERS) return;
    const slot = Pose.slotOf(id);
    const track = slotTrack(slot, layer);
    if (track.playing.clip === NONE) return;
    const nothing = Playing({ clip: NONE, loops: 0, offset: 0, speed: 0, mix: 0 });
    const time = clock.$.time;
    // A layer still fading in fades out from the share it shows
    const shown = shownOf(track, time);
    store(
      slot,
      layer,
      Track({
        playing: nothing,
        previous: track.playing,
        fadeStart: time - fade * (1 - shown),
        fade,
        weight: track.weight,
      }),
    );
  })
  .$name('poses_stop');

/** Places the line on `layer` of pose `id` at `value`, keeping its phase. */
export const mix = /* @__PURE__ */ tgpu
  .fn([d.u32, d.f32, d.u32])((id, value, layer) => {
    'use gpu';
    if (layer > LAYERS) return;
    const slot = Pose.slotOf(id);
    const track = slotTrack(slot, layer);
    const was = track.playing;
    if (was.clip === NONE || was.mix === value) return;
    const moved = Playing({ clip: was.clip, loops: 0, offset: 0, speed: 0, mix: value });
    const turn = 1 / aroundOf(was).duration - 1 / aroundOf(moved).duration;
    const offset = was.offset + clock.$.time * was.speed * turn;
    const placed = Playing({
      clip: was.clip,
      loops: was.loops,
      offset,
      speed: was.speed,
      mix: value,
    });
    store(
      slot,
      layer,
      Track({
        playing: placed,
        previous: track.previous,
        fadeStart: track.fadeStart,
        fade: track.fade,
        weight: track.weight,
      }),
    );
  })
  .$name('poses_mix');

/** Sets how much of `layer` of pose `id` shows, from 0 to 1. */
export const weigh = /* @__PURE__ */ tgpu
  .fn([d.u32, d.f32, d.u32])((id, weight, layer) => {
    'use gpu';
    if (layer > LAYERS) return;
    const slot = Pose.slotOf(id);
    const track = slotTrack(slot, layer);
    if (track.weight === weight) return;
    store(
      slot,
      layer,
      Track({
        playing: track.playing,
        previous: track.previous,
        fadeStart: track.fadeStart,
        fade: track.fade,
        weight,
      }),
    );
  })
  .$name('poses_weigh');

/** Returns how many seconds into its clip or cycle `layer` of pose `id` is, or 0 when stopped. */
export const seconds = /* @__PURE__ */ tgpu
  .fn(
    [d.u32, d.u32],
    d.f32,
  )((id, layer) => {
    'use gpu';
    if (layer > LAYERS) return 0;
    const playing = slotTrack(Pose.slotOf(id), layer).playing;
    if (playing.clip === NONE) return 0;
    const duration = aroundOf(playing).duration;
    return cycleOf(playing, phaseOf(playing, clock.$.time, duration)) * duration;
  })
  .$name('poses_seconds');

/**
 * Returns where joint `joint` of pose `id` is in model space, as `animate` placed it last. Unlike
 * `jointAt`, it includes plays from kernels and `local`. Call it from a pipeline created from
 * `root.pipe(figures.inject())`.
 */
export const jointPosition = /* @__PURE__ */ tgpu
  .fn(
    [d.u32, d.u32],
    d.vec3f,
  )((id, joint) => {
    'use gpu';
    // Read jointCount first so missing injection reports the poses rather than their table
    const joints = jointCount.$;
    const at = jointHalf.$ + (Pose.slotOf(id) * joints + joint) * 3;
    const origin = d.vec4f(rigged.$[joint].origin, 1);
    return d.vec3f(
      std.dot(poseJoints.$[at], origin),
      std.dot(poseJoints.$[at + 1], origin),
      std.dot(poseJoints.$[at + 2], origin),
    );
  })
  .$name('poses_jointPosition');

const halfNow = () => {
  'use gpu';
  return clock.$.now;
};

const halfBefore = () => {
  'use gpu';
  return clock.$.before;
};

const nothing = () => ({ clip: NONE, loops: 0, offset: 0, speed: 0, mix: 0 });

const checkMix = (value: number) => {
  if (!Number.isFinite(value))
    throw new RangeError(`The \`mix\` value must be finite. Got ${value}.`);
};

const copy = (target: d.Infer<typeof Playing>, source: d.Infer<typeof Playing>) => {
  target.clip = source.clip;
  target.loops = source.loops;
  target.offset = source.offset;
  target.speed = source.speed;
  target.mix = source.mix;
};

const matrixOf = (key: d.Infer<typeof Key>) => {
  'use gpu';
  const q = key.rotation;
  const s = key.scale;
  return d.mat4x4f(
    d.vec4f(
      1 - 2 * (q.y * q.y + q.z * q.z),
      2 * (q.x * q.y + q.w * q.z),
      2 * (q.x * q.z - q.w * q.y),
      0,
    ) * s.x,
    d.vec4f(
      2 * (q.x * q.y - q.w * q.z),
      1 - 2 * (q.x * q.x + q.z * q.z),
      2 * (q.y * q.z + q.w * q.x),
      0,
    ) * s.y,
    d.vec4f(
      2 * (q.x * q.z + q.w * q.y),
      2 * (q.y * q.z - q.w * q.x),
      1 - 2 * (q.x * q.x + q.y * q.y),
      0,
    ) * s.z,
    d.vec4f(key.position, 1),
  );
};

/**
 * A key in plain numbers for `jointAt`: rotation x, y, z, w, then position, then scale. The
 * functions below compute what `nlerp`, `blend`, `turned`, `onto`, `differenceOf` and `matrixOf`
 * compute in the kernel. Calling those on the CPU builds TypeGPU values at every step, about a
 * hundred times slower. A GPU test holds the two in step.
 */
type PlainKey = Float64Array;

const mixed = (a: number, b: number, t: number) => a * (1 - t) + b * t;

/** Interpolates the rotations at the start of both inputs. */
function plainNlerp(a: ArrayLike<number>, b: PlainKey, t: number, out: Float64Array) {
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const sign = dot >= 0 ? 1 : -1;
  let length = 0;
  for (let c = 0; c < 4; c++) {
    out[c] = mixed(a[c], sign * b[c], t);
    length += out[c] ** 2;
  }
  length = Math.sqrt(length);
  for (let c = 0; c < 4; c++) out[c] = out[c] / length;
}

// Each writes `out`, which may be one of its keys: it reads a component before writing it

function plainBlend(a: PlainKey, b: PlainKey, t: number, out: PlainKey) {
  plainNlerp(a, b, t, out);
  for (let c = 4; c < 10; c++) out[c] = mixed(a[c], b[c], t);
}

const IDENTITY_TURN = [0, 0, 0, 1];

/** `onto`, with `part` holding the share of the delta's turn. */
function plainOnto(
  key: PlainKey,
  delta: PlainKey,
  weight: number,
  part: Float64Array,
  out: PlainKey,
) {
  plainNlerp(IDENTITY_TURN, delta, weight, part);
  quatd.multiply(part, key, out);
  const length = Math.hypot(out[0], out[1], out[2], out[3]);
  for (let c = 0; c < 4; c++) out[c] = out[c] / length;
  for (let c = 4; c < 7; c++) out[c] = key[c] + delta[c] * weight;
  for (let c = 7; c < 10; c++) out[c] = key[c] * mixed(1, delta[c], weight);
}

/** `differenceOf`, with `undone` holding the first key's inverse turn. */
function plainDifference(key: PlainKey, first: PlainKey, undone: Float64Array, out: PlainKey) {
  quatd.multiply(key, quatd.conjugate(first, undone), out);
  for (let c = 4; c < 7; c++) out[c] = key[c] - first[c];
  for (let c = 7; c < 10; c++) out[c] = key[c] / first[c];
}

/** `matrixOf`, column-major, as `fromQuat`, `scale` and `setTranslation` make it. */
function plainMatrix(key: PlainKey, out: Float64Array) {
  mat4d.fromQuat(key, out);
  for (let c = 0; c < 3; c++)
    for (let r = 0; r < 4; r++) out[c * 4 + r] = key[7 + c] * out[c * 4 + r];
  out[12] = key[4];
  out[13] = key[5];
  out[14] = key[6];
  return out;
}

/** Anything `valueAt` can write numbers into. */
type Written = { [index: number]: number };

/** Holds a rotation `valueAt` normalizes before it writes it. */
const turning = new Float64Array(4);

/** Writes `channel` at `time` into `out` from `offset`: three numbers, or four for a rotation. */
function valueAt(channel: ChannelData, time: number, out: Written, offset: number) {
  const { times, values, interpolation, path } = channel;
  const rotation = path === 'rotation';
  const width = rotation ? 4 : 3;
  const cubic = interpolation === 'cubic';
  // A cubic key holds an incoming tangent, its value and an outgoing tangent
  const stride = cubic ? 3 * width : width;
  const value = cubic ? width : 0;

  // Hold the first and last keys outside them, which leaves `high` at 0
  const last = times.length - 1;
  let low = 0;
  let high = 0;
  if (last > 0 && time > times[0]) {
    if (time >= times[last]) low = last;
    else {
      high = last;
      while (high - low > 1) {
        const middle = (low + high) >>> 1;
        if (times[middle] <= time) low = middle;
        else high = middle;
      }
    }
  }
  const a = low * stride + value;
  if (high === 0 || interpolation === 'step') {
    for (let c = 0; c < width; c++) out[offset + c] = values[a + c];
    return;
  }

  const b = high * stride + value;
  const span = times[high] - times[low];
  const t = (time - times[low]) / span;
  if (cubic) {
    const t2 = t * t;
    const t3 = t2 * t;
    const into = rotation ? turning : out;
    const at = rotation ? 0 : offset;
    for (let c = 0; c < width; c++) {
      into[at + c] =
        (2 * t3 - 3 * t2 + 1) * values[a + c] +
        (t3 - 2 * t2 + t) * span * values[a + width + c] +
        (-2 * t3 + 3 * t2) * values[b + c] +
        (t3 - t2) * span * values[b - width + c];
    }
    if (rotation) normalizedInto(out, offset);
    return;
  }

  if (!rotation) {
    for (let c = 0; c < width; c++)
      out[offset + c] = values[a + c] + (values[b + c] - values[a + c]) * t;
    return;
  }

  // Interpolate rotations along the great circle between keys, as glTF requires
  let cos = 0;
  for (let c = 0; c < 4; c++) cos += values[a + c] * values[b + c];
  const sign = cos < 0 ? -1 : 1;
  cos *= sign;
  if (cos > 0.9995) {
    for (let c = 0; c < 4; c++)
      turning[c] = values[a + c] + (sign * values[b + c] - values[a + c]) * t;
    normalizedInto(out, offset);
    return;
  }
  const angle = Math.acos(cos);
  const from = Math.sin((1 - t) * angle) / Math.sin(angle);
  const to = (sign * Math.sin(t * angle)) / Math.sin(angle);
  for (let c = 0; c < 4; c++) out[offset + c] = from * values[a + c] + to * values[b + c];
}

/** Writes `turning`, normalized, into `out` from `offset`. */
function normalizedInto(out: Written, offset: number) {
  quatd.normalize(turning, turning);
  for (let c = 0; c < 4; c++) out[offset + c] = turning[c];
}

/**
 * Samples `channel` at `time` seconds on the CPU as `create` bakes it: it holds a step, mixes
 * linear keys, slerps rotations and follows cubic splines, and holds the first and last keys
 * outside them. Returns three numbers, or four for a rotation, in `out` when given.
 *
 * @param channel the keys, as `gltf.read` returns them
 * @param time seconds into the clip
 * @param out the array it fills
 * @returns the position, rotation `[x, y, z, w]` or scale
 */
export function sample(channel: ChannelData, time: number, out: number[] = []): number[] {
  out.length = channel.path === 'rotation' ? 4 : 3;
  valueAt(channel, time, out, 0);
  return out;
}

/**
 * Returns the clip named `name` among a file's clips. Throws listing the names the file has.
 *
 * @param file what `gltf.read` returns, or anything holding `clips`
 * @param name the clip's name in the file
 * @returns the clip, as the file holds it
 */
function clipNamed(file: { readonly clips: readonly ClipData[] }, name: string): ClipData {
  const found = file.clips.find((each) => each.name === name);
  if (!found) {
    const all = file.clips.map((each) => `'${each.name}'`).join(', ') || 'none';
    throw new Error(`There is no clip named '${name}'; the clips are ${all}.`);
  }
  return found;
}
// Exported under another name so parameters named `clip` don't shadow it
export { clipNamed as clip };

/** These options resample a clip on the CPU. */
export interface Resample {
  /** Sets keys per second in the clip it returns. Default 30. */
  readonly hz?: number;
}

/** Evenly spaced key times from 0 to `duration`, at least two. */
const keyTimes = (duration: number, hz: number) => {
  const gaps = Math.max(Math.round(duration * hz), 1);
  return Float32Array.from({ length: gaps + 1 }, (_, k) => (k / gaps) * duration);
};

/**
 * Returns a copy of a looping clip that starts `by` seconds in, resampled into linear keys. Use
 * it to line up the clips of a line, so feet land at the same phase.
 *
 * @param clip the looping clip
 * @param by seconds into `clip` the copy starts; a negative value starts before its end
 * @returns a clip as long as `clip`, with every channel resampled
 */
export function shift(clip: ClipData, by: number, options: Resample = {}): ClipData {
  const { duration } = clip;
  if (!(duration > 0)) return clip;
  const times = keyTimes(duration, options.hz ?? 30);
  const channels = clip.channels.map((channel) => {
    const width = channel.path === 'rotation' ? 4 : 3;
    const values = new Float32Array(times.length * width);
    for (let k = 0; k < times.length; k++) {
      const at = (((times[k] + by) % duration) + duration) % duration;
      valueAt(channel, at, values, k * width);
    }
    return { ...channel, interpolation: 'linear' as const, times, values };
  });
  return { name: clip.name, duration, channels };
}

/** These options mirror a clip. */
export interface Mirror {
  /** Selects the axis that points from one side of the rig to the other. Default `'x'`. */
  readonly across?: 'x' | 'y' | 'z';
}

/**
 * Returns a copy of `clip` mirrored to the rig's other side. Each channel moves the joint
 * `opposite` names, its positions flip along `across`, and its rotations turn the other way
 * about the two other axes. The rig's sides must be mirror images, as in most humanoid rigs.
 *
 * @param clip the clip to mirror
 * @param opposite the joint on the other side, such as `(joint) => joint.replace(/\.r$/, '.l')`,
 *   or the joint itself for one in the middle
 * @returns a clip with a channel for each of `clip`'s, on the opposite joints
 */
export function mirror(
  clip: ClipData,
  opposite: (joint: string) => string,
  options: Mirror = {},
): ClipData {
  const axis = { x: 0, y: 1, z: 2 }[options.across ?? 'x'];
  const channels = clip.channels.map((channel) => {
    const values = Float32Array.from(channel.values);
    if (channel.path === 'rotation') {
      for (let i = 0; i < values.length; i += 4) {
        for (let c = 0; c < 3; c++) if (c !== axis) values[i + c] = -values[i + c];
      }
    } else if (channel.path === 'position') {
      for (let i = axis; i < values.length; i += 3) values[i] = -values[i];
    }
    return { ...channel, joint: opposite(channel.joint), values };
  });
  return { name: clip.name, duration: clip.duration, channels };
}

/** These options lean a clip. */
export interface Lean extends Resample {
  /** Gives the clip whose first key the lean is measured from. Default `rest`, the rest pose. */
  readonly from?: ClipData;
}

const jointNamed = (skin: SkinData, name: string) => {
  const at = skin.joints.findIndex((each) => each.name === name);
  if (at < 0) {
    const all = skin.joints.map((each) => `'${each.name}'`);
    throw new Error(`The skin has no joint named '${name}'; it has ${all.join(', ')}.`);
  }
  return at;
};

/** Each joint's rotation channel in `clip`, if it has one. */
const turnsOf = (clip: ClipData, skin: SkinData) =>
  skin.joints.map(({ name }) =>
    clip.channels.find((each) => each.joint === name && each.path === 'rotation'),
  );

/**
 * Writes the rotation of `joint` in model space at `time` into `frame`, from each joint's rotation
 * channel in `turns` or its rest rotation; identity above the root. `turn` holds each step.
 */
const frameOf = (
  turns: readonly (ChannelData | undefined)[],
  skin: SkinData,
  joint: number,
  time: number,
  turn: Float64Array,
  frame: Float64Array,
) => {
  quatd.identity(frame);
  for (let at = joint; at >= 0; at = skin.joints[at].parent) {
    const channel = turns[at];
    if (channel) valueAt(channel, time, turn, 0);
    else turn.set(skin.joints[at].rotation);
    quatd.multiply(turn, frame, frame);
  }
  return frame;
};

/**
 * Returns the channels of `clip` that move `top` and the joints beneath it, for a layer masked
 * to `top`. The rotation of `top` takes on how the joints above it turn in `clip`, measured from
 * the first key of `options.from`, so the layer turns its joints as the whole clip does. That
 * rotation is resampled into linear keys.
 *
 * @param clip the clip to keep a part of
 * @param skin the joints `clip` moves
 * @param top the joint at the top of the part, as the layer's `joints` names it
 * @returns a clip with the rotation of `top` first, then the channels beneath it
 */
export function lean(clip: ClipData, skin: SkinData, top: string, options: Lean = {}): ClipData {
  const at = jointNamed(skin, top);
  const above = skin.joints[at].parent;
  const turn = quatd.create();
  const frame = quatd.create();
  const from = turnsOf(options.from ?? rest, skin);
  const standing = quatd.conjugate(frameOf(from, skin, above, 0, turn, frame));
  const turns = turnsOf(clip, skin);
  const { duration } = clip;
  const times = keyTimes(duration, options.hz ?? 30);
  const values = new Float32Array(times.length * 4);
  for (let k = 0; k < times.length; k++) {
    const leaning = quatd.multiply(
      standing,
      frameOf(turns, skin, above, times[k], turn, frame),
      frame,
    );
    const own = turns[at];
    if (own) valueAt(own, times[k], turn, 0);
    else turn.set(skin.joints[at].rotation);
    quatd.multiply(leaning, turn, turn);
    values.set(turn, k * 4);
  }

  const beneath = (name: string) => {
    let joint = skin.joints.findIndex((each) => each.name === name);
    while (joint >= 0 && joint !== at) joint = skin.joints[joint].parent;
    return joint === at;
  };
  const kept = clip.channels.filter(
    (each) => beneath(each.joint) && !(each.joint === top && each.path === 'rotation'),
  );
  const topTurn: ChannelData = {
    joint: top,
    path: 'rotation',
    interpolation: 'linear',
    times,
    values,
  };
  return { name: clip.name, duration, channels: [topTurn, ...kept] };
}

// `Array.isArray` narrows only mutable arrays
const isLine = (entry: ClipData | readonly ClipData[]): entry is readonly ClipData[] =>
  Array.isArray(entry);

function bake(skin: SkinData, clips: readonly (readonly [string, ClipData])[], hz: number) {
  const { joints } = skin;
  const byName = new Map<string, number>();
  const shared = new Set<string>();
  joints.forEach(({ name }, i) => {
    if (byName.has(name)) shared.add(name);
    byName.set(name, i);
  });

  let keys = 0;
  const laid = clips.map(([key, clip]) => {
    if (!(clip.duration >= 0 && Number.isFinite(clip.duration))) {
      throw new RangeError(
        `Clip '${key}' has a duration of ${clip.duration} seconds. Use a finite duration of 0 or more seconds.`,
      );
    }

    const moved = joints.map((): Partial<Record<ChannelData['path'], ChannelData>> => ({}));
    let used = 0;
    for (const channel of clip.channels) {
      const joint = byName.get(channel.joint);
      if (joint === undefined) continue;
      if (shared.has(channel.joint)) {
        throw new Error(
          `The clip '${key}' moves '${channel.joint}', a name two of the skin's joints have.`,
        );
      }
      moved[joint][channel.path] = channel;
      used += 1;
    }
    if (clip.channels.length > 0 && used === 0) {
      const named = [...new Set(clip.channels.map((channel) => `'${channel.joint}'`))];
      throw new Error(
        `The clip '${key}' moves none of the skin's joints: it moves ${named.slice(0, 8).join(', ')}.`,
      );
    }

    const intervals = clip.duration > 0 ? Math.max(1, Math.round(clip.duration * hz)) : 0;
    const range = { first: keys, intervals, duration: clip.duration };
    keys += intervals + 1;
    return { moved, range };
  });

  const out = new Float32Array(keys * joints.length * KEY);
  let at = 0;
  for (const { moved, range } of laid) {
    for (let k = 0; k <= range.intervals; k++) {
      const time = range.intervals > 0 ? (k / range.intervals) * range.duration : 0;
      for (let j = 0; j < joints.length; j++) {
        const { position, rotation, scale } = moved[j];
        if (rotation) valueAt(rotation, time, out, at);
        else out.set(joints[j].rotation, at);
        if (position) valueAt(position, time, out, at + 4);
        else out.set(joints[j].position, at + 4);
        if (scale) valueAt(scale, time, out, at + 8);
        else out.set(joints[j].scale, at + 8);
        at += KEY;
      }
    }
  }
  return { keys: out, ranges: laid.map(({ range }) => range) };
}

/**
 * Creates poses of `skin` that play named `clips`, for example `{ idle: general.clips[6], walk
 * }`. An entry that is an array of clips is a line, its clips sampled at one phase and mixed by
 * `mix`. It bakes clips at `hz` keys per second and blends between adjacent keys during playback.
 * Call `animate` each frame to compute joints in model space before drawing instances. The skin
 * must have 1 to 256 joints in parent-first order, with 16-float inverse binds. Invalid joint
 * data, capacity, bake rate, clips, lines of fewer than two clips, more than three layers, a layer
 * named `base` or a layer joint the skin lacks throw.
 *
 * @param skin the joints, as `gltf.read` returns them in a model's `skin`
 * @param clips the clips and lines its poses play, by the names `add` and `play` take
 * @returns poses you animate every frame and destroy
 */
export function create<const C extends string, const L extends string = never>(
  root: TgpuRoot,
  skin: SkinData,
  clips: Readonly<Record<C, ClipData | readonly ClipData[]>>,
  options: Options<L>,
): Poses<C, L> {
  const { capacity, hz = 30, local } = options;
  const joints = skin.joints.length;
  if (joints < 1 || joints > 256)
    throw new RangeError(`A skin has 1 to 256 joints, got ${joints}.`);
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > 65_535) {
    throw new RangeError(
      `Pose capacity must be an integer from 1 to 65 535. Received ${capacity}.`,
    );
  }
  if (!(hz > 0 && Number.isFinite(hz))) {
    throw new RangeError(`Clip hz must be finite and greater than 0. Received ${hz}.`);
  }
  skin.joints.forEach(({ name, parent, inverseBind }, i) => {
    if (!Number.isInteger(parent) || parent < -1 || parent >= i) {
      throw new Error(
        `Joint ${i}, '${name}', has parent ${parent}; a parent comes before its children.`,
      );
    }
    if (inverseBind.length !== 16) {
      throw new Error(
        `Joint ${i}, '${name}', has ${inverseBind.length} inverse-bind floats. Supply 16.`,
      );
    }
  });

  const names = Object.keys(clips) as C[];
  if (names.length === 0) throw new Error('Poses need a clip to play.');
  const clipOf = new Map(names.map((name, i) => [name, i]));
  const lines = names.map((name): readonly ClipData[] => {
    const entry = clips[name];
    if (!isLine(entry)) return [entry];
    if (entry.length < 2) {
      throw new Error(`The line '${name}' has ${entry.length} clips; a line mixes two or more.`);
    }
    return entry;
  });
  const flat = lines.flatMap((line, i) =>
    line.map((clip, k) => [line.length > 1 ? `${names[i]}[${k}]` : names[i], clip] as const),
  );
  const baked = bake(skin, flat, hz);

  const layerOptions: Readonly<Record<string, LayerOptions>> = options.layers ?? {};
  const layerKeys = Object.keys(layerOptions);
  if (layerKeys.length > LAYERS) {
    throw new RangeError(`Poses have at most ${LAYERS} layers, got ${layerKeys.join(', ')}.`);
  }
  if (layerKeys.includes('base')) {
    throw new Error("Choose another layer name. Poses reserves 'base' for the base layer.");
  }
  const layerNames = layerKeys as L[];
  const layerOf = new Map<string, number>([
    ['base', 0],
    ...layerNames.map((name, i) => [name, i + 1] as const),
  ]);

  const byName = new Map(skin.joints.map(({ name }, i) => [name, i]));
  const masks = new Float32Array(joints * 4);
  for (let j = 0; j < joints; j++) masks[j * 4] = 1;
  layerNames.forEach((name, i) => {
    const named = layerOptions[name].joints;
    if (!named) {
      for (let j = 0; j < joints; j++) masks[j * 4 + i + 1] = 1;
      return;
    }
    for (const joint of named) {
      const at = byName.get(joint);
      if (at === undefined) {
        const all = skin.joints.map(({ name: each }) => `'${each}'`);
        throw new Error(
          `The layer '${name}' moves '${joint}', and the skin has no joint of that name; it has ${all.join(', ')}.`,
        );
      }
      masks[at * 4 + i + 1] = 1;
    }
    // Parents come first, so one pass carries each subtree's mask down
    skin.joints.forEach(({ parent }, j) => {
      if (parent >= 0 && masks[parent * 4 + i + 1] === 1) masks[j * 4 + i + 1] = 1;
    });
  });

  const plays = table.create(root, Pose, { capacity });
  const clockData = new ArrayBuffer(12);
  const clockTime = new Float32Array(clockData, 0, 1);
  const clockHalves = new Uint32Array(clockData, 4, 2);
  const clockUniform = root.createUniform(PoseTime);
  const keyData = root
    .createBuffer(d.arrayOf(Key, baked.keys.length / KEY))
    .$usage('storage')
    .$name('poses_keys');

  // Entries first, then each clip's span after them
  let next = names.length;
  const entries = lines.map((line) => {
    const entry = { first: next, count: line.length, duration: 0 };
    next += line.length;
    return entry;
  });
  const spanList = [
    ...entries,
    ...baked.ranges.map(({ first, intervals, duration }) => ({
      first,
      count: intervals,
      duration,
    })),
  ];
  const spanData = root
    .createBuffer(d.arrayOf(Span, next), spanList)
    .$usage('storage')
    .$name('poses_spans');
  const rigData = root.createBuffer(d.arrayOf(Joint, joints)).$usage('storage').$name('poses_rig');
  const half = capacity * joints * 3;
  const jointData = root
    .createBuffer(d.arrayOf(d.vec4f, half * 2))
    .$usage('storage')
    .$name('poses_joints');

  const { queue } = root.device;
  queue.writeBuffer(root.unwrap(keyData), 0, baked.keys);
  const stride = d.sizeOf(Joint) / 4;
  const originAt = d.memoryLayoutOf(Joint, (joint) => joint.origin).offset / 4;
  const parentAt = d.memoryLayoutOf(Joint, (joint) => joint.parent).offset / 4;
  const rigFloats = new Float32Array(joints * stride);
  const rigInts = new Int32Array(rigFloats.buffer);
  skin.joints.forEach(({ inverseBind, parent }, j) => {
    rigFloats.set(inverseBind, j * stride);
    rigFloats.set(masks.subarray(j * 4, j * 4 + 4), j * stride + 16);
    // The bind's last column is where the joint sits in the model at binding
    const bind = mat4d.inverse(Float64Array.from(inverseBind));
    rigFloats.set(bind.subarray(12, 15), j * stride + originAt);
    rigInts[j * stride + parentAt] = parent;
  });
  queue.writeBuffer(root.unwrap(rigData), 0, rigFloats);
  const clockGpu = root.unwrap(clockUniform.buffer);

  const keys = keyData.as('readonly');
  const rig = rigData.as('readonly');
  const written = jointData.as('mutable');

  const keyAt = (span: d.Infer<typeof Span>, cycle: number, joint: number) => {
    'use gpu';
    const frame = cycle * d.f32(span.count);
    const before = std.min(d.u32(frame), std.max(span.count, 1) - 1);
    const after = std.min(before + 1, span.count);
    const a = keys.$[(span.first + before) * joints + joint];
    const b = keys.$[(span.first + after) * joints + joint];
    return blend(a, b, frame - d.f32(before));
  };

  const sampled = (playing: d.Infer<typeof Playing>, now: number, joint: number) => {
    'use gpu';
    const around = aroundOf(playing);
    const cycle = cycleOf(playing, phaseOf(playing, now, around.duration));
    let key = keyAt(around.a, cycle, joint);
    if (around.t > 0) key = blend(key, keyAt(around.b, cycle, joint), around.t);
    return key;
  };

  /** How a clip or line differs from its first key, at its phase. */
  const delta = (playing: d.Infer<typeof Playing>, now: number, joint: number) => {
    'use gpu';
    const around = aroundOf(playing);
    const cycle = cycleOf(playing, phaseOf(playing, now, around.duration));
    let key = keyAt(around.a, cycle, joint);
    let first = Key(keys.$[around.a.first * joints + joint]);
    if (around.t > 0) {
      key = blend(key, keyAt(around.b, cycle, joint), around.t);
      first = blend(first, keys.$[around.b.first * joints + joint], around.t);
    }
    return differenceOf(key, first);
  };

  const based = (slot: number, joint: number, now: number) => {
    'use gpu';
    const track = Pose.track0(slot);
    let key = sampled(track.playing, now, joint);
    const share = shareOf(track, now);
    if (share < 1) key = blend(sampled(track.previous, now, joint), key, share);
    return key;
  };

  const getters = [Pose.track1, Pose.track2, Pose.track3] as const;
  const layers = layerNames.map((name, i) => {
    const add = layerOptions[name].add === true;
    return { read: getters[i], lane: i + 1, key: add ? delta : sampled, lay: add ? onto : blend };
  });

  /** A joint's transform in its parent's space: the base with each layer laid over it. */
  const placedOf = (slot: number, joint: number, now: number) => {
    'use gpu';
    let placed = based(slot, joint, now);
    for (const each of tgpu.unroll(layers)) {
      const track = each.read(slot);
      const masked = track.weight * rig.$[joint].mask[each.lane];
      const share = shareOf(track, now);
      const playing = track.playing.clip !== NONE;
      const fading = track.previous.clip !== NONE && share < 1;
      if (masked > 0 && (playing || fading)) {
        let weight = masked;
        let layer = Key();
        if (playing) {
          layer = each.key(track.playing, now, joint);
          if (fading) {
            const previous = each.key(track.previous, now, joint);
            layer = blend(previous, layer, share);
          } else weight = masked * share;
        } else {
          layer = each.key(track.previous, now, joint);
          weight = masked * (1 - share);
        }
        placed = each.lay(placed, layer, weight);
      }
    }
    return placed;
  };

  const locals = tgpu.workgroupVar(d.arrayOf(d.mat4x4f, joints));

  const findJoints = tgpu.computeFn({
    workgroupSize: [joints],
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
  })(({ wid, lid }) => {
    'use gpu';
    const slot = wid.x;
    // A removed pose leaves a hole below the span, which keeps the joints it had
    if (!Pose.alive(slot)) return;
    const time = clock.$;
    const fresh = Pose.born(slot) !== 0;
    const played = placedOf(slot, lid, time.time);
    let placed = matrixOf(local === undefined ? played : local(played, lid, slot));
    locals.$[lid] = d.mat4x4f(placed);
    // Every thread reads `born` before the first clears it
    std.storageBarrier();
    std.workgroupBarrier();

    let parent = rig.$[lid].parent;
    while (parent >= 0) {
      placed = locals.$[parent] * placed;
      parent = rig.$[parent].parent;
    }

    const skinning = std.transpose(placed * rig.$[lid].inverseBind);
    const at = (slot * joints + lid) * 3;
    written.$[time.now + at] = d.vec4f(skinning.columns[0]);
    written.$[time.now + at + 1] = d.vec4f(skinning.columns[1]);
    written.$[time.now + at + 2] = d.vec4f(skinning.columns[2]);
    if (fresh) {
      // A pose added since the last frame has no last frame, so both halves hold this one
      written.$[time.before + at] = d.vec4f(skinning.columns[0]);
      written.$[time.before + at + 1] = d.vec4f(skinning.columns[1]);
      written.$[time.before + at + 2] = d.vec4f(skinning.columns[2]);
      if (lid === 0) Pose.device.$[slot].born = 0;
    }
  });

  const inject: Inject = (cfg) =>
    plays
      .inject()(cfg)
      .with(poseJoints, jointData.as('readonly'))
      .with(jointCount, joints)
      .with(jointHalf, temporal(halfNow, halfBefore))
      .with(spans, spanData.as('readonly'))
      .with(rigged, rig)
      .with(clock, clockUniform);

  // Where compute runs on the CPU, the workgroup's shared locals and barriers keep the kernel
  // there. Instead each thread finds its own joint's chain, placing its ancestors again, and two
  // small kernels copy fresh poses into the last frame's half and clear their `born`
  const alone = backends.of(root).compute === 'cpu';
  const WORKGROUP = 64;

  const localOf = (slot: number, joint: number, now: number) => {
    'use gpu';
    const played = placedOf(slot, joint, now);
    return matrixOf(local === undefined ? played : local(played, joint, slot));
  };

  const findEach = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const slot = std.intdiv(gid.x, d.u32(joints));
    const joint = gid.x % joints;
    if (!Pose.alive(slot)) return;
    const time = clock.$;
    let placed = localOf(slot, joint, time.time);
    let parent = rig.$[joint].parent;
    while (parent >= 0) {
      placed = localOf(slot, d.u32(parent), time.time) * placed;
      parent = rig.$[parent].parent;
    }
    const skinning = std.transpose(placed * rig.$[joint].inverseBind);
    const at = gid.x * 3;
    written.$[time.now + at] = d.vec4f(skinning.columns[0]);
    written.$[time.now + at + 1] = d.vec4f(skinning.columns[1]);
    written.$[time.now + at + 2] = d.vec4f(skinning.columns[2]);
  });

  // A pose added since the last frame has no last frame, so both halves hold this one
  const freshEach = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const slot = std.intdiv(gid.x, d.u32(joints * 3));
    if (!Pose.alive(slot) || Pose.born(slot) === 0) return;
    const time = clock.$;
    written.$[time.before + gid.x] = d.vec4f(written.$[time.now + gid.x]);
  });

  const bornEach = tgpu.computeFn({
    workgroupSize: [WORKGROUP],
    in: { gid: d.builtin.globalInvocationId },
  })(({ gid }) => {
    'use gpu';
    const slot = gid.x;
    if (!Pose.alive(slot)) return;
    Pose.device.$[slot].born = 0;
  });

  const finding = (
    alone
      ? root.pipe(inject).createComputePipeline({ compute: findEach })
      : root.pipe(inject).createComputePipeline({ compute: findJoints })
  ).$name('finding');
  const freshening = alone
    ? root.pipe(inject).createComputePipeline({ compute: freshEach }).$name('freshening')
    : undefined;
  const bearing = alone
    ? root.pipe(inject).createComputePipeline({ compute: bornEach }).$name('bearing')
    : undefined;

  // Retain each pose's tracks on the CPU so the next play can blend from them
  type TrackInput = d.Infer<typeof Track>;
  const mirrors: TrackInput[][] = [];
  const fields = plays.fields;
  const durations = lines.map((line) => line.map(({ duration }) => duration));

  /** The cycle's length of entry `clip` at `mix`, as the kernel finds it. */
  const lengthOf = (clip: number, value: number) => {
    const line = durations[clip];
    const last = line.length - 1;
    const m = Math.min(Math.max(value, 0), last);
    const k = Math.min(Math.floor(m), Math.max(last, 1) - 1);
    const t = m - k;
    return line[k] + (line[Math.min(k + 1, last)] - line[k]) * t;
  };
  const cycleLength = (clip: number, value: number) => Math.max(lengthOf(clip, value), 1e-6);

  const indexOf = (clip: C) => {
    const index = clipOf.get(clip);
    if (index === undefined) {
      throw new Error(`The poses have no clip '${clip}'; they play ${names.join(', ')}.`);
    }
    return index;
  };

  const laneOf = (layer: string) => {
    const lane = layerOf.get(layer);
    if (lane === undefined) {
      const all = ['base', ...layerNames].join(', ');
      throw new Error(`The poses have no layer '${layer}'; they have ${all}.`);
    }
    return lane;
  };

  const set = (
    target: d.Infer<typeof Playing>,
    clip: number,
    time: number,
    { speed = 1, loop = true, from = 0 }: Play<L>,
    value: number,
  ) => {
    target.clip = clip;
    target.loops = loop ? 1 : 0;
    target.speed = speed;
    target.mix = value;
    target.offset = (from - time * speed) / cycleLength(clip, value);
  };

  const tracksOf = (slot: number) => {
    let tracks = mirrors[slot];
    if (!tracks) {
      tracks = TRACKS.map(() => ({
        playing: nothing(),
        previous: nothing(),
        fadeStart: 0,
        fade: 0,
        weight: 1,
      }));
      mirrors[slot] = tracks;
    }
    return tracks;
  };

  /** How far a track's fade has come at `time`, as `shareOf` finds it. */
  const shareAt = (track: TrackInput, time: number) =>
    track.fade <= 0 ? 1 : Math.min(Math.max((time - track.fadeStart) / track.fade, 0), 1);

  /** How much of a track shows at `time`, before its weight and mask, as `shownOf` finds it. */
  const shownAt = (track: TrackInput, time: number) => {
    const share = shareAt(track, time);
    const fading = track.previous.clip !== NONE && share < 1;
    if (track.playing.clip !== NONE) return fading ? 1 : share;
    return fading ? 1 - share : 0;
  };

  type PlayingInput = d.Infer<typeof Playing>;

  /** Where a play is in its cycle at `now`, from 0 to 1, as `cycleOf` finds it. */
  const cycleAt = (playing: PlayingInput, now: number, length: number) => {
    const phase = playing.offset + (now * playing.speed) / length;
    return playing.loops === 1 ? phase - Math.floor(phase) : Math.min(Math.max(phase, 0), 1);
  };

  // `jointAt` reads the baked keys as the kernel does, and blends them as the kernel's functions
  // do, in keys made once: `spare` and `other` for a sample, `first` for a delta, `laid` and
  // `faded` for a layer, and `held` for a joint
  const spare: PlainKey = new Float64Array(10);
  const other: PlainKey = new Float64Array(10);
  const first: PlainKey = new Float64Array(10);
  const laid: PlainKey = new Float64Array(10);
  const faded: PlainKey = new Float64Array(10);
  const held: PlainKey = new Float64Array(10);
  const spin = new Float64Array(4);
  const placedMatrix = new Float64Array(16);
  const parentMatrix = new Float64Array(16);

  const keyOf = (index: number, out: PlainKey) => {
    const at = index * KEY;
    const k = baked.keys;
    for (let c = 0; c < 7; c++) out[c] = k[at + c];
    for (let c = 7; c < 10; c++) out[c] = k[at + c + 1];
  };

  const keyAtCpu = (span: d.Infer<typeof Span>, cycle: number, joint: number, out: PlainKey) => {
    const frame = cycle * span.count;
    const before = Math.min(Math.floor(frame), Math.max(span.count, 1) - 1);
    const after = Math.min(before + 1, span.count);
    keyOf((span.first + before) * joints + joint, out);
    keyOf((span.first + after) * joints + joint, spare);
    plainBlend(out, spare, frame - before, out);
  };

  /** The two clips around a line's mix and the share of the second, as `aroundOf` finds them. */
  const around = { a: spanList[0], b: spanList[0], t: 0, length: 0 };
  const aroundAt = (playing: PlayingInput) => {
    const entry = spanList[playing.clip];
    const last = entry.count - 1;
    const value = Math.min(Math.max(playing.mix, 0), last);
    const k = Math.min(Math.floor(value), Math.max(last, 1) - 1);
    around.a = spanList[entry.first + k];
    around.b = spanList[entry.first + Math.min(k + 1, last)];
    around.t = value - k;
    around.length = cycleLength(playing.clip, playing.mix);
    return around;
  };

  const sampledAt = (playing: PlayingInput, now: number, joint: number, out: PlainKey) => {
    const { a, b, t, length } = aroundAt(playing);
    const cycle = cycleAt(playing, now, length);
    keyAtCpu(a, cycle, joint, out);
    if (t > 0) {
      keyAtCpu(b, cycle, joint, other);
      plainBlend(out, other, t, out);
    }
  };

  const deltaAt = (playing: PlayingInput, now: number, joint: number, out: PlainKey) => {
    const { a, b, t, length } = aroundAt(playing);
    const cycle = cycleAt(playing, now, length);
    keyAtCpu(a, cycle, joint, out);
    keyOf(a.first * joints + joint, first);
    if (t > 0) {
      keyAtCpu(b, cycle, joint, other);
      plainBlend(out, other, t, out);
      keyOf(b.first * joints + joint, other);
      plainBlend(first, other, t, first);
    }
    plainDifference(out, first, spin, out);
  };

  /** A joint's transform in its parent's space, as `placedOf` finds it before `local`. */
  const placedAt = (tracks: TrackInput[], joint: number, now: number, key: PlainKey) => {
    const base = tracks[0];
    sampledAt(base.playing, now, joint, key);
    const baseShare = shareAt(base, now);
    if (baseShare < 1) {
      sampledAt(base.previous, now, joint, laid);
      plainBlend(laid, key, baseShare, key);
    }

    for (let i = 0; i < layerNames.length; i++) {
      const add = layerOptions[layerNames[i]].add === true;
      const track = tracks[i + 1];
      const masked = track.weight * masks[joint * 4 + i + 1];
      const share = shareAt(track, now);
      const playing = track.playing.clip !== NONE;
      const fading = track.previous.clip !== NONE && share < 1;
      if (masked <= 0 || (!playing && !fading)) continue;

      const value = add ? deltaAt : sampledAt;
      let weight = masked;
      if (playing) {
        value(track.playing, now, joint, laid);
        if (fading) {
          value(track.previous, now, joint, faded);
          plainBlend(faded, laid, share, laid);
        } else weight = masked * share;
      } else {
        value(track.previous, now, joint, laid);
        weight = masked * (1 - share);
      }
      if (add) plainOnto(key, laid, weight, spin, key);
      else plainBlend(key, laid, weight, key);
    }
  };

  const jointIndex = (joint: string) => {
    const at = jointNamed(skin, joint);
    if (byName.get(joint) !== at) {
      throw new Error(`Two of the skin's joints are named '${joint}'.`);
    }
    return at;
  };

  const save = (id: Id, lane: number, track: TrackInput) => {
    fields[TRACKS[lane]].write(id, track);
  };

  let flip = 0;

  const destroy = () => {
    plays.destroy();
    keyData.destroy();
    spanData.destroy();
    rigData.destroy();
    jointData.destroy();
  };

  return {
    capacity,
    joints,
    clips: Object.fromEntries(names.map((name, i) => [name, i])) as Record<C, number>,
    layers: Object.fromEntries(layerOf) as Record<L | 'base', number>,

    duration: (clip, value = 0) => lengthOf(indexOf(clip), value),

    get count() {
      return plays.count;
    },

    add(clip, time, how = {}) {
      if (plays.count >= capacity)
        throw new Error(
          `The poses are full. Increase \`capacity\` above ${capacity} or remove a pose before adding another.`,
        );
      const index = indexOf(clip);
      const { mix: value = 0 } = how;
      checkMix(value);
      const id = plays.add({ born: 1 });
      const tracks = tracksOf(plays.slotOf(id));
      for (const track of tracks) {
        Object.assign(track.playing, nothing());
        Object.assign(track.previous, nothing());
        track.fadeStart = time;
        track.fade = 0;
        track.weight = 1;
      }
      const base = tracks[0];
      set(base.playing, index, time, how, value);
      copy(base.previous, base.playing);
      plays.write(id, {
        track0: base,
        track1: tracks[1],
        track2: tracks[2],
        track3: tracks[3],
      });
      return id;
    },

    play(id, clip, time, how = {}) {
      const index = indexOf(clip);
      const lane = laneOf(how.layer ?? 'base');
      const track = tracksOf(plays.slotOf(id))[lane];
      const { mix: value = 0 } = how;
      checkMix(value);
      const fade = how.fade ?? 0;
      // A track shown whole blends from its clip, one shown in part fades in from that share
      const shown = shownAt(track, time);
      if (shown < 1) {
        Object.assign(track.previous, nothing());
        track.fadeStart = time - fade * shown;
      } else {
        copy(track.previous, track.playing);
        track.fadeStart = time;
      }
      set(track.playing, index, time, how, value);
      track.fade = fade;
      if (how.weight !== undefined) track.weight = how.weight;
      save(id, lane, track);
    },

    mix(id, value, time, { layer = 'base' } = {}) {
      checkMix(value);
      const lane = laneOf(layer);
      const track = tracksOf(plays.slotOf(id))[lane];
      const { playing } = track;
      if (playing.clip === NONE || playing.mix === value) return;
      const turn =
        1 / cycleLength(playing.clip, playing.mix) - 1 / cycleLength(playing.clip, value);
      playing.offset += time * playing.speed * turn;
      playing.mix = value;
      save(id, lane, track);
    },

    weigh(id, weight, { layer = 'base' } = {}) {
      const lane = laneOf(layer);
      if (lane === 0)
        throw new Error(
          'The base layer always has full weight. Pass a different `layer` to `weigh`.',
        );
      const track = tracksOf(plays.slotOf(id))[lane];
      if (track.weight === weight) return;
      track.weight = weight;
      save(id, lane, track);
    },

    stop(id, time, { layer = 'base', fade = 0 } = {}) {
      const lane = laneOf(layer);
      if (lane === 0) throw new Error("The base doesn't stop; play another clip on it.");
      const track = tracksOf(plays.slotOf(id))[lane];
      if (track.playing.clip === NONE) return;
      // A layer still fading in fades out from the share it shows
      const shown = shownAt(track, time);
      copy(track.previous, track.playing);
      Object.assign(track.playing, nothing());
      track.fadeStart = time - fade * (1 - shown);
      track.fade = fade;
      save(id, lane, track);
    },

    seconds(id, time, { layer = 'base' } = {}) {
      const { playing } = tracksOf(plays.slotOf(id))[laneOf(layer)];
      if (playing.clip === NONE) return 0;
      const length = cycleLength(playing.clip, playing.mix);
      return cycleAt(playing, time, length) * length;
    },

    jointAt(id, joint, time) {
      const tracks = tracksOf(plays.slotOf(id));
      let at = jointIndex(joint);
      placedAt(tracks, at, time, held);
      const placed = plainMatrix(held, placedMatrix);
      for (at = skin.joints[at].parent; at >= 0; at = skin.joints[at].parent) {
        placedAt(tracks, at, time, held);
        mat4d.multiply(plainMatrix(held, parentMatrix), placed, placed);
      }
      const column = (c: number) =>
        d.vec4f(placed[c * 4], placed[c * 4 + 1], placed[c * 4 + 2], placed[c * 4 + 3]);
      return d.mat4x4f(column(0), column(1), column(2), column(3));
    },

    remove: (id) => plays.remove(id),

    has: (id) => plays.has(id),

    slotOf: (id) => plays.slotOf(id),

    animate(pass, time) {
      plays.flush();
      if (plays.span === 0) return;
      flip ^= 1;
      clockTime[0] = time;
      clockHalves[0] = flip * half;
      clockHalves[1] = (1 - flip) * half;
      queue.writeBuffer(clockGpu, 0, clockData);
      pass.setPipeline(finding);
      if (!freshening || !bearing) {
        pass.dispatchWorkgroups(plays.span);
        return;
      }
      pass.dispatchWorkgroups(Math.ceil((plays.span * joints) / WORKGROUP));
      pass.setPipeline(freshening);
      pass.dispatchWorkgroups(Math.ceil((plays.span * joints * 3) / WORKGROUP));
      pass.setPipeline(bearing);
      pass.dispatchWorkgroups(Math.ceil(plays.span / WORKGROUP));
    },

    inject: () => inject,

    initAsync: () =>
      Promise.all([finding, freshening, bearing].map((each) => each?.initAsync())).then(() => {}),

    destroy,
    [Symbol.dispose]: destroy,
  };
}

/** The inverse of `m` and its determinant, which must not be zero. */
function inverted(m: ArrayLike<number>, what: string) {
  const determinant = mat4d.determinant(m);
  if (!(Math.abs(determinant) > 1e-30)) throw new Error(`${what} can't be inverted.`);
  return { inverse: mat4d.inverse(m), determinant };
}

/**
 * Attaches `model` to one joint of `skin`, such as a sword to a hand. It binds every mesh vertex
 * to that joint so posed instances follow it. The model coordinates must be in joint space, or
 * `offset` takes them there. Throws for an `offset` other than 16 numbers, and for a mesh whose
 * positions, normals and indices disagree on its vertices.
 *
 * @param model the model it holds
 * @param skin the skin whose joint holds it
 * @param joint the name of the joint that holds it
 * @param options.offset a column-major matrix from the model's space to the joint's, such as a turn
 *   for an item modelled facing the other way
 * @returns the model, its meshes bound to the joint
 */
export function attach(
  model: ModelData,
  skin: SkinData,
  joint: string,
  options: { readonly offset?: ArrayLike<number> } = {},
): ModelData {
  const index = jointNamed(skin, joint);
  if (options.offset && options.offset.length !== 16) {
    throw new RangeError(
      `attach's \`offset\` is a 4×4 matrix of 16 numbers, got ${options.offset.length}.`,
    );
  }
  model.meshes.forEach(({ positions, normals, indices }, i) => {
    const count = positions.length / 3;
    const why = !Number.isInteger(count)
      ? `\`positions\` holds ${positions.length} numbers, not three per vertex`
      : normals.length !== positions.length
        ? `\`normals\` holds ${normals.length} numbers, and its ${count} vertices take ${positions.length}`
        : indices.length % 3 !== 0
          ? `\`indices\` holds ${indices.length} entries, not three per triangle`
          : Array.prototype.some.call(indices, (at: number) => at >= count)
            ? `\`indices\` names a vertex past its ${count}`
            : undefined;
    if (why) throw new RangeError(`attach can't bind mesh ${i}: its ${why}.`);
  });

  // Transform joint space to model bind space before attaching the model
  const { inverseBind } = skin.joints[index];
  const bound = inverted(inverseBind, `Joint '${joint}''s inverse bind`);
  const m = options.offset ? mat4d.multiply(bound.inverse, options.offset) : bound.inverse;
  const { inverse: back, determinant } = options.offset
    ? inverted(m, `The offset for joint '${joint}'`)
    : { inverse: inverseBind, determinant: bound.determinant };

  const meshes = model.meshes.map((mesh) => {
    const count = mesh.positions.length / 3;
    const positions = new Float32Array(count * 3);
    const normals = new Float32Array(count * 3);
    for (let v = 0; v < count; v++) {
      const x = mesh.positions[v * 3];
      const y = mesh.positions[v * 3 + 1];
      const z = mesh.positions[v * 3 + 2];
      const nx = mesh.normals[v * 3];
      const ny = mesh.normals[v * 3 + 1];
      const nz = mesh.normals[v * 3 + 2];
      // Use the transpose of the inverse placement to transform attached normals
      let length = 0;
      for (let r = 0; r < 3; r++) {
        positions[v * 3 + r] = m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r];
        const n = back[r * 4] * nx + back[r * 4 + 1] * ny + back[r * 4 + 2] * nz;
        normals[v * 3 + r] = n;
        length += n * n;
      }
      const scale = length > 0 ? 1 / Math.sqrt(length) : 1;
      for (let r = 0; r < 3; r++) normals[v * 3 + r] = normals[v * 3 + r] * scale;
    }

    // Reverse winding when the bind transform mirrors triangles
    const indices = Uint32Array.from(mesh.indices);
    if (determinant < 0) {
      for (let i = 0; i < indices.length; i += 3) {
        const second = indices[i + 1];
        indices[i + 1] = indices[i + 2];
        indices[i + 2] = second;
      }
    }

    const joints = new Uint16Array(count * 4);
    const weights = new Float32Array(count * 4);
    for (let v = 0; v < count; v++) {
      joints[v * 4] = index;
      weights[v * 4] = 1;
    }
    return { ...mesh, positions, normals, indices, joints, weights };
  });

  return { meshes, images: model.images };
}

const agree = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  Array.from(a).every((value, i) => Math.abs(value - b[i]) <= 1e-4 * Math.max(1, Math.abs(value)));

/**
 * Rebinds a skinned `model` from `from` to `to` by joint name. Use this when files list the same
 * rig joints in different orders, so one set of poses can move them all. Meshes without joints
 * stay unchanged. Throws if `to` lacks a joint or has a different inverse bind.
 *
 * @param model a model skinned to `from`
 * @param from the skin it was made for
 * @param to the skin whose poses move it
 * @returns the model, skinned to `to`
 */
export function rebind(model: ModelData, from: SkinData, to: SkinData): ModelData {
  const named = new Map<string, number>();
  for (const [i, { name }] of to.joints.entries()) {
    if (named.has(name)) throw new Error(`Two of the skin's joints are named '${name}'.`);
    named.set(name, i);
  }

  const remap = from.joints.map(({ name, inverseBind }) => {
    const at = named.get(name);
    if (at === undefined) throw new Error(`The skin has no joint named '${name}'.`);
    if (!agree(inverseBind, to.joints[at].inverseBind)) {
      throw new Error(`The skins bind their joint '${name}' in different places.`);
    }
    return at;
  });

  const meshes = model.meshes.map((mesh) => {
    if (!mesh.joints) return mesh;
    const joints = Uint16Array.from(mesh.joints, (joint) => {
      const at = remap[joint];
      if (at === undefined) {
        throw new Error(`A mesh is bound to joint ${joint}, and its skin has ${remap.length}.`);
      }
      return at;
    });
    return { ...mesh, joints };
  });
  return { meshes, images: model.images };
}
