import { poses } from '@typegpu/kit';
import { quatn, vec3n } from 'wgpu-matrix';

type Vec3 = readonly [number, number, number];

/** The joint of `skin` named `name`, which must exist. */
const jointOf = (skin: poses.SkinData, name: string) => {
  const joint = skin.joints.find((each) => each.name === name);
  if (!joint) throw new Error(`The skin has no joint named '${name}'.`);
  return joint;
};

/** A channel that turns or scales `joint` linearly through `values` at `times`. */
const linear = (
  joint: string,
  path: 'rotation' | 'scale',
  times: Float32Array,
  values: ArrayLike<number>,
): poses.ChannelData => ({
  joint,
  path,
  interpolation: 'linear',
  times,
  values: Float32Array.from(values),
});

/** The rotation of joint `name` at `time` in `clip`, or its rest rotation without a channel. */
function rotationIn(clip: poses.ClipData, skin: poses.SkinData, name: string, time: number) {
  const channel = clip.channels.find((c) => c.joint === name && c.path === 'rotation');
  return channel ? poses.sample(channel, time) : [...jointOf(skin, name).rotation];
}

/** The right arm's joints, from the shoulder to the hand slot. */
const ARM = ['upperarm.r', 'lowerarm.r', 'wrist.r', 'hand.r', 'handslot.r'] as const;
/** When in the hold clip the grip is read. */
const GRIP_AT = 0.5;

/** The grip's rotation from the chest while a hold plays. Lanterns hang upright in it. */
export function gripIn(hold: poses.ClipData, skin: poses.SkinData) {
  return ARM.reduce(
    (turned, name) => quatn.multiply(turned, rotationIn(hold, skin, name, GRIP_AT)),
    quatn.identity(),
  );
}

/** Directions the raised arm points from the chest, out to the right and forward, swaying. */
const RAISED: readonly Vec3[] = [
  [-0.6, 0.75, 0.3],
  [-0.62, 0.77, 0.15],
  [-0.6, 0.75, 0.3],
];
const RAISE_SECONDS = 2.4;

/**
 * Authors a right arm raised high, swaying a little. The rig ships no such clip. The wrist turns
 * so the grip keeps the hold's angle and a lantern in it still hangs upright.
 */
export function raised(hold: poses.ClipData, skin: poses.SkinData): poses.ClipData {
  const upper = jointOf(skin, 'upperarm.r');
  const lower = jointOf(skin, 'lowerarm.r');
  const along = vec3n.normalize(vec3n.transformQuat(lower.position, upper.rotation));
  const hand = rotationIn(hold, skin, 'hand.r', GRIP_AT);
  const slot = rotationIn(hold, skin, 'handslot.r', GRIP_AT);
  const elbow = quatn.multiply(lower.rotation, quatn.fromAxisAngle([0, 0, 1], 0.25));
  const grip = quatn.multiply(gripIn(hold, skin), quatn.conjugate(quatn.multiply(hand, slot)));

  const keys = RAISED.map((direction) => {
    const toward = quatn.rotationTo(along, vec3n.normalize(direction));
    const shoulder = quatn.multiply(toward, upper.rotation);
    const wrist = quatn.multiply(quatn.conjugate(quatn.multiply(shoulder, elbow)), grip);
    return [shoulder, elbow, wrist, hand, slot];
  });
  const times = Float32Array.from(RAISED, (_, k) => (k / (RAISED.length - 1)) * RAISE_SECONDS);
  return {
    name: 'raise',
    duration: RAISE_SECONDS,
    channels: ARM.map((joint, j) =>
      linear(
        joint,
        'rotation',
        times,
        keys.flatMap((key) => key[j]),
      ),
    ),
  };
}

/** The same joint on the rig's other side, for names ending `.r` or `.l`. */
export const opposite = (joint: string) =>
  joint.replace(/\.([rl])$/, (_, side: string) => (side === 'r' ? '.l' : '.r'));

const BREATH_SECONDS = 3.6;
/** Radians the spine and chest lean back at the top of a breath. */
const LEAN = 0.035;
const SWELL = 1.04;

/**
 * Authors a slow breath for an additive layer: the spine and chest lean back and the chest
 * swells, the head keeping level. Its first key is the rest pose, so it adds only the breath.
 */
export function breathing(skin: poses.SkinData): poses.ClipData {
  const times = Float32Array.from([0, 0.45, 0.75, 1], (share) => share * BREATH_SECONDS);
  const leans = [0, 1, 0.4, 0];
  const turned = (name: string, by: number) => {
    const rest = jointOf(skin, name).rotation;
    const keys = leans.flatMap((lean) =>
      quatn.multiply(rest, quatn.fromAxisAngle([1, 0, 0], by * lean)),
    );
    return linear(name, 'rotation', times, keys);
  };
  const chest = jointOf(skin, 'chest').scale;
  const swell = leans.flatMap((lean) => {
    const s = 1 + (SWELL - 1) * lean;
    return [chest[0] * s, chest[1], chest[2] * s];
  });
  return {
    name: 'breathe',
    duration: BREATH_SECONDS,
    channels: [
      turned('spine', -LEAN),
      turned('chest', -LEAN),
      turned('head', LEAN * 1.6),
      linear('chest', 'scale', times, swell),
    ],
  };
}
