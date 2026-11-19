import { std } from 'typegpu';
import { vec3n } from 'wgpu-matrix';
import {
  CLIMB,
  ease,
  HERALD,
  HIP,
  JOINTS,
  petted,
  R,
  random,
  STANDING,
  UP,
  WALK,
  type Body,
  kick,
} from './body.ts';
import { createSeat, seat } from './spine.ts';
import { GROUND, type V3 } from './tray.ts';
import { SHOVE } from './walking.ts';

/** Thigh and shin lengths. */
const THIGH = 0.032;
const SHIN = 0.03;
/**
 * The body's bend: a spring that lags behind its motion by `lag`, leans at most `most` radii,
 * and sways by `sway` radii at rest.
 */
const BEND = { lag: 0.55, most: 0.8, sway: 0.25 };
/** The length of what grows on the head, for each kind. */
const TOP = { lengths: [R * 1.05, R * 0.6, R * 0.6] };
/** The fastest a joint moves from the body before its jump is eased, and how fast easing fades. */
const JUMP = { speed: 1.2, fade: 12 };

type V4 = [number, number, number, number];

/** One critter as the march draws it, a row of the `Critter` table that `figureOf` fills. */
export interface Figure {
  readonly body: V4;
  /** How far back the body was when the last frame was drawn, which the caller writes. */
  readonly moved: V4;
  readonly up: V4;
  readonly ahead: V4;
  readonly gaze: V4;
  readonly joints: V4[];
  readonly color: V4;
  readonly bound: V4;
  readonly bend: V4;
  readonly face: V4;
  readonly top: V4;
}

const v4 = (): V4 => [0, 0, 0, 0];

export const createFigure = (): Figure => ({
  body: v4(),
  moved: v4(),
  up: v4(),
  ahead: v4(),
  gaze: v4(),
  joints: Array.from({ length: JOINTS }, v4),
  color: v4(),
  bound: v4(),
  bend: v4(),
  face: v4(),
  top: v4(),
});

/** What a critter looks at and reaches for besides itself. */
export interface Sights {
  readonly camera: V3;
  /** The point the pointer hovers while it is fresh. */
  pointer: V3 | undefined;
  readonly knob: V3;
}

const v3 = (): V3 => [0, 0, 0];
const v3s = (count: number) => Array.from({ length: count }, v3);
const AHEAD: V3 = [0, 0, 1];

// Scratch the steps and poses write their vectors into, one critter at a time
const home = v3();
const lead = v3();
const gap = v3();
const rest = v3();
const toward = v3();
const settled = createSeat();
const contact = createSeat();
/** The body's centre and up axis, and which way it faces, as `frameOf` writes them. */
const pose = { ahead: v3(), up: v3(), centre: v3() };
const [low, high, between] = v3s(3);
const [up, side, facing, bend, shown, face, look, aim, tip] = v3s(9);
const [hips, shoulders, feet, hands, knees, steps, holds] = Array.from({ length: 7 }, () => v3s(2));
const [walkingFoot, walkingHand, heavingFoot, heavingHand, sittingFoot, sittingHand] = v3s(6);
const [forward, climb, pole] = v3s(3);
const joints = v3s(JOINTS + 1);
const placed = v3s(JOINTS + 1);
const turned = v3();
/** A walking foot's stride, as `climbStep` writes it. */
const stride = { ahead: 0, lift: 0 };

/** How much of the bend a body point takes, by its height along the up axis, as in figures.ts. */
const bendAt = (along: number) => std.clamp((along / R + 1) * 0.5, 0, 1) ** 2;

/** Rotates `v` round unit `axis` by `angle` radians into `out`, which may be `v`. */
const rotate = (v: V3, axis: V3, angle: number, out: V3) => {
  const c = Math.cos(angle);
  const along = vec3n.dot(axis, v) * (1 - c);
  vec3n.cross(axis, v, turned);
  vec3n.scale(v, c, out);
  vec3n.addScaled(out, turned, Math.sin(angle), out);
  return vec3n.addScaled(out, axis, along, out);
};

/**
 * Writes into `out` a knee between hip and foot, bent towards `bendTo`, for two bones of `a` and `b`.
 */
const knee = (hip: V3, foot: V3, a: number, b: number, bendTo: V3, out: V3) => {
  vec3n.sub(foot, hip, toward);
  const span = Math.min(vec3n.length(toward), a + b - 1e-4);
  vec3n.normalize(toward, toward);
  const along = (span * span + a * a - b * b) / (2 * span);
  const reach = Math.sqrt(Math.max(a * a - along * along, 0));
  vec3n.addScaled(bendTo, toward, -vec3n.dot(bendTo, toward), between);
  vec3n.normalize(between, between);
  vec3n.addScaled(hip, toward, along, out);
  return vec3n.addScaled(out, between, reach, out);
};

/** How far a walking foot sits ahead of the body along the jelly, and how high it lifts. */
const climbStep = (cycle: number) => {
  const u = cycle - Math.floor(cycle);
  if (u < CLIMB.planted) {
    stride.ahead = CLIMB.stride * (0.5 - u / CLIMB.planted);
    stride.lift = 0;
    return stride;
  }
  const w = (u - CLIMB.planted) / (1 - CLIMB.planted);
  stride.ahead = CLIMB.stride * (w - 0.5);
  stride.lift = Math.sin(Math.PI * w) * CLIMB.lift;
  return stride;
};

/** Steps a foot towards where it belongs under the hips once the other is planted. */
export function gait(body: Body, dt: number, across: V3) {
  vec3n.scale(body.velocity, 0.12, lead);
  for (let i = 0; i < 2; i++) {
    const foot = body.feet[i];
    vec3n.set(body.at[0], GROUND, body.at[2], home);
    vec3n.addScaled(home, across, (i === 0 ? 1 : -1) * R * 0.45, home);
    vec3n.add(home, lead, home);
    if (foot.t < 1) {
      foot.t = Math.min(1, foot.t + dt / WALK.time);
      vec3n.lerp(foot.from, foot.to, foot.t, foot.at);
      foot.at[1] += Math.sin(Math.PI * foot.t) * WALK.lift;
      if (foot.t >= 1) body.squash.velocity -= 0.25;
      continue;
    }
    const other = body.feet[1 - i];
    if (other.t >= 1 && vec3n.distance(home, foot.at) > WALK.stride) {
      vec3n.sub(home, foot.at, gap);
      vec3n.copy(foot.at, foot.from);
      vec3n.addScaled(home, gap, 0.3, foot.to);
      foot.to[1] = GROUND;
      foot.t = 0;
    }
  }
}

/** How far through its swing foot `i` is, as the height of its arc. */
const swingOf = (body: Body, i: number) =>
  body.feet[i].t < 1 ? Math.sin(Math.PI * body.feet[i].t) : 0;

/**
 * The bend lags behind the body's `accel`, sways at rest, leans over the planted foot and into a
 * petting pointer at `watched`. The squash springs back to 0.
 */
export function springs(
  body: Body,
  dt: number,
  accel: V3,
  across: V3,
  ahead: V3,
  watched: V3 | undefined,
) {
  const shift = body.mode === 'walk' ? swingOf(body, 1) - swingOf(body, 0) : 0;
  const sway =
    Math.sin(body.phase * 2.3) * BEND.sway +
    shift * 0.55 +
    (body.dizzy > 0 ? Math.sin(body.phase * 6) * 0.5 : 0) +
    (petted(body) ? Math.sin(body.phase * 9) * 0.25 : 0);
  const forth = Math.sin(body.phase * 1.6 + 1) * BEND.sway * 0.5;
  vec3n.scale(across, sway * R, rest);
  vec3n.addScaled(rest, ahead, forth * R, rest);
  if (petted(body) && watched) {
    vec3n.set(watched[0] - body.at[0], 0, watched[2] - body.at[2], gap);
    vec3n.addScaled(rest, vec3n.normalize(gap, gap), R * 0.5, rest);
  }
  kick(body.bend, accel, -dt * BEND.lag);
  body.bend.step(rest, dt);
  body.squash.step(0, dt);
}

/** Eases how happy, surprised, sleepy and dizzy it looks towards what it is doing. */
export function moods(body: Body, dt: number) {
  body.shock -= dt;
  body.joy -= dt;
  const climbing = body.mode === 'ride' && body.gripping;
  const effort = climbing && body.steep > 0.5 && body.heave > 0.35 && body.heave < 0.8;
  const happy =
    body.mode === 'perch' ||
    body.mode === 'pull' ||
    body.cheer > 0 ||
    petted(body) ||
    body.joy > 0 ||
    body.wave > 0 ||
    (body.mode === 'ride' && !body.gripping);
  let surprised = 0;
  if (body.shock > 0) surprised = 1;
  else if (body.mode === 'held') surprised = 0.5;
  let sleepy = 0;
  if (body.asleep > 0) sleepy = 1;
  else if (effort) sleepy = 0.45;
  body.mood[0] = ease(body.mood[0], happy && body.shock <= 0 ? 1 : 0, 8, dt);
  body.mood[1] = ease(body.mood[1], surprised, 10, dt);
  body.mood[2] = ease(body.mood[2], sleepy, body.asleep > 0 ? 2 : 8, dt);
  body.mood[3] = ease(body.mood[3], body.dizzy > 0 ? 1 : 0, 6, dt);
}

/** Rolls on upright out of a flip or tumble that ended early. */
export function rollUpright(body: Body, dt: number) {
  const upright = Math.round(body.spin / (Math.PI * 2)) * Math.PI * 2;
  body.spin = ease(body.spin, upright, 10, dt);
  if (Math.abs(upright - body.spin) < 0.02) {
    body.spin = 0;
    body.spinAxis = undefined;
  }
}

/** Blinks now and then, the lids shutting at once and opening over an eighth of a second. */
export function blinks(body: Body, dt: number) {
  body.blinkIn -= dt;
  if (body.blinkIn <= 0) {
    body.blink = 1;
    body.blinkIn = random(1.5, 5);
  }
  body.blink = Math.max(0, body.blink - dt * 8);
}

/**
 * Writes into `out` the point on the jelly at arc length `s`, `across` times `out` to the side and
 * `lift` off it.
 */
const onJelly = (
  body: Body,
  spine: Float32Array,
  s: number,
  out: number,
  across: number,
  lift: number,
  into: V3,
) => {
  const { surface, normal } = seat(spine, s, body.at[2], contact);
  vec3n.addScaled(surface, side, out * across, into);
  return vec3n.addScaled(into, normal, lift, into);
};

/**
 * Hands and feet on the jelly, walking up it, heaving up steep jelly or sitting to slide, into
 * `steps` and `holds`.
 */
function ridingPose(body: Body, spine: Float32Array, time: number) {
  const u = body.heave;
  for (let k = 0; k < 2; k++) {
    const out = k ? -R : R;
    // Walking up: stepping feet and pumping arms
    const step = climbStep(body.cycle + k * 0.5);
    const pump = Math.sin((body.cycle + k * 0.5) * Math.PI * 2);
    onJelly(body, spine, body.s + step.ahead, out, 0.45, R * 0.17 + step.lift, walkingFoot);
    vec3n.addScaled(shoulders[k], up, -R * 0.45, walkingHand);
    vec3n.addScaled(walkingHand, side, out * 0.35, walkingHand);
    vec3n.addScaled(walkingHand, facing, pump * R * 0.45, walkingHand);
    // Heaving: hands reach to a new hold, then the feet scrabble as the body pulls up
    const reaching = u < 0.35;
    const handAt = reaching
      ? body.grabFrom + (body.grab - body.grabFrom) * std.smoothstep(0, 0.35, u)
      : body.grab;
    const handLift = reaching ? Math.sin((Math.PI * u) / 0.35) * R * 0.5 : 0;
    const pulling = u >= 0.35 && u < 0.8;
    const scrabble = pulling ? Math.sin(time * 30 + k * Math.PI) : 0;
    onJelly(body, spine, handAt, out, 0.55, R * 0.12 + handLift, heavingHand);
    const footAt = body.s - R * 0.45 + scrabble * R * 0.2;
    const footLift = R * 0.17 + Math.max(0, scrabble) * R * 0.25;
    onJelly(body, spine, footAt, out, 0.45, footLift, heavingFoot);
    vec3n.addScaled(hips[k], facing, HIP * 0.85, sittingFoot);
    vec3n.addScaled(sittingFoot, up, -R * 0.2, sittingFoot);
    vec3n.addScaled(shoulders[k], side, out * 0.35, sittingHand);
    vec3n.addScaled(sittingHand, up, R * 0.95 + Math.sin(time * 18 + k) * 0.006, sittingHand);

    vec3n.lerp(walkingFoot, heavingFoot, body.steep, steps[k]);
    vec3n.lerp(steps[k], sittingFoot, body.seated, steps[k]);
    vec3n.lerp(walkingHand, heavingHand, body.steep, holds[k]);
    vec3n.lerp(holds[k], sittingHand, body.seated, holds[k]);
  }
}

/**
 * Eases any joint that jumps from where it was drawn, such as on grabbing the jelly or waving.
 * Takes the body's centre then its joints in `joints`, and writes them as drawn into `placed`.
 */
function smooth(body: Body, dt: number) {
  const [centre] = joints as [V3];
  // Joints relative to the centre
  for (let j = 1; j <= JOINTS; j++) vec3n.sub(joints[j], centre, joints[j]);
  const most = JUMP.speed * dt;
  const fade = Math.exp(-dt * JUMP.fade);
  for (let j = 0; j <= JOINTS; j++) {
    const now = joints[j];
    const settle = body.settle[j];
    if (body.posed) {
      const expected = j ? body.raw[j] : vec3n.addScaled(body.raw[0], body.velocity, dt, aim);
      if (vec3n.distance(now, expected) > most) {
        vec3n.add(settle, vec3n.sub(expected, now, gap), settle);
      }
    }
    vec3n.scale(settle, fade, settle);
  }
  for (let j = 0; j <= JOINTS; j++) vec3n.copy(joints[j], body.raw[j]);
  body.posed = true;
  vec3n.add(centre, body.settle[0], body.drawn);
  vec3n.copy(body.drawn, placed[0]);
  for (let j = 1; j <= JOINTS; j++) {
    vec3n.add(body.drawn, joints[j], placed[j]);
    vec3n.add(placed[j], body.settle[j], placed[j]);
  }
}

/** What grows on the head, on a spring from the top of the body at `base`, into `tip`. */
function crown(body: Body, base: V3, dt: number) {
  const length = TOP.lengths[body.kind];
  vec3n.addScaled(base, up, length, rest);
  const sprung = body.tip.value;
  if (!body.tipped) {
    vec3n.copy(rest, sprung);
    body.tipped = true;
  }
  body.tip.step(rest, dt);
  vec3n.normalize(vec3n.sub(sprung, base, gap), gap);
  vec3n.addScaled(base, gap, length, sprung);
  return vec3n.copy(sprung, tip);
}

/** Writes the body's centre and up axis, and which way it faces, for what it is doing into `pose`. */
function frameOf(body: Body, spine: Float32Array, knob: V3) {
  const { ahead, up: tilt, centre } = pose;
  vec3n.set(Math.sin(body.heading), 0, Math.cos(body.heading), ahead);
  if (body.mode === 'ride') {
    const { normal, tangent, surface } = seat(spine, body.s, body.at[2], settled);
    vec3n.scale(tangent, Math.cos(body.turn), ahead);
    vec3n.normalize(vec3n.addScaled(ahead, AHEAD, Math.sin(body.turn), ahead), ahead);
    const bob = Math.abs(Math.sin(body.cycle * Math.PI * 2)) * 0.004;
    // Walking tilt, heaving tilt steep jelly, or sitting
    vec3n.normalize(vec3n.addScaled(UP, tangent, 0.25, low), low);
    vec3n.normalize(vec3n.addScaled(normal, UP, 0.4, high), high);
    vec3n.lerp(low, high, body.steep, tilt);
    vec3n.normalize(vec3n.lerp(tilt, normal, body.seated, tilt), tilt);
    vec3n.addScaled(surface, UP, STANDING * 0.95 + bob, low);
    vec3n.addScaled(surface, normal, R * 1.1, high);
    vec3n.lerp(low, high, body.steep, centre);
    vec3n.lerp(centre, vec3n.addScaled(surface, normal, R * 0.95, low), body.seated, centre);
    return pose;
  }
  if (body.mode === 'perch' && body.base) {
    vec3n.normalize(vec3n.addScaled(body.base.upright, body.base.bend.value, 4, tilt), tilt);
    vec3n.addScaled(body.at, tilt, R * 0.9, centre);
    return pose;
  }
  if (body.mode === 'walk' && body.asleep > 0) {
    vec3n.copy(UP, tilt);
    vec3n.addScaled(body.at, UP, R * 0.92, centre);
    return pose;
  }
  if (body.mode === 'pull') {
    // Upright through the leap, then hanging from the knob, facing away from the lever
    vec3n.addScaled(body.at, UP, STANDING, centre);
    const hanging = std.smoothstep(0.5, 1, body.pulling / HERALD.leap);
    vec3n.copy(AHEAD, ahead);
    vec3n.normalize(vec3n.sub(knob, centre, high), high);
    vec3n.normalize(vec3n.lerp(UP, high, hanging, tilt), tilt);
    return pose;
  }
  vec3n.set(body.velocity[0] * 0.8, 1, body.velocity[2] * 0.8, tilt);
  vec3n.normalize(tilt, tilt);
  let bob = 0;
  if (body.mode === 'walk') {
    bob = Math.max(body.feet[0].at[1] - GROUND, body.feet[1].at[1] - GROUND);
  }
  vec3n.copy(body.at, centre);
  centre[1] += bob * 0.5;
  vec3n.addScaled(centre, tilt, STANDING, centre);
  return pose;
}

/**
 * Where the feet are off the jelly, into `feet`: out in front sitting, walking, tucked into a flip,
 * or dangling and kicking, flailing while tumbling.
 */
function feetOf(body: Body, time: number) {
  for (let k = 0; k < 2; k++) {
    const hip = hips[k];
    const foot = feet[k];
    if ((body.mode === 'walk' && body.asleep > 0) || body.mode === 'perch') {
      // Swinging a little on a friend's head
      const swing = body.mode === 'perch' ? 1 : 0;
      vec3n.addScaled(hip, facing, HIP * 0.85, foot);
      vec3n.addScaled(foot, up, -R * 0.25 + Math.sin(time * 4 + k * 2) * swing * 0.006, foot);
    } else if (body.mode === 'walk') {
      vec3n.copy(body.feet[k].at, foot);
    } else if (body.spinning && !body.spinAxis) {
      vec3n.addScaled(hip, up, -HIP * 0.45, foot);
      vec3n.addScaled(foot, facing, HIP * 0.3, foot);
    } else {
      const pace = body.spinAxis ? 30 : 16;
      vec3n.addScaled(hip, up, -HIP * 0.85, foot);
      vec3n.addScaled(foot, facing, Math.sin(time * pace + k * 2) * 0.012, foot);
    }
  }
}

/** Writes into `out` shoulder `k` reaching out `reach` radii to its side. */
const outward = (k: number, reach: number, out: V3) =>
  vec3n.addScaled(shoulders[k], side, (k ? -1 : 1) * reach * R, out);

/** Where hand `k` goes for what the critter is doing, into `hands`. */
function handOf(body: Body, i: number, k: number, time: number, knob: V3) {
  const hand = hands[k];
  const riding = body.mode === 'ride';
  const asleep = body.mode === 'walk' && body.asleep > 0;
  // Cheering, which pulling and flying start from
  outward(k, 0.35, aim);
  vec3n.addScaled(aim, up, R * 0.95 + Math.sin(time * 18 + k * 1.3) * 0.006, aim);
  if (riding) {
    vec3n.copy(holds[k], hand);
  } else if (asleep) {
    vec3n.addScaled(outward(k, -0.5, hand), facing, R * 0.8, hand);
    vec3n.addScaled(hand, up, -R * 0.3, hand);
  } else if (body.mode === 'pull') {
    vec3n.addScaled(knob, side, (k ? -1 : 1) * R * 0.3, hand);
    vec3n.lerp(aim, hand, std.smoothstep(0.3, 0.9, body.pulling / HERALD.leap), hand);
  } else if (body.mode === 'perch') {
    // Holding on to the head below, waving now and then
    const wave = Math.max(0, Math.sin(time * 1.3 + i)) * (k ? 0 : 1);
    vec3n.addScaled(outward(k, 0.2, hand), up, -R * 0.9 + wave * R * 2.1, hand);
    vec3n.addScaled(hand, facing, R * 0.3, hand);
  } else if (body.mode !== 'walk') {
    if (body.spinAxis) vec3n.addScaled(aim, side, Math.sin(time * 25 + k * 3) * R * 0.5, hand);
    else vec3n.copy(aim, hand);
  } else if (body.shove > 0) {
    vec3n.addScaled(outward(k, 0.45, hand), up, R * 0.05, hand);
    vec3n.addScaled(hand, facing, body.shove < SHOVE.windup ? 0 : R, hand);
  } else if (body.wave > 0 && k === 0) {
    vec3n.addScaled(outward(k, 0.3 + Math.sin(time * 16) * 0.3, hand), up, R * 1.3, hand);
  } else if (body.load > 0) {
    vec3n.addScaled(outward(k, 0.4, hand), up, R, hand);
  } else if (body.cheer > 0) {
    const lift = R * (1.1 + Math.sin(time * 14 + k * 2) * 0.25);
    vec3n.addScaled(outward(k, 0.4, hand), up, lift, hand);
  } else if (body.dizzy > 0) {
    vec3n.addScaled(outward(k, 0.9, hand), up, Math.sin(time * 6 + k * 2) * R * 0.4, hand);
  } else if (body.flee > 0) {
    const lift = R * 1.2 + Math.sin(time * 20 + k * 3) * R * 0.2;
    vec3n.addScaled(outward(k, 0.4, hand), up, lift, hand);
  } else {
    // Arms swing against the legs
    const swing = vec3n.dot(vec3n.sub(feet[1 - k], hips[1 - k], gap), facing);
    vec3n.addScaled(outward(k, 0.45, hand), up, -R * 0.55, hand);
    vec3n.addScaled(hand, facing, swing * 0.6 + 0.004, hand);
  }
}

/** Bends `p` in place by its height along the body from `centre`. */
const bendIn = (p: V3, centre: V3) =>
  vec3n.addScaled(p, bend, bendAt(vec3n.dot(vec3n.sub(p, centre, gap), up)), p);

/** Writes `v` grown in from the ground under the body by `grown`, and `w`, into `out`. */
const put = (body: Body, v: V3, w: number, out: V4) => {
  vec3n.sub(v, body.at, gap);
  vec3n.addScaled(body.at, gap, body.grown, out);
  out[3] = w;
};

/**
 * Poses critter `i` for drawing into its `figure`: the body bent and squashed by its springs, its
 * legs and arms placed for what it is doing, and its eyes on what it looks at.
 */
export function figureOf(
  body: Body,
  i: number,
  time: number,
  dt: number,
  spine: Float32Array,
  sights: Sights,
  figure: Figure,
) {
  const { knob } = sights;
  const g = body.grown;
  const riding = body.mode === 'ride';
  const asleep = body.mode === 'walk' && body.asleep > 0;

  const { centre } = frameOf(body, spine, knob);
  vec3n.copy(pose.up, up);
  vec3n.normalize(vec3n.cross(up, pose.ahead, side), side);
  vec3n.normalize(vec3n.cross(side, up, facing), facing);
  if (body.spin) {
    const axis = body.spinAxis ?? side;
    rotate(up, axis, body.spin, up);
    rotate(facing, axis, body.spin, facing);
    vec3n.normalize(vec3n.cross(up, facing, side), side);
  }
  vec3n.copy(up, body.upright);
  const bent = body.bend.value;
  vec3n.addScaled(bent, up, -vec3n.dot(bent, up), bend);
  if (vec3n.length(bend) > BEND.most * R) {
    vec3n.scale(vec3n.normalize(bend, bend), BEND.most * R, bend);
  }

  for (let k = 0; k < 2; k++) {
    const out = k ? -1 : 1;
    vec3n.addScaled(centre, side, out * R * 0.45, hips[k]);
    bendIn(vec3n.addScaled(hips[k], up, -R * 0.55, hips[k]), centre);
    bendIn(vec3n.addScaled(centre, side, out * R * 0.85, shoulders[k]), centre);
  }
  if (riding) {
    ridingPose(body, spine, time);
    for (let k = 0; k < 2; k++) vec3n.copy(steps[k], feet[k]);
  } else {
    feetOf(body, time);
  }
  for (let k = 0; k < 2; k++) {
    const out = k ? -1 : 1;
    vec3n.addScaled(facing, side, out * 0.3, forward);
    vec3n.addScaled(up, side, out * 0.7, climb);
    vec3n.lerp(forward, climb, body.steep, climb);
    if (riding) vec3n.lerp(climb, forward, body.seated, pole);
    else vec3n.copy(forward, pole);
    knee(hips[k], feet[k], THIGH, SHIN, pole, knees[k]);
  }
  for (let k = 0; k < 2; k++) handOf(body, i, k, time, knob);

  gazeOf(body, centre, time, sights);
  vec3n.copy(centre, joints[0]);
  for (let k = 0; k < 2; k++) {
    vec3n.copy(hips[k], joints[1 + k * 3]);
    vec3n.copy(knees[k], joints[2 + k * 3]);
    vec3n.copy(feet[k], joints[3 + k * 3]);
    vec3n.copy(shoulders[k], joints[7 + k * 2]);
    vec3n.copy(hands[k], joints[8 + k * 2]);
  }
  smooth(body, dt);
  vec3n.copy(placed[0], shown);
  vec3n.addScaled(shown, facing, R * 0.72, face);
  vec3n.addScaled(face, up, R * 0.3, face);

  // Stretching to reach and squashing at the end of each heave, and breathing in a nap
  let squash = body.squash.value;
  if (riding && body.gripping && body.steep > 0.5) {
    const u = body.heave;
    squash +=
      u < 0.35 ? Math.sin((Math.PI * u) / 0.35) * 0.15 : -std.smoothstep(0.35, 0.8, u) * 0.12;
  }
  if (asleep) squash += Math.sin(time * 2.4 + i) * 0.06;
  squash = std.clamp(squash, -0.35, 0.35);
  vec3n.addScaled(shown, up, R * 0.95 * (1 + squash), low);
  crown(body, vec3n.add(low, bend, low), dt);

  // The bound wraps the bent, squashed body with its eyes, each joint with its foot or hand,
  // and the tip with its ball or splayed ears
  let reach = R * (1 + Math.abs(squash)) * 1.3 + vec3n.length(bend);
  for (let j = 1; j <= JOINTS; j++) {
    reach = Math.max(reach, vec3n.distance(placed[j], shown) + R * 0.2);
  }
  reach = Math.max(reach, vec3n.distance(tip, shown) + R);

  put(body, shown, R * g, figure.body);
  vec3n.copy(up, figure.up);
  figure.up[3] = squash;
  vec3n.copy(facing, figure.ahead);
  figure.ahead[3] = body.blink;
  put(body, vec3n.add(face, look, face), body.mouth, figure.gaze);
  for (let j = 0; j < JOINTS; j++) put(body, placed[j + 1], 0, figure.joints[j]);
  vec3n.copy(body.color, figure.color);
  figure.color[3] = g;
  put(body, shown, reach * g, figure.bound);
  vec3n.scale(bend, g, figure.bend);
  for (let m = 0; m < 4; m++) figure.face[m] = body.mood[m];
  put(body, tip, body.kind, figure.top);
  return figure;
}

/**
 * Which way the pupils look, into `look`: along its way while riding or flying, else at what holds
 * its eye.
 */
function gazeOf(body: Body, centre: V3, time: number, sights: Sights) {
  const eyes = vec3n.addScaled(centre, facing, R * 0.72, high);
  vec3n.addScaled(eyes, up, R * 0.3, eyes);
  const riding = body.mode === 'ride';
  let target: V3 = sights.camera;
  if (riding && body.gripping) {
    vec3n.addScaled(centre, vec3n.addScaled(facing, up, 0.3, aim), 2, aim);
    target = aim;
  } else if (riding || body.mode === 'fly') {
    vec3n.copy(facing, aim);
    aim[1] -= 0.2;
    vec3n.addScaled(centre, aim, 2, aim);
    target = aim;
  } else if (body.wave > 0 && body.friend) {
    vec3n.copy(body.friend.at, aim);
    aim[1] += HIP + R;
    target = aim;
  } else if (body.cheer > 0 && body.fan?.posed) target = body.fan.drawn;
  else if (body.victim?.posed) target = body.victim.drawn;
  else if (body.grudge?.posed) target = body.grudge.drawn;
  else if (sights.pointer) target = sights.pointer;
  vec3n.normalize(vec3n.sub(target, eyes, look), look);
  if (body.dizzy > 0) {
    vec3n.scale(side, Math.cos(time * 9), aim);
    vec3n.addScaled(aim, up, Math.sin(time * 9), aim);
    vec3n.normalize(vec3n.addScaled(facing, aim, 0.8, look), look);
  }
  // Pupils stay on the front of the eyes, however far behind the target is
  const ahead = vec3n.dot(look, facing);
  if (ahead < 0.4) {
    vec3n.normalize(vec3n.addScaled(look, facing, -ahead, look), look);
    vec3n.scale(look, 0.92, look);
    vec3n.addScaled(look, facing, 0.4, look);
  }
  return look;
}
