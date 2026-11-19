import { std } from 'typegpu';
import { vec3n } from 'wgpu-matrix';
import {
  crossed,
  ease,
  FALL,
  HERALD,
  LADDER,
  LAND,
  PET,
  petted as isPetted,
  R,
  SHOO,
  STANDING,
  startle,
  topple,
  UP,
  WALK,
  jump,
  type Body,
  type Crowd,
  random,
  kick,
} from './body.ts';
import { createSeat, nearest, seat } from './spine.ts';
import { asideOf, blockOf, keepOut, passing, routeOf, STEER, stride, turnTo } from './steering.ts';
import { GROUND, type V3 } from './tray.ts';

/** How far critters come to look at the pointer, how close they stop, and bounces per second. */
const CURIOUS = { reach: 0.55, near: 0.13, bounce: 0.35 };
/** How close two critters come before they wave, for how long, and the least time between. */
const GREET = { reach: R * 5, time: 1.4, again: 9 };
/** Naps: chance per second while idle, and how long they last. */
const SLEEP = { chance: 0.04, shortest: 6, longest: 12 };
/** Follow the leader: chance per plan, followers, how far they look, spacing and duration. */
const TRAIN = { chance: 0.18, size: 3, reach: 1, gap: R * 3.2, time: 14 };

/** Shoves at a critter in the way. */
export const SHOVE = {
  /** Seconds blocked before one, and its chance then. */
  after: 0.7,
  chance: 0.4,
  /** Seconds of its wind-up, and of all of it. */
  windup: 0.35,
  time: 0.7,
  /** How fast the shover lunges and the victim stumbles. */
  lunge: 0.2,
  push: 0.25,
  /** The chance it knocks the victim over, and how hard it tips and lifts it then. */
  knock: 0.25,
  tip: 0.15,
  lift: 0.5,
  /** Seconds between shoves, and the chance of a shove back. */
  again: [10, 20],
  back: 0.35,
};

/**
 * Climbing onto the jelly, only near its fixed end, never near the free end where it rears up and
 * throws riders.
 */
const BOARD = {
  /** The chance a plan is to ride it, from the timidest critter to the boldest. */
  chance: [0.35, 0.8],
  /** The chance it goes up the approach at the fixed end, rather than from beside the jelly. */
  approach: 0.3,
  /** The world x beside the jelly it climbs on from, nearest where it is, and z off its middle. */
  beside: [-0.9, -0.45],
  edge: 0.29,
  /** Near it, it hesitates up to `hesitate` seconds, then dares at `dare`, up to 1 by nerve. */
  near: 0.3,
  hesitate: 1.2,
  dare: 0.6,
  /** Seconds a hop takes, onto a top at most `reach` along, `step` up and facing up `flat`. */
  hop: 0.5,
  reach: 0.55,
  step: 0.07,
  flat: 0.3,
  /** The most it lands off the jelly's middle. */
  lands: 0.09,
  /** Bold critters run the last `runUp` metres faster. */
  runUp: 0.35,
};

/**
 * Where a walker heads this step, within the lawn, how near it stops and how fast it goes over a
 * walk, whether it comes to look at the pointer, and the critter it follows.
 */
const intent = {
  goal: [0, 0] as [number, number],
  stop: 0,
  pace: 1,
  curious: false,
  leader: undefined as Body | undefined,
};

/** The heading a walker steers in, and whether it stops there or queues. */
const steered = { want: 0, arrived: false, waiting: false };

// Scratch the steps below write their vectors into
const forward: V3 = [0, 0, 0];
const push: V3 = [0, 0, 0];
const found = { s: 0, above: 0 };
const top = createSeat();

/** The heading from `from` towards `x, z`. */
const facingTo = (from: V3, x: number, z: number) => Math.atan2(x - from[0], z - from[2]);

/** One step of a critter on the ground. */
export function walk(body: Body, crowd: Crowd, dt: number) {
  countDown(body, dt);
  const x = body.at[0];
  const z = body.at[2];
  const due = crowd.errand === body;
  const { pointer, pointerSpeed } = crowd;
  const near = pointer ? Math.hypot(pointer[0] - x, pointer[2] - z) : Infinity;
  if (reacts(body, crowd, near, dt)) return;

  body.petted = pointer && near < PET.reach && pointerSpeed < PET.still ? body.petted + dt : 0;
  const petted = isPetted(body);
  if (body.plan <= 0 && !due) plan(body, crowd);
  intentOf(body, crowd, near, due, petted);
  const { curious, leader } = intent;
  const sociable = !body.wantsRide && !curious && !leader && !due;
  if (body.wave <= 0 && body.greeted <= 0 && sociable && body.grown > 0.5) greet(body, crowd);

  const { way, distance, toGoal } = routeOf(body, intent.goal, crowd.front, due);
  const { stop } = intent;
  if (toGoal < stop) body.arrived = true;
  else if (toGoal > stop + Math.max(0.04, stop * 0.5)) body.arrived = false;
  if (!steer(body, crowd, way, distance, toGoal, due, dt)) return;
  const { arrived, waiting } = steered;

  const aside = asideOf(body, crowd.bodies, leader);
  const turn = turnTo(body, headingOf(body, steered.want, arrived, due, petted, pointer), dt);
  let speed =
    arrived || waiting || body.wave > 0 || petted ? 0 : WALK.speed * body.grown * intent.pace;
  if (body.wantsRide && toGoal < BOARD.runUp) speed *= 1 + body.nerve * 0.6;
  if (body.load > 0) speed *= 0.75;
  if (body.dizzy > 0) {
    speed *= 0.4;
    body.heading += Math.sin(body.phase * 6) * 3 * dt;
  }
  stride(body, crowd, aside, speed * Math.max(0, Math.cos(turn)), dt);

  let mouth = 0;
  if (body.flee > 0) mouth = 0.9;
  else if (body.cheer > 0) mouth = 0.8;
  else if (body.wave > 0) mouth = 0.5;
  else if (curious && arrived) mouth = 0.3;
  body.mouth = ease(body.mouth, mouth, 6, dt);

  const settled =
    arrived &&
    vec3n.length(aside) < 0.01 &&
    body.wave <= 0 &&
    body.dizzy <= 0 &&
    !petted &&
    !leader &&
    !due &&
    body.cheer <= 0 &&
    body.load === 0 &&
    body.grown > 0.9;
  // Beside its spot or the critters there, it climbs on rather than waits
  if (body.wantsRide && arrived && toGoal < BOARD.near) brink(body, crowd, turn, dt);
  else if (settled && idles(body, curious, dt)) return;
  if (due && arrived && Math.abs(turn) < 0.3 && !crowd.world.holding) leap(body);
}

function countDown(body: Body, dt: number) {
  body.plan -= dt;
  body.greeted -= dt;
  body.wave -= dt;
  body.flee -= dt;
  body.cheer -= dt;
  body.train -= dt;
  body.shoveIn -= dt;
  body.rideIn -= dt;
  // A line breaks up when it ends, or when its leader stops walking other than onto the jelly
  const mode = body.leader?.mode;
  if (body.train <= 0 || (mode !== 'walk' && mode !== 'ride')) body.leader = undefined;
}

/** Naps, startles and shoves, which take over from walking. Returns whether one did. */
function reacts(body: Body, crowd: Crowd, near: number, dt: number) {
  const { pointer, pointerSpeed } = crowd;
  if (body.asleep > 0) {
    body.asleep -= dt;
    vec3n.zero(body.velocity);
    body.mouth = ease(body.mouth, 0, 4, dt);
    if (pointer && near < SHOO.reach && pointerSpeed > PET.still) startle(body, pointer);
    return true;
  }
  if (pointer && pointerSpeed > SHOO.speed && near < SHOO.reach && body.flee <= 0) {
    startle(body, pointer);
    return true;
  }
  if (body.shove > 0) {
    shoving(body, crowd, dt);
    return true;
  }

  // Shoves back once it has caught its balance
  const grudge = body.grudge;
  if (grudge && vec3n.length(body.stumble) < 0.03) {
    body.grudge = undefined;
    const gap = Math.hypot(grudge.at[0] - body.at[0], grudge.at[2] - body.at[2]);
    if (grudge.mode === 'walk' && gap < STEER.spacing + R * 2) startShove(body, grudge);
  }
  return false;
}

/** Starts a shove at `victim`: a lean back, then a lunge with both hands. */
function startShove(body: Body, victim: Body) {
  body.shove = 1e-3;
  body.victim = victim;
  body.blocked = 0;
  body.joy = SHOVE.time + 0.6;
  kick(body.bend, vec3n.set(Math.sin(body.heading), 0, Math.cos(body.heading), forward), -0.3);
  body.squash.velocity -= 1;
}

/** Turns to the victim, winds up and lunges, then waits a while before it shoves again. */
function shoving(body: Body, crowd: Crowd, dt: number) {
  const victim = body.victim;
  body.shove += dt;
  if (victim) turnTo(body, facingTo(body.at, victim.at[0], victim.at[2]), dt, WALK.turn * 2);
  vec3n.set(Math.sin(body.heading), 0, Math.cos(body.heading), forward);
  const lunge = body.shove > SHOVE.windup ? Math.max(0, 1 - (body.shove - SHOVE.windup) / 0.2) : 0;
  vec3n.scale(forward, lunge * SHOVE.lunge, push);
  vec3n.lerp(body.velocity, push, Math.min(1, dt * 20), body.velocity);
  vec3n.addScaled(body.at, body.velocity, dt, body.at);
  body.at[1] = GROUND;
  keepOut(body, crowd.world.obstacles, dt);
  body.mouth = ease(body.mouth, 0.6, 8, dt);

  if (crossed(body.shove, dt, SHOVE.windup)) {
    kick(body.bend, forward, 0.7);
    if (victim) shoved(victim, body);
  }
  if (body.shove >= SHOVE.time) {
    body.shove = 0;
    body.victim = undefined;
    body.shoveIn = random(SHOVE.again[0], SHOVE.again[1]);
  }
}

/** Stumbles away from a shove by `shover`, or now and then topples, and may come back to shove. */
function shoved(victim: Body, shover: Body) {
  const gap = Math.hypot(victim.at[0] - shover.at[0], victim.at[2] - shover.at[2]);
  // A critter still surprised by the last shove or bump takes no other
  if (victim.mode !== 'walk' || victim.shock > 0 || gap >= STEER.spacing + R) return;
  vec3n.set(victim.at[0] - shover.at[0], 0, victim.at[2] - shover.at[2], push);
  vec3n.normalize(push, push);
  if (Math.random() < SHOVE.knock) {
    topple(victim, vec3n.scale(push, SHOVE.tip, push), SHOVE.lift);
    return;
  }
  vec3n.scale(push, SHOVE.push, victim.stumble);
  victim.shock = 1.2;
  victim.asleep = 0;
  victim.wave = 0;
  victim.squash.velocity -= 1.5;
  // Only one free to shove shoves back, so a shove back draws none
  const back = victim.shoveIn <= 0 && Math.random() < SHOVE.back;
  victim.grudge = back ? shover : undefined;
}

/** Picks somewhere to go: onto the jelly, more often the bolder it is, or somewhere on the lawn. */
function plan(body: Body, crowd: Crowd) {
  if (body.leader) {
    body.plan = 1;
    return;
  }
  const { front } = crowd;
  const chance = BOARD.chance[0] + (BOARD.chance[1] - BOARD.chance[0]) * body.nerve;
  body.wantsRide = false;
  if (front > LADDER + 0.4 && body.rideIn <= 0 && Math.random() < chance) wantRide(body);
  else lawnSpot(front, body.goal);
  // Long enough to get there, so riders reach the ladder
  const distance = Math.hypot(body.goal[0] - body.at[0], body.goal[1] - body.at[2]);
  body.plan = (body.wantsRide ? 2 : 1) * (distance / WALK.speed) + random(1, 3);
  if (Math.random() < TRAIN.chance) startTrain(body, crowd);
}

/** Sets out to ride the jelly, up the approach or from beside it nearest where it is. */
function wantRide(body: Body) {
  body.wantsRide = true;
  body.brink = 0;
  const along = std.clamp(body.at[0] + random(-0.15, 0.15), BOARD.beside[0], BOARD.beside[1]);
  if (Math.random() < BOARD.approach) approach(body.goal);
  else {
    body.goal[0] = along;
    body.goal[1] = Math.sign(body.at[2] || 1) * BOARD.edge;
  }
}

/** Writes where critters climb on at the jelly's fixed end into `goal`. */
const approach = (goal: [number, number]) => {
  goal[0] = LADDER;
  goal[1] = random(-0.1, 0.1);
};

/** Writes somewhere on the lawn behind the front into `goal`. */
function lawnSpot(front: number, goal: [number, number]) {
  const right = std.clamp(front - 0.3, LAND.x[0], LAND.x[1]);
  goal[0] = random(LAND.x[0], right);
  goal[1] = random(LAND.z[0], LAND.z[1]);
}

/** Whether `b` is free to follow `leader` in a line. */
const followsFreely = (b: Body, leader: Body, crowd: Crowd) =>
  b !== leader &&
  b !== crowd.herald &&
  b.mode === 'walk' &&
  !b.leader &&
  b.asleep <= 0 &&
  !b.wantsRide &&
  b.grown > 0.5 &&
  Math.hypot(b.at[0] - leader.at[0], b.at[2] - leader.at[2]) < TRAIN.reach;

/** Lines up to `TRAIN.size` free critters behind `leader`, nearest first. */
function startTrain(leader: Body, crowd: Crowd) {
  let ahead = leader;
  for (let n = 0; n < TRAIN.size; n++) {
    let next: Body | undefined;
    let least = Infinity;
    for (const b of crowd.bodies) {
      if (!followsFreely(b, leader, crowd)) continue;
      const gap = Math.hypot(b.at[0] - leader.at[0], b.at[2] - leader.at[2]);
      if (gap < least) {
        least = gap;
        next = b;
      }
    }
    if (!next) break;
    next.leader = ahead;
    next.train = TRAIN.time;
    ahead = next;
  }
  if (ahead !== leader) leader.plan = Math.max(leader.plan, 6);
}

/** Works out the walker's `intent` for this step. */
function intentOf(body: Body, crowd: Crowd, near: number, due: boolean, petted: boolean) {
  const { pointer } = crowd;
  const leader = body.leader;
  const curious =
    pointer !== undefined &&
    !body.wantsRide &&
    body.dizzy <= 0 &&
    !leader &&
    !due &&
    near < CURIOUS.reach;
  aimOf(body, crowd, due, petted, curious ? pointer : undefined);
  // Whatever it heads for lies on the lawn, so it turns back short of the lawn's edge
  const { goal } = intent;
  goal[0] = std.clamp(goal[0], LAND.x[0], LAND.x[1]);
  goal[1] = std.clamp(goal[1], LAND.z[0], LAND.z[1]);
  intent.curious = curious;
  intent.leader = leader;
}

/** Sets where the walker heads in `intent`, how near it stops there and how fast it goes. */
const aim = (x: number, z: number, stop: number, pace: number) => {
  intent.goal[0] = x;
  intent.goal[1] = z;
  intent.stop = stop;
  intent.pace = pace;
};

/**
 * Where it heads, first of: away from what startled it, to the lever, staying put while petted or
 * cheering, after its leader, to the pointer it came to see, or where it planned.
 */
function aimOf(body: Body, crowd: Crowd, due: boolean, petted: boolean, seen: V3 | undefined) {
  const x = body.at[0];
  const z = body.at[2];
  const leader = body.leader;
  if (body.flee > 0) {
    vec3n.set(x - body.fleeFrom[0], 0, z - body.fleeFrom[2], push);
    vec3n.normalize(push, push);
    aim(x + push[0] * 0.5, z + push[2] * 0.5, 0.05, 1.8);
  } else if (due) {
    const knob = crowd.world.knob;
    aim(knob[0], knob[2] + HERALD.stand, R, HERALD.pace);
  } else if (petted || (body.cheer > 0 && body.fan)) {
    aim(x, z, 0.05, 1);
  } else if (leader?.mode === 'ride' && !body.wantsRide) {
    // Following the leader up the jelly, setting out once this step is done
    aim(body.goal[0], body.goal[1], 0.05, 1);
    wantRide(body);
    body.plan = 30;
    body.leader = undefined;
  } else if (leader) {
    // On the line from the leader, so the train trails it however it turns
    vec3n.set(x - leader.at[0], 0, z - leader.at[2], push);
    vec3n.normalize(push, push);
    aim(leader.at[0] + push[0] * TRAIN.gap, leader.at[2] + push[2] * TRAIN.gap, R * 0.6, 1.15);
  } else if (seen) {
    aim(seen[0], seen[2], CURIOUS.near, 1);
  } else {
    aim(body.goal[0], body.goal[1], 0.05, 1);
  }
}

/** Waves with the first free critter near enough, which waves back. */
function greet(body: Body, crowd: Crowd) {
  for (const other of crowd.bodies) {
    if (other === body || other.mode !== 'walk' || other.grown < 0.5) continue;
    if (other.wave > 0 || other.greeted > 0 || other.wantsRide || other.asleep > 0) continue;
    if (Math.hypot(other.at[0] - body.at[0], other.at[2] - body.at[2]) > GREET.reach) continue;
    waveAt(body, other);
    waveAt(other, body);
    return;
  }
}

const waveAt = (body: Body, friend: Body) => {
  body.wave = GREET.time;
  body.friend = friend;
  body.greeted = GREET.again + random(0, 5);
};

/**
 * Writes into `steered` the heading to walk in round what blocks the way, and whether to stop
 * there or queue. It stops where its goal is taken or at the edge of a crowd round it, queues
 * behind a still critter nearer the goal, and passes on a side it keeps. Blocked for long, it tries
 * the other side, shoves, and at last goes elsewhere. Returns false when it starts a shove.
 */
function steer(
  body: Body,
  crowd: Crowd,
  way: readonly [number, number],
  distance: number,
  toGoal: number,
  due: boolean,
  dt: number,
) {
  const { goal, stop, curious, leader } = intent;
  const block = blockOf(body, crowd, way, distance, leader);
  steered.want = Math.atan2(way[0], way[1]);
  steered.arrived = body.arrived;
  steered.waiting = false;
  body.dodgeFor -= dt;
  if (!block) {
    body.blocked = Math.max(0, body.blocked - dt);
    return true;
  }

  const { other, room } = block;
  const gap = Math.hypot(block.x - body.at[0], block.z - body.at[2]);
  const close = gap < room + R;
  const taken = Math.hypot(goal[0] - block.x, goal[1] - block.z) < room;
  const ahead = other && Math.hypot(goal[0] - other.at[0], goal[1] - other.at[2]) < toGoal;
  if (close && (taken || (ahead && toGoal < stop + STEER.spacing))) steered.arrived = true;
  else if (close && ahead && vec3n.length(other.velocity) < 0.08 && toGoal < stop + 0.3) {
    steered.waiting = true;
  }
  const { arrived, waiting } = steered;

  // Blocked counts while it waits or barely moves
  const stuck = !arrived && (waiting || vec3n.length(body.velocity) < 0.05);
  body.blocked = stuck ? body.blocked + dt : Math.max(0, body.blocked - dt);
  if (!arrived && !waiting) steered.want = passing(body, block, steered.want, gap, dt);

  if (other && close && !arrived && crossed(body.blocked, dt, SHOVE.after) && body.shoveIn <= 0) {
    if (due || Math.random() < SHOVE.chance) {
      startShove(body, other);
      return false;
    }
  }
  if (body.blocked > STEER.give && !curious && !due && body.flee <= 0) {
    body.blocked = 0;
    body.plan = 0;
    body.leader = undefined;
  }
  return true;
}

/**
 * Which way it faces, first of: away from the lever once there, at the jelly once beside it to
 * climb on, at a friend it waves to, at a critter flying off the jelly, at the pointer petting it
 * or once it arrives, on as it was once it arrives, or the way it walks.
 */
function headingOf(
  body: Body,
  way: number,
  arrived: boolean,
  due: boolean,
  petted: boolean,
  pointer: V3 | undefined,
) {
  if (due && arrived) return 0;
  if (body.wantsRide && arrived) {
    return body.goal[0] === LADDER ? Math.PI / 2 : facingTo(body.at, body.at[0], 0);
  }
  if (body.wave > 0 && body.friend) return facingTo(body.at, body.friend.at[0], body.friend.at[2]);
  if (body.cheer > 0 && body.fan) return facingTo(body.at, body.fan.at[0], body.fan.at[2]);
  if ((petted || arrived) && pointer) return facingTo(body.at, pointer[0], pointer[2]);
  if (arrived) return body.heading;
  return way;
}

/**
 * Now and then naps, or hops or flips for nothing, and bounces at a pointer it came to see.
 * Returns whether it did.
 */
function idles(body: Body, curious: boolean, dt: number) {
  if (!curious && Math.random() < dt * SLEEP.chance) {
    body.asleep = random(SLEEP.shortest, SLEEP.longest);
    return true;
  }
  if (Math.random() < dt * (curious ? CURIOUS.bounce : 0.05)) {
    jump(body, curious ? 0.5 : 0.65, !curious && Math.random() < 0.5);
    return true;
  }
  return false;
}

/**
 * By the jelly, facing it: hesitates as long as its nerve lets it, then beside the jelly may give
 * up, and otherwise hops onto the jelly's top. Where that top is too high or steep it tries the
 * approach, and at the approach it gives up.
 */
function brink(body: Body, crowd: Crowd, turn: number, dt: number) {
  body.brink += dt;
  if (Math.abs(turn) > 0.3 || body.brink < BOARD.hesitate * (1 - body.nerve)) return;
  body.brink = 0;
  const atApproach = body.goal[0] === LADDER;
  const surface = topNear(crowd.spine, body.at[0], body.at[2]);
  const dares = atApproach || Math.random() < BOARD.dare + (1 - BOARD.dare) * body.nerve;
  if (surface && dares) hop(body, surface);
  else if (surface || atApproach) giveUp(body, crowd.front);
  else approach(body.goal);
}

/**
 * The jelly's top nearest world x, on the side of `z`, if it lies within `BOARD.reach` of the
 * fixed end, is low enough to hop onto and faces up.
 */
function topNear(spine: Float32Array, x: number, z: number) {
  const s = Math.max(nearest(spine, x, GROUND, found).s, R);
  if (s > BOARD.reach) return undefined;
  const { surface, normal } = seat(
    spine,
    s,
    Math.sign(z) * Math.min(Math.abs(z), BOARD.lands),
    top,
  );
  if (surface[1] > GROUND + BOARD.step || normal[1] < BOARD.flat) return undefined;
  return surface;
}

/** Hops onto `surface` in `BOARD.hop` seconds, to land there and climb on. */
function hop(body: Body, surface: V3) {
  jump(body, 0, false);
  const t = BOARD.hop;
  const { at } = body;
  vec3n.set(
    (surface[0] - at[0]) / t,
    (surface[1] - at[1]) / t + (FALL * t) / 2,
    (surface[2] - at[2]) / t,
    body.velocity,
  );
  body.joy = 1;
}

/** Thinks better of riding and wanders off. */
function giveUp(body: Body, front: number) {
  body.wantsRide = false;
  lawnSpot(front, body.goal);
  body.squash.velocity -= 1;
}

/** Leaps up to the lever's knob. */
function leap(body: Body) {
  body.mode = 'pull';
  body.pulling = 0;
  vec3n.addScaled(body.at, UP, STANDING, body.leapFrom);
  body.squash.velocity -= 1.5;
}
