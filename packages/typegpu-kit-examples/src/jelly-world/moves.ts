import { clock } from '@typegpu/kit';
import { std } from 'typegpu';
import { vec3n } from 'wgpu-matrix';
import {
  CLIMB,
  CRITTERS,
  ease,
  FALL,
  HEAVE,
  HERALD,
  jump,
  KNOCK,
  land,
  R,
  random,
  STANDING,
  topple,
  turnBetween,
  UP,
  type Body,
  type Crowd,
} from './body.ts';
import { createSeat, nearest, seat, spineLength } from './spine.ts';
import { GROUND, SLOT, type V3 } from './tray.ts';
import { STEER } from './steering.ts';

/** The half-width critters ride the jelly within. */
const RIDE_Z = 0.12;
/** How far up the jelly faces where critters land on it, and below which riders fall off. */
const FACING = { land: 0, leave: -0.2 };
/**
 * The jelly throws a rider when it drops away under it faster than `away`, or pushes it up faster
 * than `toward` while moving faster than the square root of `fast`, each times the rider's grip.
 * Riders thrown or off an end hop up as they leave, at `hop.thrown` and `hop.end`.
 */
const THROW = { away: 1.2, toward: 0.9, fast: 1.4, hop: { end: 0.6, thrown: 0.2 } };
/** Seconds a critter that climbed on walks before it wants to ride again. */
const RIDDEN = { shortest: 5, longest: 12 };
/** How long a critter sits on another's head. */
const PERCH = { shortest: 4, longest: 9 };
/** Walkers this near cheer one flying off the jelly's end, this long, and again this long after. */
const CHEER = { reach: 1, time: 1.6, again: 6 };

// Scratch the steps below write their vectors into
const here = createSeat();
const there = createSeat();
const then = createSeat();
const found = { s: 0, above: 0 };
const carried: V3 = [0, 0, 0];
const along: V3 = [0, 0, 0];
const out: V3 = [0, 0, 0];
const top: V3 = [0, 0, 0];
const before: V3 = [0, 0, 0];
const hanging: V3 = [0, 0, 0];
const centre: V3 = [0, 0, 0];
const apart: V3 = [0, 0, 0];
const direction: V3 = [0, 0, 0];
const relative: V3 = [0, 0, 0];
const push: V3 = [0, 0, 0];

/**
 * Climbs the jelly or slides down it, carried as it moves, until it goes off an end, falls off
 * where it overhangs or is thrown. Critters near the end it flies off cheer. The jelly moved from
 * `previous` to `spine` over the last `over` seconds of its fixed steps.
 */
export function ride(
  body: Body,
  crowd: Crowd,
  dt: number,
  spine: Float32Array,
  previous: Float32Array,
  over: number,
) {
  const total = spineLength(spine);
  const { tangent, normal } = seat(spine, body.s, body.at[2], here);
  const slope = tangent[1];
  if (body.gripping) {
    climb(body, slope, dt);
    if (slope > CLIMB.steepest || slope < -CLIMB.over) body.gripping = false;
  } else {
    body.sliding += -FALL * slope * dt - body.sliding * CLIMB.friction * dt;
    body.s += body.sliding * dt;
    if (Math.abs(body.sliding) < 0.05 && slope > -0.1 && slope < 0.5) {
      body.gripping = true;
      body.steep = 0;
    }
  }

  body.seated = ease(body.seated, body.gripping ? 0 : 1, 8, dt);
  body.turn = ease(body.turn, !body.gripping && body.sliding < 0 ? Math.PI : 0, 6, dt);
  const effort = body.gripping && body.steep > 0.5 && body.heave > 0.35 && body.heave < 0.8;
  let mouth = body.gripping ? 0.1 : Math.min(1, Math.abs(body.sliding) * 1.5);
  if (effort) mouth = 0.25;
  body.mouth = ease(body.mouth, mouth, 8, dt);

  // The jelly carries them, and throws them when it moves away faster than they hold on
  const { surface } = seat(spine, body.s, body.at[2], there);
  vec3n.sub(surface, seat(previous, body.s, body.at[2], then).surface, carried);
  vec3n.scale(carried, 1 / over, carried);
  const lift = vec3n.dot(carried, normal);
  const hold = body.gripping ? CLIMB.grip : 1;
  const end = body.s >= total - 0.02;
  const launched = end || body.s <= 0;
  const overhang = normal[1] < FACING.leave;
  vec3n.scale(tangent, body.sliding, along);
  const thrown =
    lift < -THROW.away * hold ||
    (lift > THROW.toward * hold && vec3n.dot(carried, carried) > THROW.fast * hold);
  if (launched || overhang || thrown) {
    const hop = launched ? THROW.hop.end : thrown ? THROW.hop.thrown : 0;
    body.mode = 'fly';
    vec3n.add(along, carried, body.velocity);
    body.velocity[1] += hop;
    body.heading = Math.atan2(body.velocity[0], body.velocity[2]);
    vec3n.copy(surface, body.at);
    if (end) cheered(body, crowd.bodies);
    return;
  }
  vec3n.add(along, carried, body.velocity);
  vec3n.copy(surface, body.at);
}

/** Flies off the jelly's end for joy, cheered by walkers near enough that haven't cheered lately. */
function cheered(body: Body, bodies: readonly Body[]) {
  body.joy = 1.5;
  for (const fan of bodies) {
    // Walking counts a cheer on down past 0, to the seconds since it ended
    if (fan.mode !== 'walk' || fan.asleep > 0 || fan.cheer > -CHEER.again) continue;
    if (Math.hypot(fan.at[0] - body.at[0], fan.at[2] - body.at[2]) > CHEER.reach) continue;
    fan.cheer = CHEER.time;
    fan.fan = body;
  }
}

/** Walks up the jelly leaning into the slope, or where steep reaches, grabs and heaves itself. */
function climb(body: Body, slope: number, dt: number) {
  body.steep = ease(body.steep, std.smoothstep(CLIMB.steep[0], CLIMB.steep[1], slope), 5, dt);
  let moved: number;
  if (body.steep < 0.5) {
    moved = CLIMB.speed * (1 - 0.4 * Math.max(slope, 0)) * dt;
    body.cycle += (moved * CLIMB.planted) / CLIMB.stride;
    body.heave = 0;
    body.from = body.s + moved;
    body.grabFrom = body.s + R * 0.6;
    body.grab = body.s + HEAVE.reach;
  } else {
    body.heave += dt / HEAVE.time;
    if (body.heave >= 1) {
      body.heave -= 1;
      body.from = body.s;
      body.grabFrom = body.grab;
      body.grab = body.s + HEAVE.reach;
    }
    moved = body.from + HEAVE.pull * std.smoothstep(0.35, 0.75, body.heave) - body.s;
  }
  body.sliding = moved / dt;
  body.s += moved;
}

/**
 * Falls, off props, onto a head, onto the jelly where it faces up, or onto the ground. A critter
 * still wanting to ride, hopping on, missing the jelly, tries again from the ground.
 */
export function fly(body: Body, crowd: Crowd, dt: number, spine: Float32Array) {
  body.velocity[1] -= FALL * dt;
  vec3n.addScaled(body.at, body.velocity, dt, body.at);
  body.mouth = Math.min(1, body.mouth + dt * 4);
  if (body.spinning) {
    body.spin += body.spinning * dt;
    if (!body.spinAxis && body.spin >= Math.PI * 2) {
      body.spin = 0;
      body.spinning = 0;
    }
  }

  const x = body.at[0];
  const y = body.at[1];
  const z = body.at[2];
  for (const o of crowd.world.obstacles) {
    const gap = Math.hypot(x - o.x, z - o.z);
    if (y > GROUND + 0.25 || gap > o.radius + R || gap < 1e-4) continue;
    vec3n.set((x - o.x) / gap, 0, (z - o.z) / gap, out);
    vec3n.addScaled(body.at, out, o.radius + R - gap, body.at);
    const into = vec3n.dot(body.velocity, out);
    if (into < 0) vec3n.addScaled(body.velocity, out, -into * 1.5, body.velocity);
    body.squash.velocity -= 1;
  }
  const falling = body.velocity[1] < 0;
  if (falling && (onHead(body, crowd.bodies, x, y, z) || onJelly(body, spine, x, y, z))) return;

  const inSlot = Math.abs(x) < SLOT.x && Math.abs(z) < SLOT.z;
  const floor = inSlot ? SLOT.floor : GROUND;
  if (body.at[1] > floor) return;
  body.at[1] = floor;
  const impact = Math.abs(body.velocity[1]);
  if (inSlot) {
    // Out of the slot with a hop towards the nearer side
    vec3n.set(0, 1.1, Math.sign(z || 1) * 0.6, body.velocity);
    body.squash.velocity -= impact * 2;
    return;
  }
  body.at[1] = GROUND;
  land(body, impact);
}

/** Sits on the head of a critter under `x, y, z` it falls onto. Returns whether it did. */
function onHead(body: Body, bodies: readonly Body[], x: number, y: number, z: number) {
  for (const other of bodies) {
    if (other === body || !other.posed || other.grown < 0.5 || under(other, body)) continue;
    if (other.mode === 'fly') continue;
    vec3n.addScaled(other.drawn, other.upright, R, top);
    const gap = Math.hypot(x - top[0], z - top[2]);
    if (gap < R * 0.9 && y < top[1] + R * 0.5 && y > top[1] - R * 0.8) {
      body.mode = 'perch';
      body.base = other;
      body.perchFor = random(PERCH.shortest, PERCH.longest);
      body.spinning = 0;
      body.knocked = false;
      body.squash.velocity -= 1.5;
      return true;
    }
  }
  return false;
}

/**
 * Lands at `x, y, z` on the jelly where it faces up. One that hopped on to ride grips and climbs;
 * any other lands sliding. Returns whether it landed.
 */
function onJelly(body: Body, spine: Float32Array, x: number, y: number, z: number) {
  if (Math.abs(z) >= RIDE_Z || x <= -1.02 || x >= 1.0) return false;
  nearest(spine, x, y, found);
  const short = found.s < spineLength(spine) - 0.06;
  const facing = seat(spine, found.s, z, here).normal[1];
  if (found.above >= 0.01 || found.above <= -0.06 || !short || facing <= FACING.land) return false;

  const clambering = body.wantsRide;
  body.mode = 'ride';
  body.s = found.s;
  body.sliding = clambering ? 0 : std.clamp(body.velocity[0], -0.6, 0.6);
  body.gripping = clambering;
  body.steep = 0;
  body.turn = body.sliding < 0 ? Math.PI : 0;
  body.wantsRide = false;
  body.plan = 0;
  if (clambering) body.rideIn = random(RIDDEN.shortest, RIDDEN.longest);
  body.squash.velocity -= Math.abs(body.velocity[1]) * 2;
  body.spinning = 0;
  body.knocked = false;
  return true;
}

/** Whether `body` stands anywhere under `other` in a stack. */
function under(body: Body, other: Body) {
  for (let b: Body | undefined = body; b; b = b.base) if (b === other) return true;
  return false;
}

/** Sits on the head of its base until it jumps off, or falls when the base does. */
export function perch(body: Body, dt: number) {
  const base = body.base;
  if (!base?.posed || base.mode === 'fly') {
    body.mode = 'fly';
    body.base = undefined;
    if (base) vec3n.copy(base.velocity, body.velocity);
    else vec3n.zero(body.velocity);
    body.velocity[1] += 0.3;
    return;
  }
  body.perchFor -= dt;
  if (body.perchFor <= 0) {
    jump(body, 0.75, Math.random() < 0.6);
    vec3n.add(body.velocity, base.velocity, body.velocity);
    return;
  }
  vec3n.copy(body.at, before);
  vec3n.addScaled(base.drawn, base.upright, R * 0.98, body.at);
  vec3n.scale(vec3n.sub(body.at, before, body.velocity), 1 / dt, body.velocity);
  body.heading += turnBetween(body.heading, base.heading) * Math.min(1, dt * 2);
  body.mouth = ease(body.mouth, 0.2, 4, dt);
}

/**
 * Leaps to the lever's knob and hangs on it until the lever is down, then drops off. Returns
 * whether it is done with the lever.
 */
export function hang(body: Body, crowd: Crowd, dt: number) {
  const { world } = crowd;
  vec3n.addScaled(body.at, UP, STANDING, before);
  body.pulling += dt;
  const u = Math.min(1, body.pulling / HERALD.leap);
  vec3n.add(world.knob, HERALD.hang, hanging);
  vec3n.copy(hanging, centre);
  if (u < 1) {
    vec3n.lerp(body.leapFrom, hanging, std.smoothstep(0, 1, u), centre);
    vec3n.addScaled(centre, UP, Math.sin(Math.PI * u) * HERALD.arc, centre);
  }
  vec3n.addScaled(centre, UP, -STANDING, body.at);
  vec3n.scale(vec3n.sub(centre, before, body.velocity), 1 / dt, body.velocity);
  body.heading = 0;
  body.mouth = ease(body.mouth, u < 1 ? 0.7 : 0.35, 8, dt);

  if (world.holding) {
    // Shaken off by the pointer taking the lever
    body.mode = 'fly';
    vec3n.set(0, 0.3, 0.4, body.velocity);
    body.shock = 1;
  } else if (crowd.errand !== body || (u >= 1 && world.pulled > HERALD.done)) {
    // Drops off for joy, an ordinary critter from now on
    body.mode = 'fly';
    vec3n.scale(body.velocity, 0.5, body.velocity);
    body.velocity[1] += 0.5;
    body.velocity[2] += 0.3;
    body.joy = 2;
    body.plan = 0;
    return true;
  }
  return false;
}

/** Follows the pointer to `to`, over the jelly rather than through it. */
export function carry(body: Body, to: V3, dt: number, spine: Float32Array) {
  vec3n.copy(body.at, before);
  vec3n.addScaled(body.at, vec3n.sub(to, body.at, apart), clock.ease(dt, 18), body.at);
  if (Math.abs(body.at[2]) < RIDE_Z + 0.05) {
    nearest(spine, body.at[0], body.at[1], found);
    if (found.above < 0.01 && found.above > -0.05) {
      const on = seat(spine, found.s, body.at[2], here);
      vec3n.addScaled(on.surface, on.normal, 0.01, body.at);
    }
  }
  vec3n.sub(body.at, before, body.velocity);
  vec3n.scale(body.velocity, 1 / Math.max(dt, 1e-4), body.velocity);
  body.mouth = 1;
  body.heldFor += dt;
}

/** Pushes overlapping critters apart, and knocks over those that meet fast. */
export function collide(bodies: readonly Body[], dt: number) {
  for (let i = 0; i < CRITTERS; i++) {
    for (let j = i + 1; j < CRITTERS; j++) {
      const a = bodies[i];
      const b = bodies[j];
      if (!a.posed || !b.posed || a.grown < 0.5 || b.grown < 0.5) continue;
      if (a.base === b || b.base === a) continue;
      // Walkers meet on the ground where they are now, not where they were last drawn
      const walking = a.mode === 'walk' && b.mode === 'walk';
      if (walking) vec3n.set(a.at[0] - b.at[0], 0, a.at[2] - b.at[2], apart);
      else vec3n.sub(a.drawn, b.drawn, apart);
      const distance = vec3n.length(apart);
      if (distance >= R * 2 || distance < 1e-5) continue;

      vec3n.scale(apart, 1 / distance, direction);
      const closing = vec3n.dot(vec3n.sub(a.velocity, b.velocity, relative), direction);
      if (closing < -KNOCK.speed) {
        const shareA = shareOf(a, b);
        const shareB = shareOf(b, a);
        const impulse = -(1 + KNOCK.bounce) * closing;
        if (shareA) topple(a, vec3n.scale(direction, impulse * shareA, push));
        if (shareB) topple(b, vec3n.scale(direction, -impulse * shareB, push));
        continue;
      }
      const overlap = Math.min((R * 2 - distance) * 0.5, STEER.apart * dt);
      nudge(a, direction, overlap);
      nudge(b, direction, -overlap);
    }
  }
}

/** The share of a knock `body` takes from `other`: none while held, all from a held one. */
function shareOf(body: Body, other: Body) {
  if (body.mode === 'held') return 0;
  return other.mode === 'held' ? 1 : 0.5;
}

/** Moves a walker or a rider `by` along `n`. */
function nudge(body: Body, n: V3, by: number) {
  if (body.mode === 'walk') {
    body.at[0] += n[0] * by;
    body.at[2] += n[2] * by;
  } else if (body.mode === 'ride') body.s += n[0] * by;
}
