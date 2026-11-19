type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

const GRAVITY = 9.81;
/** The step the chain is solved in, in seconds: small steps let light links hold a heavy body. */
const SUBSTEP = 1 / 2400;
/** Air slowing the links, the body and the body's turning, per second. */
const LINK_AIR = 0.4;
const BODY_AIR = 0.05;
const TURN_AIR = 0.6;
/** A hand pulls like a critically damped spring of this frequency, in hertz, at most this hard. */
const HAND = 2.6;
const HAND_DAMPING = 0.8;
const HAND_MOST = 80;
/** The hand also holds the body's turn, loosely, like a wrist: hertz and damping ratio. */
const WRIST = 1.2;
const WRIST_DAMPING = 0.7;
/**
 * After a hand lets go, the idle swing waits this many seconds and then eases in over as many
 * again, pumping or braking the swing toward its amplitude at this rate per second.
 */
const IDLE_WAIT = 3;
const IDLE_EASE = 3;
const IDLE_RATE = 0.9;

/** Where the censer's chain hangs from, in metres. */
export const PIVOT: Vec3 = [0, 3.8, 0];
/** From the pivot to the middle of the censer's body. */
export const LENGTH = 2.6;
/** The body's size: its model is drawn in units of this many metres. */
export const SIZE = 0.4;
/**
 * The censer's outline from its finial to its foot: radius and height from the body's middle, in
 * units of `SIZE`.
 */
export const PROFILE = [
  0, 1, 0.05, 0.98, 0.075, 0.91, 0.04, 0.84, 0.1, 0.8, 0.2, 0.74, 0.3, 0.64, 0.38, 0.52, 0.44, 0.38,
  0.47, 0.26, 0.5, 0.22, 0.55, 0.2, 0.55, 0.12, 0.51, 0.1, 0.53, 0, 0.55, -0.12, 0.52, -0.28, 0.44,
  -0.42, 0.3, -0.53, 0.16, -0.58, 0.11, -0.64, 0.1, -0.72, 0.2, -0.78, 0.25, -0.84, 0.24, -0.88, 0,
  -0.89,
].flatMap((v, i, all) => (i % 2 ? [] : [[v, all[i + 1]] as const]));
/** Where the legs meet above the body's middle, and where they end on its rim, in metres. */
export const CROWN = 1.85 * SIZE;
const FEET = [0, 1, 2].map((i) => {
  const a = (i / 3) * Math.PI * 2 + 0.5;
  return [Math.cos(a) * 0.53 * SIZE, 0.16 * SIZE, Math.sin(a) * 0.53 * SIZE] as Vec3;
});
/**
 * A bronze body of a few kilograms on a light iron chain: its mass, its moments of inertia about
 * its axis and across it, and the chain's mass per metre.
 */
const MASS = 3.5;
const INERTIA: Vec3 = [0.12, 0.07, 0.12];
const DENSITY = 0.3;
/** Links in the chain to the crown and in each leg. */
const LINKS = 16;
const LEG_LINKS = 4;
/** The amplitude the idle swing settles to, in radians. */
const IDLE = 0.5;

// The chain steps thousands of times a second, so its vector maths writes into vectors made once
// rather than returning new ones; `out` may be an input

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
/** Writes `a × b` into `out`. */
const cross = (out: Vec3, a: Readonly<Vec3>, b: Readonly<Vec3>) => {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x;
  out[1] = y;
  out[2] = z;
  return out;
};
/** Writes `v` turned by the unit quaternion (x, y, z, w) into `out`. */
const turnBy = (out: Vec3, x: number, y: number, z: number, w: number, v: Readonly<Vec3>) => {
  const v0 = v[0];
  const v1 = v[1];
  const v2 = v[2];
  const t0 = (y * v2 - z * v1) * 2;
  const t1 = (z * v0 - x * v2) * 2;
  const t2 = (x * v1 - y * v0) * 2;
  out[0] = v0 + w * t0 + (y * t2 - z * t1);
  out[1] = v1 + w * t1 + (z * t0 - x * t2);
  out[2] = v2 + w * t2 + (x * t1 - y * t0);
  return out;
};
/** Writes `v` turned by the unit quaternion `q`, or by its inverse, into `out`. */
const rotate = (out: Vec3, q: Readonly<Quat>, v: Readonly<Vec3>) =>
  turnBy(out, q[0], q[1], q[2], q[3], v);
const unrotate = (out: Vec3, q: Readonly<Quat>, v: Readonly<Vec3>) =>
  turnBy(out, -q[0], -q[1], -q[2], q[3], v);
/** Turns `q` on by the small rotation vector `w`, and keeps it unit. */
const spin = (q: Quat, w: Readonly<Vec3>, scale: number) => {
  const x = w[0] * scale * 0.5;
  const y = w[1] * scale * 0.5;
  const z = w[2] * scale * 0.5;
  const a = q[0];
  const b = q[1];
  const c = q[2];
  const s = q[3];
  q[0] += x * s + y * c - z * b;
  q[1] += y * s + z * a - x * c;
  q[2] += z * s + x * b - y * a;
  q[3] -= x * a + y * b + z * c;
  const length = Math.hypot(q[0], q[1], q[2], q[3]);
  for (let i = 0; i < 4; i++) q[i] /= length;
};
/** Writes the small rotation vector that turns `from` to `to`, the shorter way, into `out`. */
const turnFrom = (out: Vec3, from: Readonly<Quat>, to: Readonly<Quat>) => {
  const a = from[0];
  const b = from[1];
  const c = from[2];
  const s = from[3];
  const x = to[0];
  const y = to[1];
  const z = to[2];
  const w = to[3];
  const sign = w * s + x * a + y * b + z * c < 0 ? -2 : 2;
  out[0] = (w * -a + x * s + y * -c - z * -b) * sign;
  out[1] = (w * -b + y * s + z * -a - x * -c) * sign;
  out[2] = (w * -c + z * s + x * -b - y * -a) * sign;
  return out;
};

/**
 * A heavy body hanging by three legs of chain from a crown, and the crown by a chain from a fixed
 * pivot, stepped by extended position-based dynamics in small steps. Links pull but never push, so
 * a chain lifted toward its pivot falls slack and drapes until it snaps taut again. The body is
 * rigid: it tilts and turns with its own inertia as the legs pull on its feet. A hand pulls it as a
 * damped spring toward a point, so it keeps its weight and momentum and flies on when let go. Left
 * alone for a while, a gentle drive keeps it swinging. It starts swung `start` radians out in the
 * x–y plane. Units are metres, kilograms and seconds.
 */
export function createChain(start = 0.4) {
  const length = LENGTH - CROWN;
  const outline = PROFILE.map(([r, y]) => [r * SIZE, y * SIZE] as const);
  const top = outline[0][1];
  const bottom = outline[outline.length - 1][1];
  /** The body's radius at a height from its middle. */
  const widthAt = (y: number) => {
    for (let i = 1; i < outline.length; i++) {
      const r1 = outline[i][0];
      const y1 = outline[i][1];
      if (y >= y1) {
        const r0 = outline[i - 1][0];
        const y0 = outline[i - 1][1];
        return r1 + ((r0 - r1) * (y - y1)) / Math.max(1e-6, y0 - y1);
      }
    }
    return 0;
  };
  const turned = (v: Vec3): Vec3 => [
    v[0] * Math.cos(start) - v[1] * Math.sin(start),
    v[0] * Math.sin(start) + v[1] * Math.cos(start),
    v[2],
  ];
  const hung = (v: Vec3): Vec3 => {
    const t = turned(v);
    return [PIVOT[0] + t[0], PIVOT[1] + t[1], PIVOT[2] + t[2]];
  };

  // The body: its middle, which is its centre of mass, its turn and their rates
  const below = -length - CROWN;
  const middle = hung([0, below, 0]);
  const halfStart = start / 2;
  const turn: Quat = [0, 0, Math.sin(halfStart), Math.cos(halfStart)];
  const velocity: Vec3 = [0, 0, 0];
  const spinning: Vec3 = [0, 0, 0];
  const lastMiddle: Vec3 = [...middle];
  const lastTurn: Quat = [...turn];

  // Particles: the pivot, the chain down to the crown, then each leg's links between crown and foot
  const positions: Vec3[] = [];
  const inverse: number[] = [];
  const linkMass = (DENSITY * length) / LINKS;
  for (let i = 0; i <= LINKS; i++) {
    positions.push(hung([0, (-length * i) / LINKS, 0]));
    inverse.push(i === 0 ? 0 : 1 / (i === LINKS ? linkMass * 2 : linkMass));
  }
  const crownAt = LINKS;
  type Rod = { a: number; b: number; foot: number; rest: number };
  const rods: Rod[] = [];
  for (let i = 0; i < LINKS; i++) rods.push({ a: i, b: i + 1, foot: -1, rest: length / LINKS });
  const legs: number[][] = [];
  FEET.forEach((foot, f) => {
    const span = Math.hypot(foot[0], foot[1] - CROWN, foot[2]);
    const legMass = (DENSITY * span) / LEG_LINKS;
    const leg = [crownAt];
    for (let i = 1; i < LEG_LINKS; i++) {
      const t = i / LEG_LINKS;
      positions.push(hung([foot[0] * t, below + CROWN + (foot[1] - CROWN) * t, foot[2] * t]));
      inverse.push(1 / legMass);
      leg.push(positions.length - 1);
    }
    for (let i = 0; i + 1 < leg.length; i++) {
      rods.push({ a: leg[i], b: leg[i + 1], foot: -1, rest: span / LEG_LINKS });
    }
    rods.push({ a: leg[leg.length - 1], b: -1, foot: f, rest: span / LEG_LINKS });
    legs.push(leg);
  });
  const count = positions.length;
  const previous = positions.map((p) => [...p] as Vec3);
  const speeds = positions.map(() => [0, 0, 0] as Vec3);

  const hand = {
    held: false,
    grip: [0, 0, 0] as Vec3,
    target: [0, 0, 0] as Vec3,
    turn: [0, 0, 0, 1] as Quat,
    // Where the target was at the last advance, where it is within this one and how fast it moves
    from: [0, 0, 0] as Vec3,
    now: [0, 0, 0] as Vec3,
    moving: [0, 0, 0] as Vec3,
  };
  let free = IDLE_WAIT + IDLE_EASE;
  let left = 0;
  const reach = length + CROWN;

  /** A point of the body given from its middle in its own frame, in the world. */
  const pointOf = (local: Vec3): Vec3 => {
    const r = rotate([0, 0, 0], turn, local);
    return [middle[0] + r[0], middle[1] + r[1], middle[2] + r[2]];
  };
  const given: Vec3 = [0, 0, 0];
  /** How freely the body gives along `n` at offset `r` from its middle. */
  const giveOf = (r: Vec3, n: Vec3) => {
    const k = unrotate(given, turn, cross(given, r, n));
    return 1 / MASS + k[0] ** 2 / INERTIA[0] + k[1] ** 2 / INERTIA[1] + k[2] ** 2 / INERTIA[2];
  };
  const shoved: Vec3 = [0, 0, 0];
  /** Pushes the body by `impulse` at offset `r` from its middle, in position. */
  const shove = (r: Vec3, impulse: Vec3) => {
    for (let i = 0; i < 3; i++) middle[i] += impulse[i] / MASS;
    const local = unrotate(shoved, turn, cross(shoved, r, impulse));
    for (let i = 0; i < 3; i++) local[i] = local[i] / INERTIA[i];
    spin(turn, rotate(local, turn, local), 1);
  };

  /** The angle from straight down the body's energy would carry it to. */
  const amplitudeOf = () => {
    const height = middle[1] - (PIVOT[1] - reach);
    const energy = dot(velocity, velocity) / 2 + GRAVITY * height;
    return Math.acos(Math.max(-1, Math.min(1, 1 - energy / (GRAVITY * reach))));
  };

  const force: Vec3 = [0, 0, 0];
  const torque: Vec3 = [0, 0, 0];
  // Scratch vectors for one substep, each for one use
  const gripped: Vec3 = [0, 0, 0];
  const moving: Vec3 = [0, 0, 0];
  const pull: Vec3 = [0, 0, 0];
  const twist: Vec3 = [0, 0, 0];
  const error: Vec3 = [0, 0, 0];
  const turning: Vec3 = [0, 0, 0];
  const wrist: Vec3 = [0, 0, 0];
  const local: Vec3 = [0, 0, 0];
  const pushed: Vec3 = [0, 0, 0];
  const delta: Vec3 = [0, 0, 0];
  const normal: Vec3 = [0, 0, 0];
  const lever: Vec3 = [0, 0, 0];
  const impulse: Vec3 = [0, 0, 0];
  const inBody: Vec3 = [0, 0, 0];
  const outward: Vec3 = [0, 0, 0];
  const substep = (h: number) => {
    // Forces on the body: its weight, a hand or the idle drive
    force[0] = 0;
    force[1] = -GRAVITY * MASS;
    force[2] = 0;
    torque[0] = torque[1] = torque[2] = 0;
    if (hand.held) {
      const r = rotate(gripped, turn, hand.grip);
      cross(moving, spinning, r);
      for (let i = 0; i < 3; i++) moving[i] = moving[i] + velocity[i] - hand.moving[i];
      const w = 2 * Math.PI * HAND;
      for (let i = 0; i < 3; i++) {
        const v = hand.now[i] - (middle[i] + r[i]);
        pull[i] = MASS * (w * w * v - 2 * HAND_DAMPING * w * moving[i]);
      }
      const strength = Math.hypot(pull[0], pull[1], pull[2]);
      const most = MASS * HAND_MOST;
      if (strength > most) for (let i = 0; i < 3; i++) pull[i] *= most / strength;
      for (let i = 0; i < 3; i++) force[i] += pull[i];
      cross(twist, r, pull);
      // The wrist turns the body back toward how it was taken hold of, about its own axes
      unrotate(error, turn, turnFrom(error, turn, hand.turn));
      unrotate(turning, turn, spinning);
      const v = 2 * Math.PI * WRIST;
      for (let k = 0; k < 3; k++) {
        wrist[k] = INERTIA[k] * (v * v * error[k] - 2 * WRIST_DAMPING * v * turning[k]);
      }
      rotate(wrist, turn, wrist);
      for (let i = 0; i < 3; i++) torque[i] += twist[i] + wrist[i];
    } else {
      const share = Math.min(1, Math.max(0, (free - IDLE_WAIT) / IDLE_EASE));
      if (share > 0) {
        // Pump or brake along the way the body moves, from straight down along x when it is still
        const speed = Math.hypot(velocity[0], velocity[1], velocity[2]);
        const still = speed <= 0.05;
        const push = share * IDLE_RATE * (IDLE - amplitudeOf()) * Math.sqrt(GRAVITY * reach) * MASS;
        force[0] += (still ? (middle[0] >= PIVOT[0] ? -1 : 1) : velocity[0] / speed) * push;
        force[1] += (still ? 0 : velocity[1] / speed) * push;
        force[2] += (still ? 0 : velocity[2] / speed) * push;
      }
    }

    // Move everything freely
    for (let i = 0; i < count; i++) {
      if (inverse[i] === 0) continue;
      const p = positions[i];
      const v = speeds[i];
      v[1] -= GRAVITY * h;
      previous[i][0] = p[0];
      previous[i][1] = p[1];
      previous[i][2] = p[2];
      for (let k = 0; k < 3; k++) p[k] += v[k] * h;
    }
    for (let k = 0; k < 3; k++) {
      velocity[k] += (force[k] / MASS) * h;
      lastMiddle[k] = middle[k];
      middle[k] += velocity[k] * h;
    }
    // Torque turns the body about its own axes, which hold its inertia
    unrotate(local, turn, spinning);
    const h0 = local[0] * INERTIA[0];
    const h1 = local[1] * INERTIA[1];
    const h2 = local[2] * INERTIA[2];
    unrotate(pushed, turn, torque);
    pushed[0] = (pushed[0] - (local[1] * h2 - local[2] * h1)) / INERTIA[0];
    pushed[1] = (pushed[1] - (local[2] * h0 - local[0] * h2)) / INERTIA[1];
    pushed[2] = (pushed[2] - (local[0] * h1 - local[1] * h0)) / INERTIA[2];
    const change = rotate(pushed, turn, pushed);
    for (let k = 0; k < 3; k++) spinning[k] += change[k] * h;
    for (let k = 0; k < 4; k++) lastTurn[k] = turn[k];
    spin(turn, spinning, h);

    // Each link pulls its ends back to its length, and never pushes
    for (let j = 0; j < rods.length; j++) {
      const rod = rods[j];
      const a = positions[rod.a];
      const wa = inverse[rod.a];
      if (rod.foot < 0) {
        const b = positions[rod.b];
        const wb = inverse[rod.b];
        for (let k = 0; k < 3; k++) delta[k] = a[k] - b[k];
        const apart = Math.hypot(delta[0], delta[1], delta[2]);
        const stretch = apart - rod.rest;
        if (stretch <= 0 || wa + wb === 0) continue;
        const move = stretch / (wa + wb) / apart;
        for (let k = 0; k < 3; k++) {
          a[k] -= delta[k] * move * wa;
          b[k] += delta[k] * move * wb;
        }
      } else {
        const r = rotate(lever, turn, FEET[rod.foot]);
        for (let k = 0; k < 3; k++) delta[k] = a[k] - (middle[k] + r[k]);
        const apart = Math.hypot(delta[0], delta[1], delta[2]);
        const stretch = apart - rod.rest;
        if (stretch <= 0) continue;
        for (let k = 0; k < 3; k++) normal[k] = delta[k] / apart;
        const lambda = stretch / (wa + giveOf(r, normal));
        for (let k = 0; k < 3; k++) a[k] -= normal[k] * lambda * wa;
        for (let k = 0; k < 3; k++) impulse[k] = normal[k] * lambda;
        shove(r, impulse);
      }
    }

    // Links rest on the body rather than passing through it
    for (let i = 1; i < count; i++) {
      const p = positions[i];
      for (let k = 0; k < 3; k++) inBody[k] = p[k] - middle[k];
      unrotate(inBody, turn, inBody);
      if (inBody[1] > top || inBody[1] < bottom) continue;
      const apart = Math.hypot(inBody[0], inBody[2]);
      const radius = widthAt(inBody[1]);
      if (apart >= radius) continue;
      // Out the shortest way: through the side, the foot or the finial
      let depth = radius - apart;
      outward[0] = apart > 1e-6 ? inBody[0] / apart : 1;
      outward[1] = 0;
      outward[2] = apart > 1e-6 ? inBody[2] / apart : 0;
      if (inBody[1] - bottom < depth) {
        depth = inBody[1] - bottom;
        outward[0] = 0;
        outward[1] = -1;
        outward[2] = 0;
      }
      if (top - inBody[1] < depth) {
        depth = top - inBody[1];
        outward[0] = 0;
        outward[1] = 1;
        outward[2] = 0;
      }
      const n = rotate(normal, turn, outward);
      for (let k = 0; k < 3; k++) lever[k] = inBody[k] + outward[k] * depth;
      const r = rotate(lever, turn, lever);
      const lambda = depth / (inverse[i] + giveOf(r, n));
      for (let k = 0; k < 3; k++) p[k] += n[k] * lambda * inverse[i];
      for (let k = 0; k < 3; k++) impulse[k] = n[k] * -lambda;
      shove(r, impulse);
    }

    // Velocities from the moves, slowed by the air
    const linkKeep = Math.exp(-LINK_AIR * h);
    for (let i = 0; i < count; i++) {
      if (inverse[i] === 0) continue;
      for (let k = 0; k < 3; k++) {
        speeds[i][k] = ((positions[i][k] - previous[i][k]) / h) * linkKeep;
      }
    }
    const bodyKeep = Math.exp(-BODY_AIR * h);
    for (let k = 0; k < 3; k++) velocity[k] = ((middle[k] - lastMiddle[k]) / h) * bodyKeep;
    // The turn from the last to this one, as a rate
    const turnKeep = Math.exp(-TURN_AIR * h);
    turnFrom(spinning, lastTurn, turn);
    for (let k = 0; k < 3; k++) spinning[k] = (spinning[k] / h) * turnKeep;
  };

  return {
    /** The body's middle, its turn as a quaternion, and their rates. Read only. */
    middle: middle as Readonly<Vec3>,
    turn: turn as Readonly<Quat>,
    velocity: velocity as Readonly<Vec3>,
    spin: spinning as Readonly<Vec3>,
    /** Every particle's position: the pivot, the chain to the crown, then each leg. Read only. */
    positions: positions as readonly Readonly<Vec3>[],
    /** The particles of the chain from the pivot to the crown. */
    chain: Array.from({ length: LINKS + 1 }, (_, i) => i),
    /** The particles of each leg from the crown, ending before its foot. */
    legs: legs as readonly (readonly number[])[],
    /** Writes where a foot is in the world into `out`. */
    footOf(index: number, out: Vec3) {
      rotate(out, turn, FEET[index]);
      for (let k = 0; k < 3; k++) out[k] = middle[k] + out[k];
      return out;
    },
    /** A point in the world in the body's own frame, from its middle. */
    localOf: (point: ArrayLike<number>): Vec3 =>
      unrotate([0, 0, 0], turn, [point[0] - middle[0], point[1] - middle[1], point[2] - middle[2]]),
    /** Whether a hand holds the body. */
    get held() {
      return hand.held;
    },
    /** Moves the censer on by `dt` seconds, in fixed steps. */
    advance(dt: number) {
      left += dt;
      // The hand's target moves evenly from where it was to where it is over the advance
      const steps = Math.floor(left / SUBSTEP);
      for (let k = 0; k < 3; k++)
        hand.moving[k] = dt > 0 ? (hand.target[k] - hand.from[k]) / dt : 0;
      let step = 0;
      while (left >= SUBSTEP) {
        const t = ++step / steps;
        for (let k = 0; k < 3; k++) {
          hand.now[k] = hand.from[k] + (hand.target[k] - hand.from[k]) * t;
        }
        substep(SUBSTEP);
        left -= SUBSTEP;
        if (!hand.held) free += SUBSTEP;
      }
      for (let k = 0; k < 3; k++) hand.from[k] = hand.target[k];
    },
    /** Takes hold of the body at a point of it, given from its middle in its own frame. */
    hold(grip: ArrayLike<number>) {
      hand.held = true;
      for (let k = 0; k < 3; k++) hand.grip[k] = grip[k];
      const at = pointOf(hand.grip);
      for (let k = 0; k < 3; k++) hand.target[k] = hand.from[k] = hand.now[k] = at[k];
      for (let k = 0; k < 4; k++) hand.turn[k] = turn[k];
    },
    /** Pulls the held point toward a point in the world. */
    pull(target: ArrayLike<number>) {
      for (let k = 0; k < 3; k++) hand.target[k] = target[k];
    },
    /** Lets go of the body, which keeps its velocity. */
    release() {
      hand.held = false;
      free = 0;
    },
    /** The angle from straight down the body would swing out to, in radians. */
    amplitude: amplitudeOf,
  };
}

export type Chain = ReturnType<typeof createChain>;
