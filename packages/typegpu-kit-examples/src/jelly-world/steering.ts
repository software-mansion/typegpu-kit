import { std } from 'typegpu';
import { vec3n } from 'wgpu-matrix';
import {
  crossed,
  LADDER,
  R,
  turnBetween,
  WALK,
  type Body,
  type Crowd,
  type Obstacle,
} from './body.ts';
import { GROUND, SLOT, type V3 } from './tray.ts';

/** How critters steer round props and each other. */
export const STEER = {
  /** Metres a second at most that props, the slot and each other move them apart. */
  apart: 0.5,
  /** How far ahead they look for what blocks their way. */
  ahead: 0.35,
  /** The room they leave around props, and around each other. */
  room: R * 1.3,
  spacing: R * 2.4,
  /** How fast they step aside from a critter too near, and how fast they reach a new speed. */
  aside: 0.12,
  speedUp: 10,
  /** Seconds they keep the side they pass on. */
  keep: 1,
  /** Going round something, they keep at it while another is under `stick` nearer, `wide` wider. */
  stick: 0.03,
  wide: 1.2,
  /** Seconds blocked before they try the other side, and before they go somewhere else. */
  flip: 2.5,
  give: 4,
  /** Radians at most they turn from their way passing a critter, so two never walk off together. */
  swerve: 1.25,
};

/** What blocks a walker's way: a prop or a critter, and the room it gives it. */
export interface Block {
  x: number;
  z: number;
  room: number;
  other: Body | undefined;
}

/** The way towards a goal in x and z, how far the walker heads along it, and how far the goal is. */
export interface Route {
  readonly way: [number, number];
  distance: number;
  toGoal: number;
}

// What the steps below write, one walker at a time
const route: Route = { way: [0, 0], distance: 0, toGoal: 0 };
const blocking: Block = { x: 0, z: 0, room: 0, other: undefined };
const sidestep: V3 = [0, 0, 0];
const going: V3 = [0, 0, 0];
const moved: V3 = [0, 0, 0];

/**
 * The way towards the goal: past the slot's nearer end, halfway in towards it, when the goal lies
 * across it; out to the goal's side first when in line with the slot past an end and the goal lies
 * along it; and back from the front.
 */
export function routeOf(body: Body, goal: readonly number[], front: number, due: boolean) {
  const x = body.at[0];
  const z = body.at[2];
  const gx = goal[0];
  const gz = goal[1];
  let towardX = gx - x;
  let towardZ = gz - z;
  const steppingIn = byLadder(body);
  const across = Math.sign(z) !== Math.sign(gz) || Math.abs(gz) < SLOT.z;
  const pastEnd = Math.abs(x) > SLOT.x && Math.abs(z) < SLOT.z + 0.1;
  if (!steppingIn && across && Math.abs(x) < SLOT.x + 0.1) {
    const end = x + gx < 0 ? -1 : 1;
    towardX = end * (SLOT.x + 0.2) - x;
    towardZ = z * 0.5 - z;
  } else if (!steppingIn && pastEnd && Math.sign(gx - x) !== Math.sign(x)) {
    towardX = Math.sign(gx - x) * 0.1;
    towardZ = Math.sign(gz || 1) * (SLOT.z + 0.2) - z;
  }
  if (x > front - 0.15 && !due) {
    towardX = -1;
    towardZ = 0;
  }

  const distance = Math.hypot(towardX, towardZ);
  const { way } = route;
  if (distance > 1e-6) {
    way[0] = towardX / distance;
    way[1] = towardZ / distance;
  } else {
    way[0] = Math.sin(body.heading);
    way[1] = Math.cos(body.heading);
  }
  route.distance = distance;
  route.toGoal = Math.hypot(gx - x, gz - z);
  return route;
}

/** Whether it is near enough the jelly's fixed end, on its way onto it, to step into the slot. */
const byLadder = (body: Body) => body.wantsRide && body.at[0] < LADDER + 0.08;

/** The walker `blockOf` looks ahead for, its way, the nearest entry so far and what it avoided. */
const search = {
  body: undefined as Body | undefined,
  way: [0, 0] as readonly [number, number],
  entry: 0,
  avoided: undefined as Body | Obstacle | undefined,
  found: false,
};

/** Takes what lies at `bx, bz` with `room` round it as the block, if the walker enters it first. */
function inWay(bx: number, bz: number, room: number, what: Body | Obstacle) {
  const { body, way } = search;
  const x = body!.at[0];
  const z = body!.at[2];
  const along = (bx - x) * way[0] + (bz - z) * way[1];
  const off = (bx - x) * way[1] - (bz - z) * way[0];
  const kept = what === search.avoided;
  const wide = kept ? room * STEER.wide : room;
  if (along <= 0 || Math.abs(off) >= wide) return;
  const enters = along - Math.sqrt(wide * wide - off * off);
  if (enters - (kept ? STEER.stick : 0) > search.entry) return;
  search.entry = Math.max(enters, 0);
  search.found = true;
  blocking.x = bx;
  blocking.z = bz;
  blocking.room = room;
  blocking.other = 'at' in what ? what : undefined;
  body!.avoiding = what;
}

/**
 * The nearest prop or walker in the way, entered within `STEER.ahead`. What it went round last
 * step keeps a wider room and wins ties by `STEER.stick`, so it doesn't swap between two.
 */
export function blockOf(
  body: Body,
  crowd: Crowd,
  way: readonly [number, number],
  distance: number,
  leader: Body | undefined,
) {
  search.body = body;
  search.way = way;
  search.entry = Math.min(distance, STEER.ahead);
  search.avoided = body.avoiding;
  search.found = false;
  body.avoiding = undefined;
  for (const o of crowd.world.obstacles) inWay(o.x, o.z, o.radius + STEER.room, o);
  for (const other of crowd.bodies) {
    if (other === body || other.mode !== 'walk' || other === leader || other.grown < 0.5) continue;
    inWay(other.at[0], other.at[2], STEER.spacing, other);
  }
  return search.found ? blocking : undefined;
}

/**
 * Turns `want` to pass `block`, `gap` away, on the side it keeps, which it flips after
 * `STEER.flip` seconds blocked.
 */
export function passing(body: Body, block: Block, want: number, gap: number, dt: number) {
  const bearing = Math.atan2(block.x - body.at[0], block.z - body.at[2]);
  const off = turnBetween(want, bearing);
  if (body.dodgeFor <= 0) {
    const lean = turnBetween(want, body.heading);
    body.dodge = Math.abs(off) > 0.05 ? -Math.sign(off) : lean >= 0 ? 1 : -1;
  }
  if (crossed(body.blocked, dt, STEER.flip)) body.dodge *= -1;
  body.dodgeFor = STEER.keep;
  const pass =
    bearing + body.dodge * (Math.asin(Math.min(1, block.room / Math.max(gap, 1e-6))) + 0.15);
  const most = block.other ? STEER.swerve : Math.PI;
  return want + std.clamp(turnBetween(want, pass), -most, most);
}

/** A step aside from walkers too near, without turning to them. */
export function asideOf(body: Body, bodies: readonly Body[], leader: Body | undefined) {
  const x = body.at[0];
  const z = body.at[2];
  vec3n.zero(sidestep);
  for (const other of bodies) {
    if (other === body || other.mode !== 'walk' || other === leader || other.grown < 0.5) continue;
    const gap = Math.hypot(x - other.at[0], z - other.at[2]);
    if (gap < STEER.spacing && gap > 1e-4) {
      const push = 1 - gap / STEER.spacing;
      sidestep[0] += ((x - other.at[0]) / gap) * push;
      sidestep[2] += ((z - other.at[2]) / gap) * push;
    }
  }
  if (vec3n.length(sidestep) > 1) vec3n.normalize(sidestep, sidestep);
  return vec3n.scale(sidestep, STEER.aside, sidestep);
}

/** Turns towards heading `want` at most `WALK.turn` radians a second. Returns the turn left. */
export function turnTo(body: Body, want: number, dt: number, rate: number = WALK.turn) {
  const turn = turnBetween(body.heading, want);
  body.heading += std.clamp(turn, -1, 1) * rate * dt;
  return turn;
}

/**
 * Eases its velocity towards walking `speed` along its heading plus `aside` and any stumble, moves,
 * and slides along what it walks into, never faster than it walks.
 */
export function stride(body: Body, crowd: Crowd, aside: V3, speed: number, dt: number) {
  const x = body.at[0];
  const z = body.at[2];
  vec3n.add(aside, body.stumble, going);
  going[0] += Math.sin(body.heading) * speed;
  going[2] += Math.cos(body.heading) * speed;
  vec3n.lerp(body.velocity, going, Math.min(1, dt * STEER.speedUp), body.velocity);
  vec3n.scale(body.stumble, Math.exp(-dt * 5), body.stumble);
  vec3n.addScaled(body.at, body.velocity, dt, body.at);
  body.at[1] = GROUND;
  keepOut(body, crowd.world.obstacles, dt);
  vec3n.set((body.at[0] - x) / dt, 0, (body.at[2] - z) / dt, moved);
  const most = vec3n.length(body.velocity);
  if (vec3n.length(moved) > most) vec3n.scale(vec3n.normalize(moved, moved), most, moved);
  vec3n.copy(moved, body.velocity);
}

/**
 * Moves a walking critter out of props and the slot, unless it is climbing onto the jelly, at
 * most `STEER.apart` a second, so one landing deep inside walks out rather than jumps.
 */
export function keepOut(body: Body, obstacles: readonly Obstacle[], dt: number) {
  const at = body.at;
  const most = STEER.apart * dt;
  for (const o of obstacles) {
    const gap = Math.hypot(at[0] - o.x, at[2] - o.z);
    const deep = Math.min(o.radius + R - gap, most);
    if (deep > 0 && gap > 1e-4) {
      at[0] += ((at[0] - o.x) / gap) * deep;
      at[2] += ((at[2] - o.z) / gap) * deep;
    }
  }
  if (byLadder(body)) return;
  const inX = SLOT.x + 0.06 - Math.abs(at[0]);
  const inZ = SLOT.z + 0.06 - Math.abs(at[2]);
  if (inX <= 0 || inZ <= 0) return;
  if (inZ < inX) at[2] += Math.sign(at[2] || 1) * Math.min(inZ, most);
  else at[0] += Math.sign(at[0] || 1) * Math.min(inX, most);
}
