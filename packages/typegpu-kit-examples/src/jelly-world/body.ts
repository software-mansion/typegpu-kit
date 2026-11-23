import { clock, type VectorSpring } from '@typegpu/kit';
import { vec3n } from 'wgpu-matrix';
import { GROUND, SLOT, type V3 } from './tray.ts';
import type { World } from './world.ts';

/** How many critters there are, at most 32 for one bit each in the march's masks. */
export const CRITTERS = 18;
/** What grows on a critter's head. */
export const ANTENNA = 0;
export const SPROUT = 1;
export const EARS = 2;

/** Critter colours in display values. */
const PALETTE: readonly (readonly [number, number, number])[] = [
  [1, 0.55, 0.7],
  [0.55, 0.85, 1],
  [1, 0.85, 0.4],
  [0.7, 0.6, 1],
  [0.6, 0.95, 0.6],
  [1, 0.65, 0.45],
];

/** Where critters climb onto the jelly, beside the slot's left end. */
export const LADDER = -1.06;
/** The lawn critters walk on, in world x and z. Wherever they head lies within it. */
export const LAND = { x: [-2.1, 2.1], z: [-1.2, 0.75] };

/**
 * The springs a critter grows in and shrinks away on, its body bends and squashes on, and what
 * grows on its head sways on.
 */
const SPRINGS = {
  grow: { stiffness: 120, damping: 9 },
  bend: { stiffness: 55, damping: 4.5 },
  squash: { stiffness: 160, damping: 10 },
  top: { stiffness: 170, damping: 6 },
};

/** Adds `by` times `k` to a vector spring's velocity, as a knock does. */
export const kick = (spring: VectorSpring<V3>, by: V3, k: number) => {
  for (let i = 0; i < 3; i++) spring.velocity[i] += by[i] * k;
};

/** Hip, knee and foot of each leg, then shoulder and hand of each arm. */
export const JOINTS = 10;

/** Body radius, and the hips' height over the feet, in world units. */
export const R = 0.04;
export const HIP = 0.05;
/** The body's centre over the ground it stands on. */
export const STANDING = HIP + R * 0.5;
/** Walking speed, step length and time, how high a foot lifts, and turns in radians a second. */
export const WALK = { speed: 0.17, stride: 0.034, time: 0.14, lift: 0.022, turn: 4 };
/** Gravity, in world units a second squared. */
export const FALL = 3.2;

/** Climbing the jelly and sliding down it. */
export const CLIMB = {
  speed: 0.15,
  /** Slopes over which walking up turns into heaving. */
  steep: [0.3, 0.5],
  /** The steepest slope critters grip. */
  steepest: 0.97,
  /** The downhill slope past a top where they sit to slide. */
  over: 0.3,
  /** Friction while sliding. */
  friction: 0.25,
  /** Step length and how high a foot lifts walking up, and the share of a step a foot is planted. */
  stride: 0.03,
  lift: 0.012,
  planted: 0.65,
  /** How much harder a grip is to shake. */
  grip: 2,
};
/** Up steep jelly: seconds per heave, how far ahead hands grab, and how far each pull lifts. */
export const HEAVE = { time: 0.75, reach: R * 1.7, pull: R * 1.1 };

/** The landing speed that makes a critter dizzy, and for how long. */
const DIZZY = { impact: 1.5, time: 2.5 };
/**
 * Bumps: the closing speed that knocks critters over, how much of it bounces back, how much
 * becomes lift, and how fast a knocked critter tumbles.
 */
export const KNOCK = { speed: 0.45, bounce: 0.4, lift: 0.5, tumble: 11 };
/** Petting: how near the pointer rests, how slowly it moves, and how long before it counts. */
export const PET = { reach: R * 1.8, still: 0.3, after: 0.35 };
/** A pointer swept fast past critters startles them away for `flee` seconds. */
export const SHOO = { speed: 1.5, reach: R * 3.5, flee: 1.2 };

/** The herald, the first critter, which pulls the lever that reveals the jelly world. */
export const HERALD = {
  /** Seconds before it pops in, and where in x and z. */
  after: 0.8,
  at: [0.72, 0.72],
  /** How much faster than a walk it trots to the lever. */
  pace: 1.6,
  /** Where it stands from the knob to leap. */
  stand: 0.17,
  /** Seconds its leap to the knob takes, and how high it arcs. */
  leap: 0.45,
  arc: 0.05,
  /** Where it hangs below the knob. */
  hang: [0, -R * 1.45, R * 0.4] as V3,
  /** How far the lever goes before it lets go. */
  done: 0.92,
};

export const UP: V3 = [0, 1, 0];

export type Mode = 'walk' | 'ride' | 'fly' | 'held' | 'perch' | 'pull';

export interface Foot {
  at: V3;
  from: V3;
  to: V3;
  /** Swing progress, 1 when planted. */
  t: number;
}

export interface Obstacle {
  readonly x: number;
  readonly z: number;
  readonly radius: number;
}

/** What critters need of the jelly world: where not to walk, and the lever one of them pulls. */
export type Ground = Pick<World, 'obstacles' | 'knob' | 'pulled' | 'holding' | 'hang'>;

/** What a critter's step reads besides itself, which the critters update each step. */
export interface Crowd {
  readonly bodies: readonly Body[];
  readonly world: Ground;
  /** The herald, and the same while it still has the lever to pull. */
  readonly herald: Body;
  errand: Body | undefined;
  /** The point the pointer hovers while it is fresh, and how fast it moves. */
  pointer: V3 | undefined;
  pointerSpeed: number;
  /** World x of the jelly world's front, which critters keep behind. */
  front: number;
  /** The jelly's spine, which critters climb onto. */
  spine: Float32Array;
}

/**
 * Critter `i` standing on the tray: the herald at its spot, the others anywhere off the slot. Its
 * times are seconds left unless they say otherwise.
 */
export function createBody(i: number) {
  const at: V3 = [random(-2, 1.6), GROUND, random(-1, 0.7)];
  if (Math.abs(at[2]) < SLOT.z + 0.1) at[2] = SLOT.z + 0.15;
  if (i === 0) [at[0], at[2]] = HERALD.at;
  const foot = (side: number): Foot => {
    const p: V3 = [at[0] + side * 0.02, GROUND, at[2]];
    return { at: p, from: [...p], to: [...p], t: 1 };
  };
  const heading = random(0, Math.PI * 2);
  const blinkIn = random(1, 4);
  const phase = random(0, 10);
  const nerve = random(0, 1);

  return {
    mode: 'walk' as Mode,
    at,
    velocity: [0, 0, 0] as V3,
    heading,
    /** How far it has grown in, from 0 to 1, and the spring it grows on. */
    grown: 0,
    growth: clock.spring(SPRINGS.grow),

    /** Arc length and speed along the spine while riding, and walking strides taken up it. */
    s: 0,
    sliding: 0,
    cycle: 0,
    /** How steep the climb is from walking to heaving, the heave's progress, and where it started. */
    steep: 0,
    heave: 0,
    from: 0,
    /** Where the hands hold the jelly, and held before the last reach. */
    grab: 0,
    grabFrom: 0,
    /** Climbing rather than sliding, how far into the sitting pose, and turned from +s towards -s. */
    gripping: true,
    seated: 0,
    turn: 0,

    feet: [foot(1), foot(-1)] as [Foot, Foot],
    /** The lean of the body's top, and the squash along its up axis. */
    bend: clock.spring({ ...SPRINGS.bend, value: [0, 0, 0] as V3 }),
    squash: clock.spring(SPRINGS.squash),
    /** A flip's or tumble's turn so far, how fast it turns, and the tumble's axis. */
    spin: 0,
    spinning: 0,
    spinAxis: undefined as V3 | undefined,
    /** Where a tumble keeps its axis. */
    axis: [0, 0, 0] as V3,
    /** What grows on its head: its tip, placed on the first pose, and which kind. */
    tip: clock.spring({ ...SPRINGS.top, value: [0, 0, 0] as V3 }),
    tipped: false,
    kind: Math.floor(i / PALETTE.length) % 3,

    /** Where it is going in x and z, until when, and whether there to climb onto the jelly. */
    goal: [at[0], at[2]] as [number, number],
    plan: 0,
    wantsRide: false,
    /**
     * How bold it is, from 0 to 1: bolder critters want to ride the jelly more often, run up to it
     * and hesitate less before climbing on. Seconds it has stood at the jelly's edge, and walking
     * before it wants to ride again.
     */
    nerve,
    brink: 0,
    rideIn: 0,
    /** Whether it stopped where it was going, until it is well away from there again. */
    arrived: false,
    /** The side it passes what blocks its way, -1 or 1, and for how long it keeps that side. */
    dodge: 0,
    dodgeFor: 0,
    /** What it last went round, which it keeps going round while another is about as near. */
    avoiding: undefined as Body | Obstacle | undefined,
    /** Seconds its way has been blocked. */
    blocked: 0,
    /** Seconds into a shove at `victim`, 0 when not shoving, and before it may shove again. */
    shove: 0,
    victim: undefined as Body | undefined,
    shoveIn: 2 + (i % 4),
    /** How fast a shove sends it stumbling, and who shoved it, to shove back. */
    stumble: [0, 0, 0] as V3,
    grudge: undefined as Body | undefined,
    /** Waving at `friend`, and before it waves again. */
    wave: 0,
    friend: undefined as Body | undefined,
    greeted: random(2, 6),
    /** Cheering for `fan`, a critter flying off the jelly's end. */
    cheer: 0,
    fan: undefined as Body | undefined,
    /** The critter it follows in a line. */
    leader: undefined as Body | undefined,
    train: 0,
    /** The critter whose head it sits on, and how many sit on its own. */
    base: undefined as Body | undefined,
    perchFor: 0,
    load: 0,
    asleep: 0,
    flee: 0,
    fleeFrom: [0, 0, 0] as V3,
    /** Seconds under a resting pointer. */
    petted: 0,
    dizzy: 0,
    /** Whether it flies knocked over, which leaves it dizzy. */
    knocked: false,
    /** Seconds held by the pointer. */
    heldFor: 0,
    /** Seconds in the leap to the lever's knob and hanging there, and where the leap started. */
    pulling: 0,
    leapFrom: [0, 0, 0] as V3,

    /** Looking surprised and looking happy. */
    shock: 0,
    joy: 0,
    /** How happy, surprised, sleepy and dizzy it looks, eased. */
    mood: [0, 0, 0, 0] as [number, number, number, number],
    blink: 0,
    blinkIn,
    mouth: 0,
    /** Seconds since it was made, which phases its sways. */
    phase,
    color: PALETTE[i % PALETTE.length],
    /**
     * The body's centre then its joints from it as last computed, once posed, offsets easing each
     * from where it was drawn after a jump, and the centre and up axis as drawn.
     */
    raw: Array.from({ length: JOINTS + 1 }, (): V3 => [0, 0, 0]),
    posed: false,
    settle: Array.from({ length: JOINTS + 1 }, (): V3 => [0, 0, 0]),
    drawn: [at[0], GROUND + STANDING, at[2]] as V3,
    upright: [0, 1, 0] as V3,
  };
}

/** One critter's state on the CPU, as `createBody` sets it up. */
export interface Body extends ReturnType<typeof createBody> {}

/** A random number from `low` to `high`. */
export const random = (low: number, high: number) => low + Math.random() * (high - low);

export const ease = (from: number, to: number, rate: number, dt: number) =>
  from + (to - from) * Math.min(1, dt * rate);

/** The turn from heading `from` to `to`, the short way round. */
export const turnBetween = (from: number, to: number) => {
  const turn = to - from;
  return turn - Math.round(turn / (Math.PI * 2)) * Math.PI * 2;
};

/** Whether a count that grew by `dt` this step reached `mark`. */
export const crossed = (count: number, dt: number, mark: number) =>
  count >= mark && count - dt < mark;

/** Whether a resting pointer has petted it long enough to show. */
export const petted = (body: Body) => body.petted > PET.after;

/** Lands on the ground with `impact`, dizzy after a hard landing or a knock. */
export function land(body: Body, impact: number) {
  body.squash.velocity -= impact * 2.5;
  kick(body.bend, body.velocity, 0.3);
  vec3n.zero(body.velocity);
  body.mode = 'walk';
  body.spinning = 0;
  body.blocked = 0;
  if (body.knocked || impact > DIZZY.impact) body.dizzy = DIZZY.time;
  body.knocked = false;
  for (const foot of body.feet) {
    foot.t = 1;
    vec3n.set(body.at[0], GROUND, body.at[2], foot.at);
  }
}

/** Jumps up at `speed`, turning a front flip on the way when `flip` is set. */
export function jump(body: Body, speed: number, flip: boolean) {
  body.mode = 'fly';
  body.base = undefined;
  body.asleep = 0;
  vec3n.set(Math.sin(body.heading) * 0.05, speed, Math.cos(body.heading) * 0.05, body.velocity);
  body.squash.velocity -= 2;
  body.spin = 0;
  body.spinAxis = undefined;
  body.spinning = flip ? (Math.PI * FALL) / speed : 0;
  body.shove = 0;
}

/** Knocks a critter over by `push` and up by `lift`, so it tumbles the way it was pushed. */
export function topple(body: Body, push: V3, lift = KNOCK.lift * vec3n.length(push)) {
  if (body.mode === 'held') return;
  body.mode = 'fly';
  body.base = undefined;
  vec3n.add(body.velocity, push, body.velocity);
  body.velocity[1] += lift;
  // Tumbles about the horizontal axis across the push
  const axis = vec3n.set(push[2], 0, -push[0], body.axis);
  const flat = vec3n.length(axis);
  body.spinAxis = flat > 1e-4 ? vec3n.normalize(axis, axis) : undefined;
  body.spin = 0;
  body.spinning = KNOCK.tumble * Math.min(1.5, vec3n.length(push) / 0.6 + 0.3);
  body.knocked = true;
  body.shock = 1.2;
  body.asleep = 0;
  body.wave = 0;
  body.leader = undefined;
  body.train = 0;
  body.wantsRide = false;
  body.shove = 0;
  body.squash.velocity -= 2;
}

/** Jumps away from `from` and flees it for a while. */
export function startle(body: Body, from: V3) {
  jump(body, 0.7, false);
  body.shock = 1;
  body.flee = SHOO.flee;
  vec3n.copy(from, body.fleeFrom);
}
