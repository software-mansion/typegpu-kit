import { solidAt, type Level } from './level.ts';

/** Half the hero's size standing, in tiles, as tall as its sprite. Ducked, it is half as tall. */
export const HALF = [0.38, 0.65] as const;
/** Speed in tiles a second the hero still slides against the way it faces while it skids. */
export const SKID = 0.1;
/**
 * Seconds a turn on a heel lasts at least, from when the skid starts, or longer while it skids.
 * A jump in the turn with no direction held backflips.
 */
export const HEEL = 0.18;
/**
 * Seconds the hero stays crouched for a backflip at least, from when it lets go of the stick in a
 * turn, unless it takes a direction again or leaves the ground.
 */
export const POISE = 0.2;

/** Top running speed, jump, stomp bounce and top falling speed, in tiles per second. */
const SPEED = { run: 7.5, jump: 22, bounce: 15, fall: 22 };
/** Top speed in tiles a second while the hero sneaks, ducked. A slide slows down to it. */
export const SNEAK = 2.5;
/** Tiles a second squared a slide loses. */
const SLIDE = 8;
/**
 * A backflip, jumped from a skid: its upward and backward speed in tiles a second, and the
 * seconds it keeps them, until it lands, before the stick steers again.
 */
const FLIP = { jump: 25, back: 5, hold: 0.5 };
/**
 * A hit's knockback: tiles a second back from what hit the hero and up. The player can't steer,
 * jump or duck until the hero lands.
 */
const KNOCK = { back: 6, up: 12 };
/** Top upward speed once jump is released, in tiles per second. */
const CUT = SPEED.jump / 2;
/** Ground and air acceleration, and gravity, in tiles per second squared. */
const ACCELERATION = { ground: 70, air: 40, gravity: 70 };
/** Seconds a jump still works after leaving a ledge, and a press waits for the ground. */
const GRACE = { coyote: 0.08, buffer: 0.12 };

/** Collision gap in tiles. Prevents contact counting as overlap. */
const SKIN = 1e-4;

/** Held inputs. A jump press also calls the hero's `jump`. */
export interface Input {
  left: boolean;
  right: boolean;
  jump: boolean;
  /** Ducks on the ground: slides when moving faster than sneaking, and sneaks otherwise. */
  down: boolean;
}

const IDLE: Input = { left: false, right: false, jump: false, down: false };

/** Moves `value` toward `target` by at most `amount`. */
const approach = (value: number, target: number, amount: number) =>
  value < target ? Math.min(value + amount, target) : Math.max(value - amount, target);

/** Runs CPU physics and tile collisions at a fixed step. */
export function createHero(level: Level) {
  const position: [number, number] = [0, 0];
  const previous: [number, number] = [0, 0];
  const start: [number, number] = [0, 0];
  const velocity: [number, number] = [0, 0];
  let sinceGround = Infinity;
  let sincePress = Infinity;
  let sinceFlip = Infinity;
  let sinceHeel = Infinity;
  let sincePoise = Infinity;
  let skidded = false;

  /**
   * The hero's box, in tiles: its sides, and its feet and head for a body `tall` tiles tall. The
   * position stays the centre of the standing body, so a duck lowers only the head.
   */
  const boxOf = (tall: number) => ({
    left: position[0] - HALF[0],
    right: position[0] + HALF[0],
    feet: position[1] - HALF[1],
    head: position[1] - HALF[1] + tall,
  });
  const tallness = () => (hero.ducked ? HALF[1] : 2 * HALF[1]);

  /** Whether any tile the hero's box `tall` tiles tall would cover is solid. */
  const blocked = (tall: number) => {
    const box = boxOf(tall);
    for (let y = Math.floor(box.feet + SKIN); y <= Math.floor(box.head - SKIN); y++) {
      for (let x = Math.floor(box.left + SKIN); x <= Math.floor(box.right - SKIN); x++) {
        if (solidAt(level, x, y)) return true;
      }
    }
    return false;
  };

  /** Moves along one axis until the first solid tile. Returns whether a collision occurred. */
  const move = (axis: 0 | 1, delta: number) => {
    if (delta === 0) return false;
    position[axis] += delta;

    const tall = tallness();
    const box = boxOf(tall);
    if (axis === 0) {
      const edge = Math.floor(delta > 0 ? box.right : box.left);
      for (let y = Math.floor(box.feet + SKIN); y <= Math.floor(box.head - SKIN); y++) {
        if (!solidAt(level, edge, y)) continue;
        position[0] = delta > 0 ? edge - HALF[0] - SKIN : edge + 1 + HALF[0] + SKIN;
        velocity[0] = 0;
        return true;
      }
      return false;
    }

    const edge = Math.floor(delta > 0 ? box.head : box.feet);
    for (let x = Math.floor(box.left + SKIN); x <= Math.floor(box.right - SKIN); x++) {
      if (!solidAt(level, x, edge)) continue;
      position[1] = delta > 0 ? edge - tall + HALF[1] - SKIN : edge + 1 + HALF[1] + SKIN;
      velocity[1] = 0;
      return true;
    }
    return false;
  };

  const hero = {
    /** Centre of the body, in tiles. */
    position,
    /** Previous fixed-step position for interpolation. */
    previous,
    /** Position before this frame's steps. The game sets it for swept critter contacts. */
    start,
    velocity,
    facing: 1,
    grounded: false,
    /**
     * Whether the hero ducks, standing, sneaking or sliding: on the ground while down is held, and
     * anywhere without room to stand.
     */
    ducked: false,
    /** Whether the hero slides, ducked, slowing down to sneaking speed. */
    sliding: false,
    /** The jump the last step took off in, if any. */
    leapt: undefined as 'jump' | 'backflip' | undefined,
    /**
     * Whether the hero backflips, away from the way it faces, from a jump in a skid with no
     * direction held. The stick doesn't steer it until it lands or `FLIP.hold` seconds pass.
     */
    flipping: false,
    /** Whether the hero flies back from a hit, out of the player's control until it lands. */
    knocked: false,
    /**
     * Whether a jump now would backflip: on the ground with no direction held, from a turn and for
     * at least `POISE` seconds.
     */
    poised: false,

    /** Whether the hero stands and slides against the way it faces, turning on a heel. */
    get skidding() {
      return hero.grounded && velocity[0] * hero.facing < -SKID;
    },

    place(x: number, y: number) {
      position[0] = previous[0] = start[0] = x;
      position[1] = previous[1] = start[1] = y - 0.5 + HALF[1] + SKIN;
      velocity[0] = velocity[1] = 0;
      hero.facing = 1;
      hero.ducked = hero.sliding = hero.flipping = hero.poised = hero.knocked = false;
      sinceHeel = sincePoise = Infinity;
      skidded = false;
    },

    jump() {
      if (!hero.knocked) sincePress = 0;
    },

    /** Knocks the hero back the way `way` points, ±1, and up, out of the player's control. */
    knock(way: number) {
      velocity[0] = way * KNOCK.back;
      velocity[1] = KNOCK.up;
      hero.knocked = true;
      hero.grounded = hero.sliding = hero.flipping = hero.poised = false;
      sinceGround = sincePress = Infinity;
    },

    bounce() {
      velocity[1] = SPEED.bounce;
      sinceGround = Infinity;
    },

    step(held: Input, dt: number) {
      previous[0] = position[0];
      previous[1] = position[1];
      hero.leapt = undefined;
      // A knocked-back hero ignores the player
      const input = hero.knocked ? IDLE : held;

      const want = Number(input.right) - Number(input.left);
      const cramped = hero.ducked && blocked(2 * HALF[1]);
      const ducking = (input.down && hero.grounded) || cramped;
      // Ducking while faster than a sneak starts a slide, which keeps its way until it slows down
      if (ducking && !hero.ducked) hero.sliding = Math.abs(velocity[0]) > SNEAK;
      hero.ducked = ducking;
      if (!ducking) hero.sliding = false;
      if (hero.sliding) {
        const way = Math.sign(velocity[0]);
        velocity[0] = approach(velocity[0], way * SNEAK, SLIDE * dt);
        if (Math.abs(velocity[0]) <= SNEAK) hero.sliding = false;
      } else if (!hero.flipping && !hero.knocked) {
        if (want !== 0) hero.facing = want;
        const acceleration = hero.grounded ? ACCELERATION.ground : ACCELERATION.air;
        const top = ducking ? SNEAK : SPEED.run;
        velocity[0] = approach(velocity[0], want * top, acceleration * dt);
      }

      sinceGround += dt;
      sincePress += dt;
      sinceFlip += dt;
      sinceHeel += dt;
      sincePoise += dt;
      const skidding = hero.skidding;
      if (skidding && !skidded) sinceHeel = 0;
      skidded = skidding;
      const turning = skidding || sinceHeel < HEEL;
      const still = hero.grounded && !ducking && want === 0;
      if (still && turning && !hero.poised) sincePoise = 0;
      hero.poised = still && (turning || sincePoise < POISE);
      if (sincePress < GRACE.buffer && sinceGround < GRACE.coyote) {
        if (hero.poised) {
          velocity[0] = -hero.facing * FLIP.back;
          velocity[1] = FLIP.jump;
          hero.flipping = true;
          hero.leapt = 'backflip';
          sinceFlip = 0;
        } else {
          velocity[1] = SPEED.jump;
          hero.leapt = 'jump';
        }
        sincePress = sinceGround = Infinity;
      }

      // A backflip and a knockback always reach their full height
      if (!input.jump && !hero.flipping && !hero.knocked && velocity[1] > CUT) velocity[1] = CUT;
      velocity[1] = Math.max(velocity[1] - ACCELERATION.gravity * dt, -SPEED.fall);

      move(0, velocity[0] * dt);
      const falling = velocity[1] < 0;
      hero.grounded = move(1, velocity[1] * dt) && falling;
      if (hero.grounded) sinceGround = 0;
      if (hero.grounded || sinceFlip >= FLIP.hold) hero.flipping = false;
      if (hero.grounded) hero.knocked = false;
      if (!hero.grounded || hero.flipping) hero.poised = false;
    },
  };

  hero.place(...level.start);
  return hero;
}

export type Hero = ReturnType<typeof createHero>;
