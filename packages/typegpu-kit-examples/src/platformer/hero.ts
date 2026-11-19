import { solidAt, type Level } from './level.ts';

/** Half the hero's size, in tiles. */
export const HALF = [0.32, 0.42] as const;

/** Top running speed, jump, stomp bounce and top falling speed, in tiles per second. */
const SPEED = { run: 7.5, jump: 22, bounce: 15, fall: 22 };
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
}

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

  /** Moves along one axis until the first solid tile. Returns whether a collision occurred. */
  const move = (axis: 0 | 1, delta: number) => {
    if (delta === 0) return false;
    position[axis] += delta;

    const across = 1 - axis;
    const low = Math.floor(position[across] - HALF[across] + SKIN);
    const high = Math.floor(position[across] + HALF[across] - SKIN);
    const edge = Math.floor(position[axis] + Math.sign(delta) * HALF[axis]);

    for (let t = low; t <= high; t++) {
      if (!(axis === 0 ? solidAt(level, edge, t) : solidAt(level, t, edge))) continue;
      position[axis] = delta > 0 ? edge - HALF[axis] - SKIN : edge + 1 + HALF[axis] + SKIN;
      velocity[axis] = 0;
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

    place(x: number, y: number) {
      position[0] = previous[0] = start[0] = x;
      position[1] = previous[1] = start[1] = y - 0.5 + HALF[1] + SKIN;
      velocity[0] = velocity[1] = 0;
      hero.facing = 1;
    },

    jump() {
      sincePress = 0;
    },

    bounce() {
      velocity[1] = SPEED.bounce;
      sinceGround = Infinity;
    },

    step(input: Input, dt: number) {
      previous[0] = position[0];
      previous[1] = position[1];

      const want = Number(input.right) - Number(input.left);
      if (want !== 0) hero.facing = want;

      const acceleration = hero.grounded ? ACCELERATION.ground : ACCELERATION.air;
      velocity[0] = approach(velocity[0], want * SPEED.run, acceleration * dt);

      sinceGround += dt;
      sincePress += dt;
      if (sincePress < GRACE.buffer && sinceGround < GRACE.coyote) {
        velocity[1] = SPEED.jump;
        sincePress = sinceGround = Infinity;
      }

      if (!input.jump && velocity[1] > CUT) velocity[1] = CUT;
      velocity[1] = Math.max(velocity[1] - ACCELERATION.gravity * dt, -SPEED.fall);

      move(0, velocity[0] * dt);
      const falling = velocity[1] < 0;
      hero.grounded = move(1, velocity[1] * dt) && falling;
      if (hero.grounded) sinceGround = 0;
    },
  };

  hero.place(...level.start);
  return hero;
}

export type Hero = ReturnType<typeof createHero>;
