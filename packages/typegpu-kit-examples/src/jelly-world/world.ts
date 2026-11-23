import { clock, table } from '@typegpu/kit';
import { std, type TgpuRoot } from 'typegpu';
import {
  ARM,
  BUSH,
  GUMDROP,
  HOUSE,
  LEVER,
  LOLLY,
  MUSHROOM,
  PIVOT,
  Prop,
  PUFF,
  TREE,
} from './props.ts';
import { missOf } from './slider.ts';
import { GROUND, type V3 } from './tray.ts';

/** Pastel jelly colours in display values. */
const PALETTE = [
  [1, 0.5, 0.66],
  [0.5, 0.8, 1],
  [1, 0.82, 0.36],
  [0.72, 0.58, 1],
  [0.3, 0.86, 0.6],
  [1, 0.6, 0.42],
] as const;

/** Trees as x, z, size, palette index and what they grow into, set back from the slot and beside it. */
const TREES = [
  [-1.45, -0.6, 1.1, 4, GUMDROP],
  [-0.95, -0.85, 0.9, 0, LOLLY],
  [-0.2, -1.25, 1.3, 4, PUFF],
  [0.5, -0.75, 0.85, 3, MUSHROOM],
  [1.3, -1.0, 1.2, 4, PUFF],
  [1.9, -0.5, 0.9, 5, BUSH],
  [-2.0, -1.15, 1.25, 2, PUFF],
  [2.3, -1.4, 1.3, 3, GUMDROP],
  [-1.5, 0.55, 0.75, 4, BUSH],
  [1.6, 0.62, 0.8, 0, MUSHROOM],
] as const;
/** Over a tree's size, how far critters keep from its middle by what it grew into. */
const FOOTPRINT = [0.03, 0.02, 0.05, 0.05, 0.12] as const;
/** Houses as x, z, heading, size and palette index. */
const HOUSES = [
  [-0.5, -0.62, 0.35, 1, 0],
  [1.0, -0.55, -0.4, 0.95, 2],
  [-1.6, -1.35, 0.2, 1.1, 1],
  [0.65, -1.4, -0.1, 1.2, 5],
  [1.75, 0.05, -0.9, 0.85, 3],
] as const;
/**
 * The lever by the slot's free end: where it stands, its colour, the angles of its arm at rest,
 * pulled all the way and where it sends the jelly over, how far back it must fall before it sends
 * it again and how far past rest it swings, how near a pointer ray must pass the knob to grab it,
 * nearer while a critter hangs there, how fast it follows the pointer, the spring that throws it
 * back, and how fast a critter's weight drags it down.
 */
const PULL = {
  at: [1.08, 0.5],
  palette: 1,
  rest: -0.2,
  full: 1.3,
  flips: 1.05,
  rearm: 0.5,
  overswing: 0.4,
  grip: [0.06, 0.035],
  follow: 25,
  stiffness: 110,
  damping: 5,
  hung: 3.4,
};
/** The lever's block squashing as it sends the jelly over: the kick, its spring and the most. */
const KICK = { push: 4, stiffness: 200, damping: 9, most: 0.3 };
/** Each kind's size over its modelled size. */
const SCALE = { tree: 1.7, house: 1.6 };

/**
 * Spring stiffness and damping of growing, which overshoots into a wobble, the most the front
 * passes a prop by in world x before it grows, so neighbours grow apart, and how much a prop
 * squashes by its growth's speed, at most `most`.
 */
const GROW = { stiffness: 140, damping: 7, stagger: 0.12, squash: 0.08, most: 0.4 };
/** How far tree tops sway, and how fast in radians a second. */
const SWAY = { reach: 0.06, rate: 2 };

/** Trees and houses as they stand, each with the radius critters keep from its middle. */
const PLANTED = [
  ...TREES.map(([x, z, size, colour, variant], i) => ({
    x,
    z,
    heading: i * 1.7,
    size: size * SCALE.tree,
    kind: TREE,
    variant,
    colour,
    radius: FOOTPRINT[variant] * size * SCALE.tree,
  })),
  ...HOUSES.map(([x, z, heading, size, colour]) => ({
    x,
    z,
    heading,
    size: size * SCALE.house,
    kind: HOUSE,
    variant: PUFF,
    colour,
    radius: 0.12 * size * SCALE.house,
  })),
];

/** The jelly world's trees and houses, grown by a front sweeping along x. */
export function createWorld(root: TgpuRoot) {
  const props = table.create(root, Prop, { capacity: PLANTED.length + 1 });
  const fields = props.fields;
  /** What grows behind the front, each a little after the last, on a spring. */
  const growing = PLANTED.map(({ x, z, heading, size, kind, variant, colour }, i) => ({
    id: props.add({ at: [x, z], heading, size, kind, variant, color: PALETTE[colour] }),
    x,
    heading,
    after: ((i * 0.618) % 1) * GROW.stagger,
    grow: clock.spring(GROW),
  }));
  const [leverX, leverZ] = PULL.at;
  const lever = props.add({
    at: [leverX, leverZ],
    size: 1,
    kind: LEVER,
    grown: 1,
    color: PALETTE[PULL.palette],
  });
  const obstacles = [...PLANTED, { x: leverX, z: leverZ, radius: 0.08 }];

  /** The lever's arm angle on its spring, and where the pointer pulls it. */
  const arm = clock.spring({ ...PULL, value: PULL.rest });
  let wanted = PULL.rest;
  let grabbed = false;
  let hung = false;
  /** Whether the lever sends the jelly over the next time it is pulled, and its block's squash. */
  let armed = true;
  const block = clock.spring(KICK);
  let moving = true;
  let elapsed = 0;
  const at: V3 = [0, 0, 0];
  const knob = () => {
    at[0] = leverX;
    at[1] = GROUND + PIVOT + Math.cos(arm.value) * ARM;
    at[2] = leverZ + Math.sin(arm.value) * ARM;
    return at;
  };

  return {
    /** Where trees, houses and the lever stand, for critters to walk around. */
    obstacles,

    /** Takes hold of the lever if a pointer ray passes its knob. */
    grab(origin: V3, direction: V3) {
      grabbed = missOf(origin, direction, knob()) < PULL.grip[hung ? 1 : 0];
      return grabbed;
    },
    /** Pulls the held lever towards where a pointer ray crosses the plane it swings in. */
    pull(origin: V3, direction: V3) {
      if (!grabbed || Math.abs(direction[0]) < 1e-4) return;
      const t = (leverX - origin[0]) / direction[0];
      const y = origin[1] + direction[1] * t - GROUND - PIVOT;
      const z = origin[2] + direction[2] * t - leverZ;
      wanted = std.clamp(Math.atan2(z, y), PULL.rest, PULL.full);
    },
    release() {
      grabbed = false;
    },
    /** Whether the pointer holds the lever. */
    get holding() {
      return grabbed;
    },

    /** Where the lever's knob is, for a critter to hang on. */
    get knob() {
      return knob();
    },
    /** How far the lever is pulled, from 0 at rest to 1 all the way. */
    get pulled() {
      return (arm.value - PULL.rest) / (PULL.full - PULL.rest);
    },
    /** Hangs a critter's weight on the knob, which drags the arm down unless the pointer holds it. */
    hang(on: boolean) {
      hung = on;
    },

    /** Whether a prop grew or squashed in the last `update`. */
    get moving() {
      return moving;
    },
    /**
     * Grows what the front at `front` in world x has passed and swings the lever. Returns true on
     * the frame the lever is pulled far enough to send the jelly over.
     */
    update(dt: number, front: number) {
      const step = Math.min(dt, 1 / 30);
      moving = false;
      // The first frame's shadows are baked from the table, so it is written even without time
      if (step <= 0) {
        props.flush();
        return false;
      }

      // The pointer or a hanging critter takes the arm where it goes, else the spring throws it back
      if (grabbed || hung) {
        const before = arm.value;
        const to = grabbed ? wanted : PULL.full;
        arm.value += (to - arm.value) * clock.ease(step, grabbed ? PULL.follow : PULL.hung);
        arm.velocity = (arm.value - before) / step;
      } else {
        arm.value = std.clamp(arm.step(PULL.rest, step), PULL.rest - PULL.overswing, PULL.full);
      }

      const flipped = armed && arm.value > PULL.flips;
      if (flipped) {
        armed = false;
        block.velocity -= KICK.push;
      }
      if (arm.value < PULL.flips - PULL.rearm) armed = true;

      const squash = block.step(0, step);
      moving = Math.abs(squash) + Math.abs(block.velocity) > 1e-4;
      fields.squash.set(lever, std.clamp(squash, -KICK.most, KICK.most));
      fields.angle.set(lever, arm.value);

      elapsed += step;
      for (const prop of growing) {
        const target = prop.x + prop.after < front ? 1 : 0;
        const { grow } = prop;
        grow.value = Math.max(0, grow.step(target, step));
        if (Math.abs(grow.velocity) + Math.abs(target - grow.value) > 1e-4) moving = true;
        fields.grown.set(prop.id, grow.value);
        fields.squash.set(prop.id, std.clamp(grow.velocity * GROW.squash, -GROW.most, GROW.most));
        fields.sway.set(prop.id, Math.sin(elapsed * SWAY.rate + prop.heading) * SWAY.reach);
      }
      props.flush();
      return flipped;
    },

    inject: props.inject,
    destroy: props.destroy,
  };
}

export type World = ReturnType<typeof createWorld>;
