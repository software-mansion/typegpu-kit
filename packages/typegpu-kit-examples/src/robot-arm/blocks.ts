import { machines } from '@typegpu/kit';
import { clamp } from 'math';
import { vec3, type Mat4, type Vec3Arg } from 'wgpu-matrix';
import { HAND, TIP } from './arm.ts';

/**
 * Fingertip height while free, reaching and carrying, which clears a tower of three blocks. The
 * hand leaves it only to dip onto a block it grabs or a spot it sets one down on.
 */
export const HOVER = 1.5;
/** Fingertip height above a block's bottom while gripping. */
const GRIP = 0.04;
/** Distance from the target at which the tip has arrived. */
const NEAR = 0.05;
/** Seconds before a move stops waiting for the tip. */
const PATIENCE = 3;
/** Seconds the fingers take to close or open. */
const SQUEEZE = 0.3;

/** Horizontal distances from the turntable axis that block centres stay within. */
const RING = [1.2, 2.0] as const;
/** Height overlap below which blocks pass over each other. */
const REST = 0.06;
/** Height a block's top may stand above a point and still hold what is set there. */
const STANDING = 0.01;
/** Fall speed in world units per second. */
const FALL = 3;
/** Centre distance resting blocks keep apart, over the sum of their sizes. */
const APART = 0.6;

/** Seconds between tasks in attract mode. */
const PAUSE = 0.6;
/**
 * Once sorted, attract mode scatters this share of the blocks onto the bench between the towers,
 * the `n`th landing `out` to `out + width` from the axis, stepping by `step` a task.
 */
const SCATTER = { share: 0.5, out: 1.3, width: 0.6, step: 0.37 };

/** A box with a square footprint and a colour, resting on the bench or another box, or held. */
export interface Block {
  /** Index of its colour, whose tower stands at the home of that index. */
  readonly colour: number;
  /** Width and depth in world units. */
  readonly size: number;
  readonly height: number;
  /** Footprint centre and bottom height. */
  x: number;
  y: number;
  z: number;
  /** Radians turned about +y. */
  yaw: number;
}

/** The arm's fingertips and wrist frame, which `step` reads as they move. */
export interface Gripper {
  readonly tip: Vec3Arg;
  /** The wrist's world transform; its +y points from the wrist to the tip. */
  readonly wrist: Mat4;
}

const topOf = (block: Block) => block.y + block.height;

/** How far `angle` is turned from the nearest quarter turn. */
const offSquare = (angle: number) => angle - Math.round(angle / (Math.PI / 2)) * (Math.PI / 2);

/** Moves `block` along its radius into `RING`. */
const ring = (block: { x: number; z: number }) => {
  const out = Math.hypot(block.x, block.z) || 1e-6;
  const kept = clamp(out, RING[0], RING[1]) / out;
  block.x *= kept;
  block.z *= kept;
};

/** Moves `block` along `dx`, `dz` until it is `reach` from where that offset starts. */
const shove = (block: Block, dx: number, dz: number, reach: number) => {
  const gap = Math.hypot(dx, dz) || 1e-6;
  const overlap = reach - gap;
  if (overlap <= 0) return;
  block.x += (dx / gap) * overlap;
  block.z += (dz / gap) * overlap;
};

/** Distance along the ray to where it enters `block`, or Infinity: a slab test in its frame. */
const entryOf = (block: Block, origin: Vec3Arg, direction: Vec3Arg) => {
  const [c, s] = [Math.cos(block.yaw), Math.sin(block.yaw)];
  const half = [block.size / 2, block.height / 2, block.size / 2];
  const [px, py, pz] = [origin[0] - block.x, origin[1] - block.y, origin[2] - block.z];
  const o = [px * c - pz * s, py - half[1], px * s + pz * c];
  const [dx, dy, dz] = [direction[0], direction[1], direction[2]];
  const d = [dx * c - dz * s, dy, dx * s + dz * c];
  let [near, far] = [-Infinity, Infinity];
  for (let k = 0; k < 3; k++) {
    const a = (-half[k] - o[k]) / d[k];
    const b = (half[k] - o[k]) / d[k];
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
  }
  return near <= far && far > 0 ? near : Infinity;
};

/**
 * Picks up and sets down `blocks` with `gripper`. A task is a machine that reaches over a block,
 * lowers onto it, closes the fingers, lifts, carries, places, opens and rises. `step` moves the
 * fingertip `target` and the `fingers`, and pushes resting blocks away from the gripper and each
 * other.
 */
export function createBlocks(
  blocks: Block[],
  gripper: Gripper,
  homes: readonly (readonly [x: number, z: number])[],
) {
  let tasks = 0;
  /** Blocks left to scatter before sorting again; 0 while sorting. */
  let scattering = 0;
  /** Block indices of the task, the one held, the one last grabbed and the one tapped; -1 for none. */
  let chosen = -1;
  let held = -1;
  let last = -1;
  let tapped = -1;
  const drop = { x: 0, z: 0, set: false };
  const target = vec3.create();
  const at = vec3.create();
  const local = vec3.create();

  /** The highest block under bench point x, z whose top is no higher than `below`, if any. */
  const surfaceAt = (x: number, z: number, below: number, except: number) => {
    let found: Block | undefined;
    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i];
      const inside = Math.hypot(block.x - x, block.z - z) < block.size / 2;
      if (i === except || i === held || !inside || topOf(block) > below + STANDING) continue;
      if (!found || topOf(block) > topOf(found)) found = block;
    }
    return found;
  };

  /** Whether nothing rests on block `i`. */
  const isBare = (i: number) => {
    const block = blocks[i];
    for (const other of blocks) {
      const touching = Math.abs(other.y - topOf(block)) < REST;
      if (touching && Math.hypot(other.x - block.x, other.z - block.z) < block.size / 2) {
        return false;
      }
    }
    return true;
  };

  /** The block that block `i` rests on, or -1 for the bench. */
  const beneath = (i: number) => {
    const block = blocks[i];
    for (let j = 0; j < blocks.length; j++) {
      const other = blocks[j];
      if (j === i || j === held || Math.abs(topOf(other) - block.y) > REST) continue;
      if (Math.hypot(other.x - block.x, other.z - block.z) < other.size / 2) return j;
    }
    return -1;
  };

  /** Whether block `i` stands in its colour's tower: at its home, on blocks of its colour only. */
  const isSorted = (i: number) => {
    const { colour } = blocks[i];
    const home = homes[colour];
    if (!home) return false;
    for (let k = i; k >= 0; k = beneath(k)) {
      const block = blocks[k];
      if (block.colour !== colour) return false;
      if (block.y < REST) return Math.hypot(block.x - home[0], block.z - home[1]) < block.size / 2;
    }
    return false;
  };
  /** Whether block `i` is on top and out of its tower, so sorting moves it. */
  const isLoose = (i: number) => i !== held && isBare(i) && !isSorted(i);
  /** Whether block `i` tops its colour's tower, so scattering takes it. */
  const isTop = (i: number) => i !== held && isBare(i) && isSorted(i);

  /** The first block from index `from` onwards, wrapping round, that `accepts` takes, or -1. */
  const nextFrom = (from: number, accepts: (i: number) => boolean) => {
    for (let k = 0; k < blocks.length; k++) {
      const i = (from + k) % blocks.length;
      if (accepts(i)) return i;
    }
    return -1;
  };

  /** Sets the held block down at bench point x, z, or on top of the blocks there. */
  const dropAt = (x: number, z: number) => {
    drop.x = x;
    drop.z = z;
    ring(drop);
    const under = surfaceAt(drop.x, drop.z, Infinity, -1);
    if (under) {
      drop.x = under.x;
      drop.z = under.z;
    }
    drop.set = true;
  };

  /**
   * Sets the held block on its colour's tower, or while scattering on the bench halfway between
   * two homes.
   */
  const placeHeld = () => {
    const home = homes[blocks[held].colour];
    if (scattering > 0 || !home) {
      scattering = Math.max(0, scattering - 1);
      const [x, z] = homes[tasks % homes.length];
      const angle = Math.atan2(z, x) + Math.PI / homes.length;
      const out = SCATTER.out + ((tasks * SCATTER.step) % 1) * SCATTER.width;
      dropAt(Math.cos(angle) * out, Math.sin(angle) * out);
    } else dropAt(home[0], home[1]);
    tasks++;
  };

  const arrived = () => vec3.distance(target, gripper.tip) < NEAR;
  /** The next block attract mode moves: one out of its tower, or a tower's top while scattering. */
  const nextTask = () => nextFrom(last + 1, scattering > 0 ? isTop : isLoose);
  const hasTask = () => nextTask() >= 0;
  const placing = () => drop.set && arrived();
  /** Moves on to `to` once the tip arrives, or after `PATIENCE` seconds. */
  const onceThere = (to: 'lower' | 'close' | 'carry' | 'open' | 'free') =>
    [
      { to, when: arrived },
      { to, after: PATIENCE },
    ] as const;

  const Task = machines.define('Task', {
    events: ['grab', 'sort'],
    states: {
      free: {
        next: [
          { to: 'reach', on: 'grab' },
          { to: 'reach', on: 'sort', after: PAUSE, when: hasTask },
        ],
      },
      reach: {
        enter: () => {
          chosen = tapped >= 0 ? tapped : nextTask();
          last = chosen;
          tapped = -1;
        },
        next: onceThere('lower'),
      },
      lower: { next: onceThere('close') },
      close: {
        exit: () => {
          held = chosen;
        },
        next: [{ to: 'lift', after: SQUEEZE }],
      },
      lift: { next: onceThere('carry') },
      carry: { next: [{ to: 'place', when: placing }] },
      place: { next: onceThere('open') },
      open: {
        exit: () => {
          held = -1;
          chosen = -1;
          drop.set = false;
        },
        next: [{ to: 'rise', after: SQUEEZE }],
      },
      rise: { next: onceThere('free') },
    },
  });
  const task = machines.run(Task);

  /** Moves resting blocks out of an upright cylinder around `centre`. */
  const push = (centre: Vec3Arg, radius: number, half: number) => {
    const y = centre[1];
    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i];
      const above = y - half > topOf(block) - REST;
      const below = y + half < block.y;
      if (i === held || i === chosen || above || below) continue;
      shove(block, block.x - centre[0], block.z - centre[2], radius + block.size / 2);
    }
  };

  /** Pushes a gripper part centred `x`, `y` in the wrist's frame. */
  const pushPart = (x: number, y: number, radius: number, half: number) => {
    vec3.transformMat4(vec3.set(x, y, 0, local), gripper.wrist, at);
    push(at, radius, half);
  };

  /** Pushes overlapping blocks apart and into the ring, then drops each onto what is under it. */
  const settle = (dt: number) => {
    for (let i = 0; i < blocks.length; i++) {
      for (let j = i + 1; j < blocks.length; j++) {
        if (i === held || j === held) continue;
        const a = blocks[i];
        const b = blocks[j];
        if (topOf(a) - REST < b.y || topOf(b) - REST < a.y) continue;
        const x = (a.x + b.x) / 2;
        const z = (a.z + b.z) / 2;
        const reach = (a.size + b.size) * (APART / 2);
        shove(a, a.x - x, a.z - z, reach);
        shove(b, b.x - x, b.z - z, reach);
      }
    }

    for (let i = 0; i < blocks.length; i++) {
      if (i === held) continue;
      const block = blocks[i];
      ring(block);
      const under = surfaceAt(block.x, block.z, block.y, i);
      block.y = Math.max(under ? topOf(under) : 0, block.y - FALL * dt);
    }
  };

  /** Points the target where the task's state heads, at bench point `aim` while free. */
  const aimFor = (aim: readonly [x: number, z: number]) => {
    const block = blocks[chosen];
    switch (task.state()) {
      case Task.free:
        vec3.set(aim[0], HOVER, aim[1], target);
        break;
      case Task.reach:
        if (block) vec3.set(block.x, HOVER, block.z, target);
        break;
      case Task.lower:
        if (block) vec3.set(block.x, block.y + GRIP, block.z, target);
        break;
      case Task.lift:
      case Task.rise:
        target[1] = HOVER;
        break;
      case Task.carry:
        if (drop.set) vec3.set(drop.x, HOVER, drop.z, target);
        else vec3.set(aim[0], HOVER, aim[1], target);
        break;
      case Task.place: {
        const under = surfaceAt(drop.x, drop.z, Infinity, -1);
        target[1] = (under ? topOf(under) : 0) + GRIP;
        break;
      }
    }
  };

  settle(Infinity);

  let fingers = HAND.open;

  return {
    /** Where the fingertips head, written by `step`. */
    target: target as Readonly<typeof target>,
    /** How far each finger stands from the palm centre, written by `step`. */
    get fingers() {
      return fingers;
    },
    get holding() {
      return held >= 0;
    },

    /** Returns the nearest block the ray hits, or -1. */
    hit(origin: Vec3Arg, direction: Vec3Arg) {
      let nearest = Infinity;
      let found = -1;
      for (let i = 0; i < blocks.length; i++) {
        const distance = entryOf(blocks[i], origin, direction);
        if (distance >= nearest) continue;
        nearest = distance;
        found = i;
      }
      return found;
    },

    /** Sends the gripper to pick up block `i` unless it is busy. */
    grab(i: number) {
      if (task.state() !== Task.free || !blocks[i]) return;
      tapped = i;
      task.send.grab();
    },

    /** Sets the held block down at bench point x, z, or on the blocks there, once it is lifted. */
    drop(x: number, z: number) {
      const state = task.state();
      if (state === Task.lift || state === Task.carry) dropAt(x, z);
    },

    /**
     * Sorts on its own for the next step: after a pause, moves a block out of its tower onto its
     * colour's, and once all are sorted scatters some across the bench to sort again.
     */
    auto() {
      if (task.state() === Task.free) {
        if (scattering === 0 && nextFrom(0, isLoose) < 0) {
          scattering = Math.ceil(blocks.length * SCATTER.share);
        } else if (scattering > 0 && nextFrom(0, isTop) < 0) scattering = 0;
      }
      task.send.sort();
      if (task.state() === Task.carry && !drop.set) placeHeld();
    },

    /**
     * Advances by `dt` seconds. While free, the fingertips head for bench point `aim` at `HOVER`
     * height.
     */
    step(dt: number, aim: readonly [x: number, z: number]) {
      // The guards measure the tip against the target of the state it is in
      aimFor(aim);
      task.step(dt);
      aimFor(aim);

      const block = blocks[chosen];
      const squeeze = Math.min(1, task.since() / SQUEEZE);
      const shut = block ? block.size / 2 + HAND.finger.width / 2 : HAND.open;
      const { wrist } = gripper;
      const yaw = Math.atan2(-wrist[2], wrist[0]);
      const state = task.state();
      if (state === Task.close) {
        fingers = HAND.open + (shut - HAND.open) * squeeze;
        if (block) block.yaw -= offSquare(block.yaw - yaw) * squeeze;
      } else if (state === Task.open) {
        fingers = shut + (HAND.open - shut) * squeeze;
      } else {
        fingers = held >= 0 ? shut : HAND.open;
      }

      const holding = blocks[held];
      if (holding) {
        vec3.set(0, TIP[1] + GRIP - holding.height / 2, 0, local);
        vec3.transformMat4(local, wrist, at);
        holding.x = at[0];
        holding.y = at[1] - holding.height / 2;
        holding.z = at[2];
        holding.yaw = yaw;
        push(at, holding.size / 2, holding.height / 2);
      }

      const { palm, finger } = HAND;
      pushPart(0, palm.at, palm.width / 3, palm.height / 2);
      pushPart(-fingers, finger.at, finger.width / 2, finger.height / 2);
      pushPart(fingers, finger.at, finger.width / 2, finger.height / 2);
      settle(dt);
    },
  };
}
