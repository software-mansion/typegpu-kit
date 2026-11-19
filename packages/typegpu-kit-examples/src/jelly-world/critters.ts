import { table } from '@typegpu/kit';
import type { Inject } from '@typegpu/kit/inject';
import { d, type TgpuRoot } from 'typegpu';
import { vec3n } from 'wgpu-matrix';
import {
  CRITTERS,
  FALL,
  createBody,
  ease,
  HERALD,
  jump,
  LAND,
  R,
  random,
  STANDING,
  UP,
  type Body,
  type Crowd,
  type Ground,
} from './body.ts';
import { Critter, nearJelly } from './figures.ts';
import { carry, collide, fly, hang, perch, ride } from './moves.ts';
import {
  blinks,
  createFigure,
  figureOf,
  gait,
  moods,
  rollUpright,
  springs,
  type Figure,
  type Sights,
} from './poses.ts';
import { BOX, POINTS, STEP } from './slider.ts';
import { spineLength } from './spine.ts';
import { GROUND, type V3 } from './tray.ts';
import { walk } from './walking.ts';

/** How near the jelly's bounds a critter's bounding sphere comes for the jelly to show it. */
const SHOWN = 0.35;
/** The lowest a held critter goes over the ground, and how far picking lifts it to the camera. */
const HELD = { floor: 0.1, lift: 0.06 };
/** The longest a tap may last and the slowest it may move a critter, to make it flip. */
const TAP = { time: 0.25, speed: 0.4 };
/** Fixed steps of the jelly's motion riders move with, a sixtieth of a second. */
const FELT = 2;

/** Whether a bounding sphere comes near enough the jelly for it to show in or through it. */
const nearJellyBox = (bound: Figure['bound']) => {
  const reach = bound[3] + SHOWN;
  return (
    bound[0] > BOX.left - reach &&
    bound[0] < BOX.right + reach &&
    bound[1] < BOX.top + reach &&
    Math.abs(bound[2]) < BOX.z + reach
  );
};

// Scratch a critter's step writes its vectors into
const was: V3 = [0, 0, 0];
const accel: V3 = [0, 0, 0];
const ahead: V3 = [0, 0, 0];
const side: V3 = [0, 0, 0];
const toward: V3 = [0, 0, 0];

/**
 * Critters walking the jelly world, sliding down the jelly and carried by the pointer. Before the
 * world is revealed, one of them pops in and pulls the lever that reveals it.
 */
export function createCritters(root: TgpuRoot, world: Ground) {
  const rows = table.create(root, Critter, { capacity: CRITTERS });
  const near = root.createUniform(d.u32);
  let nearMask = 0;
  const bodies = Array.from({ length: CRITTERS }, (_, i) => createBody(i));
  const ids = bodies.map(() => rows.add());
  const figures = bodies.map(createFigure);
  const herald = bodies[0];

  /** The point the pointer hovers, while `watching`. */
  const watched: V3 = [0, 0, 0];
  let watching = false;
  let watchedFor = 0;
  let pointerMoved = 0;
  let pointerSpeed = 0;
  let held: Body | undefined;
  const holdAt: V3 = [0, GROUND, 0];
  let depth = 0;
  /** The spine after each of its last `FELT` fixed steps and the one before, oldest first. */
  const felt: Float32Array[] = [];
  /** Seconds before the herald pops in, and whether it still has the lever to pull. */
  let heraldIn = HERALD.after;
  let heralding = true;
  const crowd: Crowd = {
    bodies,
    world,
    herald,
    errand: undefined,
    pointer: undefined,
    pointerSpeed: 0,
    front: 0,
    spine: new Float32Array(),
  };
  const sights: Sights = { camera: [0, 0, 0], pointer: undefined, knob: [0, 0, 0] };

  /** One critter's step: grown in, moved as its mode has it, its body sprung and its face eased. */
  const move = (body: Body, step: number, before: Float32Array) => {
    const { front, spine } = crowd;
    vec3n.copy(body.velocity, was);
    // Falling bends nothing: the body falls as a whole, so its bend leaves gravity out
    if (body.mode === 'fly') was[1] -= FALL * step;
    const shown = body === herald && heraldIn <= 0;
    const target = shown || body.at[0] < front + 0.05 || body.mode === 'held' ? 1 : 0;
    body.grown = Math.max(0, body.growth.step(target, step));
    body.dizzy -= step;
    // Thrown past the lawn and the front, it shrinks away there and pops back in on the lawn
    const lost = body.at[0] > Math.max(front, LAND.x[1]) + 0.05;
    if (body.mode === 'walk' && lost && body.grown < 0.05) {
      body.at[0] = LAND.x[1];
      jump(body, 0.7, false);
    }

    if (body.mode === 'walk') walk(body, crowd, step);
    else if (body.mode === 'ride') ride(body, crowd, step, spine, before, FELT * STEP);
    else if (body.mode === 'fly') fly(body, crowd, step, spine);
    else if (body.mode === 'perch') perch(body, step);
    else if (body.mode === 'pull') {
      if (hang(body, crowd, step)) heralding = false;
    } else carry(body, holdAt, step, spine);
    if (body.mode !== 'fly' && body.spin) rollUpright(body, step);

    vec3n.set(Math.sin(body.heading), 0, Math.cos(body.heading), ahead);
    vec3n.normalize(vec3n.cross(UP, ahead, side), side);
    if (body.mode === 'walk' && body.asleep <= 0) gait(body, step, side);
    vec3n.scale(vec3n.sub(body.velocity, was, accel), 1 / step, accel);
    springs(body, step, accel, side, ahead, watching ? watched : undefined);
    moods(body, step);
    blinks(body, step);
    body.phase += step;
  };

  return {
    /** The critters as last posed, for tests and tools that read them on the CPU. */
    rows,
    /** Fills the critters' table, and which are near the jelly, for pipelines that draw them. */
    inject: (): Inject => (cfg) => rows.inject()(cfg).with(nearJelly, near),

    /** The point on the ground the pointer hovers, which critters look at and come to see. */
    watch(point: V3 | undefined) {
      if (point && watching) {
        pointerMoved += Math.hypot(point[0] - watched[0], point[2] - watched[2]);
      }
      if (point) vec3n.copy(point, watched);
      watching = point !== undefined;
      watchedFor = point ? 2.5 : 0;
    },
    /** Picks up the critter nearest a pointer ray, if any is under it. */
    pick(origin: V3, direction: V3) {
      let best = Infinity;
      for (const body of bodies) {
        if (body.grown < 0.5) continue;
        const centre = body.drawn;
        const along = vec3n.dot(vec3n.sub(centre, origin, toward), direction);
        const miss = vec3n.distance(vec3n.addScaled(origin, direction, along, toward), centre);
        if (miss < R * 2.2 && along < best) {
          best = along;
          held = body;
        }
      }
      depth = best - HELD.lift;
      if (held) {
        held.mode = 'held';
        held.base = undefined;
        // Set down elsewhere, it no longer follows a line
        held.leader = undefined;
        held.asleep = 0;
        held.heldFor = 0;
        held.spinning = 0;
        held.knocked = false;
        vec3n.addScaled(held.drawn, UP, -STANDING, holdAt);
      }
      return held !== undefined;
    },
    /** Moves the held critter along a pointer ray, as far from the camera as it was picked. */
    hold(origin: V3, direction: V3) {
      if (!held) return;
      const centre = vec3n.addScaled(origin, direction, depth, toward);
      if (centre[1] < GROUND + HELD.floor && direction[1] < -1e-4) {
        vec3n.addScaled(
          origin,
          direction,
          (GROUND + HELD.floor - origin[1]) / direction[1],
          centre,
        );
      }
      vec3n.addScaled(centre, UP, -STANDING, holdAt);
    },
    /** Lets go of the held critter with the speed it was moving at; a quick tap makes it flip. */
    drop() {
      if (!held) return;
      if (held.heldFor < TAP.time && vec3n.length(held.velocity) < TAP.speed) {
        jump(held, 0.95, true);
        held.joy = 1.2;
      } else {
        held.mode = 'fly';
        vec3n.scale(held.velocity, 0.8, held.velocity);
      }
      held = undefined;
    },
    get holding() {
      return held !== undefined;
    },
    /** Makes every critter on the ground jump for joy, waking any that nap. */
    hop() {
      for (const body of bodies) {
        if (body.mode !== 'walk' || body.grown < 0.5) continue;
        if (body.asleep > 0) body.shock = 1;
        else body.joy = 2;
        jump(body, random(0.6, 0.85), Math.random() < 0.3);
      }
    },

    /**
     * Notes the spine after one of its fixed steps. Riders move with how it moved over the last
     * `FELT` steps, measured on its own clock, so the frame rate doesn't change how hard they're
     * thrown.
     */
    feel(spine: Float32Array) {
      while (felt.length <= FELT) felt.push(spine.slice());
      const oldest = felt.shift()!;
      oldest.set(spine);
      felt.push(oldest);
    },
    update(dt: number, time: number, front: number, spine: Float32Array, camera: V3) {
      const step = Math.min(dt, 1 / 30);
      if (step <= 0) return;
      watchedFor -= step;
      pointerSpeed = ease(pointerSpeed, pointerMoved / step, 12, step);
      pointerMoved = 0;
      for (const body of bodies) body.load = 0;
      for (const body of bodies) if (body.mode === 'perch' && body.base) body.base.load += 1;

      // Pops in with a hop, and is done once the world is revealed past the lever
      heraldIn -= step;
      if (heraldIn <= 0 && heraldIn + step > 0) {
        jump(herald, 0.7, false);
        herald.joy = 1.5;
      }
      if (front > world.knob[0]) heralding = false;

      const pointer = watching && watchedFor > 0 ? watched : undefined;
      crowd.errand = heralding && heraldIn <= 0 ? herald : undefined;
      crowd.pointer = pointer;
      crowd.pointerSpeed = pointerSpeed;
      crowd.front = front;
      crowd.spine = spine;
      const before = felt[0] ?? spine;
      for (const body of bodies) move(body, step, before);
      collide(bodies, step);
      // Its weight drags the lever down only while it hangs there, however it stopped
      world.hang(herald.mode === 'pull' && herald.pulling >= HERALD.leap);

      // Critters near the jelly, which it shows in and through it
      vec3n.copy(camera, sights.camera);
      vec3n.copy(world.knob, sights.knob);
      sights.pointer = pointer;
      let mask = 0;
      for (let i = 0; i < CRITTERS; i++) {
        const { body, moved } = figures[i];
        // A critter that was not drawn last frame has nowhere to move from
        const shown = body[3] > 0;
        vec3n.copy(body, moved);
        const figure = figureOf(bodies[i], i, time, step, spine, sights, figures[i]);
        if (shown) vec3n.sub(moved, body, moved);
        else vec3n.zero(moved);
        rows.write(ids[i], figure);
        if (nearJellyBox(figure.bound)) mask |= 1 << i;
      }
      rows.flush();
      if (mask !== nearMask) near.write(mask);
      nearMask = mask;
    },
    /** How much each spine point is pressed down by riders, for the slider. */
    weigh(loads: Float32Array, spine: Float32Array) {
      loads.fill(0);
      const total = spineLength(spine);
      for (const body of bodies) {
        if (body.mode !== 'ride') continue;
        const at = (body.s / total) * (POINTS - 1);
        const i = Math.floor(at);
        const t = at - i;
        if (i > 0 && i < POINTS - 1) loads[i] += 1 - t;
        if (i + 1 > 0 && i + 1 < POINTS - 1) loads[i + 1] += t;
      }
    },
    destroy() {
      rows.destroy();
    },
  };
}
