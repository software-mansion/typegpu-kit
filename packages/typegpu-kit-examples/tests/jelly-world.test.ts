import { d } from 'typegpu';
import { expect, it, vi } from 'vitest';
import { recordingRoot } from '@typegpu/kit/testing';
import { CRITTERS, LADDER, LAND } from '../src/jelly-world/body.ts';
import { createCritters } from '../src/jelly-world/critters.ts';
import { createScene } from '../src/jelly-world/scene.ts';
import { createSlider, POINTS, RANGE, STEP } from '../src/jelly-world/slider.ts';
import { SLOT, type V3 } from '../src/jelly-world/tray.ts';
import { createWorld } from '../src/jelly-world/world.ts';
import { compute, describeExample, pass, settle } from './utils/examples.ts';

const render = pass('draw');
/** The spine's field, the props' shadows while they grow or the light turns, and the tiles. */
const bake = compute(3);
const spine = compute(2);
/** What each screen tile can see, sorted out every frame. */
const tiles = compute();
/** The march, TAA's resolve and its state, the final pass into the surface and their timings. */
const tail = [
  ...render,
  ...render,
  'copy-buffer',
  ...render,
  'resolve-queries',
  'copy-buffer',
  'submit',
];

describeExample(
  'jelly-world',
  createScene,
  (build) => {
    it('bakes the spine, prints the value and marches the jelly', async () => {
      const { log, next } = await build();
      const baking = log.shaders.find((code) => code.includes('sdBezier'));
      expect(baking).toMatch(/@workgroup_size\(8, 8/);

      // The first frame bakes before the spine has stepped, and prints the value.
      next();
      // @typegpu/noise fills its Perlin cache on creation, in a pass of its own.
      const fill = [...compute(), 'submit'];
      expect(log.events).toEqual([...fill, ...render, ...bake, ...tail]);
    });

    it('bakes only after a physics step and prints the value only when it changes', async () => {
      const { log, scene, next } = await build();
      next();
      next(0.05);

      // A drag moves the end; the label follows as the jelly eases there. Nothing has grown yet.
      scene.pointing.drag?.(0.3, 0.5, true);
      let from = log.events.length;
      next(0.2);
      expect(log.events.slice(from)).toEqual([...render, ...spine, ...tail]);

      // No time passes: nothing steps, and the value stays. The timings' staging buffers map first.
      await settle();
      from = log.events.length;
      next(0);
      expect(log.events.slice(from)).toEqual([...tiles, ...tail]);
    });

    it('lets a critter pull the lever, which grows the jelly world', async () => {
      const { log, next } = await build();
      next();
      const from = log.events.length;
      for (let i = 0; i < 400; i++) next(1 / 30);
      const events = log.events.slice(from).join(' ');
      expect(events).toContain(bake.join(' '));
    });

    it('sets the detail from GPU time until it is moved by hand', async () => {
      const { scene, next } = await build();
      next();
      const { values } = scene.settings;
      expect(values.auto).toBe(true);
      expect(values.detail).toBe(1);

      values.detail = 0.8;
      await scene.settings.write!('detail');
      next();
      expect(values.auto).toBe(false);
      expect(values.detail).toBe(0.8);

      values.auto = true;
      await scene.settings.write!('auto');
      next();
      expect(values.detail).toBe(0.8);
    });

    it("keeps the host's turn, and zooms out no farther than the bare tray", async () => {
      const { scene, next } = await build();
      next();
      const { goal } = scene.view;
      const framed = goal.distance;
      goal.yaw = 0.5;
      goal.pitch = 0.4;
      goal.distance = 100;
      next(0.1);
      next(0.1);
      expect([goal.yaw, goal.pitch]).toEqual([0.5, 0.4]);
      expect(goal.distance).toBeGreaterThan(framed);
      expect(goal.distance).toBeLessThan(100);
    });
  },
  {
    features: ['timestamp-query'],
    // The two bakes, the tiles, the noise cache, the march, the label's text, TAA and the final pass.
    pipelines: 8,
  },
);

/** Which side of the line through `a` and `b` the point `p` lies on. */
const side = (a: readonly number[], b: readonly number[], p: readonly number[]) =>
  Math.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]));

it('lays the spine out at rest between its ends, so it starts untangled', () => {
  const { root } = recordingRoot();
  for (const start of [RANGE[0], 0.38, RANGE[1]]) {
    const slider = createSlider(root, { start: start });
    const at = (i: number) => [slider.spine[i * 2], slider.spine[i * 2 + 1]] as const;
    for (let i = 0; i < POINTS - 1; i++) {
      for (let j = i + 2; j < POINTS - 1; j++) {
        const [a, b, c, e] = [at(i), at(i + 1), at(j), at(j + 1)];
        // Two segments cross where each one's ends lie on opposite sides of the other
        const crosses = side(a, b, c) * side(a, b, e) < 0 && side(c, e, a) * side(c, e, b) < 0;
        expect(crosses, `segments ${i} and ${j} at ${start}`).toBe(false);
      }
    }
    expect(at(POINTS - 1)[0]).toBeCloseTo(start, 5);
    slider.destroy();
  }
});

/**
 * The jelly world's critters with `Math.random` fixed, stepped at 60 Hz with the world revealed,
 * and the jelly's end easing to `end` as the scene eases it. `carry` picks critter `k` up and sets
 * it down at `to`, moving it at `speed` when given and snapping there otherwise.
 */
function meadow(end: number = RANGE[1], rate = 60, front = 2.7) {
  let n = 0;
  vi.spyOn(Math, 'random').mockImplementation(() => (n++ * 0.618034) % 1);
  const { root } = recordingRoot();
  const world = createWorld(root);
  const slider = createSlider(root, {});
  const critters = createCritters(root, world);
  const { rows } = critters;
  /** Critter `k`'s body centre and radius, and which way it faces, as last posed. */
  const body = (k: number) => rows.fields.body.get(rows.idAt(k), d.vec4f());
  const ahead = (k: number) => rows.fields.ahead.get(rows.idAt(k), d.vec4f());
  let time = 0;
  let goal = end;
  /** Fixed steps the slider owes, which at 240 frames a second it takes every other frame. */
  let owed = 0;
  const dt = 1 / rate;
  const step = () => {
    time += dt;
    slider.drag(slider.target + (goal - slider.target) * (1 - Math.exp(-5 * dt)));
    for (owed += dt / STEP; owed >= 1 - 1e-9; owed--) {
      slider.step();
      critters.feel(slider.spine);
    }
    critters.weigh(slider.loads, slider.spine);
    world.update(dt, front);
    critters.update(dt, time, front, slider.spine, [0, 1, 2]);
  };
  /** Drags the slider towards `to` from now on. */
  const aim = (to: number) => {
    goal = to;
  };
  const carry = (k: number, [x, z]: [number, number], speed?: number) => {
    const { x: x0, z: z0 } = body(k);
    expect(critters.pick([x0, 3, z0], [0, -1, 0])).toBe(true);
    const moving = speed ? Math.ceil((Math.hypot(x - x0, z - z0) / speed) * rate) : 1;
    for (let i = 1; i < moving + rate * 0.6; i++) {
      const t = Math.min(1, i / moving);
      critters.hold([x0 + (x - x0) * t, 3, z0 + (z - z0) * t], [0, -1, 0]);
      step();
    }
    critters.drop();
  };
  const destroy = () => {
    critters.destroy();
    world.props.destroy();
    slider.destroy();
    vi.restoreAllMocks();
  };
  return { world, critters, body, ahead, step, aim, carry, destroy };
}

/**
 * Reveals the jelly world, carries the other critters near `pointer` away from it, critter 6 to
 * `blocker` if given and critter 1 to `from`, then watches `pointer` for five seconds. Returns
 * critter 1's position and heading each step.
 */
function blocked(from: [number, number], pointer: V3, blocker?: [number, number]) {
  const subject = 1;
  const { critters, body, ahead, step, carry, destroy } = meadow();
  for (let i = 0; i < 240; i++) step();
  // Everyone hops, so none is left sitting on another's head to be picked up instead of it
  critters.hop();
  for (let i = 0; i < 60; i++) step();
  // Others near the pointer would come to see it too, so they are set down far from it
  for (let k = 0; k < CRITTERS; k++) {
    const { x, z } = body(k);
    if (k === subject || Math.hypot(x - pointer[0], z - pointer[2]) > 0.8) continue;
    carry(k, [pointer[0] < 0 ? 1.6 : -1.6, -0.9]);
  }
  if (blocker) carry(6, blocker);
  carry(subject, from);
  const path: { x: number; z: number; heading: number }[] = [];
  for (let i = 0; i < 300; i++) {
    critters.watch(pointer);
    step();
    const { x, z } = body(subject);
    const facing = ahead(subject);
    path.push({ x, z, heading: Math.atan2(facing.x, facing.z) });
  }
  destroy();
  return path;
}

const wrap = (a: number) => a - Math.round(a / (Math.PI * 2)) * Math.PI * 2;

/** How often a heading turning faster than 0.3 rad/s turns the other way the next step. */
const dithering = (path: { heading: number }[]) => {
  let flips = 0;
  for (let i = 2; i < path.length; i++) {
    const a = wrap(path[i - 1].heading - path[i - 2].heading) * 60;
    const b = wrap(path[i].heading - path[i - 1].heading) * 60;
    if (Math.abs(a) > 0.3 && Math.abs(b) > 0.3 && Math.sign(a) !== Math.sign(b)) flips++;
  }
  return flips;
};

it.each([
  { name: 'the lever', from: [1.3, 0.5], pointer: [0.78, 0.08, 0.5] },
  {
    name: 'another critter',
    from: [-0.5, 0.86],
    pointer: [-0.5, 0.08, 0.35],
    blocker: [-0.5, 0.49],
  },
] as const)('walks round $name to the pointer without looking left and right', (scenario) => {
  const path = blocked(
    [...scenario.from],
    [...scenario.pointer],
    scenario.blocker && [...scenario.blocker],
  );
  const end = path[path.length - 1];
  expect(Math.hypot(end.x - scenario.pointer[0], end.z - scenario.pointer[2])).toBeLessThan(0.3);
  expect(dithering(path)).toBeLessThanOrEqual(2);
});

it("lets critters set down in a heap on the slot's edge walk apart rather than jump", () => {
  const { critters, body, step, carry, destroy } = meadow();
  for (let i = 0; i < 240; i++) step();
  // Overlapping each other and the margin critters keep from the slot
  const heap: [number, number][] = [
    [-0.3, 0.24],
    [-0.27, 0.25],
    [-0.24, 0.24],
    [-0.27, 0.27],
    [-0.3, 0.27],
  ];
  heap.forEach((to, i) => carry(2 + i, to, 0.4));
  let fastest = 0;
  const heaped = heap.map((_, i) => 2 + i);
  let last = heaped.map(body);
  for (let i = 0; i < 240; i++) {
    critters.watch([-0.27, 0.08, 0.4]);
    step();
    const now = heaped.map(body);
    now.forEach((at, k) => {
      // Once clear of the heap, a critter is free to hop, ride or be thrown
      if (Math.hypot(at.x + 0.27, at.z - 0.25) > 0.3) return;
      fastest = Math.max(fastest, Math.hypot(at.x - last[k].x, at.z - last[k].z));
    });
    last = now;
  }
  destroy();
  expect(fastest * 60).toBeLessThan(0.6);
});

it('lets a critter dropped on the jelly curled up at zero slide and fall off rather than bob', () => {
  const { critters, body, step, destroy } = meadow(RANGE[0]);
  for (let i = 0; i < 600; i++) step();
  // Beside the high side the curl comes down, which overhangs further down
  const at = body(3);
  expect(critters.pick([at.x, 1, at.z], [0, -1, 0])).toBe(true);
  const lifted = 1.07 - at.y;
  for (let i = 0; i < 90; i++) {
    critters.hold([-0.55, 0.5 + lifted, 0], [0, -1, 0]);
    step();
  }
  critters.drop();
  let highest = 0;
  for (let i = 0; i < 480; i++) {
    step();
    if (i >= 360) highest = Math.max(highest, body(3).y);
  }
  destroy();
  expect(highest).toBeLessThan(0.15);
});

it('keeps critters on the lawn, and has them climb onto the jelly from its sides near its base', () => {
  const { body, step, destroy } = meadow();
  let farthest = 0;
  /** Where in x critters came over the slot's sides onto the jelly's middle. */
  const onto: number[] = [];
  const last = Array.from({ length: CRITTERS }, (_, k) => body(k));
  for (let i = 0; i < 60 * 45; i++) {
    step();
    for (let k = 0; k < CRITTERS; k++) {
      const { x, z } = body(k);
      const out = Math.max(LAND.x[0] - x, x - LAND.x[1], LAND.z[0] - z, z - LAND.z[1]);
      farthest = Math.max(farthest, out);
      const was = last[k];
      // Over a side, short of the slot's end, where thrown riders land and walk out
      if (Math.abs(z) < 0.15 && Math.abs(was.z) >= 0.15 && x > -0.9 && x < SLOT.x - 0.1) {
        onto.push(x);
      }
      last[k] = body(k);
    }
  }
  destroy();
  expect(farthest).toBeLessThan(0.2);
  // From beside the jelly near its fixed end, never near its free end
  expect(onto.length).toBeGreaterThanOrEqual(3);
  expect(Math.max(...onto)).toBeLessThan(-0.3);
});

it('brings critters set down off the lawn or past its front back onto it', () => {
  const { body, step, carry, destroy } = meadow();
  for (let i = 0; i < 240; i++) step();
  carry(2, [-2.6, 1.2]);
  carry(3, [2.95, 0.3]);
  for (let i = 0; i < 60 * 15; i++) step();
  for (const k of [2, 3]) {
    const { x, z, w } = body(k);
    expect(x).toBeGreaterThan(LAND.x[0] - 0.2);
    expect(x).toBeLessThan(LAND.x[1] + 0.2);
    expect(z).toBeLessThan(LAND.z[1] + 0.2);
    // Grown back in, at its full radius
    expect(w).toBeGreaterThan(0.03);
  }
  destroy();
});

it('lets no crowd linger at the approach with the jelly high at 0.9', () => {
  const { body, step, destroy } = meadow(RANGE[0] + (RANGE[1] - RANGE[0]) * 0.9);
  let crowded = 0;
  let longest = 0;
  for (let i = 0; i < 60 * 40; i++) {
    step();
    let near = 0;
    for (let k = 0; k < CRITTERS; k++) {
      const { x, z } = body(k);
      if (Math.hypot(x - LADDER, z) < 0.3) near++;
    }
    crowded = near >= 5 ? crowded + 1 : 0;
    longest = Math.max(longest, crowded);
  }
  destroy();
  expect(longest / 60).toBeLessThan(12);
});

/**
 * The fastest any rider rises over a sixtieth of a second while the slider is yanked, at `rate`
 * frames a second.
 */
function peakAt(rate: number) {
  const { critters, body, step, aim, destroy } = meadow(RANGE[1], rate);
  const frames = (seconds: number) => Math.round(seconds * rate);
  const riders = [3, 4, 5, 6, 7];
  for (let i = 0; i < frames(4); i++) step();
  let peak = 0;
  for (let round = 0; round < 2; round++) {
    aim(RANGE[1]);
    for (let i = 0; i < frames(1.5); i++) step();
    // Set down on the jelly, those another critter is in the way of picking left where they are
    riders.forEach((k, i) => {
      const { x, z } = body(k);
      if (!critters.pick([x, 3, z], [0, -1, 0])) return;
      for (let f = 0; f < frames(0.6); f++) {
        critters.hold([-0.85 + i * 0.18, 3, 0], [0, -1, 0]);
        step();
      }
      critters.drop();
    });
    for (let yank = 0; yank < 4; yank++) {
      aim(yank % 2 ? RANGE[1] : RANGE[0]);
      let last = riders.map((k) => body(k).y);
      for (let i = 1; i <= frames(0.7); i++) {
        step();
        if (i % (rate / 60)) continue;
        const now = riders.map((k) => body(k).y);
        now.forEach((y, j) => (peak = Math.max(peak, (y - last[j]) * 60)));
        last = now;
      }
    }
  }
  destroy();
  return peak;
}

it('throws riders as hard at 240 frames a second as at 60', () => {
  const at60 = peakAt(60);
  const at240 = peakAt(240);
  expect(at240).toBeLessThan(at60 * 1.3);
});

it('lets the lever spring back while the critter pulling it is picked up', () => {
  // Before the reveal, with the herald on its way to the lever
  const { world, critters, body, step, destroy } = meadow(RANGE[1], 60, -2.7);
  let i = 0;
  while (world.pulled < 0.3 && i++ < 600) step();
  expect(world.pulled).toBeGreaterThanOrEqual(0.3);
  const { x, y, z } = body(0);
  expect(critters.pick([x, y + 2, z], [0, -1, 0])).toBe(true);
  for (let f = 0; f < 90; f++) {
    critters.hold([1.4, 3, 0.3], [0, -1, 0]);
    step();
  }
  expect(world.pulled).toBeLessThan(0.05);
  destroy();
});
