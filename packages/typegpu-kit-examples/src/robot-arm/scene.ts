import {
  ao,
  bloom,
  cameras,
  clock,
  finish,
  fxaa,
  instances,
  lighting,
  materials,
  meshPool,
  passes,
  pipelines,
  shading,
  table,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { srgbToLinear } from '@typegpu/color';
import { clamp } from 'math';
import { quat, vec3, type Mat4, type Quat, type Vec3Arg } from 'wgpu-matrix';
import { d, std, tgpu, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createArm, HAND, LINKS, TIP, WRIST } from './arm.ts';
import { createBlocks, HOVER, type Block } from './blocks.ts';
import { createBench, Painted } from './bench.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
const TAU = Math.PI * 2;
const UP = [0, 1, 0] as const;

/** Minimum and maximum horizontal reach from the turntable axis. */
const REACH = [0.8, 2.7] as const;
/** The way the fingers point while reaching. */
const DOWN = [0, -1, 0] as const;
/** Automatic target orbit: `turn` rad/s at radius `out`, swinging by `swing` at `pace` rad/s. */
const WANDER = { turn: 0.45, out: 1.9, swing: 0.8, pace: 0.7 };
/** Target smoothing rate. After one second, `exp(-EASE)` of the gap remains. */
const EASE = 5;
/** Minimum downward y component of a unit ray that can target the bench. */
const FALLING = { grazing: 0.01 };
/** World units the ring under the goal floats over the bench. */
const RING_LIFT = 0.02;

const paint = (r: number, g: number, b: number) => srgbToLinear(d.vec3f(r, g, b));
const PAINT = materials.glossy(paint(0.95, 0.42, 0.08), 0.35);
const STEEL = materials.glossy(paint(0.6, 0.61, 0.64), 0.3);
const DARK = materials.glossy(paint(0.36, 0.37, 0.4), 0.38);
const MARKER = materials.glow(d.vec3f(0.25, 0.75, 1), 2.5);
const LED = materials.glow(d.vec3f(1, 0.55, 0.12), 8);
/** The dim workshop beyond the lamp's pool, which the fog matches. */
const ROOM = [0.012, 0.009, 0.007] as const;

/** How far occluders darken sky light: about the gap under a block's bevel. */
const CREASE = 0.25;

/** Darkens sky light in creases, such as between stacked blocks and round the arm's foot. */
const occludedSky = (material: shading.Material, point: shading.Point) => {
  'use gpu';
  return shading.hemisphere(material, point) * ao.at(point.position);
};

/** World radius of the bulb's disc, which sets how wide the lamp's shadows soften. */
const BULB_RADIUS = 0.04;
/** Shadow samples across the bulb's disc, on a sunflower spiral, as offsets along its two axes. */
const BULB_TAPS = Array.from({ length: 12 }, (_, i) => {
  const angle = i * 2.39996;
  const out = Math.sqrt((i + 0.5) / 12) * BULB_RADIUS;
  return [Math.cos(angle) * out, Math.sin(angle) * out] as const;
});

/** The lamp's shadow at a sample of the spiral, `offset` from `point` over its surface. */
const bulbTap = (
  point: shading.Point,
  offset: d.v3f,
  toLight: d.v3f,
  distance: number,
  slot: number,
) => {
  'use gpu';
  const moved = shading.Point(point);
  moved.position = point.position + offset;
  return shading.lampShadowed(moved, toLight, distance, slot);
};

/**
 * Point light through its cone, and the lamp's shadow averaged from several spots across its bulb,
 * so shadow edges soften into a penumbra.
 */
const softShadow = (
  point: shading.Point,
  toLight: d.v3f,
  distance: number,
  range: number,
  slot: number,
) => {
  'use gpu';
  const share = shading.falloff(distance, range) * shading.beamOf(toLight, slot);
  if (share <= 0) return d.f32(0);
  // Spread the samples over the surface, so none sinks below it into its own shadow
  const facing = std.cross(point.normal, toLight);
  const spare = std.cross(point.normal, d.vec3f(0.6, 0, 0.8));
  const side = std.normalize(std.select(facing, spare, std.length(facing) < 0.05));
  const across = std.cross(point.normal, side);
  let lit = d.f32(0);
  for (const tap of tgpu.unroll(BULB_TAPS)) {
    lit += bulbTap(point, side * tap[0] + across * tap[1], toLight, distance, slot);
  }
  return (share * lit) / BULB_TAPS.length;
};

/** Each block's paint brightness, so blocks of a colour stay apart. */
const TINTS = [1, 0.86, 1.1, 0.94, 1.04, 0.9];

/** The blocks' paints, and where each colour's tower stands. */
const COLOURS = [
  paint(0.85, 0.16, 0.1),
  paint(0.95, 0.72, 0.12),
  paint(0.12, 0.32, 0.7),
  paint(0.2, 0.52, 0.26),
];
const HOME = 1.6;
const HOMES = COLOURS.map((_, k) => {
  const angle = Math.PI / 4 + (k * Math.PI) / 2;
  return [Math.cos(angle) * HOME, Math.sin(angle) * HOME] as const;
});
/** A block's width, depth and height. */
const BLOCK = 0.42;
/**
 * Blocks by colour, bench angle and distance from the axis, the height they fall from to settle
 * onto any block below, and the radians they turn past their bench angle. Some start in their
 * towers, the rest loose between them.
 */
const BLOCKS: [colour: number, angle: number, out: number, lift: number, turn: number][] = [
  [0, Math.PI / 4, HOME, 0, 0],
  [0, Math.PI / 4, HOME, 1, 0.1],
  [0, 3.3, 1.45, 0, 0.5],
  [1, (3 * Math.PI) / 4, HOME, 0, 0],
  [1, 0.1, 1.85, 0, 0.3],
  [1, 4.6, 1.4, 0, 0.7],
  [2, (5 * Math.PI) / 4, HOME, 0, 0],
  [2, (5 * Math.PI) / 4, HOME, 1, 0.15],
  [2, 1.7, 1.9, 0, 0.6],
  [3, 0, 1.35, 0, 0.2],
  [3, 2.95, 1.9, 0, 0.4],
  [3, 6.1, 1.95, 1, 0.4],
];

type Triple = [number, number, number];

/** `angle` in radians wrapped to -π through π. */
const wrapped = (angle: number) => angle - Math.round(angle / TAU) * TAU;

/**
 * A toy robot arm on a workbench under a desk lamp, sorting painted blocks into a tower of each
 * colour. Its joints are solved on the CPU.
 */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
    ldr: { format: 'rgba8unorm', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 1.3, 0.1],
    distance: 5.8,
    yaw: 0.5,
    pitch: 0.46,
    pitchRange: [0.1, 1.4],
    distanceRange: [3, 18],
  });
  /** Half-resolution depth the solid drawings draw first, for the occlusion to read. */
  const early = targets.create(root, surface, {
    depth: { format: 'depth32float', scale: 0.5, sampled: true },
  });
  const prepass = passes
    .create(root, early, { depth: passes.depth }, { label: 'prepass' })
    .pipe(view.inject());
  const occlusion = ao.create(root, view, {
    depth: () => early.textures.depth,
    settings: { radius: CREASE },
  });
  const lamps = table.create(root, lighting.SpotLight, {
    capacity: 2,
    defaults: { ...lighting.WHITE_SPOT },
  });
  const light = lighting.create(root, {
    settings: {
      sunDirection: [0.6, 0.5, 0.6],
      sunColor: [0.55, 0.65, 0.9],
      sunIntensity: 0.25,
      skyTop: [0.07, 0.075, 0.09],
      skyBottom: [0.16, 0.09, 0.045],
      skyIntensity: 1,
      fogColor: ROOM,
      fogDensity: 0.02,
    },
    shading: {
      ambient: occludedSky,
      attenuation: softShadow,
    },
    points: { table: lamps },
    pointShadow: { count: 1, filter: 4, size: 1024 },
  });
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: [...ROOM, 1], shared: { ao: occlusion.part } },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const pool = meshPool.create(root, { capacity: 64 });
  const shape = {
    pedestal: pool.add(meshes.cylinder({ radius: 0.55, height: LINKS.pedestal })),
    housing: pool.add(
      meshes.roundedBox({ width: 0.62, height: LINKS.housing, depth: 0.7, radius: 0.08 }),
    ),
    hinge: pool.add(meshes.cylinder({ radius: 0.2, height: 0.44 })),
    upper: pool.add(meshes.capsule({ radius: 0.13, height: LINKS.upper })),
    fore: pool.add(meshes.capsule({ radius: 0.11, height: LINKS.fore })),
    palm: pool.add(meshes.box(HAND.palm)),
    finger: pool.add(meshes.box(HAND.finger)),
    ring: pool.add(meshes.torus({ radius: 0.24, tube: 0.025 })),
    block: pool.add(meshes.box({ width: 1, height: 1, depth: 1 })),
    toy: pool.add(meshes.roundedBox({ width: 1, height: 1, depth: 1, radius: 0.07 })),
  };

  const things = instances.create(root, pool, { capacity: 96 });
  const renderer = things.renderer();
  const bench = createBench(root, pool);
  const painted = instances.create(root, pool, { kind: Painted, capacity: BLOCKS.length });
  const drawings = [renderer, painted.renderer(), ...bench.drawings];
  const shown = drawings.map((drawing) => scene.add(drawing));
  const casters = drawings.map((drawing) => light.pointShadow.add(drawing));
  const shapes = drawings.map((drawing) => prepass.add(drawing));
  const lamp = lamps.add({
    ...bench.lamp,
    cone: lighting.cone(0.35, 0.85),
    color: [1, 0.78, 0.5],
    intensity: 55,
    range: 24,
  });
  /** A cool fill from over the viewer's shoulder, which keeps the arm's dark side in form. */
  lamps.add({
    position: [5, 6, 7],
    color: [0.6, 0.72, 1],
    intensity: 14,
    range: 30,
  });
  light.pointShadow.cast([lamp]);
  lamps.flush();

  const boxes = BLOCKS.map(([colour], i) =>
    painted.add(shape.toy, {
      scale: BLOCK,
      color: COLOURS[colour].mul(TINTS[i % TINTS.length]),
    }),
  );
  things.add(shape.pedestal, { position: [0, LINKS.pedestal / 2, 0], ...DARK });

  /** A mesh carried by `joint`, at `at` in its frame and turned by `turn`. */
  const attach = (
    joint: number,
    mesh: typeof shape.hinge,
    at: Triple,
    fields: Parameters<typeof things.add>[1],
    turn?: Quat,
  ) => ({ joint, at, turn, id: things.add(mesh, fields) });

  /** Lays a hinge's cylinder along its joint's axis. */
  const across = quat.fromAxisAngle([0, 0, 1], Math.PI / 2);
  const parts = [
    attach(0, shape.housing, [0, LINKS.housing / 2, 0], DARK),
    attach(0, shape.block, [0, 0.28, -0.355], { ...LED, scale: [0.2, 0.05, 0.02] }),
    attach(1, shape.hinge, [0, 0, 0], STEEL, across),
    attach(1, shape.upper, [0, LINKS.upper / 2, 0], PAINT),
    attach(2, shape.hinge, [0, 0, 0], { ...STEEL, scale: 0.8 }, across),
    attach(2, shape.fore, [0, LINKS.fore / 2, 0], PAINT),
    attach(WRIST, shape.hinge, [0, 0, 0], { ...STEEL, scale: 0.6 }, across),
    attach(WRIST, shape.palm, [0, HAND.palm.at, 0], DARK),
    attach(WRIST, shape.finger, [-HAND.open, HAND.finger.at, 0], STEEL),
    attach(WRIST, shape.finger, [HAND.open, HAND.finger.at, 0], STEEL),
  ];
  const fingers = parts.slice(-2);
  const ring = things.add(shape.ring, MARKER);

  const arm = createArm();
  const stack: Block[] = BLOCKS.map(([colour, angle, out, lift, turn]) => ({
    colour,
    size: BLOCK,
    height: BLOCK,
    x: Math.cos(angle) * out,
    y: lift,
    z: Math.sin(angle) * out,
    yaw: angle + turn,
  }));
  const blocks = createBlocks(stack, { tip: arm.tip, wrist: arm.frames[WRIST] }, HOMES);

  const glow = bloom.create(root, {
    source: () => screen.textures.hdr,
    settings: { threshold: 1.5 },
  });
  const final = finish.create(root, {
    format: 'rgba8unorm',
    source: () => screen.textures.hdr,
    glow,
    settings: { exposure: 1.1, glowIntensity: 0.1, vignette: 0.35 },
  });
  /** Smooths what MSAA leaves stepped: edges of the glowing ring, the bulb and bright highlights. */
  const smooth = fxaa.create(root, { format: surface.format, source: () => screen.textures.ldr });

  await pipelines.compile([prepass, occlusion, light, scene, glow, final, smooth]);

  const values = {
    /** Maximum joint speed in radians per second. */
    speed: 3,
    /** Cyclic-coordinate-descent iterations per frame. */
    sweeps: 4,
  };

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      speed: { label: 'joint speed (rad/s)', min: 0, max: 10, step: 0.1 },
      sweeps: { label: 'IK sweeps per frame', min: 1, max: 12, step: 1 },
    },
  };

  /** The bench point x, z the fingers head for: the pointer's, or one wandering round. */
  const aim: [x: number, z: number] = [arm.tip[0], arm.tip[2]];
  /** The smoothed fingertip goal: a turn about the axis, a distance from it and a height. */
  const goal = { turn: Math.atan2(aim[1], aim[0]), out: Math.hypot(...aim), y: HOVER };
  const wrist: Triple = [0, 0, 0];
  const previous = new Float64Array(arm.angles.length);
  let following = false;

  const ray = { origin: vec3.create(), direction: vec3.create() };
  /** Where `ray` meets the bench, written by `benchHit`. */
  const spot: Triple = [0, 0, 0];

  /** Casts `ray` through surface point x, y and returns whether it meets the bench at `spot`. */
  const benchHit = (x: number, y: number) => {
    view.rayAt(x, y, ray);
    return cameras.hitHeight(ray, 0, spot, FALLING) !== undefined;
  };

  const pointing: Pointing = {
    /** Targets the bench under the pointer. Ignores points above the horizon. */
    move(x, y) {
      if (!benchHit(x, y)) return;
      aim[0] = spot[0];
      aim[1] = spot[2];
      following = true;
    },

    /** Picks up the block under the pointer, or sets the held one down there. */
    press(x, y) {
      const onBench = benchHit(x, y);
      const hit = blocks.hit(ray.origin, ray.direction);
      const block = stack[hit];
      if (block && !blocks.holding) blocks.grab(hit);
      else if (block) blocks.drop(block.x, block.z);
      else if (onBench) blocks.drop(spot[0], spot[2]);
    },

    /** Resumes automatic target movement. */
    leave() {
      following = false;
    },
  };

  /** Moves the aim round the turntable on its own. */
  const wander = (time: number) => {
    const angle = time * WANDER.turn;
    const out = WANDER.out + Math.sin(time * WANDER.pace) * WANDER.swing;
    aim[0] = Math.cos(angle) * out;
    aim[1] = Math.sin(angle) * out;
  };

  /** Eases the goal toward `target`, swinging round the axis by the shorter way. */
  const follow = (target: Vec3Arg, dt: number) => {
    const ease = clock.ease(dt, EASE);
    goal.turn += wrapped(Math.atan2(target[2], target[0]) - goal.turn) * ease;
    const out = Math.hypot(target[0], target[2]);
    goal.out += (clamp(out, REACH[0], REACH[1]) - goal.out) * ease;
    goal.y += (target[1] - goal.y) * ease;
  };

  /** Places the wrist above the goal, fingers down, within each joint's speed limit. */
  const reach = (dt: number) => {
    const { angles } = arm;
    previous.set(angles);
    wrist[0] = Math.cos(goal.turn) * goal.out;
    wrist[1] = goal.y + TIP[1];
    wrist[2] = Math.sin(goal.turn) * goal.out;
    arm.solve(wrist, WRIST, values.sweeps);
    arm.aim(WRIST, DOWN);

    const most = values.speed * dt;
    for (let i = 0; i < angles.length; i++) {
      angles[i] = previous[i] + clamp(angles[i] - previous[i], -most, most);
    }
    arm.pose();
  };

  const position: Triple = [0, 0, 0];
  const rotation: [number, number, number, number] = [0, 0, 0, 1];
  const placed = { position, rotation };
  const ringAt: Triple = [0, RING_LIFT, 0];
  const marked = { position: ringAt };

  const place = (frame: Mat4, part: (typeof parts)[number]) => {
    vec3.transformMat4(part.at, frame, position);
    quat.fromMat(frame, rotation);
    if (part.turn) quat.multiply(rotation, part.turn, rotation);
    things.write(part.id, placed);
  };

  /** Writes the arm's parts, fingers `open` from the palm centre, and the ring under the wrist. */
  const placeArm = (open: number) => {
    fingers[0].at[0] = -open;
    fingers[1].at[0] = open;
    for (const part of parts) place(arm.frames[part.joint], part);
    ringAt[0] = wrist[0];
    ringAt[2] = wrist[2];
    things.write(ring, marked);
  };

  const placeBlocks = () => {
    for (let i = 0; i < stack.length; i++) {
      const block = stack[i];
      position[0] = block.x;
      position[1] = block.y + block.height / 2;
      position[2] = block.z;
      quat.fromAxisAngle(UP, block.yaw, rotation);
      painted.write(boxes[i], placed);
    }
  };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      view.update(frame.dt);
      early.fit();
      screen.fit();

      if (!following) {
        wander(frame.time);
        blocks.auto();
      }
      blocks.step(frame.dt, aim);
      follow(blocks.target, frame.dt);
      reach(frame.dt);
      placeArm(blocks.fingers);
      placeBlocks();

      things.flush();
      painted.flush();
      light.pointShadow.record(frame, casters);
      prepass.record(frame, shapes);
      occlusion.record(frame);
      scene.record(frame, shown);
      glow.record(frame);
      final.record(frame, screen.views.ldr);
      smooth.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [
        screen,
        early,
        occlusion,
        lamps,
        light,
        pool,
        things,
        painted,
        bench,
        glow,
      ]) {
        made.destroy();
      }
    },
  };
}
