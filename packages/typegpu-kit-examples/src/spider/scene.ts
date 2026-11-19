import {
  bloom,
  cameras,
  finish,
  instances,
  lighting,
  materials,
  meshPool,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { quat, vec3, type Quat, type Vec3 } from 'math';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createTerrain } from './terrain.ts';
import { createWalker, GAIT } from './walker.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
const HAZE = [0.9, 0.74, 0.56] as const;
/** Maximum target distance from the ground centre. */
const ROAM = 13;
/**
 * Automatic wandering range of `out` world units either side of the centre along x and z.
 * Frequency is `pace` radians per second, with z one radian ahead.
 */
const WANDER = { out: [9, 8], pace: [0.13, 0.21], ahead: 1 };
/** How steeply a unit ray must fall, in its y, to target the ground, and how far it searches. */
const FALLING = { grazing: 0.01, reach: 100 };
/** World units the ring under the goal floats over the ground. */
const RING_LIFT = 0.03;
/** Tube axis in bone-local coordinates. */
const ALONG: Vec3 = [0, 1, 0];
/** Tube radii at the hip and foot ends of the coxa, femur, tibia and foot segments. */
const RADII = [
  [0.065, 0.058],
  [0.06, 0.046],
  [0.044, 0.03],
  [0.03, 0.004],
] as const;
/** The ankle lies this fraction along the solved tibia, raised `LIFT`. */
const ANKLE = 0.68;
const LIFT: Vec3 = [0, 0.07, 0];
/** Joint knob radius over the radius of the tube it starts. */
const KNOB = 1.05;

const SHELL = materials.glossy(d.vec3f(0.008, 0.007, 0.007), 0.16);
const LEG = materials.glossy(d.vec3f(0.012, 0.01, 0.009), 0.3);
const KNEE = materials.glossy(d.vec3f(0.1, 0.045, 0.012), 0.3);
const EYE = { ...materials.glossy(d.vec3f(0.004), 0.05), emission: d.vec3f(0.08, 0.015, 0.004) };
const MARKER = materials.glow(d.vec3f(1, 0.55, 0.15), 3);

const UPRIGHT: Quat = [0, 0, 0, 1];
/** The abdomen's tilt, nose down about x. */
const TILTED = quat.setAxisAngle(quat.create(), [1, 0, 0], 0.22);
/** A ball of `scale` at `at` in body coordinates, turned by `turn`. */
const piece = (at: Vec3, scale: number | Vec3, look = SHELL, turn = UPRIGHT) => ({
  at,
  scale,
  look,
  turn,
});
/** Balls mirrored across the body's middle. */
const pair = ([x, y, z]: Vec3, scale: number | Vec3, look = SHELL) => [
  piece([-x, y, z], scale, look),
  piece([x, y, z], scale, look),
];
/** Body parts in body coordinates with +z forward: carapace, waist, abdomen, fangs and eyes. */
const PARTS = [
  piece([0, 0.05, 0.04], [0.44, 0.17, 0.55]),
  piece([0, 0.18, 0.34], [0.09, 0.06, 0.1]),
  piece([0, 0.07, -0.53], 0.09),
  piece([0, 0.22, -1.2], [0.5, 0.4, 0.66], SHELL, TILTED),
  ...pair([0.09, -0.01, 0.56], [0.06, 0.09, 0.13]),
  ...pair([0.035, 0.225, 0.41], 0.028, EYE),
  ...pair([0.075, 0.215, 0.39], 0.016, EYE),
  ...pair([0.05, 0.242, 0.33], 0.015, EYE),
];

/** An open tube of unit length along +y, narrowing from radius `from` to `to`. */
const tube = (from: number, to: number) =>
  meshes.parametric(
    {
      at: (u, v) => {
        'use gpu';
        const turn = u * Math.PI * 2;
        const radius = std.mix(from, to, v);
        return d.vec3f(std.cos(turn) * radius, v - 0.5, -std.sin(turn) * radius);
      },
    },
    { cols: 12, rows: 2 },
  );

/**
 * Solves spider legs and body on the CPU with pmndrs/math. Draws instances under a sun while
 * walking toward the pointer or wandering.
 */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0.8, 0],
    distance: 9,
    yaw: 0.8,
    pitch: 0.55,
    pitchRange: [0.15, 1.4],
    distanceRange: [4, 20],
    damping: 4,
  });
  const light = lighting.create(root, {
    settings: {
      sunDirection: [-0.9, 0.36, 0.43],
      sunColor: [1, 0.72, 0.44],
      sunIntensity: 3.6,
      skyTop: [0.2, 0.34, 0.76],
      skyBottom: [0.14, 0.1, 0.07],
      skyIntensity: 0.75,
      fogColor: HAZE,
      fogDensity: 0.011,
    },
    shadow: { extent: 14 },
  });

  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: [...HAZE, 1] },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const pool = meshPool.create(root, { capacity: 64 });
  const ground = await createTerrain(root, pool);
  const { heightAt } = ground;
  const ball = pool.add(meshes.sphere({ radius: 1 }));
  const tubes = RADII.map(([from, to]) => pool.add(tube(from, to)));
  const ring = pool.add(meshes.torus({ radius: 0.3, tube: 0.03 }));

  const things = instances.create(root, pool, { capacity: 80 });
  const renderer = things.renderer();
  const shown = [scene.add(renderer), scene.add(ground.drawing)];
  const casters = [light.shadow.add(renderer)];

  const values = { ...GAIT };
  const walker = createWalker(heightAt, values);

  const parts = PARTS.map((part) => ({
    ...part,
    id: things.add(ball, { scale: part.scale, ...part.look }),
  }));

  const legs = walker.legs.map(({ chain }) => {
    const [coxa, femur, tibia] = chain.bones;
    const ankle = vec3.create();
    return {
      tibia: tibia!,
      ankle,
      joints: [coxa!.start, femur!.start, tibia!.start, ankle, tibia!.end],
      drawn: RADII.map(([from], i) => ({
        id: things.add(tubes[i], LEG),
        knob: things.add(ball, { scale: from * KNOB, ...KNEE }),
      })),
    };
  });
  const marker = things.add(ring, MARKER);

  const glow = bloom.create(root, { source: () => screen.textures.hdr });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    glow,
    settings: { exposure: 0.8, glowIntensity: 0.05 },
  });

  await pipelines.compile([light, scene, glow, final]);

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      speed: { min: 0.5, max: 3, step: 0.1 },
      stride: { min: 0.4, max: 1.2, step: 0.05 },
      step: { label: 'step duration (s)', min: 0.1, max: 0.5, step: 0.01 },
      lift: { min: 0.1, max: 1, step: 0.05 },
    },
  };

  const { body } = walker;
  const ray = { origin: vec3.create(), direction: vec3.create() };
  const position = vec3.create();
  const rotation: Quat = [0, 0, 0, 1];
  const posed = { position, rotation };
  const placed = { position };
  const stretch: Vec3 = [1, 1, 1];
  const stretched = { position, rotation, scale: stretch };
  const axis = vec3.create();
  const ringAt = vec3.create();
  const marked = { position: ringAt };
  let following = false;

  /** Walks toward ground point x, z, kept within `ROAM` of the centre. */
  const setGoal = (x: number, z: number) => {
    const out = Math.hypot(x, z);
    const kept = Math.min(1, ROAM / (out || 1));
    walker.goal[0] = x * kept;
    walker.goal[2] = z * kept;
  };

  const pointing: Pointing = {
    /** Targets ground under the pointer. Ignores points above the horizon. */
    move(x, y) {
      view.rayAt(x, y, ray);
      if (cameras.hitHeight(ray, heightAt, position, FALLING) === undefined) return;
      setGoal(position[0], position[2]);
      following = true;
    },

    /** Resumes automatic wandering. */
    leave() {
      following = false;
    },
  };

  /** Places each body part by the walker's body position and rotation. */
  const placeBody = () => {
    for (const part of parts) {
      vec3.add(position, vec3.transformQuat(position, part.at, walker.rotation), body);
      quat.multiply(rotation, walker.rotation, part.turn);
      things.write(part.id, posed);
    }
  };

  /** Stretches each leg segment's tube from one joint to the next, with a knob at its start. */
  const placeLegs = () => {
    for (const { tibia, ankle, joints, drawn } of legs) {
      vec3.add(ankle, vec3.lerp(ankle, tibia.start, tibia.end, ANKLE), LIFT);
      for (let i = 0; i < drawn.length; i++) {
        const from = joints[i];
        const to = joints[i + 1];
        vec3.copy(position, from);
        things.write(drawn[i].knob, placed);
        stretch[1] = vec3.length(vec3.sub(axis, to, from));
        quat.rotationTo(rotation, ALONG, vec3.normalize(axis, axis));
        vec3.lerp(position, from, to, 0.5);
        things.write(drawn[i].id, stretched);
      }
    }
  };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      if (!following) {
        const { out, pace, ahead } = WANDER;
        setGoal(
          Math.sin(frame.time * pace[0]) * out[0],
          Math.sin(frame.time * pace[1] + ahead) * out[1],
        );
      }
      walker.step(frame.dt);

      vec3.copy(view.goal.target, body);
      view.update(frame.dt);
      light.shadow.follow(body);

      placeBody();
      placeLegs();
      vec3.set(ringAt, walker.goal[0], walker.goal[1] + RING_LIFT, walker.goal[2]);
      things.write(marker, marked);

      screen.fit();
      things.flush();
      light.shadow.record(frame, casters);
      scene.record(frame, shown);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, light, pool, ground, things, glow]) made.destroy();
    },
  };
}
