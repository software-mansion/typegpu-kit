import {
  bloom,
  cameras,
  clock,
  finish,
  passes,
  pipelines,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { perlin3d } from '@typegpu/noise';
import { vec3n } from 'wgpu-matrix';
import { d, type Configurable, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createSwarm } from './swarm.ts';
import { MAX } from './jelly.ts';
import { water } from './tank.ts';
import { snow } from './snow.ts';
import { createBubbles } from './bubbles.ts';
import { Look, lookOf, Stir, stirOf, timeOf } from './water.ts';

type Vec3 = [number, number, number];
const zero = (): Vec3 => [0, 0, 0];

/** Where the camera looks, and from how far. */
const MIDDLE: Vec3 = [0, 0.15, 0];
const DISTANCE = 3;
const PITCH = -0.06;
/** World units of push a metre of drag gives, its most, and how fast it settles a second. */
const STIR = { gain: 1.4, most: 2.5, settle: 1.8 };
/** Steps the jellies take before the first frame, so the tank opens with its strands settled. */
const SETTLING_STEPS = 240;

/** A lit tank of jellies, drawn translucent without sorting, with bloom and a final pass. */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const values = {
    /** How many jellies swim in the tank. */
    count: 12,
    /** How strongly dragging stirs the water. */
    current: 1,
    /** Brightness of the jellies' own rim glow. */
    glow: 1,
    /** Draws the jellies through the translucency targets, or blended over in draw order. */
    oit: true,
  };
  const screen = targets.create(root, surface, {
    hdr: { format: 'rgba16float', sampled: true },
    accum: { format: 'rgba16float', sampled: true },
    reveal: { format: 'r16float', sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: MIDDLE,
    distance: DISTANCE,
    pitch: PITCH,
    fov: 0.85,
    distanceRange: [2, 9],
    damping: 3,
  });

  const time = root.createUniform(d.f32);
  const lookValues = () => ({ glow: values.glow, current: values.current });
  const look = root.createUniform(Look, lookValues());
  const stir = root.createUniform(Stir);
  const noise = perlin3d.staticCache({ root, size: d.vec3u(32, 32, 32) });
  const fill = (cfg: Configurable) =>
    noise.inject()(cfg).with(timeOf, time).with(lookOf, look).with(stirOf, stir);

  const layered = passes
    .create(root, screen, {
      hdr: passes.color,
      accum: passes.accumulation,
      reveal: passes.revealage,
    })
    .pipe(view.inject())
    .pipe(fill);
  const ordered = passes.create(root, screen, { hdr: passes.color }).pipe(view.inject()).pipe(fill);

  const sim = clock.fixedStep({ step: 1 / 60, maxSteps: 4 });
  const swarm = createSwarm(root, view, {
    size: surface,
    water: fill,
    step: sim.step,
    count: values.count,
  });
  const { bubbles, vent } = createBubbles(root, view, { step: sim.step, water: fill });

  /** Adds the tank's drawings, in draw order, to either description. */
  const tankIn = (description: typeof ordered) => [
    description.add(water),
    description.add(snow),
    description.add(swarm.tissue.tentacles),
    description.add(swarm.tissue.arms),
    description.add(swarm.tissue.bells),
    description.add(bubbles.drawing),
  ];
  const drawings = { layered: tankIn(layered), ordered: tankIn(ordered) };

  const glow = bloom.create(root, {
    source: () => screen.textures.hdr,
    settings: { threshold: 1.1, spread: 1.2 },
  });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    tone: tone.aces,
    settings: { exposure: 1.15, glowIntensity: 0.18, vignette: 0.3 },
    glow,
  });
  await pipelines.compile([layered, swarm, bubbles, glow, final]);

  // Frames keep drawing `layered` until the alternate blend pipelines compile
  let orderedReady = false;
  let ordering: Promise<void> | undefined;
  const compileOrdered = () =>
    (ordering ??= pipelines.compile([ordered]).then(() => {
      orderedReady = true;
    }));

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      count: { min: 1, max: MAX, step: 1 },
      current: { min: 0, max: 2, step: 0.05 },
      glow: { min: 0, max: 2, step: 0.05 },
      oit: { label: 'order-independent' },
    },
    async write() {
      swarm.show(values.count);
      look.write(values);
      if (!values.oit) await compileOrdered();
    },
  };

  const ray = { origin: zero(), direction: zero() };
  const toward = zero();
  const normal = zero();
  const point = zero();
  const motion = zero();
  /** Where the stir is: the drag's latest point. */
  const at = zero();
  const push = zero();
  let stirring = false;

  /** Writes where the ray through x, y meets the plane through `MIDDLE` facing the eye. */
  const planeAt = (x: number, y: number, out: Vec3) => {
    view.rayAt(x, y, ray);
    vec3n.sub(MIDDLE, ray.origin, toward);
    vec3n.normalize(toward, normal);
    const t = vec3n.dot(toward, normal) / vec3n.dot(ray.direction, normal);
    vec3n.addScaled(ray.origin, ray.direction, t, out);
  };

  /** A tap startles the jelly under it; a drag pushes the water along the plane `planeAt` meets. */
  const pointing: Pointing = {
    press: swarm.tap,
    drag(x, y, start) {
      planeAt(x, y, point);
      if (!start) {
        vec3n.addScaled(push, vec3n.sub(point, at, motion), STIR.gain, push);
        const size = vec3n.length(push);
        if (size > STIR.most) vec3n.mulScalar(push, STIR.most / size, push);
      }
      vec3n.copy(point, at);
      stirring = true;
    },
  };

  let elapsed = 0;
  let unsettled = SETTLING_STEPS;

  /** Sways the view slowly about wherever the host's orbit left it. */
  const sway = (dt: number) => {
    const yaw = Math.sin(elapsed * 0.15) * 0.22;
    const pitch = Math.sin(elapsed * 0.11) * 0.04;
    elapsed += dt;
    view.goal.yaw += Math.sin(elapsed * 0.15) * 0.22 - yaw;
    view.goal.pitch += Math.sin(elapsed * 0.11) * 0.04 - pitch;
  };

  /** Lets the stir's push die away, writing it until it has. */
  const settleStir = (dt: number) => {
    if (!stirring) return;
    vec3n.mulScalar(push, Math.exp(-dt * STIR.settle), push);
    stir.write({ at, push });
    stirring = vec3n.length(push) > 1e-4;
  };

  return {
    view,
    settings,
    pointing,

    frame(frame: Frame) {
      sway(frame.dt);
      view.update(frame.dt);
      screen.fit();

      time.write(elapsed);
      settleStir(frame.dt);

      const compute = frame.encoder.beginComputePass();
      vent(frame.dt);
      bubbles.spawn(compute);
      swarm.startle(compute);
      for (; unsettled > 0; unsettled--) swarm.step(compute);
      for (let steps = sim.update(frame.dt); steps > 0; steps--) {
        swarm.step(compute);
        bubbles.step(compute);
      }
      bubbles.compact(compute, sim.ahead);
      compute.end();

      if (values.oit || !orderedReady) {
        layered.record(frame, drawings.layered);
        layered.composite(frame);
      } else {
        ordered.record(frame, drawings.ordered);
      }

      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, noise, swarm, bubbles, glow]) made.destroy();
    },
  };
}
