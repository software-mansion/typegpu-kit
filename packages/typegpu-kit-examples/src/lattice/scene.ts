import {
  bloom,
  cameras,
  clock,
  finish,
  Particle,
  particles,
  passes,
  pipelines,
  taa,
  targets,
  tone,
  type Frame,
  type Rendered,
  type Surface,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Settings } from '../settings.ts';
import { createDepthView } from './depth.ts';
import { createLattice, HALF } from './lattice.ts';
import { createMotionView } from './motion.ts';
import { sky } from './sky.ts';

const EMBERS = 1500;
/** An ember's colour, linear and bright enough to bloom. */
const EMBER_COLOR: [number, number, number, number] = [5, 1.8, 0.5, 1];
/** World units across an ember. */
const EMBER_SIZE = 0.07;
/** World units a second an ember sways aside at most. */
const SWAY = 0.2;
/** Minimum and maximum rise speed in world units per second. */
const RISE = [0.3, 0.7] as const;
/** Camera clip planes in world units, which the depth view reads depth back with. */
const NEAR = 0.1;
const FAR = 500;
/** Each side drawn, as a fraction of the canvas's. */
const SCALES = { '100%': 1, '77%': 0.77, '67%': 0.67 };

export const defaults = {
  /** Smooths the beams' edges over several frames. */
  taa: true,
  /** Draws the scene at this share of each side of the canvas. */
  scale: '100%' as keyof typeof SCALES,
  /**
   * Resolves TAA at the canvas's size from the smaller picture. Off, TAA resolves at the drawn
   * size and the final pass stretches its picture.
   */
  upscale: true,
  /** Turns a second of the lattice. */
  speed: 0.15,
  /** Displays colour, TAA motion vectors or depth. */
  view: 'picture' as 'picture' | 'motion' | 'depth',
  sky: true,
  beams: true,
  embers: true,
  /** Chooses scene outputs for embers, or excludes their motion so underlying motion remains. */
  into: 'scene' as 'scene' | "scene.without('motion')",
  /**
   * Chooses rotation history or a plain uniform. History gives last-frame vertices the last-
   * frame angle. A uniform gives both frames the current angle.
   */
  turn: 'history' as 'history' | 'uniform',
};

/** Wraps rising embers from the top to the bottom to maintain the field. */
const rise = (ember: Particle) => {
  'use gpu';
  let next = Particle(ember);
  const sway = ember.age * 1.7 + ember.seed * 40;
  const rising = std.mix(RISE[0], RISE[1], ember.seed);
  next.velocity = d.vec3f(std.sin(sway) * SWAY, rising, std.cos(sway * 0.8) * SWAY);
  if (next.position.y > HALF) next.position.y = next.position.y - HALF * 2;
  return next;
};

/** Flickers embers, and fades them at the top and bottom to hide wrapping. */
const glowing = (ember: Particle, local: d.v2f) => {
  'use gpu';
  const flicker = 0.65 + std.sin(ember.age * 11 + ember.seed * 60) * 0.35;
  const edge = std.saturate((HALF - std.abs(ember.position.y)) * 1.5);
  return particles.softDisc(ember, local) * (flicker * edge);
};

/**
 * Draws rotating beams and rising embers over a sky. Passes write colour, depth and motion for
 * TAA to blend with the last frame.
 */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const values = { ...defaults, paused: false };
  // The motion and depth views read their targets texel for texel, so they show the full scale
  const screen = targets.create(
    root,
    surface,
    {
      hdr: { format: 'rgba16float', sampled: true },
      motion: { format: 'rg16float', sampled: true },
      depth: { format: 'depth24plus', sampled: true },
    },
    { resolution: () => (values.view === 'picture' ? SCALES[values.scale] : 1) },
  );
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0, 0],
    distance: 11,
    yaw: 0.6,
    pitch: 0.2,
    distanceRange: [4, 30],
    near: NEAR,
    far: FAR,
  });
  const sim = clock.fixedStep({ step: 1 / 60 });

  // One TAA resolves at the drawn size, the other up to the canvas's
  const resolved = {
    region: screen.regions.hdr,
    source: () => screen.textures.hdr,
    motion: () => screen.textures.motion,
    depth: () => screen.textures.depth,
  };
  const aa = taa.create(root, resolved);
  const up = taa.create(root, { ...resolved, size: surface });
  const picture: Rendered = {
    get output() {
      return up.enabled ? up.output : aa.output;
    },
    get region() {
      return up.enabled ? up.region : aa.region;
    },
  };
  // Only one TAA is on at a time, and the other's offset is zero
  const jitter = (position: d.v4f) => {
    'use gpu';
    return up.jitter(aa.jitter(position));
  };
  const scene = passes
    .create(
      root,
      screen,
      { hdr: passes.color, motion: taa.motion, depth: passes.depth },
      { jitter },
    )
    .pipe(view.inject());

  const lattice = createLattice(root);
  const embers = particles.create(root, view, {
    capacity: EMBERS,
    step: sim.step,
    update: rise,
    shade: glowing,
    settings: { gravity: [0, 0, 0], drag: 0 },
  });
  embers.burst({
    count: EMBERS,
    position: [0, 0, 0],
    area: [HALF, HALF, HALF],
    color: EMBER_COLOR,
    size: EMBER_SIZE,
    life: 1e9,
  });

  const beams = { history: scene.add(lattice.drawing), uniform: scene.add(lattice.plain) };
  // The sky leaves motion unwritten, so TAA reconstructs it from the far depth
  const skyDrawn = scene.add(sky);
  const rising = {
    scene: scene.add(embers.drawing),
    "scene.without('motion')": scene.without('motion').add(embers.drawing),
  };

  const glow = bloom.create(root, { source: picture });
  const final = finish.create(root, {
    format: surface.format,
    source: picture,
    tone: tone.aces,
    glow,
  });
  const motion = createMotionView(root, {
    format: surface.format,
    motion: () => screen.textures.motion,
  });
  const depth = createDepthView(root, view, {
    format: surface.format,
    depth: () => screen.textures.depth,
  });
  const outputs = { picture: final, motion, depth };

  await pipelines.compile([aa, up, scene, embers, glow, final, motion, depth]);

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      taa: { label: 'TAA' },
      scale: { label: 'render scale', options: Object.keys(SCALES) as (keyof typeof SCALES)[] },
      upscale: { label: 'upscale in TAA' },
      speed: { label: 'turns per second', min: 0, max: 1, step: 0.01 },
      view: { options: ['picture', 'motion', 'depth'] },
      paused: { label: 'embers paused' },
      sky: { folder: 'passes' },
      beams: { folder: 'passes' },
      embers: { folder: 'passes' },
      into: {
        label: 'ember outputs',
        folder: 'passes',
        options: ['scene', "scene.without('motion')"],
      },
      turn: { label: 'rotation source', folder: 'passes', options: ['history', 'uniform'] },
    },

    write() {
      // At full scale there is nothing to resolve up to
      const upscaling = values.upscale && values.scale !== '100%';
      aa.enable(values.taa && !upscaling);
      up.enable(values.taa && upscaling);
      sim.paused = values.paused;
    },
  };

  let turned = 0;

  return {
    view,
    settings,

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();

      if (values.speed !== 0) {
        turned += frame.dt * values.speed;
        lattice.turned.write(turned);
      }

      const compute = frame.encoder.beginComputePass();
      embers.spawn(compute);
      for (let steps = sim.update(frame.dt); steps > 0; steps--) embers.step(compute);
      embers.compact(compute, sim.ahead);
      compute.end();

      lattice.turned.record(frame);
      const pass = scene.begin(frame);
      if (values.beams) beams[values.turn].draw(pass);
      if (values.sky) skyDrawn.draw(pass);
      if (values.embers) rising[values.into].draw(pass);
      pass.end();

      aa.record(frame);
      up.record(frame);
      if (values.view === 'picture') glow.record(frame);

      outputs[values.view].record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, aa, up, lattice, embers, glow]) made.destroy();
    },
  };
}
