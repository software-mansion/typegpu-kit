import { clock, pipelines, targets, type Frame, type Surface, bloom, finish } from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createAgents } from './agents.ts';
import { createAgar, sheetAt, World } from './agar.ts';
import { createOats } from './oats.ts';
import { createTrail, TRAIL } from './trail.ts';
import { createVeins, VEINS } from './veins.ts';

const STEP = 1 / 60;
const HDR = 'rgba16float';
/** How much of the bloom the final pass adds at glow 1. */
const GLOW = 0.2;
/** Agent capacity, and the most the count setting reaches. */
const MOST = 1_000_000;
const DEGREES = Math.PI / 180;

export interface SceneOptions {
  /** Agents at the start. Default 400,000. */
  readonly count?: number;
  /** Trail grid cells along each side. Default 2048. */
  readonly grid?: number;
}

/** A Physarum slime mould whose agents follow and lay a trail, fed by oats a tap drops. */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions = {}) {
  const { count = 400_000, grid = 2048 } = options;
  const values = {
    count,
    sensorAngle: 30,
    sensorDistance: 35,
    turn: 45,
    speed: 1.2,
    deposit: 0.5,
    doubling: 2,
    endurance: 6,
    ...TRAIL,
    food: 100,
    ...VEINS,
    glow: 1,
    paused: false,
  };

  const sim = clock.fixedStep({ step: STEP });
  const world = root.createUniform(World, {
    size: d.vec2f(surface.width, surface.height),
    time: 0,
  });
  const screen = targets.create(root, surface, { hdr: { format: HDR, sampled: true } });
  const agar = createAgar(root, world, { format: HDR });
  const oats = createOats(root, { size: grid });
  const trail = createTrail(root, { size: grid });
  const agents = createAgents(root, trail, { capacity: MOST, food: oats.foodAt });
  const agentValues = () => ({
    count: values.count,
    sensorAngle: values.sensorAngle * DEGREES,
    sensorDistance: values.sensorDistance,
    turn: values.turn * DEGREES,
    speed: values.speed,
    deposit: values.deposit,
    hunger: STEP / values.endurance,
  });
  agents.set(agentValues());
  oats.set(values.food);
  const veins = createVeins(root, world, { trail, oats, format: HDR });
  const hdr = () => screen.textures.hdr;
  const glow = bloom.create(root, { source: hdr, settings: { threshold: 1, knee: 0.5 } });
  const final = finish.create(root, {
    format: surface.format,
    source: hdr,
    glow,
    settings: { glowIntensity: GLOW, vignette: 0.3 },
  });
  await pipelines.compile([agar, oats, trail, agents, veins, glow, final]);

  let scattering = true;
  /** Growth start time, in simulated seconds. */
  let born = 0;
  const regrow = () => {
    trail.clear();
    scattering = true;
    born = sim.time;
  };

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      count: { min: 50_000, max: MOST, step: 10_000, folder: 'mould' },
      sensorAngle: { min: 5, max: 90, step: 1, folder: 'mould', label: 'sensor angle (degrees)' },
      sensorDistance: {
        min: 2,
        max: 40,
        step: 0.5,
        folder: 'mould',
        label: 'sensor distance (cells)',
      },
      turn: { label: 'turn (degrees/step)', min: 0, max: 90, step: 1, folder: 'mould' },
      speed: { label: 'speed (cells/step)', min: 0.2, max: 4, step: 0.1, folder: 'mould' },
      doubling: { label: 'doubling time (s)', min: 0.5, max: 10, step: 0.1, folder: 'mould' },
      endurance: { label: 'endurance (s)', min: 1, max: 60, step: 1, folder: 'mould' },
      deposit: { min: 0, max: 2, step: 0.05, folder: 'trail' },
      decay: { min: 0, max: 0.5, step: 0.005, folder: 'trail' },
      blur: { min: 0, max: 1, step: 0.05, folder: 'trail' },
      food: { min: 0, max: 200, step: 1, folder: 'mould' },
      light: { min: 0, max: 2, step: 0.05, folder: 'look' },
      pulse: { min: 0, max: 2, step: 0.05, folder: 'look' },
      glow: { min: 0, max: 2, step: 0.05, folder: 'look' },
      paused: {},
    },
    write(key) {
      sim.paused = values.paused;
      agents.set(agentValues());
      // Changing the count resets growth to prevent agents overlapping
      if (key === 'count') regrow();
      trail.settings.write({ keep: 1 - values.decay, blur: values.blur });
      oats.set(values.food);
      veins.look.write({ light: values.light, pulse: values.pulse });
      final.settings.patch({ glowIntensity: GLOW * values.glow });
    },
    actions: { regrow },
  };

  const pointing: Pointing = {
    press(x, y) {
      const { width, height } = surface;
      oats.drop(sheetAt(d.vec2f(x * width, y * height), d.vec2f(width, height)));
    },
  };

  return {
    settings,
    pointing,

    frame(frame: Frame) {
      if (screen.fit()) world.patch({ size: d.vec2f(screen.width, screen.height) });

      const steps = sim.update(frame.dt);
      if (steps > 0) {
        world.patch({ time: sim.time });
        agents.grow(sim.time - born, values.doubling);
      }
      if (scattering) agents.scatter(frame.encoder);
      scattering = false;
      oats.paint(frame.encoder);
      for (let i = 0; i < steps; i++) {
        const pass = frame.encoder.beginComputePass();
        agents.move(pass);
        trail.spread(pass);
        pass.end();
      }

      const pass = frame.encoder.beginRenderPass({ colorAttachments: { view: screen.views.hdr } });
      agar.draw(pass);
      veins.draw(pass);
      pass.end();

      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, oats, trail, agents, glow]) made.destroy();
    },
  };
}
