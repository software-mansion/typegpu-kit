import {
  clock,
  finish,
  fxaa,
  pipelines,
  readback,
  targets,
  tone,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import type { TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import type { Bound } from '../ui.ts';
import { createClay, formatsOf } from './clay.ts';
import { SPRING } from './island.ts';
import { createRender, CHOICES } from './render.ts';
import type { ui } from './ui.ts';
import { createWater, STEP } from './water.ts';

/** Display-space target for FXAA after the final pass. */
const LDR = 'rgba8unorm';
/** A brush's centre off the surface, in radii: towards the eye to dig, above the ground to build. */
const LIFT = { dig: 0.45, build: 0.35 };
/** Metres a second a held brush moves the surface. */
const CARVE = 0.8;

/**
 * Claybook's clay and water on an island: an 8-bit SDF volume with a mip chain, ray traced in
 * compute, and SPH water on the neighbour grid. Drag to dig, or to raise clay while build is down.
 */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const formats = formatsOf(root);
  const clay = createClay(root, formats);
  // Alive now is at most the last count read back and every particle spawned since its frame
  const spawnedBy = new Map<number, number>();
  const live = () => {
    const since = spawnedBy.get(alive.latestFrame);
    return alive.latest === undefined || since === undefined
      ? spawned
      : alive.latest + spawned - since;
  };
  const water = createWater(root, clay, formats, live);
  const render = createRender(root, surface, { clay, water, formats });
  const { camera } = render;
  // The rays take one sample a pixel, so FXAA smooths the clay's and the water's silhouettes
  const display = targets.create(root, surface, { ldr: { format: LDR, sampled: true } });
  const final = finish.create(root, {
    format: LDR,
    source: render.hdr,
    tone: tone.neutral,
    settings: { exposure: 0.85, vignette: 0.2 },
  });
  const smooth = fxaa.create(root, { format: surface.format, source: () => display.textures.ldr });
  const sim = clock.fixedStep({ step: STEP, maxSteps: 2 });
  const alive = readback.create(root, water.sim.buffer, { hz: 4, decode: (s) => s.alive });

  const values = {
    ...CHOICES,
    spring: true,
  };

  let restarting = true;
  let baked = false;
  /** Steps since the water was reset, which the GPU counts too. */
  let stepped = 0;
  /** Particles the spring owes, carried between frames, and those it has given. */
  let owed = 0;
  let spawned = 0;
  /** The brush mode that edited last frame, to tell when a stroke starts. */
  let stroke = 0;

  const cursor = { x: 0, y: 0, over: false, down: false };
  let building = false;
  /**
   * A stroke's first frame only traces: the brush aims at the last traced hit, which until this
   * frame's trace lies wherever the pointer was before it landed.
   */
  let landing = false;

  const pointing = {
    move(x, y) {
      Object.assign(cursor, { x, y, over: true });
    },
    leave() {
      cursor.over = false;
    },
    drag(x, y, start) {
      Object.assign(cursor, { x, y, over: true, down: true });
      if (start) landing = true;
    },
    end() {
      cursor.down = false;
    },
  } satisfies Pointing;

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      radius: { label: 'brush radius (m)', min: 0.3, max: 1.4, step: 0.05 },
      spring: {},
      shadows: { label: 'soft shadows' },
      occlusion: { label: 'ambient occlusion' },
    },
    write() {
      render.choices.write({
        radius: values.radius,
        shadows: values.shadows ? 1 : 0,
        ao: values.occlusion ? 1 : 0,
      });
    },
    actions: {
      reset() {
        restarting = true;
      },
    },
    readouts: {
      particles: () => alive.latest ?? 0,
    },
  };

  await pipelines.compile([clay, water, render, final, smooth]);

  return {
    view: camera,
    settings,
    pointing,
    ui: {
      controls: {
        build: {
          press: () => (building = true),
          release: () => (building = false),
        },
      },
    } satisfies Bound<typeof ui>,

    frame(frame: Frame) {
      camera.update(frame.dt);
      const screen = render.fit();

      if (!baked) render.bake(frame.encoder);
      baked = true;
      if (restarting) {
        water.reset(frame.encoder);
        alive.reset();
        stepped = 0;
        owed = 0;
        spawned = 0;
      }

      // The spring wells up at its rate of simulated time, in each frame's first step, until the
      // last count read back reaches its limit
      const steps = sim.update(frame.dt);
      if (steps > 0) {
        owed += SPRING.rate * steps * STEP;
        const due = Math.floor(owed);
        owed -= due;
        const room = Math.max(SPRING.limit - (alive.latest ?? 0), 0);
        const count = values.spring ? Math.min(due, room) : 0;
        water.spawn.write({ count, first: stepped, total: spawned });
        spawned += count;
        stepped += steps;
      }

      // A stroke starts when the brush switches on or changes mode
      const mode = cursor.down && !landing ? (building ? 2 : 1) : 0;
      const fresh = mode !== 0 && mode !== stroke;
      stroke = mode;
      landing = false;
      if (mode !== 0) {
        clay.brush.write({
          radius: values.radius,
          blend: values.radius * 0.35,
          mode,
          lift: mode === 1 ? LIFT.dig : LIFT.build,
          reach: CARVE * Math.min(frame.dt, 1 / 30),
          fresh: fresh ? 1 : 0,
        });
      }
      render.view.write({
        size: [screen.width, screen.height],
        cursor: cursor.over
          ? [Math.floor(cursor.x * screen.width), Math.floor(cursor.y * screen.height)]
          : [-1, -1],
        mode: mode !== 0 ? mode : cursor.over ? 3 : 0,
      });

      // Every kernel of the frame in one compute pass: usage is checked per dispatch
      const pass = frame.encoder.beginComputePass();
      if (restarting) clay.generate(pass);
      // The brush follows the cursor's hit from last frame's trace
      if (mode !== 0) clay.edit(pass, fresh && mode === 2);
      for (let i = 0; i < steps; i++) water.step(pass);
      if (steps > 0 || restarting) water.surface(pass);
      render.record(pass);
      pass.end();
      display.fit();
      final.record(frame, display.views.ldr);
      smooth.record(frame, surface.view(frame));
      alive.capture(frame);
      spawnedBy.set(frame.index, spawned);
      for (const index of spawnedBy.keys()) {
        if (index < alive.latestFrame) spawnedBy.delete(index);
      }
      restarting = false;
    },

    destroy() {
      for (const made of [clay, water, render, display, alive]) made.destroy();
    },
  };
}
