import {
  cameras,
  clock,
  counters,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
  bloom,
  finish,
} from '@typegpu/kit';
import { d, std, type TgpuRoot } from 'typegpu';
import type { Pointing } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createCells, HEIGHT, WIDTH } from './cells.ts';
import { createRules, SPRINGS } from './rules.ts';
import { createCave, PAINTS, type Paint } from './cave.ts';
import { createEmbers } from './embers.ts';
import { createDensities } from './densities.ts';
import { createLight } from './light.ts';
import { createDrawing, type View } from './drawing.ts';

const STEP_SECONDS = 1 / 150;
const FIELD_HZ = 30;
/** Field updates after an edit, so light reflected off the walls settles. */
export const SETTLE_UPDATES = 12;
const COUNTERS_HZ = 4;
/** Simulation warm-up in seconds. Opens the cave with springs already pouring. */
const PREWARM_SECONDS = 3;
const PREWARM_CHUNK = 30;

/** Render pixels per cell. Higher resolution lets vapour move smoothly over discrete cells. */
const PICTURE_SCALE = 4;
const PICTURE = { width: WIDTH * PICTURE_SCALE, height: HEIGHT * PICTURE_SCALE };
const FIELD = { width: WIDTH, height: HEIGHT };
const ASPECT = WIDTH / HEIGHT;

const LOOK = { exposure: 1, glowIntensity: 0.2, vignette: 0.3 };
/** The diagnostic views' finish, without vignette or dither. */
const PLAIN = { vignette: 0, dither: 0 };
/** Where the tone curve reaches white, in linear radiance. */
const WHITE = 5;
const BLOOM = { levels: 5, settings: { threshold: 0.85 } };
/** Targets that later passes sample. */
const SAMPLED = { format: 'rgba16float', sampled: true } as const;

/** Per-channel Reinhard with a white point. Keeps dark colours and turns bright lava yellow. */
const reinhard = (color: d.v3f) => {
  'use gpu';
  return std.clamp((color * (1 + color / (WHITE * WHITE))) / (1 + color), d.vec3f(), d.vec3f(1));
};

/** Sand, water, lava, wood and crystal stepped cell by cell, lit through a radiance field. */
export async function createScene(root: TgpuRoot, surface: Surface) {
  const values = {
    paint: 'sand' as Paint,
    /** Brush radius in cells. */
    brush: 4,
    springs: SPRINGS,
    /** Feeds light reflected off walls and steam back into the next field update. */
    bounce: true,
    /** Displays the cave, linear radiance or field medium. */
    view: 'cave' as View,
    /** Pauses cells and embers. Edits still update the light field. */
    paused: false,
  };
  const counts = counters.create(root, ['cooled', 'burnt'], { hz: COUNTERS_HZ });
  const cells = createCells(root);
  const cave = createCave(root, cells);

  const view = cameras.flat(root, {
    size: PICTURE,
    height: HEIGHT,
    center: [WIDTH / 2, HEIGHT / 2],
  });
  const sim = clock.fixedStep({ step: STEP_SECONDS });
  const cadence = clock.cadence({ hz: FIELD_HZ });
  const seconds = root.createUniform(d.f32);
  const {
    sparks: embers,
    spark,
    sparked,
  } = createEmbers(root, {
    camera: view,
    step: STEP_SECONDS,
    cells,
    counts,
  });

  const rules = createRules(root, { cells, counts, spark, sparked });

  const glows = targets.create(root, FIELD, { emission: SAMPLED });
  const emission = passes
    .create(root, glows, { emission: passes.color }, { clear: [0, 0, 0, 0] })
    .pipe(view.inject());
  const emitting = emission.add({ ...embers.drawing, label: 'ember light' });

  const densities = createDensities(root, cells);
  const light = createLight(root, {
    cells,
    densities,
    embers: glows.textures.emission,
    seconds,
    bounce: values.bounce,
  });

  const screen = targets.create(root, PICTURE, { hdr: SAMPLED });
  const picture = createDrawing(root, { cells, light, densities, seconds });
  const scene = passes.create(root, screen, { hdr: passes.color }).pipe(view.inject());
  const cavern = scene.add(picture.drawing);
  const sparks = scene.add(embers.drawing);

  const hdr = () => screen.textures.hdr;
  const glow = bloom.create(root, { source: hdr, ...BLOOM });
  const finalOptions = { format: surface.format, source: hdr, filter: 'sharp' } as const;
  const final = finish.create(root, { ...finalOptions, tone: reinhard, glow, settings: LOOK });
  const diagnostic = finish.create(root, { ...finalOptions, tone: null, settings: PLAIN });

  let restarting = true;
  let settling = SETTLE_UPDATES;
  const settle = () => {
    settling = values.bounce ? SETTLE_UPDATES : 1;
  };

  const viewport = { x: 0, y: 0, width: surface.width, height: surface.height };
  const fit = () => {
    viewport.width = Math.min(surface.width, surface.height * ASPECT);
    viewport.height = viewport.width / ASPECT;
    viewport.x = (surface.width - viewport.width) / 2;
    viewport.y = (surface.height - viewport.height) / 2;
  };
  fit();

  let previous: readonly [number, number] | undefined;
  const pointing = {
    get reach() {
      return (values.brush * viewport.height) / (HEIGHT * surface.pixelRatio);
    },
    drag(x, y, start) {
      if (start) previous = undefined;
      fit();
      const across = (x * surface.width - viewport.x) / viewport.width;
      const down = (y * surface.height - viewport.y) / viewport.height;
      if (across < 0 || across > 1 || down < 0 || down > 1) {
        previous = undefined;
        return;
      }
      const at: readonly [number, number] = [across * WIDTH, down * HEIGHT];
      cave.stroke(previous ?? at, at, values.brush, values.paint);
      previous = at;
    },
  } satisfies Pointing;

  const settings: Settings<typeof values> = {
    values,
    knobs: {
      paint: { options: PAINTS },
      brush: { label: 'brush radius (cells)', min: 1, max: 16, step: 1 },
      springs: {},
      bounce: { label: 'reflected light' },
      view: { options: ['cave', 'radiance', 'medium'] },
      paused: {},
    },
    write(key) {
      sim.paused = values.paused;
      rules.controls.write({ springs: values.springs ? 1 : 0 });
      light.bounce.write(values.bounce ? 1 : 0);
      if (key === undefined || key === 'bounce') settle();
      picture.view(values.view);
    },
    actions: {
      reset() {
        restarting = true;
        previous = undefined;
        cave.reset();
      },
    },
    readouts: {
      'lava cooled': () => counts.latest?.cooled ?? 0,
      'wood set alight': () => counts.latest?.burnt ?? 0,
    },
  };

  await pipelines.compile([
    cells,
    cave,
    embers,
    rules,
    emission,
    densities,
    light,
    scene,
    glow,
    final,
    diagnostic,
  ]);

  const warm = root.createCommandEncoder();
  cave.record(warm);
  warm.submit();
  for (let left = PREWARM_SECONDS / STEP_SECONDS; left > 0; left -= PREWARM_CHUNK) {
    const chunk = root.createCommandEncoder();
    const pass = chunk.beginComputePass();
    for (let i = 0; i < PREWARM_CHUNK; i++) {
      rules.step(pass);
      embers.step(pass);
    }
    pass.end();
    chunk.submit();
  }

  return {
    settings,
    pointing,
    get settling() {
      return settling > 0;
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      fit();

      if (restarting) {
        counts.clear(frame);
        embers.clear(frame);
        restarting = false;
        settle();
      }
      if (cave.record(frame.encoder)) settle();

      const steps = sim.update(frame.dt);
      if (steps > 0) {
        seconds.write(sim.time);
      }

      const compute = frame.encoder.beginComputePass();
      for (let i = 0; i < steps; i++) {
        rules.step(compute);
        embers.step(compute);
      }
      embers.spawn(compute);
      embers.compact(compute, sim.ahead);
      cells.shape(compute);
      densities.measure(compute);
      compute.end();
      counts.capture(frame);

      const due = cadence.due(frame.dt);
      if (settling > 0 || (!values.paused && due)) {
        emission.record(frame, emitting);
        const gathering = frame.encoder.beginComputePass();
        light.solve(gathering);
        gathering.end();
        settling = Math.max(0, settling - 1);
      }

      const pass = scene.begin(frame);
      cavern.draw(pass);
      if (values.view === 'cave') sparks.draw(pass);
      pass.end();

      if (values.view === 'cave') glow.record(frame);

      const output = values.view === 'cave' ? final : diagnostic;
      output.record(frame, surface.view(frame), { viewport });
    },

    destroy() {
      for (const made of [
        counts,
        cells,
        cave,
        embers,
        rules,
        glows,
        densities,
        light,
        screen,
        glow,
      ]) {
        made.destroy();
      }
    },
  };
}
