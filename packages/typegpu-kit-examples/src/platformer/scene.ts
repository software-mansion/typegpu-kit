import {
  cameras,
  clock,
  coroutines,
  finish,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { clamp } from 'math';
import { d, type TgpuRoot } from 'typegpu';
import { createPlatformer, STEP, type Score } from './game.ts';
import { MAP, parseLevel } from './level.ts';
import { createAvatar } from './avatar.ts';
import { createCoins } from './coins.ts';
import { createCritters } from './critters.ts';
import { createEffects } from './effects.ts';
import { createPaper } from './paper.ts';
import { createScenery } from './scenery.ts';
import { TEXELS } from './sprites.ts';

/** View height in tiles. */
const VIEW = 12;
/** Vertical camera offset above the hero, in tiles. */
const ABOVE = 1;
/** Fixed steps a frame may catch up on, 0.1 s at `STEP`. */
const MAX_STEPS = 12;

export interface SceneOptions {
  /** The level, one string per row, as `parseLevel` reads it. Default `MAP`. */
  readonly map?: readonly string[];
  /** Receives the score at startup and whenever it changes. */
  readonly onScore?: (score: Score) => void;
  /** Seconds at which a demo run, holding right, jumps. Default none, no demo. */
  readonly demo?: readonly number[];
}

/**
 * Draws a platformer's pixel art into a low-resolution target, then prints it on paper cards under
 * a desk lamp at full resolution. Coins and critters are listed on the GPU, and dust and sparkles
 * are particles.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions = {}) {
  const { map = MAP, onScore = () => {} } = options;
  const level = parseLevel(map);

  // Match the surface aspect ratio at a fixed texel height. An even width keeps each pixel's centre
  // inside a texel, since the camera snaps its centre to the texel grid
  const pixels = {
    get width() {
      return 2 * Math.round((surface.width / surface.height) * VIEW * TEXELS * 0.5);
    },
    height: VIEW * TEXELS,
  };
  const view = cameras.flat(root, {
    size: pixels,
    height: VIEW,
    snap: 1 / TEXELS,
    center: level.start,
  });
  const art = targets.create(root, pixels, { color: { format: 'rgba8unorm', sampled: true } });
  const scene = passes.create(root, art, { color: passes.color }).pipe(view.inject());

  const time = root.createUniform(d.f32);
  const sim = clock.fixedStep({ step: STEP, maxSteps: MAX_STEPS });
  const scheduler = coroutines.create();

  const scenery = createScenery(root, time, level);
  const avatar = createAvatar(root, { time });
  const coins = createCoins(root, { time, capacity: level.coins.length, scheduler });
  const effects = createEffects(root, view, { step: sim.step });
  const critters = createCritters(root, {
    time,
    capacity: level.critters.length,
    step: STEP,
    avatar,
    dust: effects.dust,
  });
  const drawn = [
    scene.add(scenery.drawing),
    scene.add(coins.drawing),
    scene.add(critters.drawing),
    scene.add(effects.dust.drawing),
    scene.add(effects.sparks.drawing),
    scene.add(avatar.drawing),
  ];

  // The paper is already in display colour space, so skip tone mapping and encoding
  const sheet = targets.create(root, surface, { color: { format: 'rgba8unorm', sampled: true } });
  const print = passes.create(root, sheet, { color: passes.color });
  const paper = createPaper(root, { art: () => art.textures.color, size: pixels });
  const printed = [print.add(paper.drawing)];
  const final = finish.create(root, {
    format: surface.format,
    source: () => sheet.textures.color,
    tone: null,
    encode: false,
    settings: { vignette: 0.18, dither: 0 },
  });

  await pipelines.compile([scene, coins, critters, effects, paper, print, final]);

  let score = '';
  const game = createPlatformer(level, { coins, critters, effects, sim }, (state) => {
    const record = state.best === undefined ? '' : `\nbest ${state.best.toFixed(1)} s`;
    score = `coins ${state.coins} · stomps ${state.stomps}${record}`;
    onScore(state);
  });
  const { hero, input } = game;
  const hold = (key: keyof typeof input) => ({
    press: () => (input[key] = true),
    release: () => (input[key] = false),
  });

  return {
    ui: {
      score: () => score,
      controls: {
        left: hold('left'),
        right: hold('right'),
        jump: {
          press() {
            hero.jump();
            input.jump = true;
          },
          release: () => (input.jump = false),
        },
      },
    },

    /** Held inputs, set by the buttons `ui` binds. */
    input,
    hero,
    coins,
    critters,

    frame(frame: Frame) {
      if (options.demo) {
        input.right = true;
        const t = frame.time;
        const held = options.demo.some((at) => t >= at && t < at + 0.3);
        if (held && !input.jump) hero.jump();
        input.jump = held;
      }
      time.write(frame.time);
      scheduler.run(frame);
      art.fit();
      sheet.fit();

      coins.table.flush();
      critters.table.flush();

      const compute = frame.encoder.beginComputePass();
      game.update(frame, compute);
      effects.spawn(compute);
      coins.compact(compute);
      critters.compact(compute);
      effects.compact(compute, sim.ahead);
      compute.end();
      critters.contacts.capture(frame);

      // Queue writes precede frame commands, so every critter step reads the hero's whole path
      avatar.update(hero, sim.alpha, game.safe);

      const halfWidth = pixels.width / TEXELS / 2;
      view.goal.x = clamp(hero.position[0], halfWidth, level.width - halfWidth);
      view.goal.y = clamp(hero.position[1] + ABOVE, VIEW / 2, level.height - VIEW / 2);
      view.update(frame.dt);

      scene.record(frame, drawn);
      paper.record(frame);
      print.record(frame, printed);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      scheduler.destroy();
      for (const made of [art, scenery, coins, critters, effects, sheet, paper]) made.destroy();
    },
  };
}
