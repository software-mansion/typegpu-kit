import {
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
import type { Picture } from '../contract.ts';
import { createPlatformer, HEARTS, STEP, type Score } from './game.ts';
import { createHud } from './hud.ts';
import { MAP, parseLevel } from './level.ts';
import { createAvatar } from './avatar.ts';
import { createCoins } from './coins.ts';
import { createCritters } from './critters.ts';
import { createEffects } from './effects.ts';
import { createLayers, LAYERS } from './layers.ts';
import { createPaper } from './paper.ts';
import { createScenery } from './scenery.ts';
import { grid, TEXELS } from './sprites.ts';

/** View height in tiles. */
const VIEW = 12;
/** Vertical camera offset above the hero, in tiles. */
const ABOVE = 1;
/** Fixed steps a frame may catch up on, 0.1 s at `STEP`. */
const MAX_STEPS = 12;
/** How fast the view follows the hero: the share left after a second is e^-FOLLOW. */
const FOLLOW = 8;
const BACKDROP = [LAYERS.clouds, LAYERS.far, LAYERS.near];

export interface SceneOptions {
  /** The hero's sheet, from `assets.ts`. */
  readonly hero: Picture;
  /** The sponge's sheet, from `assets.ts`. */
  readonly sponge: Picture;
  /** The bubble's sheet, from `assets.ts`. */
  readonly bubble: Picture;
  /** The fork's sheet, from `assets.ts`. */
  readonly fork: Picture;
  /** The health bar's raspberries, from `assets.ts`. */
  readonly raspberry: Picture;
  /** The level, one string per row, as `parseLevel` reads it. Default `MAP`. */
  readonly map?: readonly string[];
  /** Receives the score at startup and whenever it changes. */
  readonly onScore?: (score: Score) => void;
  /** Seconds at which a demo run, holding right, jumps. Default none, no demo. */
  readonly demo?: readonly number[];
}

/**
 * Draws a platformer's scenery into low-resolution layers, each on its own texel grid, and its
 * sprites at full resolution, each on whole pixels with texels of its own. Then prints them on
 * paper cards under a desk lamp, each layer moved by whole pixels. Coins and critters are listed
 * on the GPU, and dust and sparkles are particles.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const { map = MAP, onScore = () => {} } = options;
  const level = parseLevel(map);

  // Match the surface aspect ratio at a fixed texel height
  const pixels = {
    get width() {
      return Math.round((surface.width / surface.height) * VIEW * TEXELS);
    },
    height: VIEW * TEXELS,
  };
  const layers = createLayers(root, { size: pixels, surface });
  const scene = passes
    .create(root, layers.target, { color: passes.color })
    .pipe(layers.camera.inject());
  // Sprites glide over the layers by whole pixels of the surface
  const shown = targets.create(root, surface, { color: { format: 'rgba8unorm', sampled: true } });
  const front = passes
    .create(root, shown, { color: passes.color })
    .pipe(layers.screen.inject())
    .pipe((cfg) => cfg.with(grid, layers.grid));
  const [x, y] = level.start;
  const eye = { x, y };

  const time = root.createUniform(d.f32);
  const sim = clock.fixedStep({ step: STEP, maxSteps: MAX_STEPS });
  const scheduler = coroutines.create();

  const scenery = createScenery(root, time, level, layers);
  const avatar = createAvatar(root, { time, sheet: options.hero });
  const coins = createCoins(root, { time, capacity: level.coins.length, scheduler });
  const effects = createEffects(root, layers.screen, { step: sim.step, bubble: options.bubble });
  const critters = createCritters(root, {
    time,
    capacity: level.critters.length,
    sponge: options.sponge,
    fork: options.fork,
    step: STEP,
    avatar,
    dust: effects.dust,
    bubbles: effects.bubbles,
  });
  const backdrop = [scene.add(scenery.backdrop)];
  const tiles = [scene.add(scenery.drawing)];
  const sprites = [
    front.add(coins.drawing),
    front.add(critters.drawing),
    front.add(effects.bubbles.drawing),
    front.add(effects.dust.drawing),
    front.add(effects.sparks.drawing),
    front.add(avatar.drawing),
  ];

  // The paper is already in display colour space and in the palette, so skip tone mapping,
  // encoding and the vignette
  const sheet = targets.create(root, surface, { color: { format: 'rgba8unorm', sampled: true } });
  const print = passes.create(root, sheet, { color: passes.color });
  const paper = createPaper(root, { layers, sprites: () => shown.textures.color, size: pixels });
  const hud = createHud(root, { sheet: options.raspberry, hearts: HEARTS });
  // The health bar lies over the print, in its sheet's own colours
  const printed = [print.add(paper.drawing), print.add(hud.drawing)];
  const final = finish.create(root, {
    format: surface.format,
    source: () => sheet.textures.color,
    tone: null,
    encode: false,
    settings: { vignette: 0, dither: 0 },
  });

  await pipelines.compile([scene, front, coins, critters, effects, paper, print, final]);

  let score = '';
  let health = 2 * HEARTS;
  const game = createPlatformer(level, { coins, critters, effects, sim }, (state) => {
    health = state.health;
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
        down: hold('down'),
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
    /** The run: its hero, and whether the hero can be hurt. */
    game,
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
      layers.target.fit();
      shown.fit();
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
      avatar.update(hero, sim.alpha, game.safe, frame.time);
      // The raspberries keep whole pixels a texel, about the size of the art's
      const scale = Math.max(1, Math.round(surface.height / (VIEW * TEXELS)));
      hud.update(health, surface.width, surface.height, scale);
      critters.lead(sim.ahead);

      // Follow the hero where it is drawn, between steps
      const { position, previous } = hero;
      const heroX = previous[0] + (position[0] - previous[0]) * sim.alpha;
      const heroY = previous[1] + (position[1] - previous[1]) * sim.alpha;
      const halfWidth = pixels.width / TEXELS / 2;
      const follow = clock.ease(frame.dt, FOLLOW);
      eye.x += (clamp(heroX, halfWidth, level.width - halfWidth) - eye.x) * follow;
      eye.y += (clamp(heroY + ABOVE, VIEW / 2, level.height - VIEW / 2) - eye.y) * follow;
      layers.place(eye, frame.time);

      for (const layer of BACKDROP) scene.record(frame, backdrop, { layer });
      scene.record(frame, tiles, { layer: LAYERS.level });
      front.record(frame, sprites);
      paper.record(frame);
      print.record(frame, printed);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      scheduler.destroy();
      const made = [layers, shown, scenery, avatar, coins, critters, effects, sheet, paper, hud];
      for (const module of made) module.destroy();
    },
  };
}
