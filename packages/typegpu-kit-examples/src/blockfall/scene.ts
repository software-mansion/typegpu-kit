import {
  ao,
  bloom,
  cameras,
  clock,
  coroutines,
  finish,
  lighting,
  passes,
  pipelines,
  pointShadow,
  shading,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { d, type TgpuRoot } from 'typegpu';
import { CELLS, createBlockfall, DEPTH, HEIGHT, PIECE, WIDTH, type Score } from './game.ts';
import { createBlocks } from './blocks.ts';
import { createShards } from './shards.ts';
import { createControls } from './controls.ts';
import { backdrop, createStage, cube, STAGE } from './stage.ts';
import { createWell } from './well.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
/** Well centre and shadow centre. */
const MIDDLE = [WIDTH / 2, HEIGHT / 2, DEPTH / 2] as const;
/** Seconds the glow takes to settle after blocks break, and its added intensity. */
const SWELL = { seconds: 0.7, intensity: 0.3 };
/** Camera target a little below the middle, toward the block pile. */
const TARGET = [WIDTH / 2, HEIGHT / 2 - 0.4, DEPTH / 2] as const;
/** How far occluders darken sky light, in cells: about a block, for the creases between them. */
const CREASE = 0.8;

/** Darkens the sky light in creases, between blocks and where they meet the floor. */
const occludedSky = (material: shading.Material, point: shading.Point) => {
  'use gpu';
  return shading.hemisphere(material, point) * ao.at(point.position);
};

export interface SceneOptions {
  /** Receives the score at startup and after each clear. */
  readonly onScore?: (score: Score) => void;
}

/**
 * Draws 3D Blockfall in a glass well floating in the dark, under a warm key and a cool fill, with
 * MSAA, ambient occlusion, stone chips, bloom and a final pass. Its `ui` binds the buttons `ui.ts`
 * declares to `game` and `view`.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions = {}) {
  const screen = targets.create(root, surface, {
    color: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: TARGET,
    distance: 21,
    pitch: 0.4,
    yaw: 0,
    pitchRange: [0.15, 1.45],
    distanceRange: [12, 32],
    far: 200,
  });
  const time = root.createUniform(d.f32);
  const effects = clock.fixedStep({ step: 1 / 60, maxSteps: 4 });
  const blocks = createBlocks(root, {
    time,
    capacity: CELLS + PIECE,
    ghosts: PIECE,
    plinth: {
      at: [WIDTH / 2, -0.32, DEPTH / 2],
      size: [WIDTH + 1, 0.6, DEPTH + 1],
      color: STAGE.plinth,
    },
    grid: [WIDTH, DEPTH],
    scenery: [cube],
  });
  const stage = createStage(root, view, { step: effects.step });

  /** Half-resolution depth the opaque drawings draw first, for the occlusion to read. */
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

  const light = lighting.create(root, {
    settings: STAGE.settings,
    shading: { ambient: occludedSky },
    shadow: { size: 2048, extent: 16, center: MIDDLE, filter: 9 },
    points: { table: stage.lamps },
    pointShadow: pointShadow.define({ count: 1, size: 1024, filter: 9 }),
  });
  light.pointShadow.cast([stage.key]);
  const scene = passes
    .create(
      root,
      screen,
      { color: passes.color, depth: passes.depth },
      { resolve: { color: 'hdr' }, shared: { ao: occlusion.part } },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const shards = createShards(root, {
    camera: view,
    step: effects.step,
    time,
    pool: blocks.pool,
    chip: blocks.sceneryMeshes[0],
  });
  // The backdrop fills what the blocks leave. The well, ghosts, embers and dust blend over them
  const drawn = [
    scene.add(blocks.solid),
    scene.add(shards.drawing),
    scene.add(backdrop),
    scene.add(createWell([WIDTH, HEIGHT, DEPTH])),
    scene.add(blocks.ghosts),
    scene.add(stage.embers.drawing),
    scene.add(shards.dust.drawing),
  ];
  const shapes = [prepass.add(blocks.solid), prepass.add(shards.drawing)];
  const casters = [light.shadow.add(blocks.solid), light.shadow.add(shards.drawing)];
  const lit = [light.pointShadow.add(blocks.solid), light.pointShadow.add(shards.drawing)];
  const stirred = () => !blocks.still(now) || shards.flying(now);

  const glow = bloom.create(root, { source: () => screen.textures.hdr, settings: STAGE.bloom });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    glow,
    settings: STAGE.finish,
  });

  await pipelines.compile([prepass, occlusion, light, scene, glow, final]);

  let now = 0;
  let brokeAt = -Infinity;
  let swelling = false;
  let score = '';
  const scheduler = coroutines.create();
  const game = createBlockfall({
    blocks,
    scheduler,
    onScore(state) {
      score = `${state.score}\nlines ${state.lines} · level ${state.level}`;
      options.onScore?.(state);
    },
    onBreak(x, y, z, kind) {
      shards.break(x, y, z, kind, now);
      brokeAt = now;
    },
  });

  /** Swells the glow as blocks break, settling over `SWELL.seconds`. */
  const swell = () => {
    const left = Math.max(0, 1 - (now - brokeAt) / SWELL.seconds);
    if (left === 0 && !swelling) return;

    final.settings.patch({
      glowIntensity: finish.defaults.glowIntensity + SWELL.intensity * left * left,
    });
    swelling = left > 0;
  };

  return {
    view,
    ui: { score: () => score, controls: createControls(game, view) },
    game,

    frame(frame: Frame) {
      time.write(frame.time);
      game.update(frame);
      now = frame.time;
      scheduler.run(frame);
      swell();

      view.update(frame.dt);
      screen.fit();
      early.fit();
      blocks.update(frame);
      shards.chips.flush();

      const compute = frame.encoder.beginComputePass();
      for (let steps = effects.update(frame.dt); steps > 0; steps--) {
        shards.dust.step(compute);
        stage.embers.step(compute);
      }
      shards.dust.spawn(compute);
      stage.embers.spawn(compute);
      shards.dust.compact(compute, effects.ahead);
      stage.embers.compact(compute, effects.ahead);
      compute.end();

      prepass.record(frame, shapes);
      occlusion.record(frame);
      // Shadows hold while the blocks do, and the scenery never moves
      light.shadow.record(frame, casters, { redraw: stirred });
      light.pointShadow.record(frame, lit, { redraw: stirred });
      scene.record(frame, drawn);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      scheduler.destroy();
      for (const made of [screen, early, blocks, stage, shards, occlusion, light, glow]) {
        made.destroy();
      }
    },
  };
}
