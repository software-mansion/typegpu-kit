import { clock, lighting, meshPool, particles, tone, type Frame, type Surface } from '@typegpu/kit';
import { type TgpuRoot } from 'typegpu';
import { createDominoes, LAYOUTS } from 'typegpu-kit-examples/dominoes/dominoes.ts';
import { createFloor } from 'typegpu-kit-examples/dominoes/floor.ts';
import { LIGHT, ROOM, SHADOW, sunAt, VIEWS } from 'typegpu-kit-examples/dominoes/scene.ts';
import { createTopple, LIFT, RISE, STEP } from 'typegpu-kit-examples/dominoes/topple.ts';
import { litStage } from './stage.ts';

export const STAGES = ['floor', 'one', 'run', 'shadow', 'falling'] as const;
export type Stage = (typeof STAGES)[number];

export interface Options {
  /** Which step's picture. Default `run`. */
  readonly stage?: Stage;
}

/** The camera close over the run's first domino, for the stage that stands one. */
const CLOSE = { target: [30, 2.4, 116], distance: 15, yaw: -0.6, pitch: 0.28 } as const;
/** Seconds before the push, after the figure starts or the run stands up again. */
const WAIT = 0.6;

export async function createScene(root: TgpuRoot, surface: Surface, options: Options = {}) {
  const { stage = 'run' } = options;
  const shadowed = stage === 'shadow' || stage === 'falling';
  const falling = stage === 'falling';

  const camera = stage === 'one' ? CLOSE : VIEWS.run.camera;
  const lit = litStage(root, surface, {
    camera: {
      ...camera,
      target: [...camera.target],
      fov: 0.6,
      near: 2,
      far: 4000,
      pitchRange: [0.08, 1.4],
      distanceRange: [15, 900],
    },
    samples: 4,
    clear: [...ROOM, 1],
    finish: { tone: tone.aces, settings: { exposure: 0.85, vignette: 0.45 } },
  });
  const { view, screen, final } = lit;
  const light = lighting.create(root, { settings: LIGHT, ...(shadowed ? { shadow: SHADOW } : {}) });
  const scene = lit.lit(light);

  const pool = meshPool.create(root, { capacity: 2 });
  const floor = createFloor(root, pool);
  const share = stage === 'one' ? 1 / LAYOUTS.run.length : 1;
  const dominoes =
    stage === 'floor' ? undefined : createDominoes(root, { pool, layout: 'run', share });

  const sim = clock.fixedStep({ step: STEP, maxSteps: 8 });
  const dust = falling && particles.create(root, view, { capacity: 64, step: STEP });
  const topple =
    dominoes && dust ? createTopple(root, { view, surface, dominoes, dust }) : undefined;

  const drawn = [
    scene.add(floor.renderer()),
    ...(dominoes ? [scene.add(dominoes.set.renderer())] : []),
  ];
  const casters = dominoes && light.shadow ? [light.shadow.add(dominoes.set.renderer())] : [];

  await Promise.all([
    scene.initAsync(),
    light.shadow?.initAsync(),
    final.initAsync(),
    topple?.initAsync(),
    dust && dust.initAsync(),
  ]);

  /** Seconds until the push at the first domino, or -1 once it was pushed. */
  let untilPush = WAIT;
  const onScreen = [0, 0, 0];

  return {
    view,

    /** The dominoes standing, and the draws the scene pass records for them and the floor. */
    counts: { dominoes: dominoes?.set.count ?? 0, draws: drawn.length },

    /** Puts the sun `degrees` above the floor. */
    sun(degrees: number) {
      light.settings.patch({ sunDirection: sunAt(degrees) });
    },

    /** Stands the run up again, and pushes its first domino once it stands. */
    restart() {
      if (!topple || topple.rising) return;
      topple.rise();
      untilPush = RISE + LIFT + 0.2 + WAIT;
    },

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();

      if (topple && dominoes) {
        if (untilPush >= 0) {
          untilPush -= frame.dt;
          if (untilPush < 0 && view.projectAt(dominoes.first, onScreen)) {
            topple.tap(onScreen[0], onScreen[1]);
          }
        }
        const compute = frame.encoder.beginComputePass();
        topple.record(compute, sim.update(frame.dt), frame.dt);
        compute.end();
      }

      light.shadow?.record(frame, casters);
      scene.record(frame, drawn);
      lit.present(frame);
    },

    destroy() {
      topple?.destroy();
      if (dust) dust.destroy();
      dominoes?.set.destroy();
      floor.destroy();
      pool.destroy();
      light.destroy();
      lit.destroy();
    },
  };
}
