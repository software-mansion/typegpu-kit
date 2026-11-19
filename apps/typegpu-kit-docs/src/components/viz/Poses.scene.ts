import {
  gltf,
  instances,
  lighting,
  materials,
  meshPool,
  poses,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, type TgpuRoot } from 'typegpu';
import { mat4 } from 'wgpu-matrix';
import { decoded, type Model } from 'typegpu-kit-examples/models.ts';
import type { Loading } from 'typegpu-kit-examples/contract';
import { clamp } from '../home/geometry.ts';
import { linear, litStage } from './stage.ts';

const FILES = ['minion', 'rogue', 'blade', 'skeleton-clips', 'melee-clips'] as const;
type File = (typeof FILES)[number];

/** The clips the buttons play, by the names their files give them. */
const CLIPS = {
  idle: 'Skeletons_Idle',
  walk: 'Skeletons_Walking',
  attack: 'Melee_1H_Attack_Chop',
  death: 'Skeletons_Death',
} as const;
export type Clip = keyof typeof CLIPS;
export const CLIP_KEYS = Object.keys(CLIPS) as Clip[];

export const FADE = { min: 0, max: 1, step: 0.05 } as const;
export const SPEED = { min: -2, max: 2, step: 0.1 } as const;
export const START = { clip: 'idle', fade: 0.3, speed: 1 } as const;

/** The files the figure reads, about 1.6 MB. */
export async function load({ read, decode }: Pick<Loading, 'read' | 'decode'>) {
  const model = async (name: File) => {
    const file = gltf.read(await read(`skeletons/models/${name}.glb`));
    return [name, await decoded(file, decode)] as const;
  };
  const models = Object.fromEntries(await Promise.all(FILES.map(model)));
  return { models: models as Record<File, Model> };
}

export type SkinsOptions = Awaited<ReturnType<typeof load>>;

/** A character whose vertices follow the pose its row names. */
const Fighter = instances.define('Fighter', {
  host: { pose: d.u32 },
  pose: (fighter) => {
    'use gpu';
    return fighter.pose;
  },
});

const SAMPLES = 4;

export async function createScene(root: TgpuRoot, surface: Surface, options: SkinsOptions) {
  const { models } = options;
  const rig = models.minion.skin!;
  const read = [...models['skeleton-clips'].clips, ...models['melee-clips'].clips];
  const clips = {} as Record<Clip, poses.ClipData>;
  for (const key of CLIP_KEYS) {
    const found = read.find((clip) => clip.name === CLIPS[key]);
    if (!found) throw new Error(`The clip files have no '${CLIPS[key]}'.`);
    clips[key] = found;
  }
  const loops = (clip: Clip) => clip !== 'death';

  const figures = poses.create(root, rig, clips, { capacity: 1 });
  const pose = figures.add(START.clip, 0);

  const pool = meshPool.create(root, { capacity: 8 });
  const minion = pool.add(models.minion);
  const rogue = pool.add(poses.rebind(models.rogue, models.rogue.skin!, rig));
  const sword = pool.add(
    poses.attach(models.blade, rig, 'handslot.r', { offset: mat4.rotationY(Math.PI) }),
  );
  const ground = pool.add(meshes.plane({ width: 12, depth: 12 }));

  const fighters = instances.create(root, pool, { kind: Fighter, capacity: 8 });
  // glTF characters face +z, towards the camera
  fighters.add({ meshes: [...minion.meshes, ...sword.meshes] }, { pose, position: [-0.9, 0, 0] });
  fighters.add({ meshes: [...rogue.meshes, ...sword.meshes] }, { pose, position: [0.9, 0, 0] });
  const floor = instances.create(root, pool, { capacity: 1 });
  floor.add(ground, materials.matte(linear(0.42, 0.4, 0.36)));

  const light = lighting.create(root, {
    settings: {
      sunDirection: [0.5, 0.9, 0.7],
      sunColor: [1, 0.92, 0.8],
      sunIntensity: 2,
      skyTop: [0.4, 0.48, 0.62],
      skyBottom: [0.16, 0.14, 0.12],
      skyIntensity: 0.7,
    },
    shadow: { size: 1024, extent: 8 },
  });
  const stage = litStage(root, surface, {
    camera: { target: [0, 1, 0], distance: 4.6, yaw: 0.35, pitch: 0.2, pitchRange: [0.02, 1.3] },
    samples: SAMPLES,
  });
  const { view, screen, final } = stage;
  const scene = stage.lit(light);

  const posed = fighters.renderer().pipe(figures.inject());
  const fightersShown = scene.add(posed);
  const cast = light.shadow.add(posed);
  const floorShown = scene.add(floor.renderer());
  const shown = [floorShown, fightersShown];

  await Promise.all([
    minion.ready,
    rogue.ready,
    sword.ready,
    figures.initAsync(),
    scene.initAsync(),
    light.shadow.initAsync(),
    final.initAsync(),
  ]);

  /** What the pose plays: `clip` from second `from` at `start`, at `speed`. */
  const playing = { clip: START.clip as Clip, start: 0, from: 0, speed: START.speed as number };
  let now = 0;

  /** Seconds into its clip the pose is at `time`, held at either end of a clip that doesn't loop. */
  const along = (time: number) => {
    const { clip, start, from, speed } = playing;
    const at = from + (time - start) * speed;
    return loops(clip) ? at : clamp(at, 0, clips[clip].duration);
  };

  return {
    view,

    /**
     * Plays `clip`, blending from what played over `fade` seconds. Backwards, a clip that doesn't
     * loop starts from its end.
     */
    play(clip: Clip, fade: number) {
      const loop = loops(clip);
      const from = !loop && playing.speed < 0 ? clips[clip].duration : 0;
      figures.play(pose, clip, now, { speed: playing.speed, loop, from, fade });
      Object.assign(playing, { clip, start: now, from });
    },

    /** Plays on from where the pose is, at `speed`. */
    speed(speed: number) {
      const from = along(now);
      figures.play(pose, playing.clip, now, { speed, loop: loops(playing.clip), from });
      Object.assign(playing, { start: now, from, speed });
    },

    frame(frame: Frame) {
      now = frame.time;
      view.update(frame.dt);
      screen.fit();
      fighters.flush();
      floor.flush();

      const compute = frame.encoder.beginComputePass();
      figures.animate(compute, now);
      compute.end();
      light.shadow.record(frame, cast);

      scene.record(frame, shown);
      stage.present(frame);
    },

    destroy() {
      fightersShown.destroy();
      cast.destroy();
      floorShown.destroy();
      fighters.destroy();
      floor.destroy();
      figures.destroy();
      pool.destroy();
      light.destroy();
      stage.destroy();
    },
  };
}
