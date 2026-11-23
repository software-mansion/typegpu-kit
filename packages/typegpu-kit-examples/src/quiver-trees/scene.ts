import {
  bloom,
  cameras,
  finish,
  impostors,
  instances,
  lighting,
  materials,
  meshPool,
  passes,
  pipelines,
  targets,
  type Frame,
  type Surface,
} from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, type TgpuRoot } from 'typegpu';
import type { Picture } from '../contract.ts';
import type { Settings } from '../settings.ts';
import { createGround, Ground } from './ground.ts';
import { CLEARING, MODELS, plant, RADIUS, type ModelName } from './plain.ts';

const SAMPLES = 4;
const HDR = 'rgba16float';
const HAZE = [0.71, 0.51, 0.32] as const;
const COUNT = 8_000;
/** Mesh-to-impostor transition distance and transition width in metres from the camera. */
const NEAR = 40;
const FADE = 8;
/** Metres along each side of the ground, past the plain's edge into the haze. */
const GROUND_SIDE = RADIUS * 3;
/** Impostor roughness. Captures do not include the scan textures' roughness. */
const CARD_ROUGHNESS = 0.85;
/** Texels per atlas view. Leaves need more than boulders to hold up where cards meet meshes. */
const CARD_TEXELS: Readonly<Record<ModelName, number>> = {
  'quiver-tree-01': 256,
  'quiver-tree-02': 256,
  'boulder-02': 128,
  'boulder-05': 128,
};

export interface SceneOptions {
  /** Each model as `gltf.read` returns it. */
  readonly models: Readonly<Record<ModelName, meshPool.ModelData>>;
  /** Ground texture with RGB albedo and alpha height. */
  readonly ground: Picture;
  /** Trees and boulders on the plain. Default 8 000. */
  readonly count?: number;
  /**
   * Each impostor atlas's views per side and texels per view. Default 8, and 256 texels for trees
   * and 128 for boulders.
   */
  readonly atlas?: { readonly views?: number; readonly size?: number };
  /** Receives compilation progress as a fraction after each part finishes. */
  readonly onProgress?: (fraction: number) => void;
}

/**
 * Draws nearby trees and boulders as meshes, with captured impostors farther away. GPU culling
 * tests camera and shadow views separately.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: SceneOptions) {
  const { count = COUNT } = options;
  const screen = targets.create(root, surface, {
    scene: { format: HDR, samples: SAMPLES },
    depth: { format: 'depth24plus', samples: SAMPLES },
    hdr: { format: HDR, sampled: true },
  });
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 2.5, 0],
    distance: CLEARING * 0.75,
    yaw: 0.6,
    pitch: 0.12,
    pitchRange: [0.03, 1.3],
    distanceRange: [4, 150],
  });
  const light = lighting.create(root, {
    settings: {
      sunDirection: [0.75, 0.35, 0.45],
      sunColor: [1, 0.82, 0.62],
      sunIntensity: 2.6,
      skyTop: [0.45, 0.55, 0.75],
      skyBottom: [0.35, 0.25, 0.15],
      skyIntensity: 0.5,
      fogColor: HAZE,
      fogDensity: 0.006,
    },
    shadow: { extent: 60 },
  });
  const scene = passes
    .create(
      root,
      screen,
      { scene: passes.color, depth: passes.depth },
      { resolve: { scene: 'hdr' }, clear: [...HAZE, 1] },
    )
    .pipe(view.inject())
    .beneath(light.inject());

  const plain = meshes.plane({ width: GROUND_SIDE, depth: GROUND_SIDE });
  const pool = meshPool.create(
    root,
    // The ground, the models and the card their atlases share
    meshPool.roomFor([plain, ...MODELS.map((name) => options.models[name]), impostors.CARD]),
  );
  const groundMesh = pool.add(plain);
  const models = MODELS.map((name) => pool.add(options.models[name]));
  const cardModel = pool.add(impostors.CARD);
  // Wait for model textures before capturing them
  await Promise.all([...models, cardModel].map((model) => model.ready));
  const card = cardModel.meshes[0];
  const atlases = models.map((model, i) =>
    impostors.create(root, pool, model, {
      card,
      views: 8,
      size: CARD_TEXELS[MODELS[i]],
      ...options.atlas,
    }),
  );
  const capturing = root.createCommandEncoder();
  for (const atlas of atlases) atlas.record(capturing);
  capturing.submit();

  const planted = plant(count);
  const rows = planted.reduce((sum, { model }) => sum + models[model].meshes.length, 0);
  const gravel = createGround(root, options.ground);
  const groundSet = instances.create(root, pool, { kind: Ground, capacity: 1 });
  groundSet.add(groundMesh, materials.matte(d.vec3f(1)));
  const things = instances.create(root, pool, { capacity: rows });
  const cards = models.map((_, i) =>
    instances.create(root, pool, {
      kind: impostors.Impostor,
      capacity: planted.filter(({ model }) => model === i).length,
    }),
  );
  for (const { model, position, rotation, scale, shade } of planted) {
    const placed = { position, rotation, scale, color: [shade, shade, shade] } as const;
    things.add(models[model], placed);
    cards[model].add(card, { ...placed, roughness: CARD_ROUGHNESS });
  }
  const instanceSets = [groundSet, things, ...cards];

  /** A renderer with its drawings in the scene and in the shadow. */
  const lit = (renderer: instances.Renderer) => ({
    renderer,
    shown: scene.add(renderer),
    cast: light.shadow.add(renderer),
  });
  type Lit = ReturnType<typeof lit>;
  const drawingsOf = ({ shown, cast }: Lit) => [shown, cast];

  // The ground casts no shadow, and nothing below it receives one
  const ground = scene.add(groundSet.renderer().pipe(gravel.inject()));

  /** The renderers a frame culls, shadows and draws. */
  const drawList = (renderers: readonly Lit[]) => ({
    culled: renderers.map(({ renderer }) => renderer),
    casters: renderers.map(({ cast }) => cast),
    shown: [ground, ...renderers.map(({ shown }) => shown)],
  });

  const everywhere = lit(things.renderer({ cull: true }));
  const near = lit(things.renderer({ cull: true, to: NEAR, fade: FADE, eye: view }));
  // Cards overlap across the plain, so their depth goes first and each pixel shades once
  const far = { cull: true, from: NEAR, fade: FADE, eye: view, prepass: true } as const;

  /** Adds a card renderer per model reading its atlas by `sampling`. */
  const readAs = (sampling: impostors.Sampling) =>
    cards.map((set, i) => lit(set.renderer(far).pipe(atlases[i].inject(sampling))));

  const glow = bloom.create(root, { source: () => screen.textures.hdr });
  const final = finish.create(root, {
    format: surface.format,
    source: () => screen.textures.hdr,
    glow,
  });

  const values = { impostors: true, sampling: 'blended' as impostors.Sampling };
  // The scene and the shadow compile the first mode's cards with the rest of their drawings
  let withCards = drawList([near, ...readAs(values.sampling)]);
  const readings = new Map([[values.sampling, Promise.resolve(withCards)]]);
  await pipelines.compile([light, scene, glow, final], options);
  const meshesOnly = drawList([everywhere]);

  const settings: Settings<typeof values> = {
    values,
    knobs: { impostors: {}, sampling: { options: ['nearest', 'blended', 'parallax'] } },

    /**
     * Compiles each atlas sampling mode on first use. Cards keep the last mode while
     * compilation runs. Frames read `impostors` directly.
     */
    async write(key) {
      if (key !== undefined && key !== 'sampling') return;
      const { sampling } = values;
      let reading = readings.get(sampling);
      if (!reading) {
        const renderers = readAs(sampling);
        reading = pipelines
          .compile(renderers.flatMap(drawingsOf))
          .then(() => drawList([near, ...renderers]));
        readings.set(sampling, reading);
      }
      const list = await reading;
      if (values.sampling === sampling) withCards = list;
    },
  };

  return {
    view,
    settings,

    frame(frame: Frame) {
      view.update(frame.dt);
      screen.fit();

      for (const set of instanceSets) set.flush();
      const drawn = values.impostors ? withCards : meshesOnly;
      const culling = frame.encoder.beginComputePass();
      for (const renderer of drawn.culled) renderer.cull(culling);
      culling.end();
      light.shadow.record(frame, drawn.casters);

      scene.record(frame, drawn.shown);
      glow.record(frame);
      final.record(frame, surface.view(frame));
    },

    destroy() {
      for (const made of [screen, light, pool, gravel, ...atlases, ...instanceSets, glow]) {
        made.destroy();
      }
    },
  };
}
