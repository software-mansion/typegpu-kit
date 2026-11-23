import { cameras, gltf, materials, pipelines, type Frame, type Surface } from '@typegpu/kit';
import { meshes } from '@typegpu/kit/geometry';
import { d, type TgpuRoot } from 'typegpu';
import type { Loading } from 'typegpu-kit-examples/contract';
import { decoded, type Model } from 'typegpu-kit-examples/models.ts';
import { createLook, type Stages } from './lit-models/look.ts';
import { createProps, type Placement } from './lit-models/props.ts';
import { linear, turn } from './stage.ts';

/** KayKit Medieval Hexagon Pack models by Kay Lousberg (CC0), as the hex isle ships them. */
const MODELS = [
  'grass',
  'church',
  'tavern',
  'market',
  'well',
  'home-a',
  'home-b',
  'blacksmith',
  'tree',
  'trees-medium',
  'trees-large',
  'hills',
  'mountain',
  'rock',
] as const;
type Name = (typeof MODELS)[number];

/** What stands on each hex of a patch two rings wide, by its axial coordinates. */
const PATCH: readonly (readonly [q: number, r: number, model: Name])[] = [
  [0, 0, 'church'],
  [1, 0, 'tavern'],
  [0, 1, 'market'],
  [-1, 1, 'well'],
  [-1, 0, 'home-a'],
  [0, -1, 'home-b'],
  [1, -1, 'tree'],
  [2, 0, 'trees-large'],
  [1, 1, 'blacksmith'],
  [0, 2, 'tree'],
  [-1, 2, 'trees-medium'],
  [-2, 2, 'hills'],
  [-2, 1, 'home-b'],
  [-2, 0, 'trees-large'],
  [-1, -1, 'home-a'],
  [0, -2, 'mountain'],
  [1, -2, 'trees-medium'],
  [2, -2, 'rock'],
  [2, -1, 'home-a'],
];

/** Lanterns on the corners where the church's hex meets two others, in world x and z. */
const LANTERNS = [
  [1, 0.58],
  [0, 1.15],
  [-1, 0.58],
  [0, -1.15],
  [1, -0.58],
] as const;
const POST = { radius: 0.025, height: 0.55 };
const LAMP = materials.glow(d.vec3f(1, 0.62, 0.3), 12);
const IRON = materials.matte(linear(0.09, 0.08, 0.07));

/** The centre of hex `q`, `r` in world units, for KayKit's tiles two units across. */
const hexAt = (q: number, r: number): [number, number, number] => [2 * q + r, 0, Math.sqrt(3) * r];

/** The models, about 1.1 MB. */
export async function load({ read, decode }: Pick<Loading, 'read' | 'decode'>) {
  const model = async (name: Name) => {
    const file = gltf.read(await read(`hex-isle/models/${name}.glb`));
    return [name, await decoded(file, decode)] as const;
  };
  const models = Object.fromEntries(await Promise.all(MODELS.map(model)));
  return { models: models as Record<Name, Model> };
}

export type ModelLookOptions = Awaited<ReturnType<typeof load>>;

/** The stages a reader turns on and off. */
export const STAGES = ['shadows', 'occlusion', 'taa', 'bloom'] as const;
export type Stage = (typeof STAGES)[number];

/** A grass hex and a model on each hex of the patch, and a post and lamp at each lantern. */
function placements(): Placement[] {
  const placed: Placement[] = [];
  for (const [q, r, model] of PATCH) {
    const position = hexAt(q, r);
    placed.push({ model: 'grass', values: { position } });
    // Turn each piece by a sixth so the same house doesn't repeat
    const rotation = turn([0, 1, 0], (((q * 3 + r * 5 + 12) % 6) * Math.PI) / 3);
    placed.push({ model, values: { position, rotation } });
  }
  for (const [x, z] of LANTERNS) {
    placed.push({ model: 'post', values: { position: [x, POST.height / 2, z], ...IRON } });
    placed.push({ model: 'lamp', values: { position: [x, POST.height + 0.05, z], ...LAMP } });
  }
  return placed;
}

/** Radians a second the view turns round the patch while it turns. */
const TURN = 0.08;

/**
 * A patch of a hex village from glTF models at dusk, lit by a low sun with cascaded shadows,
 * ambient occlusion, TAA, bloom and a grade, each of which the reader can turn off.
 */
export async function createScene(root: TgpuRoot, surface: Surface, options: ModelLookOptions) {
  const view = cameras.orbit(root, {
    size: surface,
    target: [0, 0.2, 0.5],
    distance: 14,
    yaw: 0.45,
    pitch: 0.62,
    fov: 0.6,
    pitchRange: [0.25, 1.3],
    distanceRange: [5, 20],
  });
  const stages: Stages = { shadows: true, occlusion: true, taa: true, bloom: true };
  const shapes = {
    post: meshes.cylinder({ radius: POST.radius, height: POST.height }),
    lamp: meshes.sphere({ radius: 0.08 }),
  };
  const props = createProps(root, { ...options.models, ...shapes }, placements());
  const look = createLook(root, surface, view, {
    far: 28,
    crease: 0.7,
    light: {
      sunDirection: [-0.75, 0.4, -0.45],
      sunColor: [1, 0.66, 0.4],
      sunIntensity: 1.6,
      skyTop: [0.16, 0.22, 0.38],
      skyBottom: [0.07, 0.055, 0.045],
      skyIntensity: 1.4,
    },
    stages,
  });
  look.add(props.renderer);
  look.apply();
  await Promise.all([props.ready, pipelines.compile([look])]);

  let turning = true;

  return {
    view,

    /** Turns `stage` on or off. */
    show(stage: Stage, shown: boolean) {
      stages[stage] = shown;
      look.apply();
    },

    /** Starts or stops the view turning round the patch. */
    rotate(rotating: boolean) {
      turning = rotating;
    },

    frame(frame: Frame) {
      if (turning) view.goal.yaw += frame.dt * TURN;
      view.update(frame.dt);
      props.flush();
      look.record(frame);
    },

    destroy() {
      look.destroy();
      props.destroy();
    },
  };
}
